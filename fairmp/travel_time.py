"""Travel-time backends and caching.

Results are memoised by
(mode, origin cell, dest cell, time bucket), and the effective time of a user is
the fastest over the user's available modes. EuclideanBackend needs no data and
is for testing; R5Backend uses real OSM + GTFS via r5py and is for results.
"""
from __future__ import annotations

import glob
import math
import os
from abc import ABC, abstractmethod

import h3

from .geo import LatLng, haversine_km


def _bundled_jdk() -> str | None:
    """Path to the portable JDK 21 under experiments/tools/jdk21, if present."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = sorted(glob.glob(os.path.join(root, "tools", "jdk21", "jdk-*")))
    return matches[0] if matches else None

# Rough mode speeds (km/h) for the data-free backend. A stand-in only; real
# travel times come from r5py. Driving/transit are deliberately modest to reflect
# urban conditions, walking/cycling are physical pace.
EUCLIDEAN_SPEED_KMH = {"walking": 4.8, "cycling": 15.0, "driving": 25.0, "transit": 18.0}


class Backend(ABC):
    @abstractmethod
    def minutes(self, origin: LatLng, dest: LatLng, mode: str, departure=None) -> float:
        """Travel minutes by a single mode, or math.inf if unreachable."""


class EuclideanBackend(Backend):
    """Straight-line distance, inflated by a detour factor, divided by mode speed.

    Not for results. It exists so the whole pipeline runs and can be validated
    without downloading networks. Swap in R5Backend for the paper.
    """

    def __init__(self, speeds: dict | None = None, detour: float = 1.3):
        self.speeds = speeds or dict(EUCLIDEAN_SPEED_KMH)
        self.detour = detour

    def minutes(self, origin, dest, mode, departure=None):
        spd = self.speeds.get(mode)
        if not spd:
            return math.inf
        km = haversine_km(origin, dest) * self.detour
        return km / spd * 60.0


class R5Backend(Backend):
    """Real multimodal travel times via r5py (OSM + GTFS).

    Requires a JDK 21 on PATH and r5py installed. This is a scaffold: the r5py
    matrix API is the efficient path, so prefer building per-origin matrices
    through `matrix()` rather than per-pair `minutes()`. Verify against the
    installed r5py version when data is in place (see scripts/DATA.md).
    """

    def __init__(self, osm_pbf: str, gtfs: list[str] | None = None):
        jdk = _bundled_jdk()
        if jdk and os.path.isdir(jdk):
            os.environ.setdefault("JAVA_HOME", jdk)
            os.environ["PATH"] = os.path.join(jdk, "bin") + os.pathsep + os.environ.get("PATH", "")
        try:
            import r5py  # noqa: F401
        except Exception as e:  # pragma: no cover - depends on env
            raise RuntimeError(
                "r5py not available. Install r5py and a JDK 21, then provide OSM/GTFS. "
                "See scripts/DATA.md."
            ) from e
        import r5py

        self._r5py = r5py
        self.network = r5py.TransportNetwork(osm_pbf, gtfs or [])

    def minutes(self, origin, dest, mode, departure=None):  # pragma: no cover
        raise NotImplementedError(
            "Use matrix-based evaluation with r5py; per-pair minutes() is intentionally "
            "not implemented for the real backend. Build a TravelTimeMatrixComputer over "
            "the candidate set instead."
        )


class PerceptionBackend(Backend):
    """Applies the perception-weighted cost of Eq 2 to schedule-based (transit) trips:

        perceived = t_in + alpha * t_out + delta * n_transfers,

    where t_out is out-of-vehicle time (access, egress, waiting) and n_transfers grows
    with trip distance. Walking, cycling, and driving pass through unchanged. The clock
    time of the wrapped backend is split into an out-of-vehicle part and an in-vehicle
    line-haul, so alpha=1, delta=0 reproduces clock time exactly, the uncalibrated
    setting used for the headline runs. Sweeping alpha and delta over this wrapper tests
    whether the method's ranking is stable across the plausible perception range.
    """

    def __init__(self, base: Backend, alpha: float = 1.0, delta: float = 0.0,
                 access_min: float = 6.0, wait_min: float = 3.0,
                 transfer_km: float = 5.0, max_transfers: int = 3):
        self.base = base
        self.alpha = alpha
        self.delta = delta
        self.access_min = access_min
        self.wait_min = wait_min
        self.transfer_km = transfer_km
        self.max_transfers = max_transfers

    def minutes(self, origin, dest, mode, departure=None):
        t = self.base.minutes(origin, dest, mode, departure)
        if mode != "transit" or not math.isfinite(t):
            return t
        d = haversine_km(origin, dest)
        n_tr = min(int(d // self.transfer_km), self.max_transfers)
        # clamp so the clock split never exceeds the total; keeps alpha=1,delta=0 == t
        out_clock = min(self.access_min + self.wait_min * n_tr, t)
        in_clock = t - out_clock
        return in_clock + self.alpha * out_clock + self.delta * n_tr


def cell_key(p: LatLng, res: int = 8) -> str:
    return h3.latlng_to_cell(p.lat, p.lng, res)


# Our modes -> r5py TransportMode names (transit needs walk access/egress).
R5_MODE = {
    "walking": ["WALK"],
    "cycling": ["BICYCLE"],
    "driving": ["CAR"],
    "transit": ["TRANSIT", "WALK"],
}


class PrecomputedBackend(Backend):
    """Serves travel times from a precomputed table keyed by (mode, origin cell,
    dest cell). Built once per instance from r5py batch matrices, then queried as
    O(1) lookups so the existing algorithm/baselines run unchanged on real times.
    """

    def __init__(self, cache_res: int = 12):
        self.cache_res = cache_res
        self.table: dict[tuple, float] = {}

    def put(self, mode: str, origin: LatLng, dest: LatLng, minutes: float):
        self.table[(mode, cell_key(origin, self.cache_res), cell_key(dest, self.cache_res))] = minutes

    def minutes(self, origin, dest, mode, departure=None):
        return self.table.get(
            (mode, cell_key(origin, self.cache_res), cell_key(dest, self.cache_res)), math.inf)


class CachedEvaluator:
    """Memoises backend calls by (mode, origin cell, dest cell, bucket) and returns
    the fastest-mode effective time. Counts backend calls for the cost metric."""

    def __init__(self, backend: Backend, cache_res: int = 12):
        # cache_res must be finer than the search resolution (fine_res), otherwise
        # distinct fine candidates collapse to one cache cell and share a travel time,
        # which silently coarsens the search. 12 (~9 m cells) is safe for fine_res <= 11.
        self.backend = backend
        self.cache_res = cache_res
        self._cache: dict[tuple, float] = {}
        self.calls = 0

    def _one(self, origin: LatLng, dest: LatLng, mode: str, bucket: str) -> float:
        key = (mode, cell_key(origin, self.cache_res), cell_key(dest, self.cache_res), bucket)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        v = self.backend.minutes(origin, dest, mode)
        self._cache[key] = v
        return v

    def effective(self, origin: LatLng, dest: LatLng, modes, bucket: str = "static") -> float:
        """tau_i(P): the fastest available mode (Eq 1)."""
        best = math.inf
        for m in modes:
            t = self._one(origin, dest, m, bucket)
            if t < best:
                best = t
        return best
