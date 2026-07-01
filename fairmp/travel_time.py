from __future__ import annotations

import glob
import json
import math
import os
import time as _time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod

import h3

from .geo import LatLng, haversine_km


def _bundled_jdk() -> str | None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matches = sorted(glob.glob(os.path.join(root, "tools", "jdk21", "jdk-*")))
    return matches[0] if matches else None


EUCLIDEAN_SPEED_KMH = {"walking": 4.8, "cycling": 15.0, "driving": 25.0, "transit": 18.0}


class Backend(ABC):
    @abstractmethod
    def minutes(self, origin: LatLng, dest: LatLng, mode: str, departure=None) -> float:
        ...


class EuclideanBackend(Backend):
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
    def __init__(self, osm_pbf: str, gtfs: list[str] | None = None):
        jdk = _bundled_jdk()
        if jdk and os.path.isdir(jdk):
            os.environ.setdefault("JAVA_HOME", jdk)
            os.environ["PATH"] = os.path.join(jdk, "bin") + os.pathsep + os.environ.get("PATH", "")
        try:
            import r5py  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "r5py not available. Install r5py and a JDK 21, then provide OSM/GTFS. "
                "See scripts/DATA.md."
            ) from e
        import r5py

        self._r5py = r5py
        self.network = r5py.TransportNetwork(osm_pbf, gtfs or [])

    def minutes(self, origin, dest, mode, departure=None):
        raise NotImplementedError(
            "Use matrix-based evaluation with r5py; per-pair minutes() is intentionally "
            "not implemented for the real backend. Build a TravelTimeMatrixComputer over "
            "the candidate set instead."
        )


class PerceptionBackend(Backend):
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
        out_clock = min(self.access_min + self.wait_min * n_tr, t)
        in_clock = t - out_clock
        return in_clock + self.alpha * out_clock + self.delta * n_tr


TFL_MODE = {
    "walking": "walking",
    "cycling": "cycling",
    "driving": "driving",
    "transit": "tube,bus,national-rail,dlr,overground,elizabeth-line,tram,river-bus",
}


class TflBackend(Backend):
    def __init__(self, app_key: str | None = None, retries: int = 3,
                 timeout: float = 20.0, backoff: float = 2.0, pause: float = 0.12,
                 user_agent: str = "Mozilla/5.0 (X11; Linux x86_64) fairmp-research/1.0"):
        self.app_key = app_key or os.environ.get("TFL_APP_KEY")
        self.retries = retries
        self.timeout = timeout
        self.backoff = backoff
        self.pause = pause
        self.user_agent = user_agent

    def minutes(self, origin, dest, mode, departure=None):
        tfl_mode = TFL_MODE.get(mode)
        if not tfl_mode:
            return math.inf
        params = {"mode": tfl_mode}
        if self.app_key:
            params["app_key"] = self.app_key
        if departure is not None:
            params["date"] = departure.strftime("%Y%m%d")
            params["time"] = departure.strftime("%H%M")
            params["timeIs"] = "Departing"
        url = (f"https://api.tfl.gov.uk/Journey/JourneyResults/"
               f"{origin.lat},{origin.lng}/to/{dest.lat},{dest.lng}?"
               + urllib.parse.urlencode(params))
        req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                journeys = data.get("journeys") or []
                if self.pause:
                    _time.sleep(self.pause)
                if not journeys:
                    return math.inf
                return float(min(j["duration"] for j in journeys))
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    _time.sleep(self.backoff * (attempt + 1))
                    continue
                return math.inf
            except Exception:
                _time.sleep(self.backoff)
        return math.inf


def cell_key(p: LatLng, res: int = 8) -> str:
    return h3.latlng_to_cell(p.lat, p.lng, res)


R5_MODE = {
    "walking": ["WALK"],
    "cycling": ["BICYCLE"],
    "driving": ["CAR"],
    "transit": ["TRANSIT", "WALK"],
}


class PrecomputedBackend(Backend):
    def __init__(self, cache_res: int = 12):
        self.cache_res = cache_res
        self.table: dict[tuple, float] = {}

    def put(self, mode: str, origin: LatLng, dest: LatLng, minutes: float):
        self.table[(mode, cell_key(origin, self.cache_res), cell_key(dest, self.cache_res))] = minutes

    def minutes(self, origin, dest, mode, departure=None):
        return self.table.get(
            (mode, cell_key(origin, self.cache_res), cell_key(dest, self.cache_res)), math.inf)


class CachedEvaluator:
    def __init__(self, backend: Backend, cache_res: int = 12):
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
        best = math.inf
        for m in modes:
            t = self._one(origin, dest, m, bucket)
            if t < best:
                best = t
        return best
