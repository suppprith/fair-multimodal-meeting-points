"""Fair meeting point: coarse-to-fine search over H3 candidates, variance objective.

Coarse-to-fine search with a variance (or Kolm-Pollak EDE) objective; see the paper, Section 4.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Point

from . import metrics
from .candidates import polyfill_centroids, prefilter, refine_cells, region_polygon
from .geo import LatLng, haversine_km


@dataclass
class Params:
    coarse_res: int = 7
    fine_res: int = 9
    k_c: int = 200       # coarse candidates kept after the geometric prefilter
    k_refine: int = 10   # top coarse cells to refine
    ring: int = 1
    t_max: float = 60.0
    gamma: float = 0.0   # 0 = pure variance; >0 adds the mean term (Eq 9)
    objective: str = "variance"  # "variance" or "ede" (Kolm-Pollak); search is metric-agnostic
    ede_epsilon: float = metrics.EDE_EPSILON
    runners_up: int = 2
    min_sep_km: float = 1.0


@dataclass
class Area:
    cell: str
    point: LatLng
    times: list
    objective: float
    feasible: bool


def _score(cell, point, origins, modes_list, ev, p, bucket, weights=None) -> Area:
    times = [ev.effective(o, point, modes, bucket) for o, modes in zip(origins, modes_list)]
    n = len(origins)
    if not metrics.all_reachable(times, n):
        obj = math.inf
    elif p.objective == "ede":
        ede = (metrics.wkolm_pollak_ede(times, weights, p.ede_epsilon) if weights is not None
               else metrics.kolm_pollak_ede(times, p.ede_epsilon))
        obj = ede + p.gamma * (metrics.wmean(times, weights) if weights is not None else metrics.mean_time(times))
    elif weights is None:
        obj = metrics.objective(times, n, p.gamma)
    else:
        obj = metrics.wvariance(times, weights) + p.gamma * metrics.wmean(times, weights)
    return Area(cell, point, times, obj, metrics.feasible(times, n, p.t_max))


def fair_meeting_point(origins, modes_list, evaluator, params: Params | None = None,
                       is_dead=None, bucket: str = "static", weights=None):
    """Returns (best Area, runners-up [Area], all scored [Area], diagnostics dict).

    weights (optional, one per origin) makes the objective a demand-weighted
    variance, used for facility siting (dark stores). None = unweighted."""
    p = params or Params()

    poly = region_polygon(origins)
    coarse = polyfill_centroids(poly, p.coarse_res)
    if is_dead:
        coarse = [(c, pt) for c, pt in coarse if not is_dead(c)]
    coarse = prefilter(coarse, origins, p.k_c)

    coarse_scored = [_score(c, pt, origins, modes_list, evaluator, p, bucket, weights) for c, pt in coarse]
    seen = {a.cell for a in coarse_scored}

    fine_scored: list[Area] = []
    finite = sorted((a for a in coarse_scored if math.isfinite(a.objective)), key=lambda a: a.objective)
    for a in finite[: p.k_refine]:
        for c, pt in refine_cells(a.cell, p.fine_res, p.ring):
            if c in seen:
                continue
            seen.add(c)
            if is_dead and is_dead(c):
                continue
            # keep refinement inside the (buffered) region so candidates stay within
            # the fine grid that the exhaustive baseline enumerates
            if not poly.contains(Point(pt.lng, pt.lat)):
                continue
            fine_scored.append(_score(c, pt, origins, modes_list, evaluator, p, bucket, weights))

    all_scored = coarse_scored + fine_scored
    # The answer is a fine cell; the coarse pass only chooses which regions to refine.
    # This keeps our candidates a subset of the exhaustive fine grid, so the optimality
    # gap is non-negative. Fall back to coarse only if nothing was refined.
    src = fine_scored if fine_scored else coarse_scored
    feasible = [a for a in src if a.feasible]
    pool = feasible if feasible else [a for a in src if math.isfinite(a.objective)]
    diag = {"coarse": len(coarse), "evaluated": len(all_scored), "routing_calls": evaluator.calls}
    if not pool:
        return None, [], all_scored, diag

    pool.sort(key=lambda a: a.objective)
    best = pool[0]
    runners: list[Area] = []
    for a in pool[1:]:
        if len(runners) >= p.runners_up:
            break
        if all(haversine_km(a.point, b.point) >= p.min_sep_km for b in [best, *runners]):
            runners.append(a)
    return best, runners, all_scored, diag
