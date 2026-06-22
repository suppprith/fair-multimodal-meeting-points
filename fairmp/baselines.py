"""Baselines. Each returns a chosen point (LatLng); the runner evaluates metrics
there. Geometric baselines ignore the evaluator; min-sum / min-max / exhaustive /
random search the candidate grid using the evaluator.
"""
from __future__ import annotations

import math
import random

from . import metrics
from .candidates import polyfill_centroids, region_polygon
from .geo import LatLng, centroid
from .travel_time import EUCLIDEAN_SPEED_KMH


def geometric_centroid(origins, modes_list=None, evaluator=None, **kw):
    return centroid(origins)


def weighted_centroid(origins, modes_list, evaluator=None, **kw):
    """Centroid weighted by inverse nominal mode speed, so slower users pull it closer."""
    ws = [1.0 / max((EUCLIDEAN_SPEED_KMH.get(m, 1.0) for m in modes), default=1.0) for modes in modes_list]
    sw = sum(ws)
    lat = sum(o.lat * w for o, w in zip(origins, ws)) / sw
    lng = sum(o.lng * w for o, w in zip(origins, ws)) / sw
    return LatLng(lat, lng)


def geometric_median(origins, modes_list=None, evaluator=None, iters=200, eps=1e-9, **kw):
    """Weiszfeld iteration on the lat/lng plane (approximate at city scale)."""
    x = sum(o.lng for o in origins) / len(origins)
    y = sum(o.lat for o in origins) / len(origins)
    for _ in range(iters):
        nx = ny = den = 0.0
        for o in origins:
            d = math.hypot(o.lng - x, o.lat - y) or eps
            w = 1.0 / d
            nx += o.lng * w
            ny += o.lat * w
            den += w
        ux, uy = nx / den, ny / den
        if math.hypot(ux - x, uy - y) < eps:
            break
        x, y = ux, uy
    return LatLng(y, x)


def _grid_search(origins, modes_list, evaluator, res, key, bucket="static"):
    n = len(origins)
    best, best_val = None, math.inf
    for _c, pt in polyfill_centroids(region_polygon(origins), res):
        times = [evaluator.effective(o, pt, modes, bucket) for o, modes in zip(origins, modes_list)]
        if not metrics.all_reachable(times, n):
            continue
        v = key(times)
        if v < best_val:
            best_val, best = v, pt
    return best


def min_sum(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    return _grid_search(origins, modes_list, evaluator, res, metrics.total_time, bucket)


def min_max(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    return _grid_search(origins, modes_list, evaluator, res, metrics.max_time, bucket)


def exhaustive_variance(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    """Minimum-variance cell over the full fine grid: the optimality ground truth."""
    return _grid_search(origins, modes_list, evaluator, res, metrics.variance, bucket)


def exhaustive_ede(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    """Minimum-EDE cell over the full fine grid: the ground truth for the EDE objective."""
    return _grid_search(origins, modes_list, evaluator, res, metrics.kolm_pollak_ede, bucket)


def min_range(origins, modes_list, evaluator, res=9, bucket="static", **kw):
    """Minimise the travel-time range (max - min). This is the N-user generalisation of
    Baudru and Bersini's pairwise-difference objective; for N>2 the range is set by a
    single outlier, so variance (whole distribution) is expected to be fairer."""
    return _grid_search(origins, modes_list, evaluator, res, metrics.spread, bucket)


def random_best(origins, modes_list, evaluator, res=9, samples=100, seed=0, bucket="static", **kw):
    rng = random.Random(seed)
    grid = polyfill_centroids(region_polygon(origins), res)
    if not grid:
        return None
    n = len(origins)
    best, best_val = None, math.inf
    for _c, pt in rng.sample(grid, min(samples, len(grid))):
        times = [evaluator.effective(o, pt, modes, bucket) for o, modes in zip(origins, modes_list)]
        if not metrics.all_reachable(times, n):
            continue
        v = metrics.variance(times)
        if v < best_val:
            best_val, best = v, pt
    return best
