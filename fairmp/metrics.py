
from __future__ import annotations

import math

def _reachable(times: list[float]) -> list[float]:
    return [t for t in times if t is not None and math.isfinite(t)]

def mean_time(times: list[float]) -> float:
    r = _reachable(times)
    return sum(r) / len(r) if r else math.inf

def variance(times: list[float]) -> float:

    r = _reachable(times)
    if not r:
        return math.inf
    m = sum(r) / len(r)
    return sum((t - m) ** 2 for t in r) / len(r)

def spread(times: list[float]) -> float:

    r = _reachable(times)
    return (max(r) - min(r)) if r else math.inf

def max_time(times: list[float]) -> float:
    r = _reachable(times)
    return max(r) if r else math.inf

def total_time(times: list[float]) -> float:
    r = _reachable(times)
    return sum(r) if r else math.inf

def jain(times: list[float]) -> float:

    r = _reachable(times)
    if not r:
        return 0.0
    s = sum(r)
    sq = sum(t * t for t in r)
    return (s * s) / (len(r) * sq) if sq > 0 else 1.0

def gini(times: list[float]) -> float:

    r = sorted(_reachable(times))
    n = len(r)
    if n == 0:
        return math.inf
    s = sum(r)
    if s == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(r))
    return (2 * cum) / (n * s) - (n + 1) / n

def theil(times: list[float]) -> float:

    r = _reachable(times)
    n = len(r)
    if n == 0:
        return math.inf
    mu = sum(r) / n
    if mu <= 0:
        return 0.0
    return sum((t / mu) * math.log(t / mu) for t in r if t > 0) / n

def maxmin_ratio(times: list[float]) -> float:
    r = _reachable(times)
    if not r:
        return math.inf
    lo = min(r)
    return (max(r) / lo) if lo > 0 else math.inf

EDE_EPSILON = 0.5

def _kp_kappa(sum_x: float, sum_x2: float, epsilon: float) -> float:

    return epsilon * sum_x / sum_x2 if sum_x2 > 0 else 0.0

def kolm_pollak_ede(times: list[float], epsilon: float = EDE_EPSILON) -> float:

    r = _reachable(times)
    n = len(r)
    if n == 0:
        return math.inf
    sum_x, sum_x2 = math.fsum(r), math.fsum(t * t for t in r)
    k = _kp_kappa(sum_x, sum_x2, epsilon)
    if k == 0:
        return 0.0

    kx = [k * t for t in r]
    m = max(kx)
    lse = m + math.log(math.fsum(math.exp(v - m) for v in kx))
    return (lse - math.log(n)) / k

def wkolm_pollak_ede(times, weights, epsilon: float = EDE_EPSILON) -> float:

    p = _pairs(times, weights)
    if not p:
        return math.inf
    w_tot = math.fsum(w for _t, w in p)
    sum_x = math.fsum(w * t for t, w in p)
    sum_x2 = math.fsum(w * t * t for t, w in p)
    k = _kp_kappa(sum_x, sum_x2, epsilon)
    if k == 0:
        return 0.0
    kx = [k * t for t, _w in p]
    m = max(kx)
    lse = m + math.log(math.fsum((w / w_tot) * math.exp(v - m) for v, (_t, w) in zip(kx, p)))
    return lse / k

def all_reachable(times: list[float], n: int) -> bool:
    return len(_reachable(times)) == n and n > 0

def feasible(times: list[float], n: int, t_max: float) -> bool:
    return all_reachable(times, n) and max_time(times) <= t_max

def objective(times: list[float], n: int, gamma: float = 0.0) -> float:

    if not all_reachable(times, n):
        return math.inf
    return variance(times) + gamma * mean_time(times)

def _pairs(times, weights):
    return [(t, w) for t, w in zip(times, weights) if t is not None and math.isfinite(t) and w > 0]

def wmean(times, weights) -> float:
    p = _pairs(times, weights)
    w = sum(x[1] for x in p)
    return sum(t * x for t, x in p) / w if w > 0 else math.inf

def wvariance(times, weights) -> float:

    p = _pairs(times, weights)
    w = sum(x[1] for x in p)
    if w <= 0:
        return math.inf
    m = sum(t * x for t, x in p) / w
    return sum(x * (t - m) ** 2 for t, x in p) / w

def percentile(times, q: float) -> float:

    r = sorted(_reachable(times))
    if not r:
        return math.inf
    k = (len(r) - 1) * q / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return r[int(k)]
    return r[lo] + (r[hi] - r[lo]) * (k - lo)

def wshare_within(times, weights, threshold: float) -> float:

    p = _pairs(times, weights)
    w = sum(x for _t, x in p)
    return sum(x for t, x in p if t <= threshold) / w if w > 0 else 0.0

def summarize(times: list[float], n: int, t_max: float, gamma: float = 0.0) -> dict:
    return {
        "variance": variance(times),
        "mean": mean_time(times),
        "max": max_time(times),
        "total": total_time(times),
        "spread": spread(times),
        "jain": jain(times),
        "gini": gini(times),
        "theil": theil(times),
        "ede": kolm_pollak_ede(times),
        "maxmin": maxmin_ratio(times),
        "feasible": feasible(times, n, t_max),
        "objective": objective(times, n, gamma),
    }
