"""Experiment harness: run our method and the baselines on an instance, collect
chosen points, per-user times, metrics, optimality gap, runtime, and routing-query
count. Each method gets a fresh cache so query counts are comparable.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

from . import baselines, metrics
from .algorithm import Params, fair_meeting_point
from .travel_time import CachedEvaluator

BASELINES = {
    "centroid": baselines.geometric_centroid,
    "weighted_centroid": baselines.weighted_centroid,
    "geometric_median": baselines.geometric_median,
    "min_sum": baselines.min_sum,
    "min_max": baselines.min_max,
    "min_range": baselines.min_range,
    "random": baselines.random_best,
    "exhaustive": baselines.exhaustive_variance,
}

# Ground-truth baselines that are only meaningful for a specific objective variant.
VARIANT_GROUND_TRUTH = {"ede": ("exhaustive_ede", baselines.exhaustive_ede)}


def solve_all(origins, modes_list, backend, params: Params | None = None, fine_res=9,
              variants=()):
    """Run every method once; return {method: {point, times, runtime_s, routing_calls, evaluated}}.

    variants names extra objective-variant runs of our algorithm (e.g. ("ede",)); each
    is reported as ours_<variant> and pulls in its exhaustive ground truth."""
    p = params or Params()
    out = {}

    def run_ours(label, prm):
        ev = CachedEvaluator(backend)
        t0 = time.perf_counter()
        best, _r, _s, diag = fair_meeting_point(origins, modes_list, ev, prm)
        dt = time.perf_counter() - t0
        if best is not None:
            out[label] = {"point": best.point, "times": best.times, "runtime_s": dt,
                          "routing_calls": ev.calls, "evaluated": diag["evaluated"]}

    run_ours("ours", p)
    for v in variants:
        run_ours(f"ours_{v}", replace(p, objective=v))

    fns = dict(BASELINES)
    for v in variants:
        if v in VARIANT_GROUND_TRUTH:
            name, fn = VARIANT_GROUND_TRUTH[v]
            fns[name] = fn
    for name, fn in fns.items():
        ev = CachedEvaluator(backend)
        t0 = time.perf_counter()
        pt = fn(origins, modes_list, ev, res=fine_res)
        dt = time.perf_counter() - t0
        if pt is None:
            continue
        times = [ev.effective(o, pt, m) for o, m in zip(origins, modes_list)]
        out[name] = {"point": pt, "times": times, "runtime_s": dt,
                     "routing_calls": ev.calls, "evaluated": None}
    return out


def rows_from_results(results, n, t_max, gamma):
    rows = []
    for name, r in results.items():
        m = metrics.summarize(r["times"], n, t_max, gamma)
        m.update(method=name, runtime_s=r["runtime_s"], routing_calls=r["routing_calls"],
                 evaluated=r["evaluated"])
        rows.append(m)
    base = next((m["variance"] for m in rows if m["method"] == "exhaustive"), None)
    if base not in (None, 0, math.inf):
        for m in rows:
            if math.isfinite(m["variance"]):
                m["opt_gap"] = (m["variance"] - base) / base
    ede_base = next((m["ede"] for m in rows if m["method"] == "exhaustive_ede"), None)
    if ede_base not in (None, 0, math.inf):
        for m in rows:
            if math.isfinite(m["ede"]):
                m["ede_gap"] = (m["ede"] - ede_base) / ede_base
    return rows


def run_instance(origins, modes_list, backend, params: Params | None = None, fine_res=9,
                 variants=()):
    p = params or Params()
    results = solve_all(origins, modes_list, backend, p, fine_res, variants)
    return rows_from_results(results, len(origins), p.t_max, p.gamma)
