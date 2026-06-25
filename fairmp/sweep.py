
from __future__ import annotations

import math
import time
from dataclasses import replace

from . import metrics
from .algorithm import Params, fair_meeting_point
from .baselines import exhaustive_variance
from .baselines import min_sum as min_sum_baseline
from .runner import run_instance
from .scenarios import assign_modes, sample_origins
from .travel_time import CachedEvaluator

def make_instance(city, n, mix="mixed", spread="clustered", seed=0, clusters=1, cluster_sd_deg=0.03):
    origins = sample_origins(city, n, seed=seed, spread=spread, clusters=clusters, cluster_sd_deg=cluster_sd_deg)
    modes = assign_modes(n, mix=mix, seed=seed)
    return origins, modes

def run_sweep(backend, cities, ns, mixes, seeds, params=None, fine_res=9,
              spread="clustered", clusters=1, cluster_sd_deg=0.03):

    rows = []
    for city in cities:
        for n in ns:
            for mix in mixes:
                for seed in seeds:
                    origins, modes = make_instance(city, n, mix, spread, seed, clusters, cluster_sd_deg)
                    for r in run_instance(origins, modes, backend, params, fine_res):
                        r.update(city=city, n=n, mix=mix, seed=seed)
                        rows.append(r)
    return rows

def gamma_sweep(origins, modes, backend, gammas, base_params=None):

    out = []
    for g in gammas:
        p = replace(base_params or Params(), gamma=g)
        ev = CachedEvaluator(backend)
        best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p)
        if best is None:
            continue
        out.append({"gamma": g, "variance": metrics.variance(best.times), "mean": metrics.mean_time(best.times)})
    return out

def pareto_matched_mean(origins, modes, backend, gammas=(0, 0.5, 1, 2, 4, 8, 16, 32, 64),
                        base_params=None, fine_res=9, mean_tol=0.05):

    p = base_params or Params()
    front = gamma_sweep(origins, modes, backend, gammas, p)
    ev = CachedEvaluator(backend)
    mpt = min_sum_baseline(origins, modes, ev, res=fine_res)
    if mpt is None or not front:
        return front, {"mean": math.inf, "variance": math.inf}, None
    mtimes = [ev.effective(o, mpt, m) for o, m in zip(origins, modes)]
    ref = {"mean": metrics.mean_time(mtimes), "variance": metrics.variance(mtimes)}
    if not math.isfinite(ref["mean"]):
        return front, ref, None
    budget = ref["mean"] * (1.0 + mean_tol)
    within = [f for f in front if f["mean"] <= budget] or [min(front, key=lambda f: f["mean"])]
    op = dict(min(within, key=lambda f: f["variance"]))
    op["mean_gap_pct"] = 100.0 * (op["mean"] - ref["mean"]) / ref["mean"] if ref["mean"] else float("nan")
    op["variance_reduction_pct"] = (100.0 * (ref["variance"] - op["variance"]) / ref["variance"]
                                    if ref["variance"] else float("nan"))
    return front, ref, op

def resolution_sweep(origins, modes, backend, fine_reses, base_params=None):

    out = []
    for fr in fine_reses:
        p = replace(base_params or Params(), fine_res=fr)
        ev = CachedEvaluator(backend)
        t0 = time.perf_counter()
        best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p)
        dt = time.perf_counter() - t0
        ev2 = CachedEvaluator(backend)
        xpt = exhaustive_variance(origins, modes, ev2, res=fr)
        xtimes = [ev2.effective(o, xpt, m) for o, m in zip(origins, modes)]
        ourv = metrics.variance(best.times)
        xv = metrics.variance(xtimes)
        out.append({"fine_res": fr, "our_variance": ourv, "exhaustive_variance": xv,
                    "opt_gap": (ourv - xv) / xv if xv > 0 else 0.0,
                    "runtime_s": dt, "routing_calls": ev.calls})
    return out

def size_sweep(backend, city, ns, seeds, base_params=None, fine_res=9):

    out = []
    for n in ns:
        runtimes, calls, gaps = [], [], []
        for seed in seeds:
            origins, modes = make_instance(city, n, seed=seed)
            rows = run_instance(origins, modes, backend, base_params, fine_res)
            ours = next((r for r in rows if r["method"] == "ours"), None)
            if not ours:
                continue
            runtimes.append(ours["runtime_s"])
            calls.append(ours["routing_calls"])
            if "opt_gap" in ours and math.isfinite(ours["opt_gap"]):
                gaps.append(ours["opt_gap"])
        if runtimes:
            out.append({"n": n,
                        "runtime_s": sum(runtimes) / len(runtimes),
                        "routing_calls": sum(calls) / len(calls),
                        "opt_gap": (sum(gaps) / len(gaps)) if gaps else float("nan")})
    return out
