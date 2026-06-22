"""Sanity checks runnable without data. Run: python tests/test_core.py"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmp import metrics  # noqa: E402
from fairmp.algorithm import Params, fair_meeting_point  # noqa: E402
from fairmp.baselines import exhaustive_variance, geometric_centroid, min_range  # noqa: E402
from fairmp.runner import run_instance  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import CachedEvaluator, EuclideanBackend  # noqa: E402


def test_metrics_basic():
    assert metrics.variance([10, 10, 10]) == 0
    assert abs(metrics.jain([10, 10, 10]) - 1.0) < 1e-9
    assert abs(metrics.gini([10, 10, 10])) < 1e-9
    assert not metrics.all_reachable([10, math.inf, 5], 3)
    assert metrics.feasible([10, 12], 2, t_max=60)
    assert not metrics.feasible([10, 80], 2, t_max=60)


def test_ede_properties():
    # No inequality: EDE equals the common value.
    assert abs(metrics.kolm_pollak_ede([12, 12, 12]) - 12.0) < 1e-6
    # EDE of an unequal "bad" exceeds the mean, and grows with inequality.
    a = [10, 20, 30]
    b = [5, 20, 35]  # same mean (20), more spread
    assert metrics.kolm_pollak_ede(a) > metrics.mean_time(a)
    assert metrics.kolm_pollak_ede(b) > metrics.kolm_pollak_ede(a)
    # Weighted EDE reduces to the value when all outcomes are equal.
    assert abs(metrics.wkolm_pollak_ede([12, 12, 12], [1, 2, 3]) - 12.0) < 1e-6


def test_eq8_two_user_variance_is_range_squared_over_four():
    # For N=2, variance == ((t1 - t2)/2)^2  (paper Eq 8)
    t1, t2 = 10.0, 20.0
    assert abs(metrics.variance([t1, t2]) - ((t1 - t2) / 2) ** 2) < 1e-9


def test_algorithm_is_fair_and_near_optimal():
    n = 5
    origins = sample_origins("london", n, seed=7, spread="clustered", clusters=1, cluster_sd_deg=0.025)
    modes = assign_modes(n, mix="mixed", seed=7)
    backend = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)

    ev = CachedEvaluator(backend)
    best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p)
    assert best is not None and best.feasible

    ev2 = CachedEvaluator(backend)
    cpt = geometric_centroid(origins, modes, ev2)
    ctimes = [ev2.effective(o, cpt, m) for o, m in zip(origins, modes)]
    cvar = metrics.variance(ctimes)

    ev3 = CachedEvaluator(backend)
    xpt = exhaustive_variance(origins, modes, ev3, res=9)
    xtimes = [ev3.effective(o, xpt, m) for o, m in zip(origins, modes)]
    xvar = metrics.variance(xtimes)

    our_var = metrics.variance(best.times)
    # ground truth is at least as fair as the centroid
    assert xvar <= cvar + 1e-9
    # coarse-to-fine should be close to exhaustive (generous bound for the test)
    assert our_var <= xvar * 1.5 + 1e-6


def test_min_range_minimises_spread():
    n = 5
    origins = sample_origins("london", n, seed=3, spread="clustered", clusters=1, cluster_sd_deg=0.025)
    modes = assign_modes(n, mix="mixed", seed=3)
    backend = EuclideanBackend()
    ev = CachedEvaluator(backend)
    rpt = min_range(origins, modes, ev, res=9)
    rtimes = [ev.effective(o, rpt, m) for o, m in zip(origins, modes)]
    ev2 = CachedEvaluator(backend)
    cpt = geometric_centroid(origins, modes, ev2)
    ctimes = [ev2.effective(o, cpt, m) for o, m in zip(origins, modes)]
    # the range-minimiser must not have a larger range than the centroid
    assert metrics.spread(rtimes) <= metrics.spread(ctimes) + 1e-9


def test_ede_objective_variant_beats_centroid_on_ede():
    n = 5
    origins = sample_origins("london", n, seed=11, spread="clustered", clusters=1, cluster_sd_deg=0.03)
    modes = assign_modes(n, mix="mixed", seed=11)
    backend = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)
    rows = {r["method"]: r for r in run_instance(origins, modes, backend, p, fine_res=9, variants=("ede",))}
    assert "ours_ede" in rows and "exhaustive_ede" in rows
    # the EDE variant beats the centroid on EDE and is close to the EDE ground truth
    assert rows["ours_ede"]["ede"] <= rows["centroid"]["ede"] + 1e-6
    assert rows["ours_ede"]["ede"] <= rows["exhaustive_ede"]["ede"] * 1.5 + 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(name, "PASS")
    print("all tests passed")
