"""Vertical-equity demonstration: the per-user weights of the objective let an operator
up-weight a low-mobility user so the fair point moves toward them and their travel time
falls. Backend is Euclidean (the effect is a property of the objective, not the times).

Run:  python scripts/run_vertical_equity.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairmp.algorithm import Params, fair_meeting_point  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import CachedEvaluator, EuclideanBackend  # noqa: E402


def main():
    b = EuclideanBackend()
    p = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)
    print(f"{'seed':>4} {'worst-off time':>14} {'after up-weight x4':>19} {'drop':>6}")
    drops = []
    for seed in range(6):
        o = sample_origins("london", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.03)
        m = assign_modes(5, "mixed", seed=seed)
        ev = CachedEvaluator(b)
        best, _r, _s, _d = fair_meeting_point(o, m, ev, p)
        i = max(range(5), key=lambda k: best.times[k])  # the worst-off user
        w = [1, 1, 1, 1, 1]
        w[i] = 4
        ev2 = CachedEvaluator(b)
        best2, _r, _s, _d = fair_meeting_point(o, m, ev2, p, weights=w)
        t, tw = best.times[i], best2.times[i]
        drops.append(t - tw)
        print(f"{seed:>4} {t:>14.1f} {tw:>19.1f} {t - tw:>6.1f}")
    print(f"\nmean drop in the up-weighted (worst-off) user's travel time: {sum(drops) / len(drops):.1f} min")


if __name__ == "__main__":
    main()
