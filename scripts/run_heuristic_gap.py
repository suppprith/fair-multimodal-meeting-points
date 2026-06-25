"""Heuristic optimality gap (paper critique 5): construct an adversarial instance on which
coarse-to-fine search returns a NONZERO gap against exhaustive fine-grid search, and
characterize when that happens.

The paper reports a zero gap on every real instance. That is a measurement, not a
guarantee, and an algorithms reviewer will (rightly) ask when the coarse-to-fine heuristic
can fail. The honest answer is: it fails when the refine budget is starved and the global
fine optimum lies under a coarse cell whose own centroid scores poorly, so that coarse cell
is never refined. This script

  1. searches random instances under a deliberately starved refine budget (small k_refine,
     coarse coarse_res) to find one with a strictly positive gap, then
  2. characterizes recovery: holding the instance fixed, it sweeps k_refine and shows the
     gap closing back to zero as the budget grows, and
  3. reports how often a positive gap occurs across many instances as a function of k_refine,

so the heuristic's failure mode is bounded empirically rather than assumed away.

The characterization that emerges: the gap is largest when the refine budget is starved
(k_refine=1). Raising the budget removes most of it at once, but a residual can survive when
the coarse grid is so coarse that only a handful of cells cover the region and the global
optimum falls in the SEAM between them, which the children-plus-ring-1 of the refined cells
never reach. That residual seam gap closes when the coarse grid is made finer (more cells,
so the optimum's neighbourhood is sampled) or the refinement ring is widened. In absolute
terms the gaps occur on near-zero-variance instances and are sub-minute in standard
deviation, the same rare-instance regime noted for the synthetic sweep.

Backend is Euclidean (data-free); the optimality gap depends only on which candidate cells
are evaluated, not on the source of the travel times, so it carries to the real network.

Run:  python scripts/run_heuristic_gap.py
Out:  outputs/heuristic_gap_instance.csv     (recovery vs refine budget k_refine)
      outputs/heuristic_gap_resolution.csv   (residual seam gap vs coarse_res and ring)
      outputs/heuristic_gap_frequency.csv    (gap frequency vs k_refine over many instances)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fairmp import metrics  # noqa: E402
from fairmp.algorithm import Params, fair_meeting_point  # noqa: E402
from fairmp.baselines import exhaustive_variance  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import CachedEvaluator, EuclideanBackend  # noqa: E402

# Starved search: a coarse coarse grid plus a small refine budget is the regime where
# coarse-to-fine can miss. fine_res matches the exhaustive baseline so the gap is well
# defined and non-negative (our candidates are a subset of the fine grid).
COARSE_RES = 6
FINE_RES = 9
K_C = 400
T_MAX = 240.0
STARVED_K_REFINE = 1


def gap_for(origins, modes, backend, k_refine, k_c=K_C, coarse_res=COARSE_RES, ring=1):
    """Optimality gap (our variance - exhaustive variance, relative) at fine_res."""
    p = Params(coarse_res=coarse_res, fine_res=FINE_RES, k_c=k_c, k_refine=k_refine,
               ring=ring, t_max=T_MAX)
    ev = CachedEvaluator(backend)
    best, _r, _s, _d = fair_meeting_point(origins, modes, ev, p)
    ev2 = CachedEvaluator(backend)
    xpt = exhaustive_variance(origins, modes, ev2, res=FINE_RES)
    xtimes = [ev2.effective(o, xpt, m) for o, m in zip(origins, modes)]
    ours = metrics.variance(best.times)
    xv = metrics.variance(xtimes)
    gap = (ours - xv) / xv if xv > 0 else 0.0
    return gap, ours, xv, ev.calls


def find_gap_instance(backend):
    """First instance (over cities/sizes/seeds) with a strictly positive starved gap."""
    best = None
    for city in ["london", "bengaluru"]:
        for n in [5, 6, 7, 8]:
            for seed in range(60):
                origins = sample_origins(city, n, seed=seed, spread="clustered",
                                         clusters=2, cluster_sd_deg=0.04)
                modes = assign_modes(n, "mixed", seed=seed)
                gap, ours, xv, _ = gap_for(origins, modes, backend, STARVED_K_REFINE)
                if gap > 1e-6:
                    cand = (gap, city, n, seed, origins, modes, ours, xv)
                    if best is None or gap > best[0]:
                        best = cand
                    # a clear gap (>5%) is illustrative enough; stop early
                    if gap > 0.05:
                        return cand
    return best


def main():
    os.makedirs("outputs", exist_ok=True)
    backend = EuclideanBackend()

    found = find_gap_instance(backend)
    if found is None:
        print("No positive-gap instance found in the searched range.")
        return
    gap, city, n, seed, origins, modes, ours, xv = found
    print("Adversarial instance found:")
    print("  city=%s  N=%d  seed=%d  (clustered, 2 clusters)" % (city, n, seed))
    print("  starved params: coarse_res=%d, fine_res=%d, k_c=%d, k_refine=%d"
          % (COARSE_RES, FINE_RES, K_C, STARVED_K_REFINE))
    print("  our variance=%.2f  exhaustive variance=%.2f  gap=%.1f%%\n"
          % (ours, xv, 100 * gap))

    import math

    def std_gap_min(o, x):
        return math.sqrt(o) - math.sqrt(x)  # absolute gap in travel-time std-dev (minutes)

    # 1. Budget recovery: hold the instance fixed, grow the refine budget. The starved
    #    gap collapses at once, then plateaus at the residual seam gap (budget alone
    #    cannot close it because too few coarse cells cover the region).
    rec = []
    for kr in [1, 2, 3, 5, 8, 10, 15, 20, 30]:
        g, o, x, calls = gap_for(origins, modes, backend, kr)
        rec.append({"k_refine": kr, "our_variance": round(o, 3),
                    "exhaustive_variance": round(x, 3), "opt_gap_pct": round(100 * g, 2),
                    "abs_gap_std_min": round(std_gap_min(o, x), 3), "routing_calls": calls})
    rec_df = pd.DataFrame(rec)
    rec_df.to_csv("outputs/heuristic_gap_instance.csv", index=False)

    # 2. Seam-gap recovery: with the budget no longer starved (k_refine=20), the residual
    #    gap closes when the coarse grid is made finer or the refinement ring is widened.
    seam = []
    for cr in [5, 6, 7, 8]:
        g, o, x, calls = gap_for(origins, modes, backend, 20, coarse_res=cr, ring=1)
        seam.append({"knob": "coarse_res", "value": cr, "opt_gap_pct": round(100 * g, 2),
                     "abs_gap_std_min": round(std_gap_min(o, x), 3), "routing_calls": calls})
    for ring in [1, 2, 3]:
        g, o, x, calls = gap_for(origins, modes, backend, 20, coarse_res=COARSE_RES, ring=ring)
        seam.append({"knob": "ring", "value": ring, "opt_gap_pct": round(100 * g, 2),
                     "abs_gap_std_min": round(std_gap_min(o, x), 3), "routing_calls": calls})
    seam_df = pd.DataFrame(seam)
    seam_df.to_csv("outputs/heuristic_gap_resolution.csv", index=False)

    # 3. Frequency: how often does a positive gap occur as k_refine grows, over many
    #    instances? This bounds the failure mode empirically.
    freq = []
    for kr in [1, 2, 3, 5, 10]:
        n_pos, gaps, abs_gaps = 0, [], []
        total = 0
        for fcity in ["london", "bengaluru"]:
            for fn in [5, 6, 7, 8]:
                for fseed in range(40):
                    o = sample_origins(fcity, fn, seed=fseed, spread="clustered",
                                       clusters=2, cluster_sd_deg=0.04)
                    md = assign_modes(fn, "mixed", seed=fseed)
                    g, ov, xv, _c = gap_for(o, md, backend, kr)
                    total += 1
                    if g > 1e-6:
                        n_pos += 1
                        gaps.append(g)
                        abs_gaps.append(std_gap_min(ov, xv))
        freq.append({"k_refine": kr, "instances": total, "positive_gap": n_pos,
                     "positive_gap_pct": round(100 * n_pos / total, 1),
                     "mean_gap_when_positive_pct": round(100 * sum(gaps) / len(gaps), 2)
                     if gaps else 0.0,
                     "max_gap_pct": round(100 * max(gaps), 2) if gaps else 0.0,
                     "max_abs_gap_std_min": round(max(abs_gaps), 3) if abs_gaps else 0.0})
    freq_df = pd.DataFrame(freq)
    freq_df.to_csv("outputs/heuristic_gap_frequency.csv", index=False)

    print("1. Budget recovery (instance fixed; gap drops then plateaus at the seam gap):")
    print(rec_df.to_string(index=False))
    print("\n2. Seam-gap recovery (k_refine=20; finer coarse grid or wider ring closes it):")
    print(seam_df.to_string(index=False))
    print("\n3. Gap frequency across instances vs refine budget:")
    print(freq_df.to_string(index=False))
    print("\nAt the paper's default k_refine=10, positive-gap rate: %.1f%%, max abs gap %.3f min std"
          % (freq_df.loc[freq_df.k_refine == 10, "positive_gap_pct"].iloc[0],
             freq_df.loc[freq_df.k_refine == 10, "max_abs_gap_std_min"].iloc[0]))


if __name__ == "__main__":
    main()
