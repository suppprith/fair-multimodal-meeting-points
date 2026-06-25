
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp.runner import run_instance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import EuclideanBackend, PerceptionBackend

ALPHAS = [1.0, 1.5, 2.0]
DELTAS = [0.0, 5.0, 10.0]
CITIES = ["london", "bengaluru"]
N = 6
SEEDS = list(range(12))

RIVALS = ["centroid", "geometric_median", "weighted_centroid", "min_sum", "min_max", "min_range"]

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")

def main():
    os.makedirs("outputs", exist_ok=True)
    base = EuclideanBackend()
    rows = []

    for alpha in ALPHAS:
        for delta in DELTAS:
            backend = PerceptionBackend(base, alpha=alpha, delta=delta)
            per_method = {}
            jain = {}
            for city in CITIES:
                for seed in SEEDS:
                    origins = sample_origins(city, N, seed=seed, spread="clustered",
                                             clusters=1, cluster_sd_deg=0.03)
                    modes = assign_modes(N, "mixed", seed=seed)
                    for r in run_instance(origins, modes, backend, fine_res=9):
                        per_method.setdefault(r["method"], []).append(r["variance"])
                        jain.setdefault(r["method"], []).append(r["jain"])

            ours = _mean(per_method.get("ours", []))
            rivals_mean = {m: _mean(per_method.get(m, [])) for m in RIVALS}

            ours_best = all(ours <= v for v in rivals_mean.values())
            worst_rival = max(rivals_mean, key=rivals_mean.get)
            rows.append({
                "alpha": alpha,
                "delta": delta,
                "ours_variance": round(ours, 2),
                "ours_jain": round(_mean(jain.get("ours", [])), 3),
                "centroid_variance": round(rivals_mean["centroid"], 2),
                "min_sum_variance": round(rivals_mean["min_sum"], 2),
                "min_range_variance": round(rivals_mean["min_range"], 2),
                "reduction_vs_centroid_pct": round(100 * (rivals_mean["centroid"] - ours)
                                                   / rivals_mean["centroid"], 1),
                "reduction_vs_min_sum_pct": round(100 * (rivals_mean["min_sum"] - ours)
                                                  / rivals_mean["min_sum"], 1),
                "reduction_vs_min_range_pct": round(100 * (rivals_mean["min_range"] - ours)
                                                    / rivals_mean["min_range"], 1),
                "ours_lowest_variance": ours_best,
                "closest_rival": worst_rival if not ours_best else "-",
            })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/perception_sensitivity.csv", index=False)
    print("Perception-weight sensitivity (mean over %d instances/grid point)\n"
          % (len(CITIES) * len(SEEDS)))
    print(df.to_string(index=False))
    stable = df["ours_lowest_variance"].all()
    print("\nRanking stable across the whole alpha-delta grid:", bool(stable))
    print("Our variance reduction vs centroid ranges %.1f%%-%.1f%% across the grid."
          % (df["reduction_vs_centroid_pct"].min(), df["reduction_vs_centroid_pct"].max()))
    print("Our variance reduction vs min-sum ranges %.1f%%-%.1f%% across the grid."
          % (df["reduction_vs_min_sum_pct"].min(), df["reduction_vs_min_sum_pct"].max()))

if __name__ == "__main__":
    main()
