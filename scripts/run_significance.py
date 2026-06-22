"""Statistical significance: many instances per scenario, 95% CIs, and paired
Wilcoxon signed-rank tests of our method against every baseline.

Backend is Euclidean (data-free, deterministic) so the test can run over 30
instances per scenario quickly; it establishes that the *ranking* is robust, while
the real-network runs (run_real_london.py) confirm the magnitude on real travel
times. Variance objective is tested via "ours"; the EDE objective via "ours_ede".

Run:  python scripts/run_significance.py
Out:  outputs/significance.csv  (per-instance rows)
      outputs/significance_tests.csv  (per-baseline test results)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fairmp.algorithm import Params  # noqa: E402
from fairmp.runner import run_instance  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import EuclideanBackend  # noqa: E402

N_INSTANCES = 30

# (label, city, n, mode-mix, params). Each is one scenario tested independently.
SCENARIOS = [
    ("social_n5", "london", 5, "mixed", Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)),
    ("social_n10", "london", 10, "mixed", Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)),
    ("rideshare_n5", "london", 5, "walking", Params(coarse_res=9, fine_res=10, k_c=400, k_refine=12, t_max=30.0)),
]


def ci95(vals):
    """Mean and half-width of the 95% t confidence interval."""
    v = [x for x in vals if x is not None and math.isfinite(x)]
    n = len(v)
    m = sum(v) / n
    if n < 2:
        return m, float("nan")
    sd = (sum((x - m) ** 2 for x in v) / (n - 1)) ** 0.5
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, h


def paired_test(ours, base):
    """One-sided paired Wilcoxon that ours < base, with the median relative gain."""
    pairs = [(a, b) for a, b in zip(ours, base) if math.isfinite(a) and math.isfinite(b)]
    if not pairs:
        return float("nan"), float("nan")
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    rel = [(y - x) / y for x, y in pairs if y > 0]
    median_gain = sorted(rel)[len(rel) // 2] if rel else float("nan")
    if all(abs(x - y) < 1e-12 for x, y in pairs):
        return float("nan"), median_gain  # identical (e.g. ours vs exhaustive)
    try:
        _stat, p = stats.wilcoxon(a, b, alternative="less", zero_method="wilcox")
    except ValueError:
        p = float("nan")
    return p, median_gain


def main():
    backend = EuclideanBackend()
    rows = []
    for label, city, n, mix, params in SCENARIOS:
        for seed in range(N_INSTANCES):
            origins = sample_origins(city, n, seed=seed, spread="clustered", clusters=1,
                                     cluster_sd_deg=0.03 if mix != "walking" else 0.01)
            modes = assign_modes(n, mix=mix, seed=seed)
            for r in run_instance(origins, modes, backend, params, fine_res=params.fine_res,
                                  variants=("ede",)):
                r.update(scenario=label, seed=seed)
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/significance.csv", index=False)

    test_rows = []
    for label in df["scenario"].unique():
        sub = df[df["scenario"] == label]
        wide = {m: g.sort_values("seed") for m, g in sub.groupby("method")}
        # variance objective -> "ours"; EDE objective -> "ours_ede"
        for our_name, metric in (("ours", "variance"), ("ours_ede", "ede")):
            if our_name not in wide:
                continue
            ours_vals = list(wide[our_name][metric])
            for base in wide:
                if base == our_name:
                    continue
                p, gain = paired_test(ours_vals, list(wide[base][metric]))
                bm, bh = ci95(list(wide[base][metric]))
                om, oh = ci95(ours_vals)
                test_rows.append({
                    "scenario": label, "metric": metric, "ours": our_name, "baseline": base,
                    "ours_mean": round(om, 2), "ours_ci95": round(oh, 2),
                    "baseline_mean": round(bm, 2), "baseline_ci95": round(bh, 2),
                    "median_rel_gain": round(gain, 3), "wilcoxon_p": p})

    tdf = pd.DataFrame(test_rows)
    tdf.to_csv("outputs/significance_tests.csv", index=False)

    pd.set_option("display.width", 160, "display.max_rows", 200)
    print(f"Significance over {N_INSTANCES} instances/scenario (Euclidean).")
    print("\nPer-method means (variance, ede):")
    print(df.groupby(["scenario", "method"])[["variance", "ede", "mean", "max", "jain", "gini"]]
          .mean(numeric_only=True).round(2).to_string())
    print("\nPaired Wilcoxon (ours < baseline), one-sided p and median relative gain:")
    print(tdf.to_string(index=False))


if __name__ == "__main__":
    main()
