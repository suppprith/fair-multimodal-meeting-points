"""Statistical significance: 100 instances per scenario, 95% CIs, and paired Wilcoxon
signed-rank tests of our method against every baseline, across all three scenario
families (social meetup, ride-share pickup, dark-store siting) and two cities.

Backend is Euclidean (data-free, deterministic) so the test can run over 100 instances
per scenario quickly; it establishes that the *ranking* holds, while the real-network
runs (run_real_*.py) confirm the magnitude on real travel times. For the meeting-point
scenarios the variance objective is tested via "ours" and the EDE objective via
"ours_ede"; for dark-store siting the demand-weighted variance ("ours" on w_variance) and
weighted EDE ("ours_ede" on w_ede) are tested against the coverage-max, weighted-centroid,
and min-sum baselines.

Run:  python scripts/run_significance.py
Out:  outputs/significance.csv         (meeting-point per-instance rows)
      outputs/significance_darkstore.csv (dark-store per-instance rows)
      outputs/significance_tests.csv    (per-baseline test results, all scenarios)
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from fairmp import darkstore  # noqa: E402
from fairmp.algorithm import Params  # noqa: E402
from fairmp.runner import run_instance  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import EuclideanBackend  # noqa: E402

N_INSTANCES = 100

# Meeting-point scenarios (variance + EDE via run_instance).
# (label, city, n, mode-mix, params). Tested on variance ("ours") and ede ("ours_ede").
MP_SCENARIOS = [
    ("social_london_n5", "london", 5, "mixed", Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)),
    ("social_london_n10", "london", 10, "mixed", Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)),
    ("social_bengaluru_n5", "bengaluru", 5, "mixed", Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)),
    ("rideshare_london_n5", "london", 5, "walking", Params(coarse_res=9, fine_res=10, k_c=400, k_refine=12, t_max=30.0)),
]

# Dark-store siting scenarios (weighted variance + weighted EDE via run_darkstore_instance).
# (label, city, n_demand_cells, params).
DS_SCENARIOS = [
    ("darkstore_london", "london", 40, Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0)),
    ("darkstore_bengaluru", "bengaluru", 40, Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0)),
]

# Which (our-method label, metric column) pairs to test, per scenario family.
MP_TESTS = [("ours", "variance"), ("ours_ede", "ede")]
DS_TESTS = [("ours", "w_variance"), ("ours_ede", "w_ede")]


def ci95(vals):
    """Mean and half-width of the 95% t confidence interval."""
    v = [x for x in vals if x is not None and math.isfinite(x)]
    n = len(v)
    if n == 0:
        return float("nan"), float("nan")
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


def tests_for(df, label, our_metric_pairs):
    """Paired tests of each our-objective against every other method in one scenario."""
    sub = df[df["scenario"] == label]
    wide = {m: g.sort_values("seed") for m, g in sub.groupby("method")}
    out = []
    for our_name, metric in our_metric_pairs:
        if our_name not in wide:
            continue
        ours_vals = list(wide[our_name][metric])
        for base in wide:
            if base == our_name:
                continue
            p, gain = paired_test(ours_vals, list(wide[base][metric]))
            bm, bh = ci95(list(wide[base][metric]))
            om, oh = ci95(ours_vals)
            out.append({
                "scenario": label, "metric": metric, "ours": our_name, "baseline": base,
                "ours_mean": round(om, 2), "ours_ci95": round(oh, 2),
                "baseline_mean": round(bm, 2), "baseline_ci95": round(bh, 2),
                "median_rel_gain": round(gain, 3), "wilcoxon_p": p})
    return out


def main():
    backend = EuclideanBackend()
    os.makedirs("outputs", exist_ok=True)

    # --- meeting-point scenarios ---
    mp_rows = []
    for label, city, n, mix, params in MP_SCENARIOS:
        for seed in range(N_INSTANCES):
            origins = sample_origins(city, n, seed=seed, spread="clustered", clusters=1,
                                     cluster_sd_deg=0.03 if mix != "walking" else 0.01)
            modes = assign_modes(n, mix=mix, seed=seed)
            for r in run_instance(origins, modes, backend, params, fine_res=params.fine_res,
                                  variants=("ede",)):
                r.update(scenario=label, seed=seed)
                mp_rows.append(r)
    mp_df = pd.DataFrame(mp_rows)
    mp_df.to_csv("outputs/significance.csv", index=False)

    # --- dark-store siting scenarios ---
    ds_rows = []
    for label, city, n_cells, params in DS_SCENARIOS:
        for seed in range(N_INSTANCES):
            demand, weights = darkstore.sample_demand(city, n_cells, seed=seed)
            for r in darkstore.run_darkstore_instance(demand, weights, backend, params, sla_min=10.0):
                r.update(scenario=label, seed=seed)
                ds_rows.append(r)
    ds_df = pd.DataFrame(ds_rows)
    ds_df.to_csv("outputs/significance_darkstore.csv", index=False)

    # --- paired tests over both families ---
    test_rows = []
    for label, *_ in MP_SCENARIOS:
        test_rows += tests_for(mp_df, label, MP_TESTS)
    for label, *_ in DS_SCENARIOS:
        test_rows += tests_for(ds_df, label, DS_TESTS)
    tdf = pd.DataFrame(test_rows)
    tdf.to_csv("outputs/significance_tests.csv", index=False)

    pd.set_option("display.width", 170, "display.max_rows", 300)
    print(f"Significance over {N_INSTANCES} instances/scenario (Euclidean), "
          f"{len(MP_SCENARIOS)} meeting-point + {len(DS_SCENARIOS)} dark-store scenarios.")
    print("\nMeeting-point per-method means (variance, ede):")
    print(mp_df.groupby(["scenario", "method"])[["variance", "ede", "mean", "max", "jain", "gini"]]
          .mean(numeric_only=True).round(2).to_string())
    print("\nDark-store per-method means (w_variance, w_ede, courier Gini, within-SLA %):")
    print(ds_df.groupby(["scenario", "method"])[["w_variance", "w_ede", "courier_gini", "pct_within_sla"]]
          .mean(numeric_only=True).round(2).to_string())
    print("\nPaired Wilcoxon (ours < baseline), one-sided p and median relative gain:")
    print(tdf.to_string(index=False))

    # Headline: does our objective beat every baseline, significantly, in every scenario?
    real = tdf[tdf["wilcoxon_p"].notna()]
    n_sig = (real["wilcoxon_p"] < 0.05).sum()
    print(f"\n{n_sig}/{len(real)} paired tests reject equality at p<0.05 in our favour "
          f"(excludes ours-vs-exhaustive ties).")


if __name__ == "__main__":
    main()
