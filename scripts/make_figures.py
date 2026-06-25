
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from fairmp.algorithm import Params
from fairmp.runner import solve_all
from fairmp.sweep import gamma_sweep, make_instance, resolution_sweep, run_sweep, size_sweep
from fairmp.travel_time import EuclideanBackend

OUT = "outputs/figures"
backend = EuclideanBackend()
P = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)

def _box(ax, data, labels, ylabel, title):
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def fig_peruser():
    origins, modes = make_instance("london", 6, seed=3)
    res = solve_all(origins, modes, backend, P)
    methods = [m for m in ["centroid", "min_max", "geometric_median", "ours"] if m in res]
    data = [res[m]["times"] for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4))
    _box(ax, data, methods, "per-user travel time (min)", "Per-user travel time by method (one instance)")
    for i, d in enumerate(data, 1):
        ax.scatter([i] * len(d), d, alpha=0.6, color="tab:blue", zorder=3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_peruser.png", dpi=150)
    plt.close(fig)

def fig_aggregate(df):
    order = [m for m in ["centroid", "min_sum", "min_max", "weighted_centroid",
                         "geometric_median", "random", "ours"] if m in set(df["method"])]
    for metric in ["variance", "jain"]:
        data = [df[df["method"] == m][metric].dropna().values for m in order]
        fig, ax = plt.subplots(figsize=(7, 4))
        _box(ax, data, order, metric, f"{metric} by method across instances")
        fig.tight_layout()
        fig.savefig(f"{OUT}/fig_{metric}_by_method.png", dpi=150)
        plt.close(fig)

def fig_pareto():
    origins, modes = make_instance("london", 6, seed=3)
    pts = gamma_sweep(origins, modes, backend, gammas=[0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2], base_params=P)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([p["mean"] for p in pts], [p["variance"] for p in pts], "o-")
    for p in pts:
        ax.annotate(f"g={p['gamma']}", (p["mean"], p["variance"]), fontsize=7)
    ax.set_xlabel("mean travel time (min)")
    ax.set_ylabel("variance")
    ax.set_title("Variance vs mean (gamma Pareto front)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_pareto.png", dpi=150)
    plt.close(fig)

def fig_resolution():
    origins, modes = make_instance("london", 6, seed=3)
    base = Params(coarse_res=6, fine_res=9, k_c=P.k_c, k_refine=P.k_refine, t_max=P.t_max)
    rows = resolution_sweep(origins, modes, backend, fine_reses=[7, 8, 9, 10], base_params=base)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["fine_res"] for r in rows], [r["our_variance"] for r in rows], "o-", label="ours")
    ax.plot([r["fine_res"] for r in rows], [r["exhaustive_variance"] for r in rows], "s--", label="exhaustive")
    ax.set_xlabel("fine H3 resolution")
    ax.set_ylabel("variance")
    ax.legend()
    ax.set_title("Variance vs fine resolution")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_resolution.png", dpi=150)
    plt.close(fig)

def fig_scaling():
    rows = size_sweep(backend, "london", ns=[3, 5, 10, 20, 40], seeds=[0, 1, 2], base_params=P, fine_res=9)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["n"] for r in rows], [r["routing_calls"] for r in rows], "o-", color="tab:blue", label="routing calls")
    ax.set_xlabel("N users")
    ax.set_ylabel("routing calls (ours)", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot([r["n"] for r in rows], [r["runtime_s"] for r in rows], "s--", color="tab:red", label="runtime s")
    ax2.set_ylabel("runtime (s)", color="tab:red")
    ax.set_title("Scaling with N (ours)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_scaling.png", dpi=150)
    plt.close(fig)

def fig_map():
    origins, modes = make_instance("london", 6, seed=3)
    res = solve_all(origins, modes, backend, P)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter([o.lng for o in origins], [o.lat for o in origins], c="black", marker="^", s=60, label="origins")
    for m in ["centroid", "min_max", "ours"]:
        if m in res:
            pt = res[m]["point"]
            ax.scatter([pt.lng], [pt.lat], s=140, label=m, edgecolor="k", zorder=4)
    ax.set_xlabel("lng")
    ax.set_ylabel("lat")
    ax.legend()
    ax.set_title("Origins and chosen points (one instance)")
    ax.set_aspect("equal", "datalim")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_map.png", dpi=150)
    plt.close(fig)

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = run_sweep(backend, ["london", "bengaluru"], [3, 5, 10], ["mixed"], list(range(5)), params=P, fine_res=9)
    df = pd.DataFrame(rows)
    df.to_csv("outputs/sweep.csv", index=False)

    fig_aggregate(df)
    fig_peruser()
    fig_pareto()
    fig_resolution()
    fig_scaling()
    fig_map()

    print("figures ->", OUT)
    print(df.groupby("method")[["variance", "jain", "opt_gap", "routing_calls"]].mean().round(3)
          .sort_values("variance").to_string())

if __name__ == "__main__":
    main()
