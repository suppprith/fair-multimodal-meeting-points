
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fairmp import metrics
from fairmp.algorithm import Params
from fairmp.runner import solve_all
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import EuclideanBackend

OUT = "outputs/figures"
os.makedirs(OUT, exist_ok=True)

def main():
    n = 6
    origins = sample_origins("london", n, seed=3, spread="clustered", clusters=1, cluster_sd_deg=0.03)
    modes = assign_modes(n, mix="mixed", seed=3)
    res = solve_all(origins, modes, EuclideanBackend(), Params(coarse_res=8, fine_res=9, k_c=300, t_max=120.0))

    cen = res["centroid"]["times"]
    our = res["ours"]["times"]
    order = sorted(range(n), key=lambda i: our[i])
    cen = [cen[i] for i in order]
    our = [our[i] for i in order]

    x = np.arange(n)
    w = 0.4
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, cen, w, color="#bdbdbd",
           label=f"geometric centroid  (spread {metrics.spread(res['centroid']['times']):.0f} min)")
    ax.bar(x + w / 2, our, w, color="#2ca25f",
           label=f"fair point (ours)  (spread {metrics.spread(res['ours']['times']):.0f} min)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"person {i+1}" for i in range(n)])
    ax.set_ylabel("travel time (minutes)")
    ax.set_title("Each person's travel time to the meeting point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_fairness_bars.png", dpi=150)
    print("wrote", f"{OUT}/fig_fairness_bars.png")

if __name__ == "__main__":
    main()
