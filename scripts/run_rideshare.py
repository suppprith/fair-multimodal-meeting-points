"""Ride-share pickup experiments.

A shared pickup is the meeting-point problem with a walking-access mode: riders walk
to one pickup, and we minimise the variance of their walk (access) time so no rider
bears a much longer walk than the others. It reuses the core pipeline with
mode = walking. A single shared pickup also cuts driver detours versus doorstep
pickup (Stiglic et al., 2015); that saving is qualitative here.

Runs on the Euclidean backend (walking is road/path-only, no GTFS). Swap in
R5Backend for real walk times.

Run:  python scripts/run_rideshare.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from fairmp.algorithm import Params  # noqa: E402
from fairmp.runner import run_instance  # noqa: E402
from fairmp.scenarios import assign_modes, sample_origins  # noqa: E402
from fairmp.travel_time import EuclideanBackend  # noqa: E402


def main():
    backend = EuclideanBackend()
    # walking is short range, so search at finer H3 resolution
    params = Params(coarse_res=9, fine_res=10, k_c=400, k_refine=12, t_max=30.0, gamma=0.0)

    rows = []
    for city in ["london", "bengaluru"]:
        for seed in range(5):
            # riders within a local area, all walking to the shared pickup
            riders = sample_origins(city, 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.01)
            modes = assign_modes(5, mix="walking", seed=seed)
            for r in run_instance(riders, modes, backend, params, fine_res=10):
                r.update(city=city, seed=seed)
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/rideshare.csv", index=False)

    cols = ["variance", "spread", "mean", "max", "jain", "opt_gap"]
    print("Ride-share pickup — rider WALK (access) time to the shared pickup, mean over instances:")
    print(df.groupby("method")[cols].mean().round(2).sort_values("variance").to_string())


if __name__ == "__main__":
    main()
