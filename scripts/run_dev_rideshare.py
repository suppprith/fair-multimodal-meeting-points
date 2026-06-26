
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp.algorithm import Params
from fairmp.runner import run_instance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import EuclideanBackend

def main():
    backend = EuclideanBackend()

    params = Params(coarse_res=9, fine_res=10, k_c=400, k_refine=12, t_max=30.0, gamma=0.0)

    rows = []
    for city in ["london", "bengaluru"]:
        for seed in range(5):

            riders = sample_origins(city, 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.01)
            modes = assign_modes(5, mix="walking", seed=seed)
            for r in run_instance(riders, modes, backend, params, fine_res=10):
                r.update(city=city, seed=seed)
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/dev_rideshare.csv", index=False)

    cols = ["variance", "spread", "mean", "max", "jain", "opt_gap"]
    print("Ride-share pickup: rider WALK (access) time to the shared pickup, mean over instances:")
    print(df.groupby("method")[cols].mean().round(2).sort_values("variance").to_string())

if __name__ == "__main__":
    main()
