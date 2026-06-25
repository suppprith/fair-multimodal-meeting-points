
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp.algorithm import Params
from fairmp.sweep import run_sweep
from fairmp.travel_time import EuclideanBackend

def main():
    backend = EuclideanBackend()
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0)
    rows = run_sweep(
        backend,
        cities=["london", "bengaluru"],
        ns=[3, 5, 10],
        mixes=["mixed"],
        seeds=list(range(5)),
        params=params,
        fine_res=9,
    )
    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/sweep.csv", index=False)
    print("wrote outputs/sweep.csv", df.shape)
    agg = df.groupby("method")[["variance", "jain", "gini", "opt_gap", "routing_calls"]].mean().round(3)
    print(agg.sort_values("variance").to_string())

if __name__ == "__main__":
    main()
