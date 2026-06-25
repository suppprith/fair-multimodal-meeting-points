
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
    city, n = "london", 5
    origins = sample_origins(city, n, seed=42, spread="clustered", clusters=1, cluster_sd_deg=0.025)
    modes = assign_modes(n, mix="mixed", seed=42)
    backend = EuclideanBackend()
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0, gamma=0.0)

    rows = run_instance(origins, modes, backend, params, fine_res=9)
    df = pd.DataFrame(rows)
    cols = ["method", "variance", "jain", "gini", "mean", "max", "feasible",
            "routing_calls", "runtime_s", "opt_gap"]
    df = df[[c for c in cols if c in df.columns]].sort_values("variance").round(3)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print(f"City={city}  N={n}  modes={[m[0] for m in modes]}")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
