
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import darkstore
from fairmp.algorithm import Params
from fairmp.travel_time import EuclideanBackend

def main():
    backend = EuclideanBackend()
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0, gamma=0.0)
    sla_min = 10.0

    rows = []
    for city in ["london", "bengaluru"]:
        for seed in range(4):
            demand, weights = darkstore.sample_demand(city, 40, seed=seed)
            for r in darkstore.run_darkstore_instance(demand, weights, backend, params, sla_min):
                r.update(city=city, seed=seed)
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/dev_darkstore.csv", index=False)

    cols = ["w_variance", "p90", "pct_within_sla", "courier_gini", "routing_calls"]
    print("Dark-store siting (mean over instances, SLA = {:.0f} min):".format(sla_min))
    print(df.groupby("method")[cols].mean().round(2).sort_values("w_variance").to_string())

if __name__ == "__main__":
    main()
