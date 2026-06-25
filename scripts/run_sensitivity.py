
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from fairmp import metrics
from fairmp.algorithm import Params, fair_meeting_point
from fairmp.baselines import exhaustive_variance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.sweep import resolution_sweep, size_sweep
from fairmp.travel_time import CachedEvaluator, EuclideanBackend

def main():
    b = EuclideanBackend()
    os.makedirs("outputs", exist_ok=True)

    n_rows = size_sweep(b, "london", ns=[3, 5, 10, 20, 50], seeds=[0, 1, 2, 3], fine_res=9)
    pd.DataFrame(n_rows).to_csv("outputs/sensitivity_n.csv", index=False)

    o = sample_origins("london", 6, seed=4, spread="clustered", clusters=1, cluster_sd_deg=0.03)
    m = assign_modes(6, "mixed", seed=4)
    res_rows = resolution_sweep(o, m, b, fine_reses=[8, 9, 10, 11])
    pd.DataFrame(res_rows).to_csv("outputs/sensitivity_resolution.csv", index=False)

    ev0 = CachedEvaluator(b)
    xpt = exhaustive_variance(o, m, ev0, res=9)
    xvar = metrics.variance([ev0.effective(oo, xpt, mm) for oo, mm in zip(o, m)])
    hp_rows = []
    for kc in [100, 200, 300, 500]:
        for kr in [5, 10, 20]:
            p = Params(coarse_res=8, fine_res=9, k_c=kc, k_refine=kr, t_max=120.0)
            ev = CachedEvaluator(b)
            best, _r, _s, _d = fair_meeting_point(o, m, ev, p)
            ov = metrics.variance(best.times)
            hp_rows.append({"k_c": kc, "k_refine": kr, "our_variance": round(ov, 3),
                            "opt_gap": round((ov - xvar) / xvar if xvar > 0 else 0.0, 4),
                            "routing_calls": ev.calls})
    pd.DataFrame(hp_rows).to_csv("outputs/sensitivity_hyperparams.csv", index=False)

    print("N-scaling:")
    print(pd.DataFrame(n_rows).to_string(index=False))
    print("\nH3 fine-resolution:")
    print(pd.DataFrame(res_rows).to_string(index=False))
    print("\nK_c / K robustness (opt_gap vs exhaustive):")
    print(pd.DataFrame(hp_rows).to_string(index=False))

if __name__ == "__main__":
    main()
