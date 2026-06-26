
from __future__ import annotations

import datetime as dt
import glob
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_jdk = sorted(glob.glob(os.path.join(ROOT, "tools", "jdk21", "jdk-*")))
if _jdk:
    os.environ.setdefault("JAVA_HOME", _jdk[0])
    os.environ["PATH"] = os.path.join(_jdk[0], "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from fairmp import baselines
from fairmp.algorithm import Params
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.runner import run_instance
import fairmp.scenarios as scenarios
from fairmp.scenarios import CITY_BBOX, assign_modes, sample_origins

CITY_BBOX["bengaluru"] = (12.92, 77.55, 13.02, 77.67)

scenarios.MODES = ["driving", "walking", "cycling"]
from fairmp.sweep import pareto_matched_mean
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend

OSM = os.path.join(ROOT, "data", "bengaluru", "blr_city.osm.pbf")

GTFS = []

def candidates_for(origins, modes, coarse, fine):
    region = region_polygon(origins)
    pts = {}
    for res in (coarse, fine):
        for _c, p in polyfill_centroids(region, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p
    for bp in (baselines.geometric_centroid(origins, modes),
               baselines.weighted_centroid(origins, modes),
               baselines.geometric_median(origins, modes)):
        pts[(round(bp.lat, 6), round(bp.lng, 6))] = bp
    return list(pts.values())

def precompute(r5, origins, modes, cands, departure):
    r5py = r5._r5py
    net = r5.network
    pre = PrecomputedBackend()
    dest_gdf = gpd.GeoDataFrame(
        {"id": [f"c{i}" for i in range(len(cands))]},
        geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    by_mode = defaultdict(list)
    for i, (o, m) in enumerate(zip(origins, modes)):
        by_mode[m[0]].append((i, o))
    for mode, users in by_mode.items():
        tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
        src_gdf = gpd.GeoDataFrame(
            {"id": [f"o{i}" for i, _ in users]},
            geometry=[Point(o.lng, o.lat) for _, o in users], crs="EPSG:4326")
        ttm = r5py.TravelTimeMatrix(net, origins=src_gdf, destinations=dest_gdf,
                                    departure=departure, transport_modes=tmodes)
        df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
        omap = {f"o{i}": o for i, o in users}
        cmap = {f"c{i}": cands[i] for i in range(len(cands))}
        for r in df.itertuples(index=False):
            o, c = omap.get(r.from_id), cmap.get(r.to_id)
            if o is None or c is None:
                continue
            t = r.travel_time
            pre.put(mode, o, c, float(t) if t == t else float("inf"))
    return pre

def main():
    print("loading Bengaluru network (Karnataka OSM + BMTC + synthetic metro)...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=120.0, gamma=0.0)

    n_instances = int(os.environ.get("N_INSTANCES", "100"))
    rows, pareto_rows = [], []
    for seed in range(n_instances):
        origins = sample_origins("bengaluru", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.02)
        modes = assign_modes(5, "mixed", seed=seed)
        cands = candidates_for(origins, modes, 8, 9)
        print(f"seed {seed}: {len(cands)} candidates, modes {[m[0] for m in modes]} -> r5 matrices...")
        pre = precompute(r5, origins, modes, cands, departure)
        for r in run_instance(origins, modes, pre, params, fine_res=9, variants=("ede",)):
            r.update(seed=seed)
            rows.append(r)
        _front, ref, op = pareto_matched_mean(origins, modes, pre, base_params=params, fine_res=9)
        if op is not None:
            pareto_rows.append({"seed": seed, "minsum_mean": ref["mean"], "minsum_variance": ref["variance"],
                                "gamma": op["gamma"], "op_mean": op["mean"], "op_variance": op["variance"],
                                "mean_gap_pct": op["mean_gap_pct"],
                                "variance_reduction_pct": op["variance_reduction_pct"]})

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/bengaluru.csv", index=False)
    cols = [c for c in ["variance", "jain", "gini", "ede", "mean", "max", "feasible", "opt_gap"] if c in df.columns]
    print("\nREAL Bengaluru social meetup (mean over instances):")
    print(df.groupby("method")[cols].mean(numeric_only=True).round(3).sort_values("variance").to_string())

    if pareto_rows:
        pdf = pd.DataFrame(pareto_rows)
        pdf.to_csv("outputs/bengaluru_pareto.csv", index=False)
        print(f"\nPareto matched-mean: mean variance reduction {pdf['variance_reduction_pct'].mean():.0f}%")

if __name__ == "__main__":
    main()
