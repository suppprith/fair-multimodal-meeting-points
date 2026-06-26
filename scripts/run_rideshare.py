
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

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
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]
WALK = ["walking"]

def candidates_for(riders, modes, coarse, fine):
    region = region_polygon(riders)
    pts = {}
    for res in (coarse, fine):
        for _c, p in polyfill_centroids(region, res):
            pts[(round(p.lat, 6), round(p.lng, 6))] = p

    for bp in (baselines.geometric_centroid(riders, modes),
               baselines.weighted_centroid(riders, modes),
               baselines.geometric_median(riders, modes)):
        pts[(round(bp.lat, 6), round(bp.lng, 6))] = bp
    return list(pts.values())

def precompute(r5, riders, cands, departure):
    r5py = r5._r5py
    net = r5.network
    pre = PrecomputedBackend()
    dest_gdf = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                                geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    src_gdf = gpd.GeoDataFrame({"id": [f"o{i}" for i in range(len(riders))]},
                               geometry=[Point(o.lng, o.lat) for o in riders], crs="EPSG:4326")
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE["walking"]]
    ttm = r5py.TravelTimeMatrix(net, origins=src_gdf, destinations=dest_gdf,
                                departure=departure, transport_modes=tmodes)
    df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
    omap = {f"o{i}": o for i, o in enumerate(riders)}
    cmap = {f"c{i}": cands[i] for i in range(len(cands))}
    for r in df.itertuples(index=False):
        o, c = omap.get(r.from_id), cmap.get(r.to_id)
        if o is None or c is None:
            continue
        t = r.travel_time
        pre.put("walking", o, c, float(t) if t == t else float("inf"))
    return pre

def main():
    print("loading London network (cached .dat if present)...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    params = Params(coarse_res=9, fine_res=10, k_c=400, k_refine=12, t_max=30.0, gamma=0.0)

    n_instances = int(os.environ.get("N_INSTANCES", "100"))
    rows = []
    for seed in range(n_instances):
        riders = sample_origins("london", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.012)
        modes = assign_modes(5, "walking", seed=seed)
        cands = candidates_for(riders, modes, 9, 10)
        print(f"seed {seed}: {len(cands)} candidates -> r5 walk matrix...")
        pre = precompute(r5, riders, cands, departure)
        for r in run_instance(riders, modes, pre, params, fine_res=10, variants=("ede",)):
            r.update(seed=seed)
            rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/rideshare.csv", index=False)
    cols = ["variance", "spread", "ede", "mean", "max", "jain", "opt_gap"]
    print("\nREAL London ride-share pickup (rider WALK access time, mean over instances):")
    print(df.groupby("method")[cols].mean(numeric_only=True).round(2).sort_values("variance").to_string())

if __name__ == "__main__":
    main()
