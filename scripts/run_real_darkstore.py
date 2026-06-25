"""Real dark-store siting on the London road network (couriers cycling on OSM).

Demand is synthetic (local service area, population-like weights); travel times are
real r5py cycling times on the London road network. Reuses the precompute approach:
one r5 batch matrix (all demand -> all candidates), then run_darkstore_instance as
lookups.

Run:  python scripts/run_real_darkstore.py
Out:  outputs/real_darkstore.csv
"""
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

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

from fairmp import darkstore  # noqa: E402
from fairmp.algorithm import Params  # noqa: E402
from fairmp.candidates import polyfill_centroids, region_polygon  # noqa: E402
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend  # noqa: E402

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]
COURIER = ["cycling"]


def candidates_for(demand, weights, coarse, fine):
    region = region_polygon(demand)
    pts = {}
    for _c, p in polyfill_centroids(region, coarse):
        pts[(round(p.lat, 6), round(p.lng, 6))] = p
    for _c, p in polyfill_centroids(region, fine):
        pts[(round(p.lat, 6), round(p.lng, 6))] = p
    wc = darkstore._weighted_centroid(demand, weights)
    pts[(round(wc.lat, 6), round(wc.lng, 6))] = wc
    return list(pts.values())


def precompute(r5, origins, mode, cands, departure):
    r5py = r5._r5py
    net = r5.network
    pre = PrecomputedBackend()
    dest_gdf = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                                geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    src_gdf = gpd.GeoDataFrame({"id": [f"o{i}" for i in range(len(origins))]},
                               geometry=[Point(o.lng, o.lat) for o in origins], crs="EPSG:4326")
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
    ttm = r5py.TravelTimeMatrix(net, origins=src_gdf, destinations=dest_gdf,
                                departure=departure, transport_modes=tmodes)
    df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
    omap = {f"o{i}": o for i, o in enumerate(origins)}
    cmap = {f"c{i}": cands[i] for i in range(len(cands))}
    for r in df.itertuples(index=False):
        o, c = omap.get(r.from_id), cmap.get(r.to_id)
        if o is None or c is None:
            continue
        t = r.travel_time
        pre.put(mode, o, c, float(t) if t == t else float("inf"))
    return pre


def main():
    print("loading London network...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    params = Params(coarse_res=8, fine_res=9, k_c=300, k_refine=10, t_max=30.0, gamma=0.0)
    sla = 10.0

    n_instances = int(os.environ.get("N_INSTANCES", "100"))
    n_cells = int(os.environ.get("DEMAND_CELLS", "40"))
    # WORLDPOP=1 draws demand from the real WorldPop population raster (inner-London
    # service area) instead of the synthetic generator: real-demand, larger-scale siting.
    use_worldpop = bool(os.environ.get("WORLDPOP"))
    wp_raster = os.path.join(ROOT, "data", "london", "worldpop_gbr.tif")
    wp_bbox = (51.48, -0.16, 51.54, -0.06)
    rows = []
    for seed in range(n_instances):
        if use_worldpop:
            demand, weights = darkstore.sample_demand_worldpop(wp_raster, wp_bbox, n_cells, seed=seed)
        else:
            demand, weights = darkstore.sample_demand("london", n_cells, seed=seed)
        cands = candidates_for(demand, weights, 8, 9)
        print(f"seed {seed}: {len(cands)} candidates, r5 cycling matrix...")
        pre = precompute(r5, demand, "cycling", cands, departure)
        for r in darkstore.run_darkstore_instance(demand, weights, pre, params, sla, courier=COURIER):
            r.update(seed=seed)
            rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/real_darkstore.csv", index=False)
    cols = ["w_variance", "w_ede", "p90", "pct_within_sla", "courier_gini", "max"]
    print("\nREAL London dark-store siting (real cycling times, synthetic demand; SLA 10 min):")
    print(df.groupby("method")[cols].mean(numeric_only=True).round(2).sort_values("w_variance").to_string())


if __name__ == "__main__":
    main()
