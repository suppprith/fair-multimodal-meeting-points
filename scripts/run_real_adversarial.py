
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
from fairmp.geo import LatLng
from fairmp.runner import run_instance
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]

THAMES_N = (51.5105, -0.0380)
THAMES_S = (51.4960, -0.0420)

def _jitter(rng, lat, lng, sd=0.004):
    return LatLng(rng.gauss(lat, sd), rng.gauss(lng, sd))

def make_instance(kind, seed):
    import random
    rng = random.Random(seed)
    if kind == "river":
        n_n = 2 + seed % 2
        origins = [_jitter(rng, *THAMES_N) for _ in range(n_n)] + \
                  [_jitter(rng, *THAMES_S) for _ in range(5 - n_n)]
        modes = [["walking"], ["cycling"], ["walking"], ["transit"], ["cycling"]]
    elif kind == "mismatch":
        c_lat, c_lng = 51.515 + rng.uniform(-0.02, 0.02), -0.12 + rng.uniform(-0.02, 0.02)
        origins = [_jitter(rng, c_lat, c_lng, sd=0.02) for _ in range(5)]
        modes = [["driving"]] + [["walking"]] * 4
    elif kind == "linear":
        lat0 = 51.50 + rng.uniform(-0.01, 0.01)
        lng0 = -0.20 + rng.uniform(-0.02, 0.02)
        origins = [LatLng(lat0 + rng.gauss(0, 0.002), lng0 + i * 0.06) for i in range(5)]
        modes = [["transit"], ["driving"], ["cycling"], ["transit"], ["walking"]]
    else:
        raise ValueError(kind)
    return origins, modes

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
    dest_gdf = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                                geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    by_mode = defaultdict(list)
    for i, (o, m) in enumerate(zip(origins, modes)):
        by_mode[m[0]].append((i, o))
    for mode, users in by_mode.items():
        tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
        src_gdf = gpd.GeoDataFrame({"id": [f"o{i}" for i, _ in users]},
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
    print("loading London network...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    params = Params(coarse_res=8, fine_res=9, k_c=400, k_refine=12, t_max=120.0)
    n_instances = int(os.environ.get("N_INSTANCES", "4"))

    rows = []
    for kind in ("river", "mismatch", "linear"):
        for seed in range(n_instances):
            origins, modes = make_instance(kind, seed)
            cands = candidates_for(origins, modes, 8, 9)
            print(f"{kind} seed {seed}: {len(cands)} candidates -> r5 matrices...")
            pre = precompute(r5, origins, modes, cands, departure)
            for r in run_instance(origins, modes, pre, params, fine_res=9, variants=("ede",)):
                r.update(kind=kind, seed=seed)
                rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/real_adversarial.csv", index=False)

    print("\nREAL London adversarial topologies (mean over instances):")
    for kind in ("river", "mismatch", "linear"):
        sub = df[df["kind"] == kind]
        piv = sub.groupby("method")[["variance", "max", "jain"]].mean(numeric_only=True)
        ours = piv.loc["ours", "variance"]
        cen = piv.loc["centroid", "variance"] if "centroid" in piv.index else float("nan")
        blow = cen / ours if ours else float("nan")
        print(f"\n[{kind}] centroid variance / ours variance = {blow:.1f}x "
              f"(ours {ours:.1f}, centroid {cen:.1f})")
        print(piv.round(1).sort_values("variance").to_string())

if __name__ == "__main__":
    main()
