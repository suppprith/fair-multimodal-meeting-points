
from __future__ import annotations

import datetime as dt
import glob
import math
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

from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import centroid, haversine_km
from fairmp.metrics import variance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend, TflBackend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]


def small_candidates(origins, res, cap):
    region = region_polygon(origins)
    pts = [p for _c, p in polyfill_centroids(region, res)]
    c = centroid(origins)
    pts.sort(key=lambda p: haversine_km(c, p))
    return pts[:cap]


def static_transit(r5, origins, cands, departure):
    r5py = r5._r5py
    pre = PrecomputedBackend()
    dest_gdf = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                                geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    src_gdf = gpd.GeoDataFrame({"id": [f"o{i}" for i in range(len(origins))]},
                               geometry=[Point(o.lng, o.lat) for o in origins], crs="EPSG:4326")
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE["transit"]]
    ttm = r5py.TravelTimeMatrix(r5.network, origins=src_gdf, destinations=dest_gdf,
                                departure=departure, transport_modes=tmodes)
    df = ttm.compute_travel_times() if hasattr(ttm, "compute_travel_times") else ttm
    omap = {f"o{i}": o for i, o in enumerate(origins)}
    cmap = {f"c{i}": cands[i] for i in range(len(cands))}
    for r in df.itertuples(index=False):
        o, c = omap.get(r.from_id), cmap.get(r.to_id)
        if o is None or c is None:
            continue
        t = r.travel_time
        pre.put("transit", o, c, float(t) if t == t else float("inf"))
    return pre


def live_transit(tfl, origins, cands, departure):
    pre = PrecomputedBackend()
    for o in origins:
        for c in cands:
            pre.put("transit", o, c, tfl.minutes(o, c, "transit", departure))
    return pre


def best_point(pre, origins, cands):
    best_p, best_v = None, math.inf
    for cand in cands:
        ts = [pre.minutes(o, cand, "transit") for o in origins]
        if any(not math.isfinite(t) for t in ts):
            continue
        v = variance(ts)
        if v < best_v:
            best_v, best_p = v, cand
    return best_p, best_v


def var_at(pre, origins, cand):
    ts = [pre.minutes(o, cand, "transit") for o in origins]
    if any(not math.isfinite(t) for t in ts):
        return math.inf
    return variance(ts)


def main():
    if not os.environ.get("TFL_APP_KEY"):
        print("warning: TFL_APP_KEY not set; using the unregistered TfL rate limit", flush=True)
    print("loading London network...")
    r5 = R5Backend(OSM, GTFS)
    tfl = TflBackend()
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)

    n_instances = int(os.environ.get("N_INSTANCES", "15"))
    cap = int(os.environ.get("LIVE_CANDS", "30"))
    rows = []
    for seed in range(n_instances):
        origins = sample_origins("london", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.03)
        _modes = assign_modes(5, "transit", seed=seed)
        cands = small_candidates(origins, 8, cap)
        print(f"seed {seed}: {len(cands)} candidates, r5 static + TfL live transit...", flush=True)
        pre_s = static_transit(r5, origins, cands, departure)
        pre_l = live_transit(tfl, origins, cands, departure)

        diffs = []
        for o in origins:
            for c in cands:
                ts, tl = pre_s.minutes(o, c, "transit"), pre_l.minutes(o, c, "transit")
                if math.isfinite(ts) and math.isfinite(tl):
                    diffs.append(abs(tl - ts))

        p_s, var_s = best_point(pre_s, origins, cands)
        p_l, var_l = best_point(pre_l, origins, cands)
        if p_s is None or p_l is None:
            print(f"  seed {seed}: no feasible candidate in both, skipped", flush=True)
            continue
        var_l_at_s = var_at(pre_l, origins, p_s)
        rows.append({
            "seed": seed,
            "n_cands": len(cands),
            "n_pairs_matched": len(diffs),
            "mean_abs_dt_min": round(sum(diffs) / len(diffs), 2) if diffs else float("nan"),
            "displacement_m": round(haversine_km(p_s, p_l) * 1000.0, 1),
            "var_static": round(var_s, 3),
            "var_live": round(var_l, 3),
            "live_regret_pct": round((var_l_at_s - var_l) / var_l * 100.0, 1) if var_l > 1e-6 else 0.0,
        })
        pd.DataFrame(rows).to_csv("outputs/live_tfl.csv", index=False)

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/live_tfl.csv", index=False)
    print(f"\nLive TfL vs static r5py snapshot (London transit, {len(df)} instances):")
    if len(df):
        print(f"  median per-pair |live - static|:   {df['mean_abs_dt_min'].median():.1f} min")
        print(f"  median fair-point displacement:    {df['displacement_m'].median():.0f} m")
        print(f"  median live-regret of static point: {df['live_regret_pct'].median():.1f} %")


if __name__ == "__main__":
    main()
