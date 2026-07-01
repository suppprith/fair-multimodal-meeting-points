
from __future__ import annotations

import datetime as dt
import glob
import math
import os
import random
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
from fairmp.candidates import polyfill_centroids, region_polygon
from fairmp.geo import haversine_km
from fairmp.metrics import variance
from fairmp.scenarios import assign_modes, sample_origins
from fairmp.travel_time import R5_MODE, PrecomputedBackend, R5Backend

OSM = os.path.join(ROOT, "data", "london", "network.osm.pbf")
GTFS = [os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")]


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


def _matrix(r5, origins, mode, cands, departure, pre):
    r5py = r5._r5py
    dest_gdf = gpd.GeoDataFrame({"id": [f"c{i}" for i in range(len(cands))]},
                                geometry=[Point(p.lng, p.lat) for p in cands], crs="EPSG:4326")
    src_gdf = gpd.GeoDataFrame({"id": [f"o{i}" for i in range(len(origins))]},
                               geometry=[Point(o.lng, o.lat) for o in origins], crs="EPSG:4326")
    tmodes = [getattr(r5py.TransportMode, x) for x in R5_MODE[mode]]
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
        pre.put(mode, o, c, float(t) if t == t else float("inf"))


def precompute_common(r5, origins, modes, cands, departure):
    pre = PrecomputedBackend()
    by_mode = defaultdict(list)
    for o, m in zip(origins, modes):
        by_mode[m[0]].append(o)
    for mode, os_ in by_mode.items():
        _matrix(r5, os_, mode, cands, departure, pre)
    return pre


def precompute_staggered(r5, origins, modes, cands, departures):
    pre = PrecomputedBackend()
    for o, m, dep in zip(origins, modes, departures):
        _matrix(r5, [o], m[0], cands, dep, pre)
    return pre


def eff(pre, o, cand, ms):
    return min(pre.minutes(o, cand, m) for m in ms)


def best_point(pre, origins, modes, cands):
    best_p, best_v = None, math.inf
    for cand in cands:
        ts = [eff(pre, o, cand, m) for o, m in zip(origins, modes)]
        if any(not math.isfinite(t) for t in ts):
            continue
        v = variance(ts)
        if v < best_v:
            best_v, best_p = v, cand
    return best_p, best_v


def times_at(pre, origins, modes, cand):
    return [eff(pre, o, cand, m) for o, m in zip(origins, modes)]


def main():
    print("loading London network...")
    r5 = R5Backend(OSM, GTFS)
    print("network ready")
    today = dt.date.today()
    wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
    t0 = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
    window = int(os.environ.get("STAGGER_WINDOW_MIN", "60"))
    half = window // 2

    n_instances = int(os.environ.get("N_INSTANCES", "30"))
    rows = []
    for seed in range(n_instances):
        origins = sample_origins("london", 5, seed=seed, spread="clustered", clusters=1, cluster_sd_deg=0.03)
        modes = assign_modes(5, "mixed", seed=seed)
        cands = candidates_for(origins, modes, 8, 9)

        rng = random.Random(1000 + seed)
        offsets = [rng.randint(-half, half) for _ in origins]
        deps = [t0 + dt.timedelta(minutes=off) for off in offsets]

        print(f"seed {seed}: {len(cands)} candidates, stagger {max(offsets) - min(offsets)} min...", flush=True)
        pre0 = precompute_common(r5, origins, modes, cands, t0)
        preS = precompute_staggered(r5, origins, modes, cands, deps)

        p_snap, _ = best_point(pre0, origins, modes, cands)
        p_true, var_true_best = best_point(preS, origins, modes, cands)
        if p_snap is None or p_true is None:
            print(f"  seed {seed}: no feasible candidate, skipped", flush=True)
            continue

        centroid = baselines.geometric_centroid(origins, modes)
        var_snap = variance(times_at(preS, origins, modes, p_snap))
        var_centroid = variance(times_at(preS, origins, modes, centroid))

        std_best = math.sqrt(var_true_best)
        std_snap = math.sqrt(var_snap)
        rows.append({
            "seed": seed,
            "stagger_span_min": max(offsets) - min(offsets),
            "displacement_m": round(haversine_km(p_snap, p_true) * 1000.0, 1),
            "var_snapshot_under_stagger": round(var_snap, 4),
            "var_staggered_optimum": round(var_true_best, 4),
            "regret_std_min": round(std_snap - std_best, 3),
            "regret_pct": round((var_snap - var_true_best) / var_true_best * 100.0, 1) if var_true_best > 1e-6 else 0.0,
            "beats_centroid_under_stagger": bool(var_snap <= var_centroid),
        })

    df = pd.DataFrame(rows)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/departure_sensitivity.csv", index=False)
    print(f"\nStaggered-departure sensitivity (London, {window}-min window, {len(df)} instances):")
    print(f"  median point displacement:      {df['displacement_m'].median():.0f} m")
    print(f"  median excess std vs staggered:  {df['regret_std_min'].median():.2f} min")
    print(f"  median regret:                   {df['regret_pct'].median():.1f} %")
    print(f"  snapshot still beats centroid:   {100.0 * df['beats_centroid_under_stagger'].mean():.0f}% of instances")


if __name__ == "__main__":
    main()
