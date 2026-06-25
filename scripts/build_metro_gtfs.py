
from __future__ import annotations

import csv
import io
import json
import math
import os
import zipfile

from shapely.geometry import LineString, Point

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "bengaluru", "_metro_src", "metro-lines-stations.geojson")
OUT = os.path.join(ROOT, "data", "bengaluru", "gtfs", "namma_metro.zip")

ASSIGN_DEG = 0.0035
SPEED_KMH = 33.0
DWELL_S = 25
SERVICE_START = "05:00:00"
SERVICE_END = "23:00:00"
HEADWAY_S = 360
START_DATE, END_DATE = "20240101", "20271231"
EARTH_KM = 6371.0

def haversine_km(a, b):
    (lng1, lat1), (lng2, lat2) = a, b
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    h = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(h)))

def hms(total_s):
    h, rem = divmod(int(total_s), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def load():
    feats = json.load(open(SRC, encoding="utf-8"))["features"]
    lines, stations = [], []
    for f in feats:
        g = f["geometry"]
        name = (f["properties"].get("Name") or "").strip()
        if g["type"] == "LineString":
            lines.append((name, f["properties"].get("description") or "", g["coordinates"]))
        elif g["type"] == "Point":
            lng, lat = g["coordinates"][0], g["coordinates"][1]
            stations.append((name, lng, lat))
    return lines, stations

def main():
    lines, stations = load()

    stop_id = {}
    stops = []
    for name, lng, lat in stations:
        if name not in stop_id:
            sid = f"S{len(stops)+1}"
            stop_id[name] = sid
            stops.append((sid, name, lat, lng))

    routes, trips, stop_times, freqs = [], [], [], []
    for li, (lname, color, coords) in enumerate(lines):
        line = LineString([(c[0], c[1]) for c in coords])
        on = []
        for name, lng, lat in stations:
            p = Point(lng, lat)
            if line.distance(p) < ASSIGN_DEG:
                on.append((line.project(p), name, lng, lat))
        on.sort(key=lambda t: t[0])

        seq = []
        for _d, name, lng, lat in on:
            if not seq or seq[-1][0] != name:
                seq.append((name, lng, lat))
        if len(seq) < 2:
            continue
        rid = f"R{li+1}"
        routes.append((rid, lname[:60] or rid, color or "metro"))
        for direction, ordered in ((0, seq), (1, list(reversed(seq)))):
            tid = f"{rid}_d{direction}"
            trips.append((rid, "DAILY", tid, direction))
            t = 0.0
            base = 5 * 3600
            prev = None
            for k, (name, lng, lat) in enumerate(ordered):
                if prev is not None:
                    t += haversine_km(prev, (lng, lat)) / SPEED_KMH * 3600.0
                arr = base + t
                dep = arr + DWELL_S
                stop_times.append((tid, hms(arr), hms(dep), stop_id[name], k + 1))
                t += DWELL_S
                prev = (lng, lat)
            freqs.append((tid, SERVICE_START, SERVICE_END, HEADWAY_S))

    _write(stops, routes, trips, stop_times, freqs)
    print("wrote", OUT)
    print(f"  stops {len(stops)}  routes {len(routes)}  trips {len(trips)}  "
          f"stop_times {len(stop_times)}  frequencies {len(freqs)}")

def _csv(rows, header):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()

def _write(stops, routes, trips, stop_times, freqs):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    files = {
        "agency.txt": _csv([["BMRCL", "Namma Metro (synthetic)", "https://english.bmrc.co.in", "Asia/Kolkata"]],
                           ["agency_id", "agency_name", "agency_url", "agency_timezone"]),
        "stops.txt": _csv([[s[0], s[1], f"{s[2]:.6f}", f"{s[3]:.6f}"] for s in stops],
                          ["stop_id", "stop_name", "stop_lat", "stop_lon"]),
        "routes.txt": _csv([[r[0], "BMRCL", r[1], 1] for r in routes],
                           ["route_id", "agency_id", "route_long_name", "route_type"]),
        "trips.txt": _csv([[t[0], t[1], t[2], t[3]] for t in trips],
                          ["route_id", "service_id", "trip_id", "direction_id"]),
        "stop_times.txt": _csv([[s[0], s[1], s[2], s[3], s[4]] for s in stop_times],
                               ["trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"]),
        "frequencies.txt": _csv([[f[0], f[1], f[2], f[3]] for f in freqs],
                                ["trip_id", "start_time", "end_time", "headway_secs"]),
        "calendar.txt": _csv([["DAILY", 1, 1, 1, 1, 1, 1, 1, START_DATE, END_DATE]],
                             ["service_id", "monday", "tuesday", "wednesday", "thursday", "friday",
                              "saturday", "sunday", "start_date", "end_date"]),
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for fname, content in files.items():
            z.writestr(fname, content)

if __name__ == "__main__":
    main()
