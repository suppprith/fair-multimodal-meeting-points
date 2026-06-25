
from __future__ import annotations

import datetime as dt
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_jdk = sorted(glob.glob(os.path.join(ROOT, "tools", "jdk21", "jdk-*")))
if _jdk:
    os.environ.setdefault("JAVA_HOME", _jdk[0])
    os.environ["PATH"] = os.path.join(_jdk[0], "bin") + os.pathsep + os.environ.get("PATH", "")
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Xmx4g")

import geopandas as gpd
from shapely.geometry import Point
import r5py

print("r5py", getattr(r5py, "__version__", "?"))

osm = os.path.join(ROOT, "data", "london", "network.osm.pbf")
gtfs = os.path.join(ROOT, "data", "london", "gtfs", "london_bus.zip")
print("building network from", os.path.basename(osm), "+", os.path.basename(gtfs), "...")
net = r5py.TransportNetwork(osm, [gtfs])
print("network built OK")

pts = [(51.5074, -0.1278), (51.5155, -0.0922), (51.4975, -0.1357), (51.5230, -0.1580)]
gdf = gpd.GeoDataFrame(
    {"id": [str(i) for i in range(len(pts))]},
    geometry=[Point(lng, lat) for lat, lng in pts],
    crs="EPSG:4326",
)

today = dt.date.today()
wed = today + dt.timedelta((2 - today.weekday()) % 7 + 7)
departure = dt.datetime(wed.year, wed.month, wed.day, 8, 30)
modes = [r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK]

if hasattr(r5py, "TravelTimeMatrix"):
    obj = r5py.TravelTimeMatrix(net, origins=gdf, destinations=gdf, departure=departure, transport_modes=modes)
    df = obj.compute_travel_times() if hasattr(obj, "compute_travel_times") else obj
else:
    obj = r5py.TravelTimeMatrixComputer(net, origins=gdf, destinations=gdf, departure=departure, transport_modes=modes)
    df = obj.compute_travel_times()
print("departure", departure)
print(df.to_string(index=False))
print("OK: finite times present ->", df["travel_time"].notna().any())
