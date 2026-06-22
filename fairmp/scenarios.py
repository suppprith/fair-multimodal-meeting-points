"""Synthetic instance generation. Cities are used only to place synthetic origins
on a real extent; in the paper they appear by topology, not name.
"""
from __future__ import annotations

import random

from .geo import LatLng

# Approximate bounding boxes: (lat_min, lng_min, lat_max, lng_max).
CITY_BBOX = {
    "london": (51.40, -0.30, 51.62, 0.10),
    "bengaluru": (12.85, 77.45, 13.10, 77.75),
}

MODES = ["transit", "driving", "walking", "cycling"]


def sample_origins(city, n, seed=0, spread="uniform", clusters=2, cluster_sd_deg=0.02):
    rng = random.Random(seed)
    lat0, lng0, lat1, lng1 = CITY_BBOX[city]
    pts = []
    if spread == "uniform":
        for _ in range(n):
            pts.append(LatLng(rng.uniform(lat0, lat1), rng.uniform(lng0, lng1)))
    else:  # clustered Gaussian mixture
        centers = [(rng.uniform(lat0, lat1), rng.uniform(lng0, lng1)) for _ in range(clusters)]
        for _ in range(n):
            clat, clng = rng.choice(centers)
            lat = min(max(rng.gauss(clat, cluster_sd_deg), lat0), lat1)
            lng = min(max(rng.gauss(clng, cluster_sd_deg), lng0), lng1)
            pts.append(LatLng(lat, lng))
    return pts


def assign_modes(n, mix="mixed", seed=0):
    """Each user gets one mode (M_i = [mode]), i.e. one member of the group.

    mix = "mixed" assigns each user a random mode; otherwise pass a single mode
    name ("transit", "driving", "walking", "cycling") for a uniform-mode group.
    """
    rng = random.Random(seed)
    chosen = [rng.choice(MODES) for _ in range(n)] if mix == "mixed" else [mix] * n
    return [[m] for m in chosen]
