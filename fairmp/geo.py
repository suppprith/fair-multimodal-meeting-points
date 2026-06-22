"""Geometry helpers: haversine distance, centroid, and spread."""
from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_KM = 6371.0


@dataclass(frozen=True)
class LatLng:
    lat: float
    lng: float


def centroid(points: list[LatLng]) -> LatLng:
    n = len(points)
    return LatLng(sum(p.lat for p in points) / n, sum(p.lng for p in points) / n)


def haversine_km(a: LatLng, b: LatLng) -> float:
    dlat = math.radians(b.lat - a.lat)
    dlng = math.radians(b.lng - a.lng)
    lat1 = math.radians(a.lat)
    lat2 = math.radians(b.lat)
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return 2 * EARTH_KM * math.asin(min(1.0, math.sqrt(h)))


def spread_km(points: list[LatLng]) -> float:
    """Greatest distance from the centroid to any point (the origin spread)."""
    c = centroid(points)
    return max((haversine_km(c, p) for p in points), default=0.0)
