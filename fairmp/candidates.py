"""Candidate generation: region bound, H3 polyfill, prefilter, refinement.

Implements the coarse-to-fine candidate pipeline of the paper (Section 4.2-4.4):
region hull plus margin, H3 polyfill, a max-haversine geometric prefilter, and
child-plus-ring refinement.
"""
from __future__ import annotations

import h3
from shapely.geometry import MultiPoint

from .geo import LatLng, haversine_km, spread_km

KM_PER_DEG = 111.0


def region_polygon(origins: list[LatLng], margin_frac: float = 0.2, min_buffer_km: float = 1.0):
    """Convex hull of origins, buffered by a margin that grows with the spread.

    Returns a shapely polygon in (lng, lat) order. The buffer is approximate
    (degrees), which is fine at city scale.
    """
    pts = MultiPoint([(o.lng, o.lat) for o in origins])
    hull = pts.convex_hull
    buffer_km = margin_frac * max(spread_km(origins), min_buffer_km) + min_buffer_km
    return hull.buffer(buffer_km / KM_PER_DEG)


def _to_h3poly(poly):
    # shapely coords are (lng, lat); h3.LatLngPoly wants (lat, lng)
    outer = [(lat, lng) for lng, lat in poly.exterior.coords]
    holes = [[(lat, lng) for lng, lat in r.coords] for r in poly.interiors]
    return h3.LatLngPoly(outer, *holes)


def polyfill_centroids(poly, res: int) -> list[tuple[str, LatLng]]:
    """H3 cells covering the region, returned as (cell, centroid LatLng)."""
    out = []
    for c in h3.polygon_to_cells(_to_h3poly(poly), res):
        lat, lng = h3.cell_to_latlng(c)
        out.append((c, LatLng(lat, lng)))
    return out


def coarse_score(point: LatLng, origins: list[LatLng]) -> float:
    """Cheap pre-rank: worst-case straight-line distance to any origin (minimax proxy)."""
    return max(haversine_km(o, point) for o in origins)


def prefilter(candidates: list[tuple[str, LatLng]], origins: list[LatLng], k_c: int):
    """Keep the K_c candidates with the lowest geometric proxy, before any routing."""
    ranked = sorted(candidates, key=lambda cp: coarse_score(cp[1], origins))
    return ranked[:k_c]


def refine_cells(cell: str, fine_res: int, ring: int = 1) -> list[tuple[str, LatLng]]:
    """A coarse cell's children at fine_res plus their k-ring neighbours.

    Guards fine_res to be at least the cell's own resolution (children at a coarser
    resolution are undefined)."""
    fine = max(fine_res, h3.get_resolution(cell))
    cells = set(h3.cell_to_children(cell, fine))
    for c in list(cells):
        cells.update(h3.grid_disk(c, ring))
    out = []
    for c in cells:
        lat, lng = h3.cell_to_latlng(c)
        out.append((c, LatLng(lat, lng)))
    return out
