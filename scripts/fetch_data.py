"""Fetch the auto-downloadable open data for London and Bengaluru.

Run:  python scripts/fetch_data.py

Covers the no-account, direct-URL sources. The rest (large extracts, account-gated,
or browser-only) are printed as manual steps at the end. See ../../data-sources.md.
"""
from __future__ import annotations

import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

AUTO = [
    ("london/network.osm.pbf", "https://download.bbbike.org/osm/bbbike/London/London.osm.pbf"),
    ("london/gtfs/london_bus.zip", "https://data.bus-data.dft.gov.uk/timetable/download/gtfs-file/london/"),
    ("bengaluru/gtfs/bmtc.zip", "https://raw.githubusercontent.com/Vonter/bmtc-gtfs/main/gtfs/bmtc.zip"),
]

MANUAL = """
Manual steps (large + crop, account, or browser-only):
  - Bengaluru OSM: download karnataka.osm.pbf from osm.fr (extracts/asia/india), then
    run scripts/crop_bengaluru.py -> data/bengaluru/blr_city.osm.pbf
  - London tube/rail GTFS: already bundled in the London region feed above; no separate
    download needed.
  - Bengaluru metro GTFS (synthetic): built by scripts/build_metro_gtfs.py.
  - WorldPop rasters (siting scenario): worldpop.org -> data/<city>/worldpop.tif
  - London Fire incidents (emergency, real): London Datastore -> data/london/incidents.csv
"""


def fetch(rel: str, url: str):
    dest = os.path.join(DATA, rel)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("exists  ", rel)
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print("download", rel)
    urllib.request.urlretrieve(url, dest)
    print("        ", round(os.path.getsize(dest) / 1e6, 1), "MB")


def main():
    for rel, url in AUTO:
        try:
            fetch(rel, url)
        except Exception as e:  # noqa: BLE001
            print("FAILED  ", rel, "->", e)
    print(MANUAL)


if __name__ == "__main__":
    main()
