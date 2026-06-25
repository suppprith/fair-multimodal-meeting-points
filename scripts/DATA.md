# Data download: London, Bengaluru, Tokyo, Bay Area

Full rationale in `../../data-sources.md`. Put files under `data/<city>/`. Record source
URL and date for each. This is the checklist; `scripts/fetch_data.py` auto-pulls the
no-account, direct-URL items and prints the rest as manual steps.

The four cities are chosen for contrasting street and transit topology: London (compact,
mature multimodal), Bengaluru (sprawling, organic; bus + partial metro), Tokyo (dense
rail), Bay Area (car-dominant sprawl; rail + bus). London/Bengaluru are fetched; Tokyo and
Bay Area are scaffolded (bboxes + fetch wiring in place) and need their data pulled and a
JDK 21 + r5py run.

## Both cities

- **Road network (OSM):** draw a city bounding box at https://extract.bbbike.org
  and download `.osm.pbf` -> `data/<city>/network.osm.pbf`.
- **Population (siting demand):** WorldPop 100 m raster at https://www.worldpop.org
  -> `data/<city>/worldpop.tif`.

## London

- **Public transit (GTFS):** Bus Open Data Service London region feed,
  https://data.bus-data.dft.gov.uk/timetable/download/gtfs-file/london/ . This one feed
  already bundles the Underground, DLR, tram, ferry, and bus, so no separate tube/rail
  download is needed. -> `data/london/gtfs/`.
- **Emergency incidents (real):** London Fire Brigade Incident Records,
  https://data.london.gov.uk/dataset/london-fire-brigade-incident-records
  -> `data/london/incidents.csv`.
- **Ride OD seed (aggregate):** "Origin and destination of public transport journeys",
  London Datastore -> seeds synthetic rider origins.

## Bengaluru

- **Buses (BMTC GTFS):** https://github.com/Vonter/bmtc-gtfs (or DULT
  https://tdh.dult-karnataka.com / OpenCity https://data.opencity.in)
  -> `data/bengaluru/gtfs/`.
- **Metro (Namma Metro):** TUMI Datahub https://hub.tumidata.org/dataset/gtfs-bengaluru
  or assemble from https://github.com/geohacker/namma-metro. May be partial; if so,
  model from station coords + headways and label synthetic.
- **Emergency incidents:** no open dataset -> generate synthetic from WorldPop.
- **Ride OD:** no open trip data -> synthetic OD.

## Tokyo

- **Road network (OSM):** bbbike Tokyo extract (auto) -> `data/tokyo/network.osm.pbf`, or
  Geofabrik `asia/japan/kanto` and crop to the 23-wards bbox.
- **Transit (GTFS):** Open Data Platform for Transportation (ODPT, https://www.odpt.org)
  publishes GTFS-JP for Tokyo Metro, Toei, and JR-East but needs a free API token. The
  Mobility Database (https://mobilitydatabase.org) mirrors some feeds without a token.
  -> `data/tokyo/gtfs/`. Dense-rail city, so transit coverage matters here.
- **Population (siting):** WorldPop -> `data/tokyo/worldpop.tif`.

## Bay Area

- **Road network (OSM):** bbbike SanFrancisco extract (auto) -> `data/bayarea/network.osm.pbf`,
  or Geofabrik `north-america/us/california` cropped to the SF + inner East Bay bbox.
- **Transit (GTFS):** BART rail and SFMTA Muni publish open, no-account GTFS (auto-fetched)
  -> `data/bayarea/gtfs/`. For full regional coverage, the 511.org all-agency feed
  (https://511.org/open-data/transit) needs a free API token. Car-dominant city, so this
  is the test of whether the advantage survives a sparse-transit network.
- **Population (siting):** WorldPop or US Census blocks -> `data/bayarea/worldpop.tif`.

## Runtime requirement

r5py needs a **JDK 21** on PATH. The machine currently has Java 20; install
Temurin 21 (https://adoptium.net) before running the r5py backend.

## Licences (keep with the data)

OSM ODbL; BODS and London Datastore UK OGL; WorldPop CC-BY; Vonter BMTC per its repo.
