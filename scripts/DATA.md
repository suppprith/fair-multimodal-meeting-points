# Data download: London and Bengaluru

Full rationale in `../../data-sources.md`. Put files under `data/london/` and
`data/bengaluru/`. Record source URL and date for each. None of this is fetched
yet; this is the checklist.

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

## Runtime requirement

r5py needs a **JDK 21** on PATH. The machine currently has Java 20; install
Temurin 21 (https://adoptium.net) before running the r5py backend.

## Licences (keep with the data)

OSM ODbL; BODS and London Datastore UK OGL; WorldPop CC-BY; Vonter BMTC per its repo.
