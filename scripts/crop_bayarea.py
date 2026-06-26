import os
import sys

import osmium

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "bayarea", "network.osm.pbf")
DST = os.path.join(ROOT, "data", "bayarea", "sf_city.osm.pbf")

BBOX = (-122.53, 37.71, -122.37, 37.83)


class Collect(osmium.SimpleHandler):
    def __init__(self):
        super().__init__()
        self.keep_nodes = set()
        self.keep_ways = set()

    def way(self, w):
        inbox = False
        refs = []
        for n in w.nodes:
            loc = n.location
            if loc.valid():
                refs.append(n.ref)
                if BBOX[0] <= loc.lon <= BBOX[2] and BBOX[1] <= loc.lat <= BBOX[3]:
                    inbox = True
        if inbox:
            self.keep_ways.add(w.id)
            self.keep_nodes.update(refs)


def main():
    print("pass 1: finding ways and nodes inside the San Francisco box...")
    c = Collect()
    c.apply_file(SRC, locations=True)
    print(f"  keep {len(c.keep_ways)} ways, {len(c.keep_nodes)} nodes")

    print("pass 2: writing the cropped extract...")
    header = osmium.io.Header()
    header.add_box(osmium.osm.Box(BBOX[0], BBOX[1], BBOX[2], BBOX[3]))
    writer = osmium.SimpleWriter(DST, header=header)
    keep_nodes, keep_ways = c.keep_nodes, c.keep_ways

    class Write(osmium.SimpleHandler):
        def node(self, n):
            if n.id in keep_nodes:
                writer.add_node(n)

        def way(self, w):
            if w.id in keep_ways:
                writer.add_way(w)

    Write().apply_file(SRC)
    writer.close()
    print(f"wrote {DST} ({os.path.getsize(DST) / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
