# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Acquire an unlabelled NRW test pool from current Geobasis NRW orthophotos.

Building footprints come from the official ALKIS-vereinfacht WFS.

The pool is stratified: random small buildings per region give a realistic PV
prevalence and false-positive denominator; buildings that OpenStreetMap tags with
rooftop solar enrich the PV share. Every crop centre lies at least
MIN_SOURCE_DISTANCE_M (1 km plus the patch diagonal) from every location already
used for training, validation, testing or holdouts; the ten regions farthest from
those locations are used. Masks start as all-255
(ignore) placeholders; labels come only from blind human annotation afterwards.

    uv run --locked --no-sync python scripts/build_nrw_test_pool.py data/nrw-test-v1
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image
from pyproj import Transformer

import numpy as np

from pvextent.data.nrw import LICENSE_URL, MIN_SOURCE_DISTANCE_M, _bounds, _fetch, _inside, _map_url, _metadata, _roof

SEED = 20261022
RANDOM_PER_REGION = 25
OSM_PER_REGION = 8
MIN_CENTER_DISTANCE_M = 80.0
WFS_URL = "https://www.wfs.nrw.de/geobasis/wfs_nw_alkis_vereinfacht"
GML = "{http://www.opengis.net/gml/3.2}"
AVE = "{http://repository.gdi-de.org/schemas/adv/produkt/alkis-vereinfacht/2.0}"
OVERPASS_URLS = ("https://overpass-api.de/api/interpreter", "https://overpass.private.coffee/api/interpreter")
# Residential districts across NRW's landscapes (lon, lat of the district centre).
REGIONS = {
    "muenster-gievenbeck": (7.570, 51.965), "bielefeld-sennestadt": (8.585, 51.945),
    "paderborn": (8.755, 51.718), "aachen-brand": (6.165, 50.750), "bonn-beuel": (7.130, 50.735),
    "krefeld": (6.565, 51.335), "wesel": (6.620, 51.660), "lippstadt": (8.345, 51.675),
    "siegen-weidenau": (8.020, 50.875), "guetersloh": (8.385, 51.905), "rheine": (7.440, 52.285),
    "moenchengladbach": (6.440, 51.185), "hamm": (7.820, 51.680), "detmold": (8.880, 51.935),
    "kleve": (6.140, 51.790), "euskirchen": (6.790, 50.660), "arnsberg": (8.065, 51.395),
    "bocholt": (6.615, 51.840), "minden": (8.915, 52.285), "meschede": (8.285, 51.350),
}
TARGET_REGIONS = 10
HALF_WIDTH_DEG, HALF_HEIGHT_DEG = 0.006, 0.004  # about 830 m x 890 m


def excluded_points(project: Transformer) -> list[tuple[float, float]]:
    points = []
    partition = json.loads(Path("data/pv4ger-seg-v0.9.1/provenance/default_partition.json").read_text())
    for ids in partition.values():
        for sample in ids:
            points.append(project.transform(*map(float, sample.split(","))))
    for manifest in Path("data").glob("*/manifest.jsonl"):
        for line in manifest.open(encoding="utf-8"):
            transform = (json.loads(line).get("georeference") or {}).get("transform")
            if transform:
                points.append((transform[2] + 16.0, transform[5] - 16.0))
    return points


def with_retries(call, *args, attempts: int = 8):
    """Retry a flaky public geodata request with a growing pause."""
    for attempt in range(attempts):
        try:
            return call(*args)
        except Exception as error:  # urllib, socket and DatasetImportError wrappers
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1} after {type(error).__name__}: {error}", flush=True)
            time.sleep(min(90, 10 * (attempt + 1)))


def wfs_buildings(bbox_utm: tuple[float, float, float, float]) -> list[dict]:
    """Read ALKIS buildings from the official WFS into the GeoJSON-like shape ``_roof`` expects."""
    query = urlencode({"SERVICE": "WFS", "VERSION": "2.0.0", "REQUEST": "GetFeature",
                       "TYPENAMES": "ave:GebaeudeBauwerk", "COUNT": "3000",
                       "BBOX": ",".join(f"{v:.1f}" for v in bbox_utm) + ",urn:ogc:def:crs:EPSG::25832"})
    with urlopen(WFS_URL + "?" + query, timeout=120) as response:
        root = ET.fromstring(response.read())
    features = []
    for building in root.iter(AVE + "GebaeudeBauwerk"):
        polygons = []
        for polygon in building.iter(GML + "Polygon"):
            rings = []
            for ring in polygon.iter(GML + "posList"):
                values = [float(v) for v in ring.text.split()]
                rings.append([[values[i], values[i + 1]] for i in range(0, len(values), 2)])
            polygons.append(rings)
        features.append({"id": building.findtext(AVE + "oid"),
                         "properties": {"gebnutzbez": building.findtext(AVE + "gebnutzbez"),
                                        "funktion": building.findtext(AVE + "funktion") or ""},
                         "geometry": {"type": "MultiPolygon", "coordinates": polygons}})
    return features


def overpass_solar(bbox_wgs84: tuple[float, float, float, float]) -> list[tuple[float, float]] | None:
    west, south, east, north = bbox_wgs84
    cache = Path("data/cache/overpass") / f"{west:.4f}_{south:.4f}_{east:.4f}_{north:.4f}.json"
    if cache.exists():
        return [tuple(p) for p in json.loads(cache.read_text())]
    query = (f'[out:json][timeout:60];(nwr["generator:source"="solar"]({south},{west},{north},{east});'
             f'nwr["plant:source"="solar"]({south},{west},{north},{east}););out center;')
    for attempt in range(4):
        url = OVERPASS_URLS[attempt % len(OVERPASS_URLS)]
        request = Request(url, data=urlencode({"data": query}).encode(),
                          headers={"User-Agent": "spacement-pvextent research (Valeford)"})
        try:
            with urlopen(request, timeout=90) as response:
                elements = json.loads(response.read())["elements"]
            break
        except OSError as error:  # Public Overpass instances throttle with 429/502/504
            print(f"  overpass retry {attempt + 1}: {error}", flush=True)
            time.sleep(min(90, 15 * (attempt + 1)))
    else:
        return None  # Enrichment is optional; the region falls back to random buildings
    points = [(e["lon"], e["lat"]) if "lon" in e else (e["center"]["lon"], e["center"]["lat"]) for e in elements]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(points))
    return points


def main(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    project = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    excluded = np.array(excluded_points(project))
    rng = random.Random(SEED)

    def clearance(x: float, y: float) -> float:
        return float(np.hypot(excluded[:, 0] - x, excluded[:, 1] - y).min())

    regions = sorted(((name, lon, lat, round(clearance(*project.transform(lon, lat))))
                      for name, (lon, lat) in REGIONS.items()), key=lambda r: -r[3])
    used, osm_status = [], {}

    (output / "images").mkdir(parents=True)
    (output / "masks").mkdir()
    records, today = [], date.today()
    for name, lon, lat, _ in regions:
        if len(used) == TARGET_REGIONS:
            break
        bbox = (lon - HALF_WIDTH_DEG, lat - HALF_HEIGHT_DEG, lon + HALF_WIDTH_DEG, lat + HALF_HEIGHT_DEG)
        corners = [project.transform(bbox[0], bbox[1]), project.transform(bbox[2], bbox[3])]
        buildings = with_retries(wfs_buildings, (corners[0][0], corners[0][1], corners[1][0], corners[1][1]))
        roofs = {}
        for feature in buildings:
            roof = _roof(feature)
            if roof and clearance(*roof[0]) >= MIN_SOURCE_DISTANCE_M:
                roofs[feature["id"]] = (roof[0], feature["geometry"]["coordinates"])
        if len(roofs) < OSM_PER_REGION + RANDOM_PER_REGION:
            print(f"{name}: skipped ({len(roofs)} eligible roofs)", flush=True)
            continue
        used.append(name)
        solar_wgs84 = overpass_solar(bbox)
        osm_status[name] = "ok" if solar_wgs84 is not None else "unavailable"
        solar = [project.transform(*p) for p in solar_wgs84 or []]
        osm_ids = sorted({bid for bid, (_, polygons) in roofs.items()
                          for point in solar if any(_inside(point, polygon) for polygon in polygons)})
        random_ids = sorted(set(roofs) - set(osm_ids))
        rng.shuffle(osm_ids)
        rng.shuffle(random_ids)
        chosen, centers = [], []
        for stratum, pool, quota in (("osm_solar", osm_ids, OSM_PER_REGION), ("random", random_ids, RANDOM_PER_REGION)):
            taken = 0
            for bid in pool:
                if taken == quota + (OSM_PER_REGION - min(OSM_PER_REGION, len(osm_ids)) if stratum == "random" else 0):
                    break
                center = roofs[bid][0]
                if any(math.hypot(center[0] - cx, center[1] - cy) < MIN_CENTER_DISTANCE_M for cx, cy in centers):
                    continue
                centers.append(center)
                chosen.append((stratum, bid, center))
                taken += 1
        for stratum, bid, center in chosen:
            bounds = _bounds(center)
            payload = with_retries(_fetch, _map_url(bounds), 4 * 1024 * 1024)[0]
            image = Image.open(__import__("io").BytesIO(payload)).convert("RGB")
            if image.size != (320, 320):
                continue
            key = hashlib.sha256(f"{bid}:{bounds}".encode()).hexdigest()[:24]
            image.save(output / "images" / f"{key}.png")
            Image.new("L", (320, 320), 255).save(output / "masks" / f"{key}.png")
            meta = _metadata(with_retries(_fetch, _map_url(bounds, metadata=True), 64 * 1024)[0], bounds, today)
            records.append({"id": f"nrw-test-{key}", "image": f"images/{key}.png", "mask": f"masks/{key}.png",
                            "building_ids": [f"geobasis-nrw:alkis:{bid}"], "region": name, "stratum": stratum,
                            "capture_date": meta["capture_date"], "native_gsd_m": meta["native_gsd_m"],
                            "georeference": {"crs": "EPSG:25832", "transform": [0.1, 0, bounds[0], 0, -0.1, bounds[3]]},
                            "gsd_m": 0.1, "width": 320, "height": 320, "split": "test",
                            "annotation_status": "unreviewed", "source_id": "geobasis-nrw-dop"})
            time.sleep(0.2)
        print(name, {s: sum(1 for c in chosen if c[0] == s) for s in ("osm_solar", "random")}, flush=True)

    rng.shuffle(records)  # Annotation order must not reveal the stratum
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "dataset.toml").write_text(
        f'name = "nrw-test-v1"\nlicense = "Datenlizenz Deutschland - Zero - Version 2.0"\nlicense_url = "{LICENSE_URL}"\n'
        f'seed = {SEED}\nregions = {json.dumps(used)}\nosm_enrichment = {json.dumps(json.dumps(osm_status))}\n', encoding="utf-8")
    print(f"{len(records)} crops written to {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
