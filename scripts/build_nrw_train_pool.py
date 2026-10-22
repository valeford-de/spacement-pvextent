# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Acquire an unlabelled NRW training pool with paired colour-infrared crops.

Random small buildings from the ALKIS-vereinfacht WFS in districts that the NRW
test set does not use. Every crop centre keeps MIN_SOURCE_DISTANCE_M from every
evaluation location (PV4GER validation and test crops, NRW test set v1, the
frozen holdouts), so training never sees an evaluation neighbourhood; nearness to
earlier training crops is allowed. The CIR crop (same
flight) is stored for review only; models receive the RGB crop.

Whole regions are assigned to train or val so validation stays spatially
independent. Masks start as all-255 placeholders.

    uv run --locked --no-sync python scripts/build_nrw_train_pool.py data/nrw-train-pool-v1
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image
from pyproj import Transformer

from build_nrw_test_pool import wfs_buildings, with_retries
from pvextent.data.nrw import LICENSE_URL, MIN_SOURCE_DISTANCE_M, _bounds, _fetch, _map_url, _metadata, _roof

SEED = 20261023
PER_REGION = 80
MIN_CENTER_DISTANCE_M = 80.0
HALF_WIDTH_DEG, HALF_HEIGHT_DEG = 0.009, 0.006  # about 1.25 km x 1.33 km
# Districts not used by NRW test set v1 (lon, lat of the district centre).
REGIONS = {
    "aachen-brand": (6.165, 50.750), "krefeld": (6.565, 51.335), "wesel": (6.620, 51.660),
    "lippstadt": (8.345, 51.675), "siegen-weidenau": (8.020, 50.875), "guetersloh": (8.385, 51.905),
    "moenchengladbach": (6.440, 51.185), "kleve": (6.140, 51.790), "euskirchen": (6.790, 50.660),
    "bocholt": (6.615, 51.840), "coesfeld": (7.165, 51.945), "ahaus": (7.010, 52.075),
    "warendorf": (7.990, 51.950), "soest": (8.110, 51.570), "brilon": (8.570, 51.395),
    "olpe": (7.845, 51.030), "gummersbach": (7.565, 51.025), "dueren": (6.485, 50.800),
    "heinsberg": (6.095, 51.065), "herford": (8.670, 52.115), "hoexter": (9.380, 51.775),
    "lemgo": (8.900, 52.025), "iserlohn": (7.700, 51.375), "unna": (7.690, 51.535),
    "ahlen": (7.890, 51.765), "luebbecke": (8.620, 52.305), "viersen": (6.395, 51.255),
    "bergisch-gladbach": (7.135, 50.990),
}
VAL_REGIONS = {"warendorf", "olpe", "heinsberg", "lemgo"}


HOLDOUTS = ("nrw-test-v1", "positive-holdout-frozen-v1", "negative-holdout-frozen-v2",
            "negative-holdout-curation-v1", "nrw-negative-holdout-candidates-v1")


def protected_points(project: Transformer) -> list[tuple[float, float]]:
    """UTM centres of every crop used for validation, testing or holdouts."""
    centers = {}
    for line in open("data/pv4ger-seg-v0.9.1/provenance/samples.jsonl", encoding="utf-8"):
        sample = json.loads(line)
        centers[sample["id"]] = project.transform(*sample["approximate_center"])
    points = []
    for manifest in Path("data").glob("*/manifest.jsonl"):
        for line in manifest.open(encoding="utf-8"):
            record = json.loads(line)
            if manifest.parent.name not in HOLDOUTS and record.get("split") not in ("val", "test"):
                continue
            transform = (record.get("georeference") or {}).get("transform")
            if transform:
                points.append((transform[2] + 16.0, transform[5] - 16.0))
            elif record["id"] in centers:
                points.append(centers[record["id"]])
            else:
                raise SystemExit(f"No location for evaluation crop {record['id']} in {manifest}")
    return points


def cir_url(bounds: tuple) -> str:
    return _map_url(bounds).replace("LAYERS=nw_dop_rgb", "LAYERS=nw_dop_cir")


def fetch_crop(job: tuple) -> dict | None:
    output, region, bid, center, today = job
    bounds = _bounds(center)
    key = hashlib.sha256(f"{bid}:{bounds}".encode()).hexdigest()[:24]
    rgb = Image.open(io.BytesIO(with_retries(_fetch, _map_url(bounds), 4 * 1024 * 1024)[0])).convert("RGB")
    cir = Image.open(io.BytesIO(with_retries(_fetch, cir_url(bounds), 4 * 1024 * 1024)[0])).convert("RGB")
    if rgb.size != (320, 320) or cir.size != (320, 320):
        return None
    rgb.save(output / "images" / f"{key}.png")
    cir.save(output / "cir" / f"{key}.png")
    Image.new("L", (320, 320), 255).save(output / "masks" / f"{key}.png")
    meta = _metadata(with_retries(_fetch, _map_url(bounds, metadata=True), 64 * 1024)[0], bounds, today)
    return {"id": f"nrw-train-{key}", "image": f"images/{key}.png", "cir": f"cir/{key}.png",
            "mask": f"masks/{key}.png", "building_ids": [f"geobasis-nrw:alkis:{bid}"], "region": region,
            "capture_date": meta["capture_date"], "native_gsd_m": meta["native_gsd_m"],
            "georeference": {"crs": "EPSG:25832", "transform": [0.1, 0, bounds[0], 0, -0.1, bounds[3]]},
            "gsd_m": 0.1, "width": 320, "height": 320,
            "split": "val" if region in VAL_REGIONS else "train",
            "spatial_group": f"nrw-train-pool-v1-{region}",
            "annotation_status": "unreviewed", "source_id": "geobasis-nrw-dop"}


def main(output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    project = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    excluded = np.array(protected_points(project))
    print(f"{len(excluded)} protected evaluation locations", flush=True)
    rng = random.Random(SEED)

    def clearance(x: float, y: float) -> float:
        return float(np.hypot(excluded[:, 0] - x, excluded[:, 1] - y).min())

    for sub in ("images", "cir", "masks"):
        (output / sub).mkdir(parents=True)
    records, counts, today = [], {}, date.today()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for name, (lon, lat) in REGIONS.items():
            bbox = (lon - HALF_WIDTH_DEG, lat - HALF_HEIGHT_DEG, lon + HALF_WIDTH_DEG, lat + HALF_HEIGHT_DEG)
            corners = [project.transform(bbox[0], bbox[1]), project.transform(bbox[2], bbox[3])]
            buildings = with_retries(wfs_buildings, (corners[0][0], corners[0][1], corners[1][0], corners[1][1]))
            roofs = {}
            for feature in buildings:
                roof = _roof(feature)
                if roof and clearance(*roof[0]) >= MIN_SOURCE_DISTANCE_M:
                    roofs[feature["id"]] = roof[0]
            ids = sorted(roofs)
            rng.shuffle(ids)
            chosen, centers = [], []
            for bid in ids:
                if len(chosen) == PER_REGION:
                    break
                center = roofs[bid]
                if any(math.hypot(center[0] - cx, center[1] - cy) < MIN_CENTER_DISTANCE_M for cx, cy in centers):
                    continue
                centers.append(center)
                chosen.append((output, name, bid, center, today))
            fetched = [r for r in pool.map(fetch_crop, chosen) if r]
            records += fetched
            counts[name] = len(fetched)
            print(f"{name}: {len(fetched)} crops ({len(roofs)} eligible roofs)", flush=True)
            time.sleep(1)

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "dataset.toml").write_text(
        f'name = "nrw-train-pool-v1"\nlicense = "Datenlizenz Deutschland - Zero - Version 2.0"\nlicense_url = "{LICENSE_URL}"\n'
        f'seed = {SEED}\nval_regions = {json.dumps(sorted(VAL_REGIONS))}\ncounts = {json.dumps(json.dumps(counts))}\n',
        encoding="utf-8")
    print(f"{len(records)} crops written to {output}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
