# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Rasterize labeling-page polygons into the NRW test masks.

Input is a JSON export of the labeling page's ``labels`` collection: an object
mapping crop id to ``{"polygons": [{"cls": "pv"|"ignore", "pts": [[x, y], ...]}],
"unsure": bool, "status": "done", ...}``. Masks follow the project convention:
PV ``1``, ignore ``255``, background ``0``. Crops without a label stay all-255
and ``unreviewed``.

    uv run --locked --no-sync python scripts/finalize_nrw_test_labels.py data/nrw-test-v1 labels.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 320


def rasterize(polygons: list[dict]) -> Image.Image:
    mask = Image.new("L", (SIZE, SIZE), 0)
    draw = ImageDraw.Draw(mask)
    # PV first, ignore last: an ignore region overrides any PV it overlaps
    for cls, value in (("pv", 1), ("ignore", 255)):
        for polygon in polygons:
            if polygon.get("cls") == cls and len(polygon.get("pts", [])) >= 3:
                draw.polygon([(float(x), float(y)) for x, y in polygon["pts"]], fill=value)
    return mask


def main(dataset: Path, export: Path) -> None:
    labels = json.loads(export.read_text(encoding="utf-8"))
    manifest = dataset / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.open(encoding="utf-8")]
    labeled = 0
    for record in records:
        label = labels.get(record["id"])
        if not label or label.get("status") != "done":
            continue
        mask = rasterize(label.get("polygons", []))
        mask.save(dataset / record["mask"])
        record["mask_sha256"] = hashlib.sha256((dataset / record["mask"]).read_bytes()).hexdigest()
        record["annotation_status"] = "reviewed"
        record["annotation"] = {"polygons": label.get("polygons", []), "unsure": bool(label.get("unsure")),
                                "updated_at": label.get("updatedAt"), "method": "blind-human-ipad-v1"}
        labeled += 1
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"{labeled} of {len(records)} crops rasterized")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
