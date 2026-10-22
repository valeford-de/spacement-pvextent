# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Build training masks for the NRW training pool and merge it into a training dataset.

Implements the label construction in docs/research/nrw-train-v1-protocol.md from
triage.json (candidate components) and review.json (one verdict per component:
"pv", "not_pv" or "unsure", plus "missed" for crops with uncovered PV), then
writes a new dataset directory: the parent dataset's records plus the pool
crops, hard-linked, with the pool declared as an extra source.

    python scripts/assemble_nrw_train_v1.py data/nrw-train-pool-v1 data/pvextent-training-v4 data/pvextent-training-v7
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from triage_nrw_pool import LOW_THRESHOLD, MIN_COMPONENT_PX, components

PV_THRESHOLD = 0.505
IGNORE_ABOVE = 0.1
SOURCE = """
[sources."nrw-train-pool-v1"]
name = "Geobasis NRW current digital orthophotos"
url = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
version = "nrw-train-pool-v1; labels nrw-train-v1 (docs/research/nrw-train-v1-protocol.md)"
image_license = "dl-de-zero-2.0"
annotation_license = "Apache-2.0"
attribution = "Geobasis NRW/openNRW. Labels: copyright 2026 Valeford, Apache-2.0. Model-proposed components judged by an AI reviewer from RGB and colour-infrared crops; label accuracy measured on a human-labelled audit sample."
"""


def dilate(mask: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        grown = mask.copy()
        grown[1:] |= mask[:-1]
        grown[:-1] |= mask[1:]
        grown[:, 1:] |= mask[:, :-1]
        grown[:, :-1] |= mask[:, 1:]
        mask = grown
    return mask


def build_mask(p_tinden: np.ndarray, p_vidda: np.ndarray, verdicts: list[str]) -> np.ndarray:
    mask = np.where(p_tinden > IGNORE_ABOVE, 255, 0).astype(np.uint8)
    found = [c for c in components((p_tinden >= LOW_THRESHOLD) | (p_vidda >= 0.5)) if len(c) >= MIN_COMPONENT_PX]
    found.sort(key=len, reverse=True)  # Same order as triage.json
    if len(found) != len(verdicts):
        raise ValueError("Component count differs from triage")
    pv_layers = []
    for pixels, verdict in zip(found, verdicts):
        region = np.zeros(mask.shape, dtype=bool)
        region[pixels[:, 0], pixels[:, 1]] = True
        if verdict == "not_pv":
            mask[dilate(region, 3)] = 0
        elif verdict == "unsure":
            mask[dilate(region, 2)] = 255
        elif verdict == "pv":
            pv_layers.append(region)
        else:
            raise ValueError(f"Unknown verdict {verdict!r}")
    for region in pv_layers:  # PV last, so a hard-negative ring never erases it
        mask[dilate(region, 2)] = 255
        mask[region & (p_tinden >= PV_THRESHOLD)] = 1
    return mask


def main(pool: Path, parent: Path, output: Path) -> None:
    if output.exists():
        raise SystemExit(f"Output already exists: {output}")
    triage = {t["key"]: t for t in json.loads((pool / "triage.json").read_text())}
    review = json.loads((pool / "review.json").read_text())
    (output / "masks" / "nrw-train-pool-v1").mkdir(parents=True)
    (output / "images" / "nrw-train-pool-v1").mkdir(parents=True)
    for item in parent.iterdir():  # Parent data stays hard-linked and unchanged
        if item.name in ("images", "masks"):
            for sub in item.iterdir():
                shutil.copytree(sub, output / item.name / sub.name, copy_function=lambda s, d: Path(d).hardlink_to(s)) \
                    if sub.is_dir() else (output / item.name / sub.name).hardlink_to(sub)
        elif item.name == "provenance":
            shutil.copytree(item, output / "provenance")
    records = [json.loads(line) for line in open(parent / "manifest.jsonl", encoding="utf-8")]
    stats = {"auto_negative": 0, "reviewed": 0, "missed_excluded": 0, "pv_px": 0, "ignore_px": 0}
    for line in open(pool / "manifest.jsonl", encoding="utf-8"):
        record = json.loads(line)
        key = Path(record["image"]).stem
        p_tinden = np.asarray(Image.open(pool / "prob" / "tinden" / f"{key}.png")) / 255
        p_vidda = np.asarray(Image.open(pool / "prob" / "vidda" / f"{key}.png")) / 255
        verdict = review.get(key, {"components": [], "missed": False})
        if triage[key]["candidates"] and key not in review:
            raise SystemExit(f"Flagged crop {key} has no review verdict")
        if verdict["missed"]:
            stats["missed_excluded"] += 1
            continue
        stats["reviewed" if triage[key]["candidates"] else "auto_negative"] += 1
        mask = build_mask(p_tinden, p_vidda, verdict["components"])
        stats["pv_px"] += int((mask == 1).sum())
        stats["ignore_px"] += int((mask == 255).sum())
        image_rel, mask_rel = f"images/nrw-train-pool-v1/{key}.png", f"masks/nrw-train-pool-v1/{key}.png"
        (output / image_rel).hardlink_to(pool / record["image"])
        Image.fromarray(mask).save(output / mask_rel)
        records.append({key_: record[key_] for key_ in ("building_ids", "capture_date", "georeference", "gsd_m",
                                                        "height", "width", "split", "spatial_group")}
                       | {"id": record["id"], "image": image_rel, "mask": mask_rel, "source_id": "nrw-train-pool-v1",
                          "annotation_status": "reviewed",
                          "image_sha256": hashlib.sha256((output / image_rel).read_bytes()).hexdigest(),
                          "mask_sha256": hashlib.sha256((output / mask_rel).read_bytes()).hexdigest()})
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    toml = (parent / "dataset.toml").read_text(encoding="utf-8")
    toml = toml.replace(toml.split("\n")[2], f'dataset_version = "2026-10-22.training-v7"')
    (output / "dataset.toml").write_text(toml.rstrip() + "\n" + SOURCE, encoding="utf-8")
    (output / "provenance" / "nrw-train-v1-assembly.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
