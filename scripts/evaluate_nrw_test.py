# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Evaluate frozen checkpoints on the labelled NRW test set v1.

Implements the metrics frozen in docs/research/nrw-test-v1-protocol.md: pixel
IoU, precision and recall (ignore excluded), crop-level detection on PV crops,
false alarms (a predicted component of at least 1 m²) on PV-free crops, each
overall and per stratum, with and without crops flagged unsure.

    uv run --locked --no-sync python scripts/evaluate_nrw_test.py data/nrw-test-v1 \\
        --model vidda=runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt \\
        --model tinden=runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt,runs/deeplabv3plus-resnet50-v6-nrw-pseudo-001/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pvextent.inference.tiler import predict_tiled
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb

MIN_COMPONENT_PX = 100  # 1 m² at 0.1 m/px
# Settings documented for each release profile in the sealed PV4GER benchmark
PROFILES = {"single": {"tta": "none", "temperature": 1.0, "threshold": 0.5},
            "ensemble": {"tta": "d4", "temperature": 0.4068, "threshold": 0.505}}


def component_sizes(mask: np.ndarray) -> list[int]:
    """Pixel counts of 4-connected components in a boolean mask."""
    seen, sizes = np.zeros_like(mask, dtype=bool), []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, size = [(y, x)], 0
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            size += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        sizes.append(size)
    return sizes


def load(path: str, device: str) -> torch.nn.Module:
    model = build_deeplabv3plus_resnet50(None)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state_dict"])
    return model.to(device).eval()


def predict(models: list, rgb: np.ndarray, device: str, profile: dict) -> np.ndarray:
    probs = [predict_tiled(model=m, image=rgb, tile_size=400, overlap=0.5, tta_mode=profile["tta"],
                           temperature=profile["temperature"], device=device, normalize_fn=normalize_rgb)["probabilities"]
             for m in models]
    return np.mean(probs, axis=0) >= profile["threshold"]


def summarize(rows: list[dict]) -> dict:
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    pv, free = [r for r in rows if r["ref_px"] > 0], [r for r in rows if r["ref_px"] == 0]
    return {"crops": len(rows), "pv_crops": len(pv), "pv_free_crops": len(free),
            "iou": tp / (tp + fp + fn) if tp + fp + fn else None,
            "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "crop_detection_rate": sum(r["tp"] > 0 for r in pv) / len(pv) if pv else None,
            "false_alarm_rate": sum(r["false_alarm"] for r in free) / len(free) if free else None,
            "tp": tp, "fp": fp, "fn": fn}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--model", action="append", required=True, help="name=ckpt[,ckpt...]")
    parser.add_argument("--output")
    args = parser.parse_args()
    dataset = Path(args.dataset)
    records = [json.loads(line) for line in open(dataset / "manifest.jsonl", encoding="utf-8")]
    records = [r for r in records if r["annotation_status"] == "reviewed"]
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    report = {"dataset": str(dataset), "labelled_crops": len(records), "models": {}}
    for spec in args.model:
        name, paths = spec.split("=", 1)
        checkpoints = paths.split(",")
        models = [load(p, device) for p in checkpoints]
        profile = PROFILES["ensemble" if len(models) > 1 else "single"]
        rows = []
        for record in records:
            rgb = np.asarray(Image.open(dataset / record["image"]).convert("RGB"), dtype=np.float32) / 255.0
            mask = np.asarray(Image.open(dataset / record["mask"]))
            valid, ref = mask != 255, mask == 1
            pred = predict(models, rgb, device, profile) & valid
            sizes = component_sizes(pred) if ref.sum() == 0 else []
            rows.append({"id": record["id"], "stratum": record["stratum"],
                         "unsure": record.get("annotation", {}).get("unsure", False), "ref_px": int(ref.sum()),
                         "tp": int((pred & ref).sum()), "fp": int((pred & ~ref).sum()), "fn": int((ref & ~pred).sum()),
                         "false_alarm": bool(ref.sum() == 0 and any(s >= MIN_COMPONENT_PX for s in sizes))})
        result = {"checkpoints": checkpoints, "profile": profile, "all": summarize(rows),
                  "without_unsure": summarize([r for r in rows if not r["unsure"]])}
        for stratum in ("random", "osm_solar"):
            result[stratum] = summarize([r for r in rows if r["stratum"] == stratum])
        result["per_crop"] = rows
        report["models"][name] = result
        print(name, json.dumps({k: v for k, v in result.items() if k not in ("per_crop", "checkpoints")}, indent=1))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
