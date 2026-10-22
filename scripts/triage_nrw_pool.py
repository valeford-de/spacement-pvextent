# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Run the frozen release models over an unlabelled NRW pool and flag crops for review.

Writes per-crop probability maps (uint8 PNG, probability x 255) for Tinden
(ensemble profile) and Vidda (single profile), and a triage file listing the
candidate components: 4-connected regions of at least MIN_COMPONENT_PX where
Tinden >= LOW_THRESHOLD or Vidda >= 0.5. The low threshold and the union with
the higher-recall Vidda keep weakly detected PV in review instead of silently
labelling it background. Crops without candidates are the automatic negatives.

    python scripts/triage_nrw_pool.py data/nrw-train-pool-v1 \\
        --vidda runs/deeplabv3plus-resnet50-v5-photometric-001/best.pt \\
        --v6 runs/deeplabv3plus-resnet50-v6-nrw-pseudo-001/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from evaluate_nrw_test import PROFILES, load
from pvextent.inference.tiler import predict_tiled
from pvextent.models.resnet_unet import normalize_rgb

LOW_THRESHOLD = 0.3
MIN_COMPONENT_PX = 50  # 0.5 m² at 0.1 m/px


def components(mask: np.ndarray) -> list[np.ndarray]:
    """Pixel index arrays of 4-connected components in a boolean mask."""
    seen, result = np.zeros_like(mask, dtype=bool), []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack, pixels = [(y, x)], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        result.append(np.array(pixels))
    return result


def probability(models: list, rgb: np.ndarray, device: str, profile: dict) -> np.ndarray:
    return np.mean([predict_tiled(model=m, image=rgb, tile_size=400, overlap=0.5, tta_mode=profile["tta"],
                                  temperature=profile["temperature"], device=device,
                                  normalize_fn=normalize_rgb)["probabilities"] for m in models], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--vidda", required=True)
    parser.add_argument("--v6", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vidda, v6 = load(args.vidda, device), load(args.v6, device)
    (dataset / "prob" / "tinden").mkdir(parents=True, exist_ok=True)
    (dataset / "prob" / "vidda").mkdir(parents=True, exist_ok=True)
    triage = []
    records = [json.loads(line) for line in open(dataset / "manifest.jsonl", encoding="utf-8")]
    for index, record in enumerate(records, 1):
        key = Path(record["image"]).stem
        rgb = np.asarray(Image.open(dataset / record["image"]).convert("RGB"), dtype=np.float32) / 255.0
        # Candidates use the stored 8-bit maps, so label assembly can reproduce them exactly
        q_tinden = np.round(probability([vidda, v6], rgb, device, PROFILES["ensemble"]) * 255).astype(np.uint8)
        q_vidda = np.round(probability([vidda], rgb, device, PROFILES["single"]) * 255).astype(np.uint8)
        Image.fromarray(q_tinden).save(dataset / "prob" / "tinden" / f"{key}.png")
        Image.fromarray(q_vidda).save(dataset / "prob" / "vidda" / f"{key}.png")
        p_tinden, p_vidda = q_tinden / 255, q_vidda / 255
        candidates = []
        for pixels in components((p_tinden >= LOW_THRESHOLD) | (p_vidda >= 0.5)):
            if len(pixels) < MIN_COMPONENT_PX:
                continue
            ys, xs = pixels[:, 0], pixels[:, 1]
            candidates.append({"px": len(pixels), "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                               "tinden_max": round(float(p_tinden[ys, xs].max()), 3),
                               "tinden_px": int((p_tinden[ys, xs] >= PROFILES["ensemble"]["threshold"]).sum()),
                               "vidda_px": int((p_vidda[ys, xs] >= 0.5).sum())})
        candidates.sort(key=lambda c: -c["px"])
        triage.append({"id": record["id"], "key": key, "split": record["split"], "region": record["region"],
                       "candidates": candidates})
        if index % 100 == 0:
            print(f"{index}/{len(records)} crops, {sum(bool(t['candidates']) for t in triage)} flagged", flush=True)
    (dataset / "triage.json").write_text(json.dumps(triage, indent=0), encoding="utf-8")
    print(f"{sum(bool(t['candidates']) for t in triage)} of {len(triage)} crops flagged")


if __name__ == "__main__":
    main()
