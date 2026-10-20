# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Create an offline validation-only contact sheet from a verified best.pt.

This is a qualitative inspection artifact, not an independent evaluation or an
inference service. Real imagery and predictions must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from pvextent.data.segmentation import SegmentationDataset
from pvextent.models.resnet_unet import build_resnet_unet, normalize_rgb
from pvextent.training.baseline import _read_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset")
    parser.add_argument("checkpoint", help="Verified local best.pt and .sha256 companion")
    parser.add_argument("output", help="New PNG path under artifacts/ or outside Git")
    parser.add_argument("--samples", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.samples <= 12:
        parser.error("--samples must be between 1 and 12")
    output = Path(args.output)
    if output.exists() or output.suffix.lower() != ".png":
        parser.error("output must be a new PNG path")
    document, checkpoint_hash = _read_checkpoint(Path(args.checkpoint), torch, resume=False)
    config = document["config"]
    dataset = SegmentationDataset(args.dataset, split="val",
                                  max_samples=config["max_val_samples"] or None,
                                  seed=config["seed"])
    if (dataset.dataset_fingerprint != config["dataset_fingerprint"]
            or dataset.selection_fingerprint != config["val_selection_fingerprint"]):
        parser.error("source validation snapshot differs from the checkpoint")
    selected = sorted(range(len(dataset)), key=lambda index: hashlib.sha256(
        f"{config['seed']}:{dataset.records[index]['id']}".encode()).hexdigest())[:args.samples]
    torch.set_num_threads(1)
    model = build_resnet_unet(None).cpu().eval()
    model.load_state_dict(document["model_state_dict"], strict=True)
    width, height = dataset.dimensions
    gap, top, row_height = 10, 62, height + 34
    canvas = Image.new("RGB", (4 * (width + gap) + gap, top + len(selected) * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), "Spacement PVExtent: validation examples only; upstream labels, fixed threshold", fill="black")
    draw.text((gap, 22), "Error map: green=TP; red=FP; blue=FN; gray=TN; orange=ignore", fill="black")
    for column, label in enumerate(("RGB image", "Source PV labels", "Predicted PV", "Pixel errors")):
        draw.text((gap + column * (width + gap), 42), label, fill="black")
    audit = []
    with torch.no_grad():
        for row_index, index in enumerate(selected):
            item = dataset[index]
            probability = model(normalize_rgb(item["image"].unsqueeze(0))).sigmoid()[0, 0]
            prediction = probability.ge(config["threshold"])
            target, valid = item["target"][0].eq(1), item["valid"][0]
            image = Image.frombytes("RGB", (width, height), bytes(
                item["image"].mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().flatten().tolist()))
            truth_view = Image.frombytes("L", (width, height), bytes(target.to(torch.uint8).mul(255).flatten().tolist()))
            prediction_view = Image.frombytes("L", (width, height), bytes(prediction.to(torch.uint8).mul(255).flatten().tolist()))
            errors = torch.zeros((height, width, 3), dtype=torch.uint8)
            errors[:] = torch.tensor((45, 45, 45), dtype=torch.uint8)
            errors[valid & prediction & target] = torch.tensor((0, 190, 120), dtype=torch.uint8)
            errors[valid & prediction & ~target] = torch.tensor((220, 45, 45), dtype=torch.uint8)
            errors[valid & ~prediction & target] = torch.tensor((50, 110, 230), dtype=torch.uint8)
            errors[~valid] = torch.tensor((245, 170, 50), dtype=torch.uint8)
            error_view = Image.frombytes("RGB", (width, height), bytes(errors.flatten().tolist()))
            y = top + row_index * row_height
            draw.text((gap, y), item["id"], fill="black")
            for column, panel in enumerate((image, truth_view.convert("RGB"), prediction_view.convert("RGB"), error_view)):
                canvas.paste(panel, (gap + column * (width + gap), y + 18))
            audit.append({"id": item["id"], "split": "val"})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        canvas.save(stream, format="PNG")
    print(json.dumps({"ok": True, "output": str(output.resolve()), "checkpoint_sha256": checkpoint_hash,
                      "dataset_fingerprint": dataset.dataset_fingerprint, "threshold": config["threshold"],
                      "samples": audit, "test_used": False}, indent=2))


if __name__ == "__main__":
    main()
