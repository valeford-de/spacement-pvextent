# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Run CAD-grade vectorization on holdout crops using best DeepLabV3+ Focal+Lovasz checkpoint."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from pvextent.cad.vectorize import render_cad_overlay, vectorize_mask
from pvextent.models.deeplabv3plus import build_deeplabv3plus_resnet50
from pvextent.models.resnet_unet import normalize_rgb

CKPT_PATH = "runs/deeplabv3plus-resnet50-focal-lovasz-v4-001/best.pt"
OUTPUT_DIR = Path("artifacts/cad-vectorization-v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CROPS = [
    {
        "id": "nrw-DENW07AL0000ytXZBL",
        "name": "Crop 1 (ytXZBL)",
        "image": "data/positive-holdout-frozen-v1/images/nrw-DENW07AL0000ytXZBL.png",
        "georef": {
            "crs": "EPSG:25832",
            "transform": [0.1, 0, 479139.2, 0, -0.1, 5672049.5],
        },
    },
    {
        "id": "nrw-DENW53AL0000StB8BL",
        "name": "Crop 2 (StB8BL)",
        "image": "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000StB8BL.png",
        "georef": {
            "crs": "EPSG:25832",
            "transform": [0.1, 0, 479226.0, 0, -0.1, 5671867.7],
        },
    },
    {
        "id": "nrw-DENW53AL0000mo4NBL",
        "name": "Crop 3 (mo4NBL)",
        "image": "data/positive-holdout-frozen-v1/images/nrw-DENW53AL0000mo4NBL.png",
        "georef": {
            "crs": "EPSG:25832",
            "transform": [0.1, 0, 479332.9, 0, -0.1, 5672106.2],
        },
    },
]


def main():
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = build_deeplabv3plus_resnet50(None)
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)["model_state_dict"]
    model.load_state_dict(sd)
    model.eval()

    all_summaries = {}

    for crop in CROPS:
        cid = crop["id"]
        cname = crop["name"]
        img_path = Path(crop["image"])
        print(f"\nProcessing {cname} ({cid})...")

        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        img_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        norm_t = normalize_rgb(img_t)

        with torch.no_grad():
            logits = model(norm_t).squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()

        # Threshold at 0.25 (calibrated operating point for Focal+Lovasz)
        binary_mask = (probs >= 0.25).astype(np.uint8)
        predicted_pv_px = int(binary_mask.sum())
        print(f"  Predicted PV pixels (@0.25): {predicted_pv_px} px")

        # CAD vectorization
        geojson = vectorize_mask(
            binary_mask,
            georeference=crop["georef"],
            gsd_m=0.1,
            min_area_px=50.0,
            regularize=True,
            tolerance_deg=20.0,
            module_efficiency=0.205,
        )

        n_arrays = geojson["summary"]["detected_pv_arrays"]
        tot_area = geojson["summary"]["total_pv_area_m2"]
        tot_kwp = geojson["summary"]["total_capacity_kwp"]
        tot_mods = geojson["summary"]["total_estimated_modules"]
        print(f"  CAD Vectorization Result:")
        print(f"    Detected PV Arrays: {n_arrays}")
        print(f"    Total Surface Area: {tot_area:.2f} m²")
        print(f"    Estimated DC Capacity: {tot_kwp:.2f} kWp")
        print(f"    Estimated Module Count: {tot_mods} modules (400W eq)")

        # Save GeoJSON
        geojson_path = OUTPUT_DIR / f"{cid}.geojson"
        with open(geojson_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)
        print(f"  Exported GeoJSON: {geojson_path}")

        # Render CAD Overlay
        overlay_path = OUTPUT_DIR / f"{cid}_cad_overlay.png"
        render_cad_overlay(img, geojson, overlay_path)
        print(f"  Rendered CAD Overlay: {overlay_path}")

        all_summaries[cid] = {
            "name": cname,
            "summary": geojson["summary"],
            "features": geojson["features"],
        }

    summary_file = OUTPUT_DIR / "cad_vectorization_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nAll holdouts processed successfully! Saved summary to {summary_file}")


if __name__ == "__main__":
    main()
