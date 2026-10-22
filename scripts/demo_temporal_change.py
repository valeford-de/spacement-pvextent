# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Live demonstration script for multi-temporal change detection and 4D CAD asset tracking."""

import json
from pathlib import Path
import numpy as np
from PIL import Image

from pvextent.cad.temporal import (
    CHANGE_TYPE_DECOMMISSIONED,
    CHANGE_TYPE_EXPANSION,
    CHANGE_TYPE_NEW,
    CHANGE_TYPE_STABLE,
    detect_temporal_changes,
    render_temporal_cad_overlay,
)
from pvextent.cad.vectorize import vectorize_mask


def run_demo(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base aerial imagery simulation (or use an existing sample image)
    sample_img_p = Path("data/positive-holdout-frozen-v1/images/nrw-DENW07AL0000ytM5BL.png")
    if sample_img_p.exists():
        base_img = Image.open(sample_img_p).convert("RGB")
    else:
        # Fallback synthetic aerial roof tile
        base_img = Image.new("RGB", (400, 400), (95, 100, 105))

    img_t1 = base_img.copy()
    img_t2 = base_img.copy()

    # Create synthetic T1 and T2 masks representing a realistic 4D evolution:
    # Array A: Stable array at [(40, 40) -> (140, 100)] (100x60 = 6000 px = 60 m²)
    # Array B: Historical small array at [(250, 50) -> (320, 110)] (70x60 = 4200 px = 42 m²) -> will be EXPANDED in T2 to (250, 50) -> (390, 110)
    # Array C: Decommissioned array at [(50, 250) -> (120, 320)] (70x70 = 4900 px = 49 m²) -> removed in T2
    # Array D: Brand new installation (Zubau) in T2 at [(220, 240) -> (360, 350)] (140x110 = 15400 px = 154 m²)

    mask_t1 = np.zeros((400, 400), dtype=np.uint8)
    mask_t1[40:100, 40:140] = 1   # Array A (Stable)
    mask_t1[50:110, 250:320] = 1  # Array B (Before expansion)
    mask_t1[250:320, 50:120] = 1  # Array C (To be decommissioned)

    mask_t2 = np.zeros((400, 400), dtype=np.uint8)
    mask_t2[40:100, 40:140] = 1   # Array A (Stable)
    mask_t2[50:110, 250:390] = 1  # Array B (Expanded: +70px width)
    # Array C is 0 (Decommissioned)
    mask_t2[240:350, 220:360] = 1 # Array D (Brand new Zubau)

    # Vectorize both epochs
    georef = {
        "crs": "EPSG:25832",
        "transform": [0.1, 0.0, 350000.0, 0.0, -0.1, 5650000.0],
    }

    gj_t1 = vectorize_mask(mask_t1, georeference=georef, gsd_m=0.10)
    gj_t2 = vectorize_mask(mask_t2, georeference=georef, gsd_m=0.10)

    # Run multi-temporal change detection
    date_t1 = "2024-05-15"
    date_t2 = "2025-09-20"
    temporal_result = detect_temporal_changes(
        geojson_t1=gj_t1,
        geojson_t2=gj_t2,
        date_t1=date_t1,
        date_t2=date_t2,
    )

    # Save outputs
    gj_out = output_dir / "temporal_changes.geojson"
    with open(gj_out, "w", encoding="utf-8") as f:
        json.dump(temporal_result, f, indent=2)

    overlay_out = output_dir / "temporal_cad_overlay.png"
    render_temporal_cad_overlay(
        image_t2=img_t2,
        temporal_geojson=temporal_result,
        output_path=overlay_out,
        image_t1=img_t1,
    )

    print("Demo completed successfully:")
    print(json.dumps(temporal_result["summary"], indent=2))
    return temporal_result


if __name__ == "__main__":
    run_demo(Path("runs/demo_temporal"))
