# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Live demonstration script for realistic multi-temporal 4D photovoltaic change detection."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from pvextent.cad.temporal import detect_temporal_changes, render_temporal_cad_overlay
from pvextent.cad.vectorize import vectorize_mask


def generate_realistic_aerial_scene(width: int = 500, height: int = 500) -> tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    """Generate a realistic aerial survey pair showing distinct buildings across survey epochs T1 and T2."""
    # Background terrain (soft aerial asphalt/grass textures)
    base_t1 = Image.new("RGB", (width, height), (135, 140, 130))
    draw_t1 = ImageDraw.Draw(base_t1)

    # Road and lawns
    draw_t1.rectangle([0, 235, width, 265], fill=(70, 72, 75))
    draw_t1.rectangle([0, 0, 240, 235], fill=(125, 145, 115))
    draw_t1.rectangle([260, 0, width, 235], fill=(120, 140, 110))
    draw_t1.rectangle([0, 265, 240, height], fill=(125, 142, 112))
    draw_t1.rectangle([260, 265, width, height], fill=(122, 143, 114))

    # Four distinct residential/commercial buildings:
    # Building 1: Top-Left (Red clay tile roof)
    # Building 2: Top-Right (Dark slate tile roof)
    # Building 3: Bottom-Left (Terracotta tile roof)
    # Building 4: Bottom-Right (Zinc gray roof)
    draw_t1.rectangle([30, 30, 210, 200], fill=(175, 75, 55), outline=(130, 50, 35), width=2)
    draw_t1.rectangle([290, 30, 470, 200], fill=(95, 105, 115), outline=(65, 75, 85), width=2)
    draw_t1.rectangle([30, 290, 210, 460], fill=(165, 80, 60), outline=(125, 55, 40), width=2)
    draw_t1.rectangle([290, 290, 470, 460], fill=(130, 135, 140), outline=(95, 100, 105), width=2)

    # Copy base to T2
    base_t2 = base_t1.copy()
    draw_t2 = ImageDraw.Draw(base_t2)

    # Masks for T1 and T2
    mask_t1 = np.zeros((height, width), dtype=np.uint8)
    mask_t2 = np.zeros((height, width), dtype=np.uint8)

    # --- Array 1 (Building 1, Top-Left): STABLE (Bestand) ---
    # Present in both T1 and T2 at [50, 50, 190, 140]
    draw_t1.rectangle([50, 50, 190, 140], fill=(25, 35, 55), outline=(40, 55, 85), width=1)
    draw_t2.rectangle([50, 50, 190, 140], fill=(25, 35, 55), outline=(40, 55, 85), width=1)
    mask_t1[50:140, 50:190] = 1
    mask_t2[50:140, 50:190] = 1

    # --- Array 2 (Building 2, Top-Right): EXPANSION (Erweiterung) ---
    # T1 has small array at [310, 50, 380, 140] (70x90)
    draw_t1.rectangle([310, 50, 380, 140], fill=(25, 35, 55), outline=(40, 55, 85), width=1)
    mask_t1[50:140, 310:380] = 1
    # T2 expands array across roof to [310, 50, 450, 140] (140x90, doubled!)
    draw_t2.rectangle([310, 50, 450, 140], fill=(25, 35, 55), outline=(40, 55, 85), width=1)
    mask_t2[50:140, 310:450] = 1

    # --- Array 3 (Building 3, Bottom-Left): DECOMMISSIONED (Rückbau) ---
    # T1 had an older array at [50, 310, 190, 400]
    draw_t1.rectangle([50, 310, 190, 400], fill=(35, 45, 60), outline=(50, 65, 85), width=1)
    mask_t1[310:400, 50:190] = 1
    # In T2, panels were completely dismantled; bare terracotta tiles remain!
    # (mask_t2 remains 0 here)

    # --- Array 4 (Building 4, Bottom-Right): NEW INSTALLATION (Zubau) ---
    # In T1, roof was empty (mask_t1 is 0 here)
    # In T2, a brand-new high-capacity array was installed at [310, 310, 450, 430]!
    draw_t2.rectangle([310, 310, 450, 430], fill=(20, 30, 50), outline=(35, 50, 80), width=1)
    mask_t2[310:430, 310:450] = 1

    return base_t1, base_t2, mask_t1, mask_t2


def run_demo(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    img_t1, img_t2, mask_t1, mask_t2 = generate_realistic_aerial_scene()

    georef = {
        "crs": "EPSG:25832",
        "transform": [0.1, 0.0, 350000.0, 0.0, -0.1, 5650000.0],
    }

    # Vectorize both epochs
    gj_t1 = vectorize_mask(mask_t1, georeference=georef, gsd_m=0.10)
    gj_t2 = vectorize_mask(mask_t2, georeference=georef, gsd_m=0.10)

    # Execute longitudinal multi-temporal change detection
    temporal_result = detect_temporal_changes(
        geojson_t1=gj_t1,
        geojson_t2=gj_t2,
        date_t1="2024-05-15",
        date_t2="2025-09-20",
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

    print("Realistic multi-temporal change detection demo completed:")
    print(json.dumps(temporal_result["summary"], indent=2))
    return temporal_result


if __name__ == "__main__":
    run_demo(Path("runs/demo_temporal"))
