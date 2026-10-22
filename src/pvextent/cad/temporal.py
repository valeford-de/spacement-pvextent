# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Multi-Temporal (Bi-Temporal) 4D Change Detection and Photovoltaic Asset Tracking.

Enables longitudinal tracking over multi-epoch aerial surveys:
- Automatic detection of new solar PV installations (Zubau)
- Capacity expansion and array augmentation tracking (Erweiterung)
- Decommissioning and module removal detection (Rückbau)
- Stable baseline asset verification (Bestand)
- Bayesian differential probability and epistemic uncertainty modulation
- Net capacity (ΔkWp), module count (Δmodules), and 3D area (Δm²) accounting
- Multi-temporal high-contrast CAD visual overlay rendering
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CHANGE_TYPE_NEW = "new_installation"
CHANGE_TYPE_EXPANSION = "expansion"
CHANGE_TYPE_DECOMMISSIONED = "decommissioned"
CHANGE_TYPE_STABLE = "stable"


def polygon_iou_raster(
    poly1: Sequence[tuple[float, float]],
    poly2: Sequence[tuple[float, float]],
) -> float:
    """Calculate the exact spatial Intersection-over-Union (IoU) of two 2D polygons via fast binary rasterization."""
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0

    xs = [p[0] for p in poly1] + [p[0] for p in poly2]
    ys = [p[1] for p in poly1] + [p[1] for p in poly2]

    min_x = max(0, int(math.floor(min(xs))))
    max_x = int(math.ceil(max(xs))) + 1
    min_y = max(0, int(math.floor(min(ys))))
    max_y = int(math.ceil(max(ys))) + 1

    w = max_x - min_x + 2
    h = max_y - min_y + 2

    # Fast 1-bit monochrome rasterization
    img1 = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img1).polygon([(x - min_x + 1, y - min_y + 1) for x, y in poly1], outline=1, fill=1)

    img2 = Image.new("1", (w, h), 0)
    ImageDraw.Draw(img2).polygon([(x - min_x + 1, y - min_y + 1) for x, y in poly2], outline=1, fill=1)

    arr1 = np.array(img1, dtype=bool)
    arr2 = np.array(img2, dtype=bool)

    intersection = int(np.logical_and(arr1, arr2).sum())
    union = int(np.logical_or(arr1, arr2).sum())

    return float(intersection / union) if union > 0 else 0.0


def calculate_temporal_deltas(props_t1: dict, props_t2: dict) -> dict[str, Any]:
    """Compute difference metrics between a paired T1 and T2 feature."""
    area_t1 = props_t1.get("area_m2", 0.0)
    area_t2 = props_t2.get("area_m2", 0.0)
    area_3d_t1 = props_t1.get("area_true_3d_m2", area_t1)
    area_3d_t2 = props_t2.get("area_true_3d_m2", area_t2)

    cap_t1 = props_t1.get("fitted_capacity_kwp") or props_t1.get("estimated_capacity_kwp", 0.0)
    cap_t2 = props_t2.get("fitted_capacity_kwp") or props_t2.get("estimated_capacity_kwp", 0.0)

    mods_t1 = props_t1.get("fitted_module_count") or props_t1.get("estimated_module_count", 0)
    mods_t2 = props_t2.get("fitted_module_count") or props_t2.get("estimated_module_count", 0)

    mwh_t1 = props_t1.get("estimated_annual_yield_mwh", 0.0)
    mwh_t2 = props_t2.get("estimated_annual_yield_mwh", 0.0)

    co2_t1 = props_t1.get("annual_co2_savings_tons", 0.0)
    co2_t2 = props_t2.get("annual_co2_savings_tons", 0.0)

    return {
        "delta_area_m2": round(area_t2 - area_t1, 2),
        "delta_true_3d_area_m2": round(area_3d_t2 - area_3d_t1, 2),
        "delta_capacity_kwp": round(cap_t2 - cap_t1, 3),
        "delta_module_count": mods_t2 - mods_t1,
        "delta_annual_yield_mwh": round(mwh_t2 - mwh_t1, 3),
        "delta_co2_savings_tons": round(co2_t2 - co2_t1, 2),
    }


def detect_temporal_changes(
    geojson_t1: dict[str, Any],
    geojson_t2: dict[str, Any],
    date_t1: str | None = None,
    date_t2: str | None = None,
    probabilities_t1: np.ndarray | None = None,
    probabilities_t2: np.ndarray | None = None,
    uncertainty_t1: np.ndarray | None = None,
    uncertainty_t2: np.ndarray | None = None,
    iou_stable_threshold: float = 0.70,
    iou_match_threshold: float = 0.15,
) -> dict[str, Any]:
    """Execute longitudinal change detection between two multi-epoch vectorization results.

    Identifies new installations, expansions, decommissionings, and stable arrays with full delta metrics.

    Args:
        geojson_t1: GeoJSON FeatureCollection at initial survey epoch T1.
        geojson_t2: GeoJSON FeatureCollection at subsequent survey epoch T2.
        date_t1: ISO date string for T1 survey (e.g. "2024-06-15").
        date_t2: ISO date string for T2 survey (e.g. "2025-06-20").
        probabilities_t1: Raster probabilities at T1 (optional).
        probabilities_t2: Raster probabilities at T2 (optional).
        uncertainty_t1: Total uncertainty raster at T1 (optional).
        uncertainty_t2: Total uncertainty raster at T2 (optional).
        iou_stable_threshold: Spatial overlap threshold above which an array is deemed STABLE (default: 0.70).
        iou_match_threshold: Spatial overlap threshold to consider an array a related EXPANSION (default: 0.15).

    Returns:
        dict: Multi-temporal GeoJSON FeatureCollection containing all detected events and net balance summary.
    """
    feats_t1 = geojson_t1.get("features", [])
    feats_t2 = geojson_t2.get("features", [])

    matched_t1_indices: set[int] = set()
    temporal_features = []

    # Calculate temporal interval in days if dates provided
    interval_days = None
    if date_t1 and date_t2:
        try:
            d1 = datetime.fromisoformat(date_t1.split("T")[0])
            d2 = datetime.fromisoformat(date_t2.split("T")[0])
            interval_days = abs((d2 - d1).days)
        except Exception:
            pass

    # 1. Match T2 features against T1 features
    for idx_t2, f_t2 in enumerate(feats_t2):
        poly_t2 = f_t2["properties"].get("pixel_polygon", [])
        if len(poly_t2) < 3:
            continue

        best_iou = 0.0
        best_t1_idx = -1

        for idx_t1, f_t1 in enumerate(feats_t1):
            poly_t1 = f_t1["properties"].get("pixel_polygon", [])
            if len(poly_t1) < 3:
                continue

            iou = polygon_iou_raster(poly_t1, poly_t2)
            if iou > best_iou:
                best_iou = iou
                best_t1_idx = idx_t1

        # Classify event
        props_t2 = f_t2["properties"]
        geom_t2 = f_t2["geometry"]
        feat_id = f"temporal-event-{len(temporal_features) + 1:03d}"

        if best_iou >= iou_stable_threshold and best_t1_idx >= 0:
            change_type = CHANGE_TYPE_STABLE
            matched_t1_indices.add(best_t1_idx)
            props_t1 = feats_t1[best_t1_idx]["properties"]
            deltas = calculate_temporal_deltas(props_t1, props_t2)
            matched_t1_id = props_t1.get("feature_id")
        elif best_iou >= iou_match_threshold and best_t1_idx >= 0:
            change_type = CHANGE_TYPE_EXPANSION
            matched_t1_indices.add(best_t1_idx)
            props_t1 = feats_t1[best_t1_idx]["properties"]
            deltas = calculate_temporal_deltas(props_t1, props_t2)
            matched_t1_id = props_t1.get("feature_id")
        else:
            change_type = CHANGE_TYPE_NEW
            matched_t1_id = None
            empty_t1 = {}
            deltas = calculate_temporal_deltas(empty_t1, props_t2)

        # Modulate change confidence using probability difference and uncertainty if available
        change_conf = 1.0
        if probabilities_t1 is not None and probabilities_t2 is not None:
            xs = [p[0] for p in poly_t2]
            ys = [p[1] for p in poly_t2]
            min_x, max_x = max(0, int(min(xs))), min(probabilities_t2.shape[1] - 1, int(max(xs)))
            min_y, max_y = max(0, int(min(ys))), min(probabilities_t2.shape[0] - 1, int(max(ys)))
            sub_p1 = probabilities_t1[min_y : max_y + 1, min_x : max_x + 1]
            sub_p2 = probabilities_t2[min_y : max_y + 1, min_x : max_x + 1]
            dp = float(np.mean(sub_p2 - sub_p1))
            if uncertainty_t1 is not None and uncertainty_t2 is not None:
                sub_u1 = uncertainty_t1[min_y : max_y + 1, min_x : max_x + 1]
                sub_u2 = uncertainty_t2[min_y : max_y + 1, min_x : max_x + 1]
                mean_u = float(np.mean(np.sqrt(sub_u1**2 + sub_u2**2)))
            else:
                mean_u = 0.02
            change_conf = max(0.1, min(1.0, round(abs(dp) * (1.0 - min(0.5, mean_u * 5.0)), 4)))

        feature = {
            "type": "Feature",
            "id": feat_id,
            "geometry": geom_t2,
            "properties": {
                "event_id": feat_id,
                "change_type": change_type,
                "spatial_overlap_iou": round(best_iou, 3),
                "matched_t1_feature": matched_t1_id,
                "date_t1": date_t1,
                "date_t2": date_t2,
                "interval_days": interval_days,
                "change_confidence": change_conf,
                **deltas,
                **props_t2,
            },
        }
        temporal_features.append(feature)

    # 2. Check for decommissioned T1 features (arrays in T1 missing in T2)
    for idx_t1, f_t1 in enumerate(feats_t1):
        if idx_t1 not in matched_t1_indices:
            poly_t1 = f_t1["properties"].get("pixel_polygon", [])
            if len(poly_t1) < 3:
                continue

            props_t1 = f_t1["properties"]
            geom_t1 = f_t1["geometry"]
            feat_id = f"temporal-event-{len(temporal_features) + 1:03d}"
            deltas = calculate_temporal_deltas(props_t1, {})

            feature = {
                "type": "Feature",
                "id": feat_id,
                "geometry": geom_t1,
                "properties": {
                    "event_id": feat_id,
                    "change_type": CHANGE_TYPE_DECOMMISSIONED,
                    "spatial_overlap_iou": 0.0,
                    "matched_t1_feature": props_t1.get("feature_id"),
                    "date_t1": date_t1,
                    "date_t2": date_t2,
                    "interval_days": interval_days,
                    "change_confidence": 0.95,
                    **deltas,
                    **props_t1,
                },
            }
            temporal_features.append(feature)

    # Summarize net temporal changes
    count_new = sum(1 for f in temporal_features if f["properties"]["change_type"] == CHANGE_TYPE_NEW)
    count_exp = sum(1 for f in temporal_features if f["properties"]["change_type"] == CHANGE_TYPE_EXPANSION)
    count_dec = sum(1 for f in temporal_features if f["properties"]["change_type"] == CHANGE_TYPE_DECOMMISSIONED)
    count_sta = sum(1 for f in temporal_features if f["properties"]["change_type"] == CHANGE_TYPE_STABLE)

    net_cap = sum(f["properties"]["delta_capacity_kwp"] for f in temporal_features)
    net_mods = sum(f["properties"]["delta_module_count"] for f in temporal_features)
    net_area = sum(f["properties"]["delta_area_m2"] for f in temporal_features)
    net_3d_area = sum(f["properties"]["delta_true_3d_area_m2"] for f in temporal_features)
    net_mwh = sum(f["properties"]["delta_annual_yield_mwh"] for f in temporal_features)
    net_co2 = sum(f["properties"]["delta_co2_savings_tons"] for f in temporal_features)

    return {
        "type": "FeatureCollection",
        "crs": geojson_t2.get("crs", {"type": "name", "properties": {"name": "EPSG:25832"}}),
        "summary": {
            "survey_epoch_t1": date_t1,
            "survey_epoch_t2": date_t2,
            "interval_days": interval_days,
            "total_events": len(temporal_features),
            "new_installations": count_new,
            "expansions": count_exp,
            "decommissioned_arrays": count_dec,
            "stable_arrays": count_sta,
            "net_delta_capacity_kwp": round(net_cap, 3),
            "net_delta_module_count": net_mods,
            "net_delta_pv_area_m2": round(net_area, 2),
            "net_delta_true_3d_area_m2": round(net_3d_area, 2),
            "net_delta_annual_yield_mwh": round(net_mwh, 3),
            "net_delta_co2_savings_tons": round(net_co2, 2),
        },
        "features": temporal_features,
    }


def render_temporal_cad_overlay(
    image_t2: Image.Image,
    temporal_geojson: dict[str, Any],
    output_path: Path | str,
    image_t1: Image.Image | None = None,
) -> None:
    """Render high-contrast 4D longitudinal CAD vector overlay with color-coded temporal change indicators.

    Palette:
        - Emerald Green: New installation (Zubau)
        - Electric Cyan: Expansion (Erweiterung)
        - Crimson Red: Decommissioned (Rückbau)
        - Subtle Slate Blue: Stable baseline (Bestand)
    """
    img = image_t2.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Change-class color codes
    COLOR_MAP = {
        CHANGE_TYPE_NEW: {
            "fill": (0, 230, 120, 100),       # Vibrant neon emerald
            "border": (0, 255, 140, 255),
            "badge_bg": (10, 35, 20, 230),
            "badge_text": (240, 255, 245, 255),
            "symbol": "ZUBAU (+)",
        },
        CHANGE_TYPE_EXPANSION: {
            "fill": (0, 200, 255, 90),        # Cyan electric blue
            "border": (0, 230, 255, 255),
            "badge_bg": (10, 25, 40, 230),
            "badge_text": (240, 250, 255, 255),
            "symbol": "ERWEITERUNG",
        },
        CHANGE_TYPE_DECOMMISSIONED: {
            "fill": (255, 40, 60, 90),         # Alert coral crimson
            "border": (255, 70, 90, 255),
            "badge_bg": (40, 10, 15, 230),
            "badge_text": (255, 240, 240, 255),
            "symbol": "RUECKBAU (-)",
        },
        CHANGE_TYPE_STABLE: {
            "fill": (80, 120, 180, 50),        # Subtle slate blue
            "border": (120, 160, 220, 200),
            "badge_bg": (20, 25, 35, 210),
            "badge_text": (230, 240, 255, 255),
            "symbol": "BESTAND (=)",
        },
    }

    for feature in temporal_geojson.get("features", []):
        props = feature.get("properties", {})
        change_type = props.get("change_type", CHANGE_TYPE_STABLE)
        pts = props.get("pixel_polygon", [])
        if len(pts) < 3:
            continue

        colors = COLOR_MAP.get(change_type, COLOR_MAP[CHANGE_TYPE_STABLE])
        poly_pts = [(p[0], p[1]) for p in pts]

        # Draw translucent filled boundary
        draw.polygon(poly_pts, fill=colors["fill"])

        # Draw individual physical module grids if present
        for mod in props.get("fitted_modules_px", []):
            mod_pts = [(p[0], p[1]) for p in mod]
            draw.line(mod_pts + [mod_pts[0]], fill=colors["border"], width=1)

        # Draw outer boundary
        draw.line(poly_pts + [poly_pts[0]], fill=colors["border"], width=2)

        # Construct callout text
        cx, cy = props.get("centroid_pixel", (pts[0][0], pts[0][1]))
        symbol = colors["symbol"]
        cap = props.get("fitted_capacity_kwp") or props.get("estimated_capacity_kwp", 0.0)
        d_cap = props.get("delta_capacity_kwp", 0.0)

        if change_type == CHANGE_TYPE_NEW:
            label = f"{symbol}: +{cap:.1f} kWp | +{props.get('fitted_module_count', 0)} mod"
        elif change_type == CHANGE_TYPE_EXPANSION:
            label = f"{symbol}: +{d_cap:.1f} kWp (Total {cap:.1f} kWp)"
        elif change_type == CHANGE_TYPE_DECOMMISSIONED:
            label = f"{symbol}: {d_cap:.1f} kWp (Removed)"
        else:
            label = f"{symbol}: {cap:.1f} kWp"

        badge_w, badge_h = len(label) * 7 + 12, 18
        bx1 = max(0, cx - badge_w / 2)
        by1 = max(0, cy - badge_h / 2)
        bx2 = min(img.width, bx1 + badge_w)
        by2 = min(img.height, by1 + badge_h)

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=3, fill=colors["badge_bg"], outline=colors["border"], width=1)
        draw.text((bx1 + 6, by1 + 2), label, fill=colors["badge_text"])

    composite = Image.alpha_composite(img, overlay).convert("RGB")

    # If side-by-side comparison with T1 requested
    if image_t1 is not None:
        t1_resized = image_t1.convert("RGB").resize(composite.size, Image.Resampling.BILINEAR)
        total_w = composite.width * 2 + 10
        side_by_side = Image.new("RGB", (total_w, composite.height), (20, 20, 20))
        side_by_side.paste(t1_resized, (0, 0))
        side_by_side.paste(composite, (composite.width + 10, 0))

        # Add headers: T1 (Before) and T2 (After with Change Detection)
        draw_sbs = ImageDraw.Draw(side_by_side)
        h_t1 = f"EPOCH T1 (HISTORICAL): {temporal_geojson.get('summary', {}).get('survey_epoch_t1', '')}"
        h_t2 = f"EPOCH T2 (CURRENT + DETECTIONS): {temporal_geojson.get('summary', {}).get('survey_epoch_t2', '')}"
        draw_sbs.rectangle([5, 5, len(h_t1) * 7 + 15, 25], fill=(0, 0, 0, 200))
        draw_sbs.text((10, 8), h_t1, fill=(255, 255, 255))
        draw_sbs.rectangle([composite.width + 15, 5, composite.width + 15 + len(h_t2) * 7 + 10, 25], fill=(0, 0, 0, 200))
        draw_sbs.text((composite.width + 20, 8), h_t2, fill=(0, 255, 170))
        side_by_side.save(output_path)
    else:
        composite.save(output_path)
