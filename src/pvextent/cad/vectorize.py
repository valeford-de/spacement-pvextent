# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade vectorization from raster masks to regularized geospatial polygons."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .elevation import (
    DEFAULT_GHI_KWH_M2,
    azimuth_to_cardinal,
    calculate_3d_surface_area,
    calculate_annual_solar_yield,
    fit_roof_plane_from_dsm,
)
from .metrics import DEFAULT_GSD_M, DEFAULT_MODULE_EFFICIENCY, calculate_pv_metrics, polygon_area_pixels
from .regularize import detect_dominant_orientation, minimum_bounding_box, regularize_manhattan, simplify_douglas_peucker
from .spectral import extract_polygon_spectral_stats


def marching_squares_contours(binary_mask: np.ndarray, min_area_px: float = 30.0) -> list[list[tuple[float, float]]]:
    """Extract closed, non-self-intersecting polygon boundary rings using Marching Squares.

    Returns:
        list of polygon rings, each being a list of (x, y) float tuples.
    """
    h, w = binary_mask.shape
    # Pad by 1 pixel of zeros to guarantee closed loops at image boundaries
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:h + 1, 1:w + 1] = (binary_mask > 0).astype(np.uint8)

    # 4 corners of each 2x2 cell
    tl = padded[:-1, :-1]
    tr = padded[:-1, 1:]
    br = padded[1:, 1:]
    bl = padded[1:, :-1]

    # Cell configuration: 0 to 15
    cell_types = (tl << 3) | (tr << 2) | (br << 1) | bl

    # Edge midpoints in cell coordinates (r, c):
    # Top: (c + 0.5, r)
    # Right: (c + 1.0, r + 0.5)
    # Bottom: (c + 0.5, r + 1.0)
    # Left: (c, r + 0.5)
    # Lookup table for directed segments (p1 -> p2) for each of the 16 cases
    # 0 = Top, 1 = Right, 2 = Bottom, 3 = Left
    LOOKUP: dict[int, list[tuple[int, int]]] = {
        0: [],
        1: [(3, 2)],          # BL
        2: [(2, 1)],          # BR
        3: [(3, 1)],          # BL + BR
        4: [(1, 0)],          # TR
        5: [(3, 0), (1, 2)],  # BL + TR (saddle)
        6: [(2, 0)],          # BR + TR
        7: [(3, 0)],          # BL + BR + TR
        8: [(0, 3)],          # TL
        9: [(0, 2)],          # TL + BL
        10: [(0, 1), (2, 3)], # TL + BR (saddle)
        11: [(0, 1)],         # TL + BL + BR
        12: [(1, 3)],         # TL + TR
        13: [(1, 2)],         # TL + TR + BL
        14: [(2, 3)],         # TL + TR + BR
        15: [],
    }

    # Extract all directed line segments
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    rows, cols = np.where((cell_types > 0) & (cell_types < 15))
    for r, c in zip(rows, cols):
        ctype = cell_types[r, c]
        edges = LOOKUP.get(int(ctype), [])
        
        midpoints = {
            0: (float(c) + 0.5 - 1.0, float(r) - 1.0),         # Top
            1: (float(c) + 1.0 - 1.0, float(r) + 0.5 - 1.0),   # Right
            2: (float(c) + 0.5 - 1.0, float(r) + 1.0 - 1.0),   # Bottom
            3: (float(c) - 1.0, float(r) + 0.5 - 1.0),         # Left
        }
        for e_from, e_to in edges:
            segments.append((midpoints[e_from], midpoints[e_to]))

    if not segments:
        return []

    # Chain segments into closed loops
    # Build forward adjacency mapping with coordinate quantization to avoid float mismatch
    def quantize(pt: tuple[float, float]) -> tuple[int, int]:
        return (int(round(pt[0] * 2.0)), int(round(pt[1] * 2.0)))

    adj: dict[tuple[int, int], list[tuple[tuple[float, float], tuple[float, float]]]] = {}
    for s_from, s_to in segments:
        k = quantize(s_from)
        adj.setdefault(k, []).append((s_from, s_to))

    visited_segments = set()
    loops: list[list[tuple[float, float]]] = []

    for idx, (start_pt, next_pt) in enumerate(segments):
        seg_id = (quantize(start_pt), quantize(next_pt))
        if seg_id in visited_segments:
            continue

        loop = [start_pt]
        curr = next_pt
        visited_segments.add(seg_id)

        while True:
            loop.append(curr)
            curr_k = quantize(curr)
            if curr_k == quantize(start_pt) or len(loop) > len(segments) + 2:
                break
            # Find next outgoing segment
            candidates = adj.get(curr_k, [])
            found_next = False
            for cand_from, cand_to in candidates:
                cand_id = (curr_k, quantize(cand_to))
                if cand_id not in visited_segments:
                    visited_segments.add(cand_id)
                    curr = cand_to
                    found_next = True
                    break
            if not found_next:
                break

        # Check if valid closed loop with sufficient area
        if len(loop) >= 4 and quantize(loop[0]) == quantize(loop[-1]):
            clean_ring = loop[:-1]
            if polygon_area_pixels(clean_ring) >= min_area_px:
                loops.append(clean_ring)

    return loops


def transform_pixel_to_geo(
    vertices_px: Sequence[tuple[float, float]],
    transform: Sequence[float],
) -> list[tuple[float, float]]:
    """Transform pixel coordinates (x, y) to map CRS using GDAL 6-parameter affine transform.

    [a, b, c, d, e, f] where:
      X_geo = c + x * a + y * b
      Y_geo = f + x * d + y * e
    """
    a, b, c, d, e, f = transform
    out = []
    for x, y in vertices_px:
        gx = c + x * a + y * b
        gy = f + x * d + y * e
        out.append((round(gx, 3), round(gy, 3)))
    return out


def vectorize_mask(
    mask: np.ndarray,
    georeference: dict | None = None,
    gsd_m: float = DEFAULT_GSD_M,
    min_area_px: float = 30.0,
    min_area_m2: float = 0.0,
    regularize: bool = True,
    tolerance_deg: float = 20.0,
    module_efficiency: float = DEFAULT_MODULE_EFFICIENCY,
    probabilities: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    fit_modules: bool = True,
    dsm: np.ndarray | str | Path | None = None,
    default_tilt_deg: float = 0.0,
    default_azimuth_deg: float | None = None,
    ghi_kwh_m2: float = DEFAULT_GHI_KWH_M2,
    rgb: np.ndarray | Image.Image | str | Path | None = None,
    nir: np.ndarray | Image.Image | str | Path | None = None,
    veto_vegetated_roofs: bool = False,
) -> dict[str, Any]:
    """Execute end-to-end CAD-grade geospatial vectorization on a PV segmentation mask.

    Returns:
        dict: GeoJSON FeatureCollection with CAD-regularized polygon geometries, 3D surface area,
              physical BIM module packing, annual photovoltaic energy yield modeling, and multispectral NIR verification.
    """
    effective_min_area_px = max(min_area_px, (min_area_m2 / (gsd_m ** 2))) if min_area_m2 > 0 else min_area_px
    raw_rings = marching_squares_contours(mask, min_area_px=effective_min_area_px)
    features = []

    transform = georeference.get("transform") if georeference else None
    crs_name = georeference.get("crs", "EPSG:25832") if georeference else "EPSG:25832"

    dsm_arr = None
    if dsm is not None:
        if isinstance(dsm, np.ndarray):
            dsm_arr = dsm
        elif isinstance(dsm, (str, Path)):
            dp = Path(dsm)
            if dp.suffix == ".npy":
                dsm_arr = np.load(dp)
            else:
                dsm_arr = np.array(Image.open(dp), dtype=np.float32)

    rgb_arr = None
    if rgb is not None:
        if isinstance(rgb, np.ndarray):
            rgb_arr = rgb
        elif isinstance(rgb, Image.Image):
            rgb_arr = np.array(rgb.convert("RGB"))
        elif isinstance(rgb, (str, Path)):
            rgb_arr = np.array(Image.open(rgb).convert("RGB"))

    nir_arr = None
    if nir is not None:
        if isinstance(nir, np.ndarray):
            nir_arr = nir
        elif isinstance(nir, Image.Image):
            nir_arr = np.array(nir.convert("L"))
        elif isinstance(nir, (str, Path)):
            nir_arr = np.array(Image.open(nir).convert("L"))

    for idx, ring in enumerate(raw_rings, 1):
        # 1. Simplify
        simplified = simplify_douglas_peucker(ring, tolerance=1.0)
        if len(simplified) < 3:
            continue

        # 2. Regularize
        if regularize:
            regularized = regularize_manhattan(simplified, tolerance_deg=tolerance_deg)
        else:
            regularized = simplified

        if len(regularized) < 3:
            continue

        # 3. Physical & CAD metrics
        metrics = calculate_pv_metrics(
            regularized,
            gsd_m=gsd_m,
            module_efficiency=module_efficiency,
        )
        if min_area_m2 > 0.0 and metrics["area_m2"] < min_area_m2:
            continue

        # Compute dominant azimuth & OBB
        _, azimuth_deg = detect_dominant_orientation(regularized)
        obb = minimum_bounding_box(regularized)
        obb_len_m = round(obb["length"] * gsd_m, 2)
        obb_wid_m = round(obb["width"] * gsd_m, 2)

        # Estimate standard module table grid layout (standard module 1.76m x 1.04m)
        table_rows = max(1, round(obb_wid_m / 1.04)) if obb_wid_m >= 0.8 else 1
        table_cols = max(1, round(obb_len_m / 1.76)) if obb_len_m >= 1.2 else 1

        metrics["dominant_azimuth_deg"] = round(azimuth_deg, 1)
        metrics["aspect_ratio"] = round(obb["aspect_ratio"], 2)
        metrics["obb_length_m"] = obb_len_m
        metrics["obb_width_m"] = obb_wid_m
        metrics["estimated_table_layout"] = f"{table_rows}x{table_cols}"
        metrics["is_regularized"] = regularize

        # 4. 3D Elevation, Roof Tilt and Orientation
        if dsm_arr is not None:
            plane_fit = fit_roof_plane_from_dsm(dsm_arr, regularized, gsd_m=gsd_m)
            tilt_deg = plane_fit["tilt_deg"] if plane_fit["is_valid_3d_fit"] else float(default_tilt_deg)
            az_deg = plane_fit["azimuth_deg"] if plane_fit["is_valid_3d_fit"] else (float(default_azimuth_deg) if default_azimuth_deg is not None else float(azimuth_deg))
            metrics["dsm_fit_r2"] = plane_fit.get("fit_r2")
            metrics["dsm_fit_rmse_m"] = plane_fit.get("fit_rmse_m")
            metrics["elevation_mean_m"] = plane_fit.get("elevation_mean_m")
            metrics["elevation_min_m"] = plane_fit.get("elevation_min_m")
            metrics["elevation_max_m"] = plane_fit.get("elevation_max_m")
            metrics["elevation_source"] = "dsm_raster"
        else:
            tilt_deg = float(default_tilt_deg)
            az_deg = float(default_azimuth_deg) if default_azimuth_deg is not None else float(azimuth_deg)
            metrics["elevation_source"] = "manual_or_default"

        area_3d = calculate_3d_surface_area(metrics["area_m2"], tilt_deg)
        metrics["area_true_3d_m2"] = area_3d["true_3d_area_m2"]
        metrics["area_expansion_factor"] = area_3d["area_expansion_factor"]
        metrics["roof_tilt_deg"] = area_3d["tilt_deg"]
        metrics["roof_azimuth_deg"] = round(az_deg, 1)
        metrics["roof_azimuth_cardinal"] = azimuth_to_cardinal(az_deg, tilt_deg)

        # 5. Physical PV module grid fitting on 3D sloped surface
        modules_px = []
        if fit_modules:
            from .packing import fit_module_grid
            packed = fit_module_grid(regularized, az_deg, gsd_m=gsd_m, tilt_deg=tilt_deg)
            metrics["fitted_module_count"] = packed["module_count"]
            metrics["fitted_capacity_kwp"] = packed["total_capacity_kwp"]
            metrics["module_packing_orientation"] = packed["orientation"]
            modules_px = packed["modules_px"]

        # 6. Annual Solar Energy Yield & Carbon Displacement
        eff_kwp = metrics.get("fitted_capacity_kwp") or metrics["estimated_capacity_kwp"]
        yield_metrics = calculate_annual_solar_yield(
            capacity_kwp=eff_kwp,
            tilt_deg=tilt_deg,
            azimuth_deg=az_deg,
            ghi_kwh_m2=ghi_kwh_m2,
        )
        metrics["transposition_factor"] = yield_metrics["transposition_factor"]
        metrics["specific_annual_yield_kwh_per_kwp"] = yield_metrics["specific_annual_yield_kwh_per_kwp"]
        metrics["estimated_annual_yield_kwh"] = yield_metrics["estimated_annual_yield_kwh"]
        metrics["estimated_annual_yield_mwh"] = yield_metrics["estimated_annual_yield_mwh"]
        metrics["annual_co2_savings_tons"] = yield_metrics["annual_co2_savings_tons"]

        # Array-level Bayesian probability and uncertainty assessment
        if probabilities is not None:
            xs = [p[0] for p in regularized]
            ys = [p[1] for p in regularized]
            min_x, max_x = max(0, int(min(xs))), min(mask.shape[1] - 1, int(max(xs)))
            min_y, max_y = max(0, int(min(ys))), min(mask.shape[0] - 1, int(max(ys)))
            sub_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
            sub_p = probabilities[min_y : max_y + 1, min_x : max_x + 1]
            arr_p = sub_p[sub_mask > 0] if np.any(sub_mask > 0) else sub_p.ravel()
            mean_p = float(np.mean(arr_p)) if arr_p.size > 0 else 0.5
            p90_p = float(np.percentile(arr_p, 90)) if arr_p.size > 0 else mean_p
            metrics["mean_detection_probability"] = round(mean_p, 4)
            metrics["p90_detection_probability"] = round(p90_p, 4)
        else:
            mean_p = 1.0

        if uncertainty is not None:
            xs = [p[0] for p in regularized]
            ys = [p[1] for p in regularized]
            min_x, max_x = max(0, int(min(xs))), min(mask.shape[1] - 1, int(max(xs)))
            min_y, max_y = max(0, int(min(ys))), min(mask.shape[0] - 1, int(max(ys)))
            sub_mask = mask[min_y : max_y + 1, min_x : max_x + 1]
            sub_u = uncertainty[min_y : max_y + 1, min_x : max_x + 1]
            arr_u = sub_u[sub_mask > 0] if np.any(sub_mask > 0) else sub_u.ravel()
            mean_u = float(np.mean(arr_u)) if arr_u.size > 0 else 0.0
            metrics["mean_uncertainty"] = round(mean_u, 5)
        else:
            mean_u = 0.0

        # Multispectral NIR and Spectral Signature Analysis
        if rgb_arr is not None:
            spec_stats = extract_polygon_spectral_stats(
                polygon_px=regularized,
                rgb_arr=rgb_arr,
                nir_arr=nir_arr,
            )
            metrics.update(spec_stats)
            if veto_vegetated_roofs and spec_stats.get("is_spectral_false_positive"):
                continue
            is_spectral_fp = spec_stats.get("is_spectral_false_positive", False)
        else:
            is_spectral_fp = False

        if mean_p >= 0.80 and mean_u <= 0.025 and not is_spectral_fp:
            conf_tier = "high"
        elif mean_p >= 0.50 and mean_u <= 0.060 and not is_spectral_fp:
            conf_tier = "medium"
        else:
            conf_tier = "review_recommended"
        metrics["confidence_tier"] = conf_tier

        # 4. Geospatial projection
        if transform:
            geo_ring = transform_pixel_to_geo(regularized, transform)
            # Ensure closed ring for GeoJSON standard
            geo_coords = geo_ring + [geo_ring[0]]
        else:
            geo_coords = [(round(x, 2), round(y, 2)) for x, y in regularized] + [(round(regularized[0][0], 2), round(regularized[0][1], 2))]

        feature = {
            "type": "Feature",
            "id": f"pv-array-{idx:03d}",
            "geometry": {
                "type": "Polygon",
                "coordinates": [geo_coords],
            },
            "properties": {
                "feature_id": f"pv-array-{idx:03d}",
                "pixel_polygon": regularized,
                "fitted_modules_px": modules_px,
                **metrics,
            },
        }
        features.append(feature)

    high_conf = sum(1 for f in features if f["properties"].get("confidence_tier") == "high")
    review_rec = sum(1 for f in features if f["properties"].get("confidence_tier") == "review_recommended")
    mean_scene_conf = float(np.mean([f["properties"].get("mean_detection_probability", 1.0) for f in features])) if features else 1.0
    total_fitted_mods = sum(f["properties"].get("fitted_module_count", 0) for f in features)

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name},
        },
        "summary": {
            "detected_pv_arrays": len(features),
            "high_confidence_arrays": high_conf,
            "review_recommended_arrays": review_rec,
            "total_pv_area_m2": round(sum(f["properties"]["area_m2"] for f in features), 2),
            "total_true_3d_area_m2": round(sum(f["properties"].get("area_true_3d_m2", f["properties"]["area_m2"]) for f in features), 2),
            "total_capacity_kwp": round(sum(f["properties"]["estimated_capacity_kwp"] for f in features), 3),
            "total_fitted_capacity_kwp": round(sum(f["properties"].get("fitted_capacity_kwp", 0.0) for f in features), 3),
            "total_estimated_modules": sum(f["properties"]["estimated_module_count"] for f in features),
            "total_fitted_modules": total_fitted_mods,
            "total_annual_yield_mwh": round(sum(f["properties"].get("estimated_annual_yield_mwh", 0.0) for f in features), 3),
            "total_annual_co2_savings_tons": round(sum(f["properties"].get("annual_co2_savings_tons", 0.0) for f in features), 2),
            "mean_roof_tilt_deg": round(float(np.mean([f["properties"].get("roof_tilt_deg", 0.0) for f in features])), 1) if features else 0.0,
            "ghi_kwh_m2": ghi_kwh_m2,
            "mean_scene_confidence": round(mean_scene_conf, 4),
            "multispectral_nir_enabled": nir_arr is not None,
            "spectral_silicon_confirmed": sum(1 for f in features if f["properties"].get("spectral_verdict") == "silicon_photovoltaic_confirmed"),
            "vegetated_roof_vetoes": sum(1 for f in features if f["properties"].get("spectral_verdict") == "vegetated_roof_veto"),
            "regularized": regularize,
            "gsd_m": gsd_m,
        },
        "features": features,
    }
    return geojson


def render_cad_overlay(
    image: Image.Image,
    geojson: dict,
    output_path: Path | str,
) -> None:
    """Render high-contrast CAD vector overlay with rectilinear polygon outlines, module grids and kWp callout badges."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color palette
    FILL_COLOR = (0, 220, 140, 90)       # High-visibility neon emerald transparent fill
    BORDER_COLOR = (0, 255, 170, 240)    # Crisp CAD cyan-green border
    MODULE_COLOR = (0, 255, 230, 160)    # Fine cyan-blue module line
    CALLOUT_BG = (10, 25, 20, 210)       # Dark slate callout badge
    TEXT_COLOR = (240, 255, 250, 255)    # Pristine white text

    for feature in geojson.get("features", []):
        props = feature.get("properties", {})
        pts = props.get("pixel_polygon", [])
        if len(pts) < 3:
            continue

        poly_pts = [(p[0], p[1]) for p in pts]
        # Draw translucent filled polygon
        draw.polygon(poly_pts, fill=FILL_COLOR)
        # Draw individual physical PV module grid outlines
        for mod in props.get("fitted_modules_px", []):
            mod_pts = [(p[0], p[1]) for p in mod]
            draw.line(mod_pts + [mod_pts[0]], fill=MODULE_COLOR, width=1)
        # Draw sharp rectilinear boundary line
        draw.line(poly_pts + [poly_pts[0]], fill=BORDER_COLOR, width=2)

        # Callout badge at centroid
        cx, cy = props.get("centroid_pixel", (pts[0][0], pts[0][1]))
        kwp = props.get("fitted_capacity_kwp") or props.get("estimated_capacity_kwp", 0.0)
        area_3d = props.get("area_true_3d_m2") or props.get("area_m2", 0.0)
        tilt = props.get("roof_tilt_deg", 0.0)
        cardinal = props.get("roof_azimuth_cardinal", "")
        mwh = props.get("estimated_annual_yield_mwh", 0.0)
        if tilt > 3.0:
            label = f"{kwp:.1f} kWp | {area_3d:.1f} m² | {tilt:.0f}° {cardinal} | {mwh:.1f} MWh/a"
        else:
            label = f"{kwp:.1f} kWp | {area_3d:.1f} m² | {mwh:.1f} MWh/a"

        # Badge rectangle
        badge_w, badge_h = len(label) * 7 + 10, 18
        bx1 = max(0, cx - badge_w / 2)
        by1 = max(0, cy - badge_h / 2)
        bx2 = min(img.width, bx1 + badge_w)
        by2 = min(img.height, by1 + badge_h)

        draw.rounded_rectangle([bx1, by1, bx2, by2], radius=3, fill=CALLOUT_BG, outline=BORDER_COLOR, width=1)
        draw.text((bx1 + 5, by1 + 2), label, fill=TEXT_COLOR)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(output_path)
