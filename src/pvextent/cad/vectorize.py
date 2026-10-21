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

from .metrics import DEFAULT_GSD_M, DEFAULT_MODULE_EFFICIENCY, calculate_pv_metrics, polygon_area_pixels
from .regularize import detect_dominant_orientation, minimum_bounding_box, regularize_manhattan, simplify_douglas_peucker


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
    regularize: bool = True,
    tolerance_deg: float = 20.0,
    module_efficiency: float = DEFAULT_MODULE_EFFICIENCY,
) -> dict[str, Any]:
    """Execute end-to-end CAD-grade geospatial vectorization on a PV segmentation mask.

    Returns:
        dict: GeoJSON FeatureCollection with CAD-regularized polygon geometries and physical metrics.
    """
    raw_rings = marching_squares_contours(mask, min_area_px=min_area_px)
    features = []

    transform = georeference.get("transform") if georeference else None
    crs_name = georeference.get("crs", "EPSG:25832") if georeference else "EPSG:25832"

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

        # Compute dominant azimuth
        _, azimuth_deg = detect_dominant_orientation(regularized)
        obb = minimum_bounding_box(regularized)

        metrics["dominant_azimuth_deg"] = round(azimuth_deg, 1)
        metrics["aspect_ratio"] = round(obb["aspect_ratio"], 2)
        metrics["obb_length_m"] = round(obb["length"] * gsd_m, 2)
        metrics["obb_width_m"] = round(obb["width"] * gsd_m, 2)
        metrics["is_regularized"] = regularize

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
                **metrics,
            },
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {"name": crs_name},
        },
        "summary": {
            "detected_pv_arrays": len(features),
            "total_pv_area_m2": round(sum(f["properties"]["area_m2"] for f in features), 2),
            "total_capacity_kwp": round(sum(f["properties"]["estimated_capacity_kwp"] for f in features), 3),
            "total_estimated_modules": sum(f["properties"]["estimated_module_count"] for f in features),
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
    """Render high-contrast CAD vector overlay with rectilinear polygon outlines and kWp callout badges."""
    img = image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Color palette
    FILL_COLOR = (0, 220, 140, 90)       # High-visibility neon emerald transparent fill
    BORDER_COLOR = (0, 255, 170, 240)    # Crisp CAD cyan-green border
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
        # Draw sharp rectilinear boundary line
        draw.line(poly_pts + [poly_pts[0]], fill=BORDER_COLOR, width=2)

        # Callout badge at centroid
        cx, cy = props.get("centroid_pixel", (pts[0][0], pts[0][1]))
        kwp = props.get("estimated_capacity_kwp", 0.0)
        area = props.get("area_m2", 0.0)
        label = f"{kwp:.1f} kWp | {area:.1f} m²"

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
