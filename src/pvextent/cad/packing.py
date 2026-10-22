# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade physical photovoltaic module grid packing and BIM layout generation."""

from __future__ import annotations

import math
from typing import Sequence

# Standard commercial 400W monocrystalline silicon module dimensions
STANDARD_MODULE_LENGTH_M = 1.76   # 1760 mm
STANDARD_MODULE_WIDTH_M = 1.04    # 1040 mm
STANDARD_MODULE_GAP_M = 0.02      # 20 mm inter-module clamping gap
STANDARD_MODULE_WATTAGE_KW = 0.400  # 400 Wp


def point_in_polygon(x: float, y: float, polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting algorithm to determine if point (x, y) is strictly inside 2D polygon."""
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def fit_module_grid(
    polygon_px: Sequence[tuple[float, float]],
    dominant_angle_deg: float,
    gsd_m: float = 0.10,
    module_length_m: float = STANDARD_MODULE_LENGTH_M,
    module_width_m: float = STANDARD_MODULE_WIDTH_M,
    gap_m: float = STANDARD_MODULE_GAP_M,
    tilt_deg: float = 0.0,
) -> dict:
    """Pack standardized physical PV module rectangles inside a regularized CAD polygon.

    Evaluates both portrait and landscape orientations to maximize panel packing density.

    Returns:
        dict: {
            'module_count': int,
            'orientation': 'portrait' | 'landscape',
            'modules_px': list of 4-point polygon rings [(x, y), ...] in pixel space,
            'total_capacity_kwp': float,
        }
    """
    if len(polygon_px) < 3:
        return {"module_count": 0, "orientation": "portrait", "modules_px": [], "total_capacity_kwp": 0.0}

    # Centroid and bounding extent
    cx = sum(p[0] for p in polygon_px) / len(polygon_px)
    cy = sum(p[1] for p in polygon_px) / len(polygon_px)

    theta_rad = math.radians(dominant_angle_deg)
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    # Transform polygon to OBB local coordinate frame:
    # u = (x - cx) * cos + (y - cy) * sin
    # v = -(x - cx) * sin + (y - cy) * cos
    local_pts = [
        ((x - cx) * cos_t + (y - cy) * sin_t, -(x - cx) * sin_t + (y - cy) * cos_t)
        for x, y in polygon_px
    ]

    min_u = min(p[0] for p in local_pts)
    max_u = max(p[0] for p in local_pts)
    min_v = min(p[1] for p in local_pts)
    max_v = max(p[1] for p in local_pts)

    span_u_m = (max_u - min_u) * gsd_m
    span_v_m = (max_v - min_v) * gsd_m

    best_result = {
        "module_count": 0,
        "orientation": "portrait",
        "modules_px": [],
        "total_capacity_kwp": 0.0,
    }

    # Test both orientations: Portrait (L along u, W along v) vs Landscape (W along u, L along v)
    orientations = [
        ("portrait", module_length_m, module_width_m),
        ("landscape", module_width_m, module_length_m),
    ]

    cos_tilt = math.cos(math.radians(max(0.0, min(85.0, float(tilt_deg)))))

    for ori_name, mod_u_m, mod_v_m in orientations:
        # The slope dimension (v) is foreshortened in 2D projection by cos(tilt)
        mod_v_proj = mod_v_m * cos_tilt
        gap_v_proj = gap_m * cos_tilt

        step_u_m = mod_u_m + gap_m
        step_v_m = mod_v_proj + gap_v_proj

        step_u_px = step_u_m / gsd_m
        step_v_px = step_v_m / gsd_m
        half_u_px = (mod_u_m / gsd_m) * 0.5
        half_v_px = (mod_v_proj / gsd_m) * 0.5

        n_u = max(1, int(span_u_m / step_u_m))
        n_v = max(1, int(span_v_m / step_v_m))

        start_u = min_u + (span_u_m / gsd_m - (n_u * step_u_px - gap_m / gsd_m)) * 0.5 + half_u_px
        start_v = min_v + (span_v_m / gsd_m - (n_v * step_v_px - gap_m / gsd_m)) * 0.5 + half_v_px

        packed_modules = []

        for i_u in range(n_u):
            for i_v in range(n_v):
                m_u = start_u + i_u * step_u_px
                m_v = start_v + i_v * step_v_px

                # Transform center back to pixel space
                px_x = cx + m_u * cos_t - m_v * sin_t
                px_y = cy + m_u * sin_t + m_v * cos_t

                # Verify module center is inside the regularized PV array polygon
                if point_in_polygon(px_x, px_y, polygon_px):
                    # Compute 4 corners in local frame
                    local_corners = [
                        (m_u - half_u_px, m_v - half_v_px),
                        (m_u + half_u_px, m_v - half_v_px),
                        (m_u + half_u_px, m_v + half_v_px),
                        (m_u - half_u_px, m_v + half_v_px),
                    ]
                    # Rotate back to global pixel space
                    global_corners = [
                        (
                            round(cx + lu * cos_t - lv * sin_t, 2),
                            round(cy + lu * sin_t + lv * cos_t, 2),
                        )
                        for lu, lv in local_corners
                    ]
                    packed_modules.append(global_corners)

        if len(packed_modules) > best_result["module_count"]:
            best_result = {
                "module_count": len(packed_modules),
                "orientation": ori_name,
                "modules_px": packed_modules,
                "total_capacity_kwp": round(len(packed_modules) * STANDARD_MODULE_WATTAGE_KW, 3),
            }

    return best_result
