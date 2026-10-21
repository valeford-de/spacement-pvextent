# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Physical photovoltaic and geospatial metric calculations."""

from __future__ import annotations

import math
from typing import Sequence

# Typical modern monocrystalline silicon module efficiency (20.5% STC)
DEFAULT_MODULE_EFFICIENCY = 0.205  # 205 W/m² under 1000 W/m² STC
DEFAULT_MODULE_WATTAGE_KW = 0.400  # 400 W standard commercial module rating
DEFAULT_GSD_M = 0.10  # 10 cm ground sampling distance for DOP10


def polygon_area_pixels(vertices: Sequence[tuple[float, float]]) -> float:
    """Calculate the 2D polygon area in pixel units using the Shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) * 0.5


def polygon_perimeter_pixels(vertices: Sequence[tuple[float, float]]) -> float:
    """Calculate perimeter in pixel units."""
    n = len(vertices)
    if n < 2:
        return 0.0
    perimeter = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        perimeter += math.hypot(x2 - x1, y2 - y1)
    return perimeter


def polygon_centroid(vertices: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """Calculate the 2D centroid (center of mass) of a polygon."""
    n = len(vertices)
    if n < 3:
        if n == 0:
            return (0.0, 0.0)
        return (sum(p[0] for p in vertices) / n, sum(p[1] for p in vertices) / n)
    signed_area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        signed_area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    signed_area *= 0.5
    if abs(signed_area) < 1e-9:
        return (sum(p[0] for p in vertices) / n, sum(p[1] for p in vertices) / n)
    cx /= (6.0 * signed_area)
    cy /= (6.0 * signed_area)
    return (cx, cy)


def calculate_pv_metrics(
    vertices_px: Sequence[tuple[float, float]],
    gsd_m: float = DEFAULT_GSD_M,
    module_efficiency: float = DEFAULT_MODULE_EFFICIENCY,
    module_wattage_kw: float = DEFAULT_MODULE_WATTAGE_KW,
    confidence_scores: Sequence[float] | None = None,
) -> dict:
    """Compute CAD-grade surface area, nominal capacity (kWp), module count and geometric properties."""
    area_px = polygon_area_pixels(vertices_px)
    area_m2 = area_px * (gsd_m ** 2)
    perimeter_px = polygon_perimeter_pixels(vertices_px)
    perimeter_m = perimeter_px * gsd_m
    centroid_px = polygon_centroid(vertices_px)

    # Photovoltaic capacity: Area (m²) * Efficiency * 1.0 kW/m² irradiance
    capacity_kwp = area_m2 * module_efficiency
    estimated_modules = max(1, round(capacity_kwp / module_wattage_kw)) if area_m2 > 0.5 else 0

    mean_conf = float(sum(confidence_scores) / len(confidence_scores)) if confidence_scores else 1.0

    return {
        "area_pixels": round(area_px, 2),
        "area_m2": round(area_m2, 2),
        "perimeter_m": round(perimeter_m, 2),
        "centroid_pixel": (round(centroid_px[0], 2), round(centroid_px[1], 2)),
        "estimated_capacity_kwp": round(capacity_kwp, 3),
        "estimated_module_count": estimated_modules,
        "module_efficiency": module_efficiency,
        "reference_module_rating_w": int(module_wattage_kw * 1000),
        "mean_confidence": round(mean_conf, 4),
        "gsd_m": gsd_m,
    }
