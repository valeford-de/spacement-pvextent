# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""CAD-grade 3D roof geometry, elevation profile fitting, and solar yield modeling.

Enables 3D rooftop analysis:
- Surface normal and plane fitting from Digital Surface Model (DSM / DOM) rasters
- Roof tilt (pitch) and azimuth (compass orientation) calculation
- True 3D surface area expansion: A_true = A_proj / cos(tilt)
- PVGIS-grade annual photovoltaic solar yield modeling (kWh/kWp/year, MWh/year)
- Annual grid CO2 displacement accounting
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np

# Standard Central European / German solar irradiation parameters (DWD / PVGIS)
DEFAULT_GHI_KWH_M2 = 1050.0  # Annual Global Horizontal Irradiation in kWh/m²
DEFAULT_SYSTEM_PERFORMANCE_RATIO = 0.82  # Standard 82% performance ratio (inverter, wiring, temp derating)
DEFAULT_CO2_EMISSION_FACTOR_KG_PER_KWH = 0.380  # 380 g CO2/kWh displaced German grid mix


def azimuth_to_cardinal(azimuth_deg: float, tilt_deg: float = 30.0) -> str:
    """Convert a 0-360 degree compass azimuth into a human-readable cardinal direction.

    Conventions:
        0° / 360° = North, 90° = East, 180° = South, 270° = West.
        If tilt <= 3°, the roof is functionally FLAT.
    """
    if tilt_deg <= 3.0:
        return "FLAT"

    norm_az = (azimuth_deg % 360.0 + 360.0) % 360.0
    directions = [
        ("N", 0.0),
        ("NNE", 22.5),
        ("NE", 45.0),
        ("ENE", 67.5),
        ("E", 90.0),
        ("ESE", 112.5),
        ("SE", 135.0),
        ("SSE", 157.5),
        ("S", 180.0),
        ("SSW", 202.5),
        ("SW", 225.0),
        ("WSW", 247.5),
        ("W", 270.0),
        ("WNW", 292.5),
        ("NW", 315.0),
        ("NNW", 337.5),
    ]

    best_card = "N"
    min_diff = 360.0
    for card, angle in directions:
        diff = abs((norm_az - angle + 180.0) % 360.0 - 180.0)
        if diff < min_diff:
            min_diff = diff
            best_card = card

    return best_card


def calculate_3d_surface_area(projected_area_m2: float, tilt_deg: float) -> dict[str, float]:
    """Calculate the true 3D physical roof surface area accounting for roof tilt.

    Formula:
        A_true = A_proj / cos(tilt)
    """
    # Clamp tilt between 0 and 88.5 degrees to avoid non-physical infinity
    clamped_tilt = max(0.0, min(88.5, float(tilt_deg)))
    cos_tilt = math.cos(math.radians(clamped_tilt))

    expansion_factor = 1.0 / cos_tilt if cos_tilt > 1e-4 else 1.0
    true_3d_area_m2 = projected_area_m2 * expansion_factor

    return {
        "projected_area_m2": round(projected_area_m2, 2),
        "true_3d_area_m2": round(true_3d_area_m2, 2),
        "area_expansion_factor": round(expansion_factor, 4),
        "tilt_deg": round(clamped_tilt, 1),
    }


def calculate_annual_solar_yield(
    capacity_kwp: float,
    tilt_deg: float = 30.0,
    azimuth_deg: float = 180.0,
    ghi_kwh_m2: float = DEFAULT_GHI_KWH_M2,
    performance_ratio: float = DEFAULT_SYSTEM_PERFORMANCE_RATIO,
    co2_factor_kg_per_kwh: float = DEFAULT_CO2_EMISSION_FACTOR_KG_PER_KWH,
) -> dict[str, Any]:
    """Model the annual electricity production and environmental impact using PVGIS transposition.

    Args:
        capacity_kwp: Nominal DC peak capacity in kWp.
        tilt_deg: Roof pitch in degrees (0° = flat, 90° = vertical).
        azimuth_deg: Compass bearing of panel orientation (180° = South, 90° = East, 270° = West).
        ghi_kwh_m2: Annual Global Horizontal Irradiation in kWh/m² (default: 1050 kWh/m²).
        performance_ratio: Total system performance ratio accounting for inverter and cabling losses (default: 0.82).
        co2_factor_kg_per_kwh: Grid displacement carbon intensity (default: 0.380 kg CO2/kWh).

    Returns:
        dict: Physical energy metrics and carbon savings.
    """
    clamped_tilt = max(0.0, min(85.0, float(tilt_deg)))
    norm_azimuth = (float(azimuth_deg) % 360.0 + 360.0) % 360.0

    # Deviation from optimal South (180°) in degrees [0, 180]
    azimuth_dev_rad = math.radians(abs(((norm_azimuth - 180.0 + 180.0) % 360.0) - 180.0))
    tilt_rad = math.radians(clamped_tilt)

    # PVGIS transposition model approximation for Central Europe (Latitude ~51°N):
    # Optimal tilt for South is ~35°, yielding ~1.12x global horizontal irradiance.
    # Flat horizontal roof (tilt=0) has factor 1.00.
    # Deviation towards East/West drops yield by ~12-18%. North drops yield significantly.
    cos_az_dev = math.cos(azimuth_dev_rad)
    sin_tilt = math.sin(tilt_rad)

    # Transposition factor on tilted surface
    if clamped_tilt <= 3.0:
        transposition_factor = 1.00
    else:
        # Surface transposition coefficient
        gain_component = 0.14 * sin_tilt * cos_az_dev
        loss_component = 0.08 * sin_tilt * (1.0 - cos_az_dev)
        overhang_loss = 0.12 * (math.sin(tilt_rad * 0.5) ** 2) * max(0.0, -cos_az_dev)
        transposition_factor = 1.0 + gain_component - loss_component - overhang_loss

    # Floor at 0.50 (diffuse-only northern vertical slope) and cap at 1.15
    transposition_factor = max(0.50, min(1.15, transposition_factor))

    # Specific annual yield in kWh / kWp
    specific_yield_kwh_per_kwp = round(ghi_kwh_m2 * transposition_factor * performance_ratio, 1)

    # Total energy generated
    annual_yield_kwh = round(capacity_kwp * specific_yield_kwh_per_kwp, 1)
    annual_yield_mwh = round(annual_yield_kwh / 1000.0, 3)

    # CO2 savings
    annual_co2_kg = annual_yield_kwh * co2_factor_kg_per_kwh
    annual_co2_tons = round(annual_co2_kg / 1000.0, 2)

    cardinal = azimuth_to_cardinal(norm_azimuth, clamped_tilt)

    return {
        "tilt_deg": round(clamped_tilt, 1),
        "azimuth_deg": round(norm_azimuth, 1),
        "azimuth_cardinal": cardinal,
        "transposition_factor": round(transposition_factor, 3),
        "specific_annual_yield_kwh_per_kwp": specific_yield_kwh_per_kwp,
        "estimated_annual_yield_kwh": annual_yield_kwh,
        "estimated_annual_yield_mwh": annual_yield_mwh,
        "annual_co2_savings_tons": annual_co2_tons,
        "global_horizontal_irradiance_kwh_m2": ghi_kwh_m2,
        "system_performance_ratio": performance_ratio,
    }


def fit_roof_plane_from_dsm(
    dsm: np.ndarray,
    polygon_px: Sequence[tuple[float, float]],
    gsd_m: float = 0.10,
) -> dict[str, Any]:
    """Fit a 3D surface plane Z = a*X + b*Y + c to a Digital Surface Model (DSM) within a polygon footprint.

    Calculates:
        - Surface normal vector: n = (-a, -b, 1) / sqrt(a^2 + b^2 + 1)
        - Roof tilt / pitch angle theta = arctan(sqrt(a^2 + b^2))
        - Azimuth compass orientation phi = direction of steepest descent
        - Goodness-of-fit R² and RMSE in meters

    Args:
        dsm: 2D numpy array of elevation heights in meters (H, W).
        polygon_px: List of (x, y) polygon vertices in pixel coordinates.
        gsd_m: Ground sampling distance in meters per pixel.

    Returns:
        dict: 3D geometric profile including tilt, azimuth, elevation range, and fit statistics.
    """
    if len(polygon_px) < 3 or dsm.ndim != 2:
        return {
            "tilt_deg": 0.0,
            "azimuth_deg": 180.0,
            "azimuth_cardinal": "FLAT",
            "fit_r2": 0.0,
            "fit_rmse_m": 0.0,
            "elevation_mean_m": 0.0,
            "elevation_min_m": 0.0,
            "elevation_max_m": 0.0,
            "is_valid_3d_fit": False,
        }

    from .packing import point_in_polygon

    h, w = dsm.shape
    xs = [p[0] for p in polygon_px]
    ys = [p[1] for p in polygon_px]

    min_x = max(0, int(math.floor(min(xs))))
    max_x = min(w - 1, int(math.ceil(max(xs))))
    min_y = max(0, int(math.floor(min(ys))))
    max_y = min(h - 1, int(math.ceil(max(ys))))

    # Collect all valid (X, Y, Z) points inside polygon
    pts_x = []
    pts_y = []
    pts_z = []

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if point_in_polygon(float(x), float(y), polygon_px):
                val = float(dsm[y, x])
                # Filter out nodata or non-finite values
                if math.isfinite(val) and val > -100.0:
                    # Real-world metric coordinates: X points East, Y_geo points North (-y in image coords)
                    pts_x.append(x * gsd_m)
                    pts_y.append(-y * gsd_m)  # Negative because image y increases Southward
                    pts_z.append(val)

    if len(pts_z) < 4:
        # Fallback to horizontal
        mean_elev = float(np.mean(pts_z)) if pts_z else 0.0
        return {
            "tilt_deg": 0.0,
            "azimuth_deg": 180.0,
            "azimuth_cardinal": "FLAT",
            "fit_r2": 0.0,
            "fit_rmse_m": 0.0,
            "elevation_mean_m": round(mean_elev, 2),
            "elevation_min_m": round(mean_elev, 2),
            "elevation_max_m": round(mean_elev, 2),
            "is_valid_3d_fit": False,
        }

    X = np.array(pts_x, dtype=np.float64)
    Y = np.array(pts_y, dtype=np.float64)
    Z = np.array(pts_z, dtype=np.float64)

    # Design matrix: Z = a*X + b*Y + c
    A = np.column_stack([X, Y, np.ones_like(X)])
    try:
        # Robust least squares
        coeffs, residuals, rank, s = np.linalg.lstsq(A, Z, rcond=None)
        a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

        # Fitted Z and residuals
        z_pred = a * X + b * Y + c
        ss_tot = float(np.sum((Z - np.mean(Z)) ** 2))
        ss_res = float(np.sum((Z - z_pred) ** 2))
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 1e-6 else 1.0
        rmse = float(math.sqrt(ss_res / len(Z)))

        # Gradients:
        # a = dZ / dX (East gradient)
        # b = dZ / dY_geo (North gradient)
        # Slope magnitude
        slope_mag = math.sqrt(a * a + b * b)
        tilt_deg = math.degrees(math.atan(slope_mag))

        # Direction of steepest descent (aspect/azimuth):
        # The vector pointing downhill is (-a, -b).
        # Geographic azimuth: 0° = North (+Y_geo), 90° = East (+X), 180° = South, 270° = West
        # compass = atan2(-a, -b) in standard compass coordinates:
        azimuth_rad = math.atan2(-a, -b)
        azimuth_deg = (math.degrees(azimuth_rad) + 360.0) % 360.0

        cardinal = azimuth_to_cardinal(azimuth_deg, tilt_deg)

        return {
            "tilt_deg": round(tilt_deg, 1),
            "azimuth_deg": round(azimuth_deg, 1),
            "azimuth_cardinal": cardinal,
            "fit_r2": round(r2, 4),
            "fit_rmse_m": round(rmse, 3),
            "elevation_mean_m": round(float(np.mean(Z)), 2),
            "elevation_min_m": round(float(np.min(Z)), 2),
            "elevation_max_m": round(float(np.max(Z)), 2),
            "sample_points": len(Z),
            "plane_coefficients": {"a": round(a, 6), "b": round(b, 6), "c": round(c, 4)},
            "is_valid_3d_fit": True,
        }
    except Exception:
        mean_elev = float(np.mean(Z))
        return {
            "tilt_deg": 0.0,
            "azimuth_deg": 180.0,
            "azimuth_cardinal": "FLAT",
            "fit_r2": 0.0,
            "fit_rmse_m": 0.0,
            "elevation_mean_m": round(mean_elev, 2),
            "elevation_min_m": round(mean_elev, 2),
            "elevation_max_m": round(mean_elev, 2),
            "is_valid_3d_fit": False,
        }
