# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Multispectral analysis, NIR / CIR color infrared processing and spectral PV signature verification.

Implements spectral diagnostics for aerial imagery:
- 4-Band (RGB + Near-Infrared / NIR) ingestion
- Normalized Difference Vegetation Index (NDVI) calculation
- Photovoltaic Spectral Index (PVSI) and NIR-to-Red band ratio
- False-positive suppression for green roofs (Dachbegrünung), dark slate, and water reflections
- False-color Color Infrared (CIR) rendering (NIR -> Red, Red -> Green, Green -> Blue)
- Deep learning weight transfer / conv1 expansion for 4-channel multispectral backbones
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
from PIL import Image, ImageDraw


# Radiometric and Biophysical Thresholds
NDVI_VEGETATION_THRESHOLD = 0.35      # Polygons with mean NDVI > 0.35 are living vegetation (Dachbegrünung/trees)
WATER_PUDDLE_NIR_MAX = 0.04           # Extreme NIR absorption (< 0.04) marks standing water / wet puddles
SILICON_PV_NDVI_MAX = 0.25            # Crystalline silicon PV panels exhibit low NDVI (-0.25 to +0.25)
SILICON_PV_NIR_RED_RATIO_MIN = 0.70   # Silicon anti-reflective coatings reflect more in NIR than visible red


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Calculate the Normalized Difference Vegetation Index: (NIR - Red) / (NIR + Red).

    Args:
        nir: Near-infrared band raster normalized to [0.0, 1.0].
        red: Red band raster normalized to [0.0, 1.0].

    Returns:
        np.ndarray: NDVI raster with values in [-1.0, 1.0].
    """
    nir_f = np.asarray(nir, dtype=np.float32)
    red_f = np.asarray(red, dtype=np.float32)
    denominator = nir_f + red_f
    safe_denom = np.where(denominator < 1e-6, 1e-6, denominator)
    ndvi = (nir_f - red_f) / safe_denom
    return np.clip(ndvi, -1.0, 1.0)


def calculate_pvsi(nir: np.ndarray, blue: np.ndarray) -> np.ndarray:
    """Calculate the Photovoltaic Spectral Index: (NIR - Blue) / (NIR + Blue).

    Exploits the steep bandgap absorption cutoff of silicon cells between visible blue and NIR.
    """
    nir_f = np.asarray(nir, dtype=np.float32)
    blue_f = np.asarray(blue, dtype=np.float32)
    denominator = nir_f + blue_f
    safe_denom = np.where(denominator < 1e-6, 1e-6, denominator)
    pvsi = (nir_f - blue_f) / safe_denom
    return np.clip(pvsi, -1.0, 1.0)


def calculate_ndre(nir: np.ndarray, red_edge: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Red Edge index: (NIR - RE) / (NIR + RE)."""
    nir_f = np.asarray(nir, dtype=np.float32)
    re_f = np.asarray(red_edge, dtype=np.float32)
    denominator = nir_f + re_f
    safe_denom = np.where(denominator < 1e-6, 1e-6, denominator)
    ndre = (nir_f - re_f) / safe_denom
    return np.clip(ndre, -1.0, 1.0)


def compose_cir_composite(
    red: np.ndarray,
    green: np.ndarray,
    nir: np.ndarray,
) -> Image.Image:
    """Create standard Color Infrared (CIR) composite image from Red, Green, and NIR bands.

    Band Assignment (Standard German Landesvermessung / Geobasis NRW CIR):
        - Output Red Channel   <- NIR (highlights vegetation in vibrant red/magenta)
        - Output Green Channel <- Red band
        - Output Blue Channel  <- Green band
    """
    def _to_u8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        scaled = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else np.clip(arr / 255.0, 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    nir_u8 = _to_u8(nir)
    red_u8 = _to_u8(red)
    green_u8 = _to_u8(green)

    cir_arr = np.stack([nir_u8, red_u8, green_u8], axis=-1)
    return Image.fromarray(cir_arr, mode="RGB")


def extract_polygon_spectral_stats(
    polygon_px: Sequence[tuple[float, float]],
    rgb_arr: np.ndarray,
    nir_arr: np.ndarray | None = None,
) -> dict[str, Any]:
    """Extract biophysical and radiometric spectral statistics inside a candidate PV polygon.

    Args:
        polygon_px: List of (x, y) coordinates forming the polygon in pixel space.
        rgb_arr: Array of shape (H, W, 3) with float [0, 1] or uint8 [0, 255].
        nir_arr: Optional array of shape (H, W) representing the calibrated NIR band.

    Returns:
        dict: Spectral properties, vegetation indices, and false-positive veto classification.
    """
    if len(polygon_px) < 3:
        return {"spectral_status": "insufficient_vertices"}

    h, w = rgb_arr.shape[:2]
    # Rasterize polygon into binary mask inside its bounding box
    xs = [p[0] for p in polygon_px]
    ys = [p[1] for p in polygon_px]

    min_x, max_x = max(0, int(min(xs))), min(w - 1, int(math.ceil(max(xs))))
    min_y, max_y = max(0, int(min(ys))), min(h - 1, int(math.ceil(max(ys))))

    if max_x <= min_x or max_y <= min_y:
        return {"spectral_status": "degenerate_bounding_box"}

    box_w = max_x - min_x + 1
    box_h = max_y - min_y + 1

    mask_img = Image.new("1", (box_w, box_h), 0)
    shifted_poly = [(x - min_x, y - min_y) for x, y in polygon_px]
    ImageDraw.Draw(mask_img).polygon(shifted_poly, outline=1, fill=1)
    poly_mask = np.array(mask_img, dtype=bool)

    if not np.any(poly_mask):
        return {"spectral_status": "empty_polygon_mask"}

    # Extract sub-regions
    sub_rgb = rgb_arr[min_y : max_y + 1, min_x : max_x + 1]
    if sub_rgb.dtype == np.uint8:
        sub_rgb = sub_rgb.astype(np.float32) / 255.0

    r_vals = sub_rgb[..., 0][poly_mask]
    g_vals = sub_rgb[..., 1][poly_mask]
    b_vals = sub_rgb[..., 2][poly_mask]

    mean_r = float(np.mean(r_vals))
    mean_g = float(np.mean(g_vals))
    mean_b = float(np.mean(b_vals))

    result: dict[str, Any] = {
        "mean_red_reflectance": round(mean_r, 4),
        "mean_green_reflectance": round(mean_g, 4),
        "mean_blue_reflectance": round(mean_b, 4),
        "multispectral_nir_available": nir_arr is not None,
    }

    if nir_arr is None:
        result["spectral_verdict"] = "unverified_rgb_only"
        result["is_spectral_false_positive"] = False
        return result

    sub_nir = nir_arr[min_y : max_y + 1, min_x : max_x + 1]
    if sub_nir.dtype == np.uint8:
        sub_nir = sub_nir.astype(np.float32) / 255.0

    nir_vals = sub_nir[poly_mask]
    mean_nir = float(np.mean(nir_vals))
    p90_nir = float(np.percentile(nir_vals, 90))

    # Compute pixel-wise NDVI inside polygon
    sub_ndvi = calculate_ndvi(sub_nir, sub_rgb[..., 0])
    ndvi_vals = sub_ndvi[poly_mask]
    mean_ndvi = float(np.mean(ndvi_vals))
    p90_ndvi = float(np.percentile(ndvi_vals, 90))

    # Compute pixel-wise PVSI inside polygon
    sub_pvsi = calculate_pvsi(sub_nir, sub_rgb[..., 2])
    pvsi_vals = sub_pvsi[poly_mask]
    mean_pvsi = float(np.mean(pvsi_vals))

    nir_red_ratio = float(mean_nir / max(1e-4, mean_r))

    result["mean_nir_reflectance"] = round(mean_nir, 4)
    result["p90_nir_reflectance"] = round(p90_nir, 4)
    result["mean_ndvi"] = round(mean_ndvi, 4)
    result["p90_ndvi"] = round(p90_ndvi, 4)
    result["mean_pvsi"] = round(mean_pvsi, 4)
    result["nir_red_ratio"] = round(nir_red_ratio, 3)

    # Radiometric decision engine
    if mean_ndvi > NDVI_VEGETATION_THRESHOLD:
        # High chlorophyll reflection -> Green roof (Dachbegrünung) / overhanging trees
        verdict = "vegetated_roof_veto"
        is_fp = True
    elif mean_nir < WATER_PUDDLE_NIR_MAX and mean_r < 0.05:
        # Total NIR absorption -> Water puddle / wet roof ponding
        verdict = "water_puddle_veto"
        is_fp = True
    elif mean_ndvi <= SILICON_PV_NDVI_MAX and nir_red_ratio >= SILICON_PV_NIR_RED_RATIO_MIN:
        # Matches anti-reflective silicon solar cell reflectance profile
        verdict = "silicon_photovoltaic_confirmed"
        is_fp = False
    else:
        # Standard roof material (e.g. slate or tile) without vegetation
        verdict = "standard_spectral_response"
        is_fp = False

    result["spectral_verdict"] = verdict
    result["is_spectral_false_positive"] = is_fp
    return result


def expand_conv1_for_nir(
    model: Any,
    pretrained_state_dict: dict[str, Any],
    conv1_key: str = "conv1.weight",
) -> dict[str, Any]:
    """Expand a 3-channel Conv1 weight tensor to 4 channels for multispectral RGB+NIR ingestion.

    Initializes the 4th channel (NIR) as the balanced radiometric combination of Red and Green weights,
    preserving activation variance while maintaining immediate backwards-compatibility with 3-channel encoders.
    """
    import torch

    expanded_state = dict(pretrained_state_dict)
    w3 = pretrained_state_dict[conv1_key]
    if w3.shape[1] == 4:
        return expanded_state

    # Shape: (out_channels, 3, k_h, k_w)
    out_ch, in_ch, kh, kw = w3.shape
    if in_ch != 3:
        raise ValueError(f"Expected 3 input channels in {conv1_key}, got {in_ch}")

    # Standard remote sensing expansion: 0.5 * (w_red + w_green)
    w_nir = 0.5 * (w3[:, 0:1, :, :] + w3[:, 1:2, :, :])
    w4 = torch.cat([w3, w_nir], dim=1)  # (out_channels, 4, kh, kw)
    expanded_state[conv1_key] = w4
    return expanded_state
