# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Crop-level pixel error diagnostics, without changing labels or thresholds.

Area bins describe the total valid source-PV area in a crop, not individual
arrays. Boundary diagnostics use a fixed three-pixel Chebyshev radius. Actual
ignored pixels exclude their whole neighborhood from boundary/interior
classification; space outside the image supplies neither labels nor ignore.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import math
from statistics import median
from typing import Any

import torch
from torch.nn import functional as F

from .pixels import PixelConfusion

BOUNDARY_RADIUS_PX = 3
AREA_STRATA = ("lt_10_m2", "10_to_lt_50_m2", "ge_50_m2")
COUNT_FIELDS = ("tp", "fp", "fn", "tn")
BOUNDARY_FIELDS = (
    "boundary_band_pixels", "boundary_error_pixels", "interior_error_pixels",
    "unclassified_error_pixels",
)


def _finite_nonnegative(value: Any) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0)
    except OverflowError:
        return False


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _present_in_neighborhood(mask: torch.Tensor) -> torch.Tensor:
    # max_pool2d pads with negative infinity, not a semantic label. A positive
    # pooled value therefore requires an actual in-image True pixel.
    return F.max_pool2d(
        mask.to(dtype=torch.float32)[None, None],
        kernel_size=2 * BOUNDARY_RADIUS_PX + 1, stride=1, padding=BOUNDARY_RADIUS_PX,
    )[0, 0].gt(0)


@torch.no_grad()
def sample_statistics(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, *, gsd_m: float,
) -> dict[str, Any]:
    """Return masked pixel counts, source area and fixed-radius error classes.

    Inputs are matching dense boolean [H,W] rasters on one device. Predictions
    have already been thresholded by the caller. ``foreground_fraction`` uses
    valid source-PV pixels divided by valid pixels. Areas exclude ignored pixels.
    """
    if not all(isinstance(value, torch.Tensor) for value in (prediction, target, valid)):
        raise ValueError("prediction, target and valid must be torch tensors.")
    if any(value.dtype != torch.bool or value.ndim != 2 or not all(value.shape)
           or value.layout != torch.strided for value in (prediction, target, valid)):
        raise ValueError("Expected nonempty dense boolean [H,W] rasters.")
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target and valid must have the same shape.")
    if not (prediction.device == target.device == valid.device):
        raise ValueError("prediction, target and valid must be on the same device.")
    if not bool(valid.any()):
        raise ValueError("At least one supervised valid pixel is required.")
    if not _finite_nonnegative(gsd_m) or gsd_m == 0:
        raise ValueError("gsd_m must be finite and greater than zero.")
    pixel_area = float(gsd_m) * float(gsd_m)
    if not math.isfinite(pixel_area) or pixel_area == 0:
        raise ValueError("gsd_m must produce a finite positive pixel area.")

    positive, background = target & valid, ~target & valid
    tp = int(torch.count_nonzero(prediction & positive).item())
    fp = int(torch.count_nonzero(prediction & background).item())
    fn = int(torch.count_nonzero(~prediction & positive).item())
    tn = int(torch.count_nonzero(~prediction & background).item())
    confusion = PixelConfusion(tp=tp, fp=fp, fn=fn, tn=tn, samples=1).to_dict()
    target_pixels, predicted_pixels = tp + fn, tp + fp
    target_area, predicted_area = target_pixels * pixel_area, predicted_pixels * pixel_area
    if not math.isfinite(target_area) or not math.isfinite(predicted_area):
        raise ValueError("PV pixel areas must remain finite.")

    classified = valid & ~_present_in_neighborhood(~valid)
    boundary = classified & _present_in_neighborhood(positive) & _present_in_neighborhood(background)
    errors = (prediction ^ target) & valid
    edge_pv = (bool(positive[0, :].any()) or bool(positive[-1, :].any())
               or bool(positive[:, 0].any()) or bool(positive[:, -1].any()))
    return {
        **confusion,
        "target_pv_pixels": target_pixels, "predicted_pv_pixels": predicted_pixels,
        "target_pv_area_m2": target_area, "predicted_pv_area_m2": predicted_area,
        "foreground_fraction": target_pixels / confusion["valid_pixels"],
        "target_touches_crop_edge": edge_pv, "error_pixels": fp + fn,
        "boundary_band_pixels": int(torch.count_nonzero(boundary).item()),
        "boundary_error_pixels": int(torch.count_nonzero(errors & boundary).item()),
        "interior_error_pixels": int(torch.count_nonzero(errors & classified & ~boundary).item()),
        "unclassified_error_pixels": int(torch.count_nonzero(errors & ~classified).item()),
    }


def _validate_row(row: Any) -> None:
    if not isinstance(row, Mapping):
        raise ValueError("Expected crop statistics mappings from sample_statistics.")
    required = {*COUNT_FIELDS, *BOUNDARY_FIELDS, "valid_pixels", "error_pixels",
                "target_pv_pixels", "predicted_pv_pixels", "target_pv_area_m2",
                "predicted_pv_area_m2", "target_touches_crop_edge"}
    if not required <= row.keys():
        raise ValueError("Crop statistics are missing required counts, areas or edge status.")
    integers = (*COUNT_FIELDS, *BOUNDARY_FIELDS, "valid_pixels", "error_pixels",
                "target_pv_pixels", "predicted_pv_pixels")
    if any(type(row[key]) is not int or row[key] < 0 for key in integers):
        raise ValueError("Crop pixel counts must be nonnegative integers.")
    if (row["valid_pixels"] == 0 or sum(row[key] for key in COUNT_FIELDS) != row["valid_pixels"]
            or row["target_pv_pixels"] != row["tp"] + row["fn"]
            or row["predicted_pv_pixels"] != row["tp"] + row["fp"]
            or row["error_pixels"] != row["fp"] + row["fn"]
            or sum(row[key] for key in BOUNDARY_FIELDS[1:]) != row["error_pixels"]
            or row["boundary_error_pixels"] > row["boundary_band_pixels"]
            or row["boundary_band_pixels"] > row["valid_pixels"]):
        raise ValueError("Crop pixel counts are inconsistent.")
    if type(row["target_touches_crop_edge"]) is not bool:
        raise ValueError("Crop edge status must be boolean.")
    for role in ("target", "predicted"):
        area, count = row[f"{role}_pv_area_m2"], row[f"{role}_pv_pixels"]
        if not _finite_nonnegative(area) or (area == 0) != (count == 0):
            raise ValueError("Crop PV areas must be finite and agree with zero/nonzero pixel counts.")


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {key: sum(row[key] for row in rows) for key in COUNT_FIELDS}
    confusion = PixelConfusion(**totals, samples=len(rows)).to_dict()
    ious = [value for row in rows
            if (value := PixelConfusion(**{key: row[key] for key in COUNT_FIELDS}).to_dict()["iou"]) is not None]
    try:
        target_area = math.fsum(row["target_pv_area_m2"] for row in rows)
        predicted_area = math.fsum(row["predicted_pv_area_m2"] for row in rows)
    except OverflowError as error:
        raise ValueError("Aggregated PV areas must remain finite.") from error
    if not math.isfinite(target_area) or not math.isfinite(predicted_area):
        raise ValueError("Aggregated PV areas must remain finite.")
    error_pixels = totals["fp"] + totals["fn"]
    boundary = {key: sum(row[key] for row in rows) for key in BOUNDARY_FIELDS}
    return {
        **confusion, "samples": len(rows),
        "target_pv_pixels": totals["tp"] + totals["fn"],
        "predicted_pv_pixels": totals["tp"] + totals["fp"],
        "target_pv_area_m2": target_area, "predicted_pv_area_m2": predicted_area,
        "foreground_fraction": _ratio(totals["tp"] + totals["fn"], confusion["valid_pixels"]),
        "macro_iou": math.fsum(ious) / len(ious) if ious else None,
        "median_iou": median(ious) if ious else None,
        "min_iou": min(ious) if ious else None,
        "defined_iou_samples": len(ious), "undefined_iou_samples": len(rows) - len(ious),
        "error_pixels": error_pixels, **boundary,
        "boundary_error_share": _ratio(boundary["boundary_error_pixels"], error_pixels),
        "interior_error_share": _ratio(boundary["interior_error_pixels"], error_pixels),
        "unclassified_error_share": _ratio(boundary["unclassified_error_pixels"], error_pixels),
        "boundary_error_rate": _ratio(boundary["boundary_error_pixels"], boundary["boundary_band_pixels"]),
    }


def aggregate_statistics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate crop statistics globally and by fixed source-area/edge strata.

    Macro/median/min IoU omit crops where source and prediction are both empty;
    ``defined_iou_samples`` and ``undefined_iou_samples`` expose that denominator.
    Empty strata have zero counts and null ratios. Extra row metadata is allowed.
    """
    try:
        records = list(rows)
    except TypeError as error:
        raise ValueError("Expected an iterable of crop statistics.") from error
    for row in records:
        _validate_row(row)
    area_rows = {key: [] for key in AREA_STRATA}
    edge_rows = {"touches_edge": [], "does_not_touch_edge": []}
    for row in records:
        area = row["target_pv_area_m2"]
        area_key = AREA_STRATA[0] if area < 10 else AREA_STRATA[1] if area < 50 else AREA_STRATA[2]
        area_rows[area_key].append(row)
        edge_rows["touches_edge" if row["target_touches_crop_edge"] else "does_not_touch_edge"].append(row)
    return {
        **_summary(records),
        "boundary_radius_px": BOUNDARY_RADIUS_PX, "boundary_distance": "chebyshev",
        "area_strata_basis": "Total valid source PV area per crop; not individual array size.",
        "area_strata": {key: _summary(values) for key, values in area_rows.items()},
        "crop_edge_strata": {key: _summary(values) for key, values in edge_rows.items()},
    }
