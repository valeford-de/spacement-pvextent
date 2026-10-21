# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Predictions from images and trained checkpoints."""

from .candidates import export_candidate_predictions
from .calibration import (
    calibrate_validation_split,
    compute_calibration_curve,
    find_optimal_thresholds,
    fit_platt_scaling,
    fit_temperature_scaling,
)
from .tiler import (
    compute_tile_slices,
    generate_hann_window,
    pad_to_stride,
    predict_tiled,
    unpad,
)
from .tta import TTAMode, apply_tta_transforms, invert_tta_prediction, predict_with_tta

__all__ = [
    "TTAMode",
    "apply_tta_transforms",
    "calibrate_validation_split",
    "compute_calibration_curve",
    "compute_tile_slices",
    "export_candidate_predictions",
    "find_optimal_thresholds",
    "fit_platt_scaling",
    "fit_temperature_scaling",
    "generate_hann_window",
    "invert_tta_prediction",
    "pad_to_stride",
    "predict_tiled",
    "predict_with_tta",
    "unpad",
]
