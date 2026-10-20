# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Dataset-level binary pixel confusion counts, excluding ignored pixels."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from pvextent.data.training import TrainingDataError


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass
class PixelConfusion:
    """Accumulate integer counts; ratios are computed after dataset aggregation.

    Predictions are sigmoid(logits) >= threshold. Undefined ratios are null,
    including PV IoU/Dice when neither predictions nor valid labels contain PV.
    The accumulator stores no tensors or computation graphs.
    """

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    samples: int = 0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
               threshold: float = 0.5) -> None:
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not 0 <= threshold <= 1 or not math.isfinite(threshold)):
            raise TrainingDataError("threshold must be a finite number between 0 and 1.")
        if not all(isinstance(value, torch.Tensor) for value in (logits, target, valid)):
            raise TrainingDataError("logits, target and valid must be torch tensors.")
        if (logits.ndim != 4 or logits.shape[1] != 1 or not all(logits.shape)
                or logits.shape != target.shape or logits.shape != valid.shape):
            raise TrainingDataError("Expected matching, nonempty [N,1,H,W] logits, target and valid shapes.")
        if not logits.is_floating_point() or target.is_complex() or valid.dtype != torch.bool:
            raise TrainingDataError("logits must be floating, target real-valued, and valid boolean.")
        if not (logits.device == target.device == valid.device):
            raise TrainingDataError("Metric tensors must be on the same device.")
        if not bool(torch.isfinite(logits).all()):
            raise TrainingDataError("logits must be finite.")
        labels = target[valid]
        if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
            raise TrainingDataError("Valid target pixels must contain finite binary 0/1 labels.")
        prediction = logits[valid].sigmoid().ge(threshold)
        positive = labels.eq(1)
        # Reduction results are converted to Python integers, not probabilities
        # or batch means. Unequal batch sizes and ignore counts remain exact.
        self.tp += int(torch.count_nonzero(prediction & positive).item())
        self.fp += int(torch.count_nonzero(prediction & ~positive).item())
        self.fn += int(torch.count_nonzero(~prediction & positive).item())
        self.tn += int(torch.count_nonzero(~prediction & ~positive).item())
        self.samples += logits.shape[0]

    def to_dict(self) -> dict[str, int | float | None]:
        valid = self.tp + self.fp + self.fn + self.tn
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
            "valid_pixels": valid,
            "iou": _ratio(self.tp, self.tp + self.fp + self.fn),
            "dice": _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn),
            "precision": _ratio(self.tp, self.tp + self.fp),
            "recall": _ratio(self.tp, self.tp + self.fn),
            "pixel_accuracy": _ratio(self.tp + self.tn, valid),
        }
