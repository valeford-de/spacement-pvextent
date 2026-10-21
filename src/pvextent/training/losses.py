# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Binary segmentation losses which exclude ignored pixels completely."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from pvextent.data.training import TrainingDataError


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Compute gradient of the Lovasz extension w.r.t sorted errors (Berman et al.)."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-7)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge_flat(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary Lovasz hinge loss over 1D flat supervised predictions."""
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = labels[perm.data]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss


def masked_bce_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict:
    """BCE-with-logits plus global soft Dice, using only valid N1HW pixels.

    Target values outside ``valid`` are never used. Dice smoothing is 1.0.
    Half-precision inputs are accumulated as float32; no probability clipping
    or thresholding changes the differentiable training objective.
    """
    if not all(isinstance(value, torch.Tensor) for value in (logits, target, valid)):
        raise TrainingDataError("logits, target and valid must be torch tensors.")
    if (logits.ndim != 4 or logits.shape[1] != 1 or not all(logits.shape)
            or target.shape != logits.shape or valid.shape != logits.shape):
        raise TrainingDataError("Expected matching, nonempty [N,1,H,W] logits, target and valid shapes.")
    if not logits.is_floating_point() or not target.is_floating_point() or valid.dtype != torch.bool:
        raise TrainingDataError("logits/target must be floating tensors and valid must be boolean.")
    if not (logits.device == target.device == valid.device):
        raise TrainingDataError("Training tensors must be on the same device.")
    if not bool(torch.isfinite(logits).all()):
        raise TrainingDataError("logits must be finite.")
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        raise TrainingDataError("Masked loss requires at least one valid supervised pixel.")
    labels = target[valid]
    if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise TrainingDataError("Supervised target values must be finite binary 0/1 labels.")
    selected = logits[valid]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    labels = labels.to(selected.dtype)
    # Divide before summing: many large finite BCE terms must not overflow
    # during reduction when their mean remains representable.
    bce = (F.binary_cross_entropy_with_logits(selected, labels, reduction="none") / valid_pixels).sum()
    probability = selected.sigmoid()
    dice_loss = 1 - (2 * (probability * labels).sum() + 1.0) / (probability.sum() + labels.sum() + 1.0)
    loss = bce + dice_loss
    if not bool(torch.isfinite(loss)):
        raise TrainingDataError("Masked loss produced a non-finite result.")
    return {"loss": loss, "bce": bce, "dice_loss": dice_loss, "valid_pixels": valid_pixels}


def masked_focal_lovasz(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    lovasz_weight: float = 0.5,
) -> dict:
    """Boundary-focused loss combining Focal cross-entropy with the Lovasz hinge surrogate.

    Specifically designed for sharp rectangular panel delineation and hard negative suppression.
    Ignored pixels outside ALKIS/supervised boundaries are strictly excluded.
    """
    if not all(isinstance(value, torch.Tensor) for value in (logits, target, valid)):
        raise TrainingDataError("logits, target and valid must be torch tensors.")
    if (logits.ndim != 4 or logits.shape[1] != 1 or not all(logits.shape)
            or target.shape != logits.shape or valid.shape != logits.shape):
        raise TrainingDataError("Expected matching, nonempty [N,1,H,W] logits, target and valid shapes.")
    if not logits.is_floating_point() or not target.is_floating_point() or valid.dtype != torch.bool:
        raise TrainingDataError("logits/target must be floating tensors and valid must be boolean.")
    if not (logits.device == target.device == valid.device):
        raise TrainingDataError("Training tensors must be on the same device.")
    if not bool(torch.isfinite(logits).all()):
        raise TrainingDataError("logits must be finite.")
    valid_pixels = int(valid.sum().item())
    if valid_pixels == 0:
        raise TrainingDataError("Masked loss requires at least one valid supervised pixel.")
    labels = target[valid]
    if not bool(torch.isfinite(labels).all()) or not bool(((labels == 0) | (labels == 1)).all()):
        raise TrainingDataError("Supervised target values must be finite binary 0/1 labels.")
    selected = logits[valid]
    if selected.dtype in (torch.float16, torch.bfloat16):
        selected = selected.float()
    labels = labels.to(selected.dtype)

    # 1. Binary Focal Loss with hard negative modulation
    p = selected.sigmoid()
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    alpha_t = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    focal_weight = alpha_t * ((1.0 - p_t) ** gamma)
    bce = F.binary_cross_entropy_with_logits(selected, labels, reduction="none")
    focal_loss = (focal_weight * bce).sum() / valid_pixels

    # 2. Lovasz Hinge loss over sorted error distribution
    lovasz_loss = _lovasz_hinge_flat(selected, labels)

    loss = focal_loss + lovasz_weight * lovasz_loss
    if not bool(torch.isfinite(loss)):
        raise TrainingDataError("Masked focal Lovasz loss produced a non-finite result.")
    return {
        "loss": loss,
        "focal_loss": focal_loss,
        "lovasz_loss": lovasz_loss,
        "valid_pixels": valid_pixels,
    }
