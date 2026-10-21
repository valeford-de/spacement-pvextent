# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Probability calibration (Temperature & Platt scaling) and threshold optimization for aerial segmentation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np
import torch


def compute_nll_and_brier(
    probabilities: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-7,
) -> dict[str, float]:
    """Compute Negative Log-Likelihood (BCE) and Brier Score."""
    p = np.clip(np.asarray(probabilities, dtype=np.float64).ravel(), eps, 1.0 - eps)
    y = np.asarray(targets, dtype=np.float64).ravel()
    if len(p) == 0:
        return {"nll": 0.0, "brier": 0.0}
    nll = float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))
    return {"nll": round(nll, 5), "brier": round(brier, 5)}


def compute_calibration_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Calculate reliability diagram metrics, Expected Calibration Error (ECE), and Maximum Calibration Error (MCE).

    Args:
        probabilities: 1D array of predicted probabilities in [0, 1].
        targets: 1D binary array of ground-truth labels (0 or 1).
        n_bins: Number of equal-width confidence bins in [0, 1].

    Returns:
        dict: {
            'ece': float (Expected Calibration Error in [0, 1]),
            'mce': float (Maximum Calibration Error in [0, 1]),
            'nll': float (Negative Log-Likelihood),
            'brier': float (Brier Score),
            'bins': list of dicts with bin statistics (confidence, accuracy, count),
        }
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.int64).ravel()

    if len(p) != len(y) or len(p) == 0:
        return {"ece": 0.0, "mce": 0.0, "nll": 0.0, "brier": 0.0, "bins": []}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data = []
    total_samples = len(p)
    weighted_error_sum = 0.0
    max_error = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            in_bin = (p >= low) & (p <= high)
        else:
            in_bin = (p >= low) & (p < high)

        count = int(np.sum(in_bin))
        if count > 0:
            avg_conf = float(np.mean(p[in_bin]))
            avg_acc = float(np.mean(y[in_bin]))
            err = abs(avg_acc - avg_conf)
            weighted_error_sum += (count / total_samples) * err
            max_error = max(max_error, err)
            bins_data.append({
                "bin_index": i,
                "bin_range": [round(float(low), 3), round(float(high), 3)],
                "count": count,
                "confidence": round(avg_conf, 4),
                "accuracy": round(avg_acc, 4),
                "error": round(err, 4),
            })

    scores = compute_nll_and_brier(p, y)

    return {
        "ece": float(weighted_error_sum),
        "mce": float(max_error),
        "nll": scores["nll"],
        "brier": scores["brier"],
        "bins": bins_data,
        "n_samples": total_samples,
    }


def fit_temperature_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.05,
) -> float:
    """Find the optimal scalar temperature T > 0 minimizing Binary Cross-Entropy loss on validation logits."""
    # Ensure 1D flat tensors on CPU for numerical stability
    z = logits.detach().cpu().flatten().float()
    y = targets.detach().cpu().flatten().float()

    # Parameterize log_t so T is strictly positive: T = exp(log_t)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        t = torch.exp(log_t).clamp(min=1e-3, max=100.0)
        loss = bce(z / t, y)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        # Fallback to Adam if LBFGS encounters line-search step failure
        opt_adam = torch.optim.Adam([log_t], lr=lr)
        for _ in range(max_iter):
            opt_adam.zero_grad()
            t = torch.exp(log_t).clamp(min=1e-3, max=100.0)
            loss = bce(z / t, y)
            loss.backward()
            opt_adam.step()

    optimal_t = float(torch.exp(log_t).item())
    return max(0.01, min(10.0, optimal_t))


def fit_platt_scaling(
    logits: torch.Tensor,
    targets: torch.Tensor,
    max_iter: int = 100,
    lr: float = 0.05,
) -> tuple[float, float]:
    """Fit affine Platt scaling parameters (a, b) such that p = sigmoid(a * z + b) minimizes BCE loss."""
    z = logits.detach().cpu().flatten().float()
    y = targets.detach().cpu().flatten().float()

    a = torch.tensor([1.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], lr=lr, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        scaled = a * z + b
        loss = bce(scaled, y)
        loss.backward()
        return loss

    try:
        optimizer.step(closure)
    except Exception:
        opt_adam = torch.optim.Adam([a, b], lr=lr)
        for _ in range(max_iter):
            opt_adam.zero_grad()
            scaled = a * z + b
            loss = bce(scaled, y)
            loss.backward()
            opt_adam.step()

    return float(a.item()), float(b.item())


def find_optimal_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    n_steps: int = 100,
) -> dict[str, Any]:
    """Grid search for the Bayes-optimal decision thresholds maximizing IoU, F1/Dice, and balanced precision/recall."""
    p = np.asarray(probabilities, dtype=np.float32).ravel()
    y = np.asarray(targets, dtype=bool).ravel()

    thresholds = np.linspace(0.01, 0.99, n_steps)
    best_iou = -1.0
    best_f1 = -1.0
    best_tau_iou = 0.5
    best_tau_f1 = 0.5
    balanced_tau = 0.5
    min_diff_pr = math.inf

    n_pos = int(np.sum(y))
    curve = []

    for tau in thresholds:
        pred = (p >= tau)
        tp = int(np.sum(pred & y))
        fp = int(np.sum(pred & ~y))
        fn = n_pos - tp

        denom_iou = tp + fp + fn
        iou = tp / denom_iou if denom_iou > 0 else 0.0

        denom_f1 = 2 * tp + fp + fn
        f1 = 2 * tp / denom_f1 if denom_f1 > 0 else 0.0

        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / n_pos if n_pos > 0 else 0.0

        diff_pr = abs(prec - rec)
        if diff_pr < min_diff_pr:
            min_diff_pr = diff_pr
            balanced_tau = float(tau)

        if iou > best_iou:
            best_iou = iou
            best_tau_iou = float(tau)

        if f1 > best_f1:
            best_f1 = f1
            best_tau_f1 = float(tau)

        curve.append({
            "threshold": round(float(tau), 3),
            "iou": round(float(iou), 4),
            "f1": round(float(f1), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        })

    return {
        "optimal_threshold_iou": round(best_tau_iou, 3),
        "max_iou": round(best_iou, 4),
        "optimal_threshold_f1": round(best_tau_f1, 3),
        "max_f1": round(best_f1, 4),
        "balanced_threshold": round(balanced_tau, 3),
        "threshold_curve": curve,
    }


def calibrate_validation_split(
    model: Any,
    dataset: Any,
    device: str = "cpu",
    normalize_fn: Any = None,
    max_samples: int = 0,
    subsample_pixels: int = 0,
) -> dict[str, Any]:
    """Execute complete probability calibration and threshold optimization over the validation dataset."""
    model.eval()
    all_logits = []
    all_targets = []

    n_eval = min(len(dataset), max_samples) if max_samples > 0 else len(dataset)
    indices = list(range(n_eval))

    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            img = sample["image"].unsqueeze(0).to(device)
            target = sample["target"].to(device)
            valid = sample["valid"].to(device)

            if normalize_fn:
                img = normalize_fn(img)

            logits = model(img).squeeze()
            valid_mask = valid.squeeze().bool()

            valid_logits = logits[valid_mask].cpu()
            valid_target = target.squeeze()[valid_mask].cpu()

            if valid_logits.numel() > 0:
                if subsample_pixels > 0 and valid_logits.numel() > subsample_pixels:
                    sub_idx = torch.randperm(valid_logits.numel())[:subsample_pixels]
                    valid_logits = valid_logits[sub_idx]
                    valid_target = valid_target[sub_idx]
                all_logits.append(valid_logits)
                all_targets.append(valid_target)

    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)

    # 1. Uncalibrated curve
    raw_probs = torch.sigmoid(logits_tensor).numpy()
    targets_np = targets_tensor.numpy()
    raw_curve = compute_calibration_curve(raw_probs, targets_np)
    raw_thresh = find_optimal_thresholds(raw_probs, targets_np)

    # 2. Temperature scaling
    optimal_t = fit_temperature_scaling(logits_tensor, targets_tensor)
    cal_t_probs = torch.sigmoid(logits_tensor / optimal_t).numpy()
    cal_t_curve = compute_calibration_curve(cal_t_probs, targets_np)
    cal_thresh = find_optimal_thresholds(cal_t_probs, targets_np)

    # 3. Platt scaling
    platt_a, platt_b = fit_platt_scaling(logits_tensor, targets_tensor)
    platt_probs = torch.sigmoid(platt_a * logits_tensor + platt_b).numpy()
    platt_curve = compute_calibration_curve(platt_probs, targets_np)

    return {
        "samples_evaluated": n_eval,
        "total_pixels_evaluated": int(targets_tensor.numel()),
        "pv_positive_pixels": int(targets_tensor.sum().item()),
        "raw_metrics": {
            "ece": round(raw_curve["ece"], 4),
            "mce": round(raw_curve["mce"], 4),
            "nll": raw_curve["nll"],
            "brier": raw_curve["brier"],
            "optimal_threshold_iou": raw_thresh["optimal_threshold_iou"],
            "max_iou": raw_thresh["max_iou"],
            "optimal_threshold_f1": raw_thresh["optimal_threshold_f1"],
            "max_f1": raw_thresh["max_f1"],
            "balanced_threshold": raw_thresh["balanced_threshold"],
        },
        "temperature_scaling": {
            "temperature": round(optimal_t, 4),
            "ece": round(cal_t_curve["ece"], 4),
            "mce": round(cal_t_curve["mce"], 4),
            "nll": cal_t_curve["nll"],
            "brier": cal_t_curve["brier"],
            "ece_reduction_pct": round((1.0 - cal_t_curve["ece"] / max(raw_curve["ece"], 1e-8)) * 100, 2),
            "optimal_threshold_iou": cal_thresh["optimal_threshold_iou"],
            "max_iou": cal_thresh["max_iou"],
            "optimal_threshold_f1": cal_thresh["optimal_threshold_f1"],
            "max_f1": cal_thresh["max_f1"],
            "balanced_threshold": cal_thresh["balanced_threshold"],
        },
        "platt_scaling": {
            "a": round(platt_a, 4),
            "b": round(platt_b, 4),
            "ece": round(platt_curve["ece"], 4),
            "mce": round(platt_curve["mce"], 4),
            "nll": platt_curve["nll"],
            "brier": platt_curve["brier"],
            "ece_reduction_pct": round((1.0 - platt_curve["ece"] / max(raw_curve["ece"], 1e-8)) * 100, 2),
        },
        "raw_calibration_bins": raw_curve["bins"],
        "temperature_calibration_bins": cal_t_curve["bins"],
        "platt_calibration_bins": platt_curve["bins"],
        "threshold_curve": cal_thresh["threshold_curve"],
    }
