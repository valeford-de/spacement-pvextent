# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Test-Time Augmentation (TTA) with dihedral group D4 symmetry for aerial segmentation."""

from __future__ import annotations

from typing import Any, Literal
import torch


TTAMode = Literal["none", "flips", "rot4", "d4"]


def apply_tta_transforms(image_tensor: torch.Tensor, mode: TTAMode = "d4") -> tuple[torch.Tensor, list[str]]:
    """Generate transformed batch of an input image tensor (C, H, W) or (1, C, H, W).

    Returns:
        tuple[torch.Tensor, list[str]]: Batched tensor of shape (K, C, H, W) and names of applied transforms.
    """
    if image_tensor.ndim == 3:
        x = image_tensor.unsqueeze(0)
    else:
        x = image_tensor

    if mode == "none":
        return x, ["identity"]

    transforms: list[tuple[str, torch.Tensor]] = [("identity", x)]

    if mode in ("flips", "d4"):
        transforms.append(("hflip", torch.flip(x, dims=[-1])))
        transforms.append(("vflip", torch.flip(x, dims=[-2])))
        transforms.append(("hvflip", torch.flip(x, dims=[-2, -1])))

    if mode in ("rot4", "d4"):
        transforms.append(("rot90", torch.rot90(x, k=1, dims=[-2, -1])))
        transforms.append(("rot180", torch.rot90(x, k=2, dims=[-2, -1])))
        transforms.append(("rot270", torch.rot90(x, k=3, dims=[-2, -1])))

    if mode == "d4":
        # Transpose (hflip + rot90)
        hflip = transforms[1][1]
        transforms.append(("transpose", torch.rot90(hflip, k=1, dims=[-2, -1])))

    names = [name for name, _ in transforms]
    batch = torch.cat([t for _, t in transforms], dim=0)
    return batch, names


def invert_tta_prediction(pred_tensor: torch.Tensor, transform_name: str) -> torch.Tensor:
    """Invert the spatial transformation applied to a single prediction map (1, H, W) or (H, W)."""
    p = pred_tensor
    if transform_name == "identity":
        return p
    if transform_name == "hflip":
        return torch.flip(p, dims=[-1])
    if transform_name == "vflip":
        return torch.flip(p, dims=[-2])
    if transform_name == "hvflip":
        return torch.flip(p, dims=[-2, -1])
    if transform_name == "rot90":
        return torch.rot90(p, k=3, dims=[-2, -1])  # Inverse of 90 CW is 270 CW (3x90)
    if transform_name == "rot180":
        return torch.rot90(p, k=2, dims=[-2, -1])
    if transform_name == "rot270":
        return torch.rot90(p, k=1, dims=[-2, -1])
    if transform_name == "transpose":
        unrot = torch.rot90(p, k=3, dims=[-2, -1])
        return torch.flip(unrot, dims=[-1])
    raise ValueError(f"Unknown TTA transform: {transform_name}")


def predict_with_tta(
    model: Any,
    image_tensor: torch.Tensor,
    mode: TTAMode = "d4",
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """Run model inference with Test-Time Augmentation (TTA).

    Args:
        model: PyTorch segmentation model expecting (B, C, H, W) input.
        image_tensor: Single input image tensor (C, H, W) or (1, C, H, W).
        mode: TTA mode ('none', 'flips', 'rot4', 'd4').
        batch_size: Batch size for batched forward passes.

    Returns:
        dict: {
            'probabilities': torch.Tensor of shape (H, W) in range [0, 1],
            'uncertainty': torch.Tensor of shape (H, W) variance across TTA passes,
            'transforms_used': int (e.g. 8 for 'd4'),
        }
    """
    batch, names = apply_tta_transforms(image_tensor, mode=mode)
    k = len(names)

    preds: list[torch.Tensor] = []
    model.eval()

    with torch.no_grad():
        for start in range(0, k, batch_size):
            chunk = batch[start:start + batch_size]
            logits = model(chunk)
            # Squeeze channel dimension: (B, 1, H, W) -> (B, H, W)
            if logits.ndim == 4:
                logits = logits.squeeze(1)
            probs = torch.sigmoid(logits)
            for i in range(probs.shape[0]):
                t_name = names[start + i]
                single_prob = probs[i:i + 1]
                inv_prob = invert_tta_prediction(single_prob, t_name)
                preds.append(inv_prob.squeeze(0))

    # Stack all inverted predictions: (K, H, W)
    stacked = torch.stack(preds, dim=0)

    # Calculate ensemble mean probability and variance
    mean_prob = torch.mean(stacked, dim=0)
    variance = torch.var(stacked, dim=0, unbiased=False) if k > 1 else torch.zeros_like(mean_prob)

    return {
        "probabilities": mean_prob,
        "uncertainty": variance,
        "transforms_used": k,
        "mode": mode,
    }
