# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Seamless overlapping tiled inference with 2D Hann window blending for arbitrary resolution aerial orthophotos."""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence
import numpy as np
import torch


def generate_hann_window(height: int, width: int, power: float = 1.0) -> np.ndarray:
    """Generate a 2D separable Hann (Hanning) apodization window for smooth overlap blending.

    Args:
        height: Window height in pixels.
        width: Window width in pixels.
        power: Exponent to sharpen or soften the window taper (default: 1.0).

    Returns:
        2D numpy array of shape (height, width) with values in (0, 1].
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid window dimensions: {height}x{width}")

    # Separable 1D Hann windows centered in pixel coordinates
    # Using (i + 0.5) ensures non-zero edge weights and symmetry
    wy = np.sin((np.arange(height, dtype=np.float32) + 0.5) / height * math.pi) ** 2
    wx = np.sin((np.arange(width, dtype=np.float32) + 0.5) / width * math.pi) ** 2

    window = np.outer(wy, wx)
    if power != 1.0:
        window = np.power(window, power)

    # Floor at 1e-4 to avoid division by zero near corners
    return np.clip(window, 1e-4, 1.0).astype(np.float32)


def compute_tile_slices(
    total_size: int,
    tile_size: int,
    stride: int,
) -> list[slice]:
    """Compute 1D slice coordinates covering total_size with given tile_size and stride."""
    if total_size <= tile_size:
        return [slice(0, total_size)]

    slices: list[slice] = []
    start = 0
    while start + tile_size <= total_size:
        slices.append(slice(start, start + tile_size))
        start += stride

    # Ensure last tile covers the final boundary exactly
    if slices[-1].stop < total_size:
        slices.append(slice(total_size - tile_size, total_size))

    return slices


def pad_to_stride(
    image: torch.Tensor,
    stride: int = 32,
    mode: str = "reflect",
    square: bool = False,
) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
    """Pad image tensor (B, C, H, W) or (C, H, W) so H and W are multiples of stride (and optionally square).

    Returns:
        padded_tensor, (pad_left, pad_right, pad_top, pad_bottom)
    """
    h, w = image.shape[-2:]
    if square:
        target = max(h, w)
        target = ((target + stride - 1) // stride) * stride
        target_h, target_w = target, target
    else:
        target_h = ((h + stride - 1) // stride) * stride
        target_w = ((w + stride - 1) // stride) * stride

    pad_h = target_h - h
    pad_w = target_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h == 0 and pad_w == 0:
        return image, (0, 0, 0, 0)

    # torch.nn.functional.pad expects (pad_left, pad_right, pad_top, pad_bottom)
    padded = torch.nn.functional.pad(image, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)
    return padded, (pad_left, pad_right, pad_top, pad_bottom)


def unpad(
    tensor: torch.Tensor | np.ndarray,
    padding: tuple[int, int, int, int],
) -> torch.Tensor | np.ndarray:
    """Remove padding added by pad_to_stride."""
    pad_left, pad_right, pad_top, pad_bottom = padding
    if pad_left == 0 and pad_right == 0 and pad_top == 0 and pad_bottom == 0:
        return tensor

    if isinstance(tensor, torch.Tensor):
        h, w = tensor.shape[-2:]
        return tensor[..., pad_top : h - pad_bottom, pad_left : w - pad_right]
    else:
        h, w = tensor.shape[-2:]
        return tensor[..., pad_top : h - pad_bottom, pad_left : w - pad_right]


def predict_tiled(
    model: Any,
    image: torch.Tensor | np.ndarray,
    tile_size: int = 400,
    overlap: float = 0.50,
    tta_mode: str = "d4",
    temperature: float = 1.0,
    scales: Sequence[float] = (1.0,),
    device: str = "cpu",
    normalize_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Run seamless tiled sliding-window inference with 2D Hann window blending over an arbitrary resolution image.

    Args:
        model: PyTorch segmentation model.
        image: Input image as (H, W, 3) numpy array or (C, H, W) / (1, C, H, W) torch tensor.
               Values should be in range [0, 1] or [0, 255].
        tile_size: Square tile dimension in pixels (default: 400).
        overlap: Overlap fraction between adjacent tiles in [0.0, 0.75] (default: 0.50).
        tta_mode: Test-Time Augmentation mode ('none', 'flips', 'rot4', 'd4').
        temperature: Temperature scaling factor for calibrated probabilities (default: 1.0).
        device: PyTorch device ('cpu', 'mps', 'cuda').
        normalize_fn: Image normalization function (e.g. normalize_rgb).
        batch_size: Number of tiles to process simultaneously in batched forward passes.

    Returns:
        dict: {
            'probabilities': np.ndarray of shape (H, W) in [0, 1],
            'uncertainty': np.ndarray of shape (H, W) empirical variance,
            'tiles_processed': int,
            'resolution': tuple[int, int] (H, W),
        }
    """
    from .tta import predict_with_tta

    # Convert image input to torch tensor (1, 3, H, W) float32 in [0, 1]
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[:, :, :3]  # Strip alpha if present
            img_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif arr.ndim == 3 and arr.shape[0] == 3:
            img_tensor = torch.from_numpy(arr).unsqueeze(0)
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    elif isinstance(image, torch.Tensor):
        img_tensor = image.detach().cpu().float()
        if img_tensor.max() > 1.5:
            img_tensor /= 255.0
        if img_tensor.ndim == 3:
            img_tensor = img_tensor.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    _, _, orig_h, orig_w = img_tensor.shape

    # Fast path: if image fits in a single tile or is very close
    if orig_h <= tile_size and orig_w <= tile_size:
        padded, pad_spec = pad_to_stride(img_tensor, stride=32, square=True)
        norm_padded = normalize_fn(padded) if normalize_fn else padded
        norm_padded = norm_padded.to(device)

        tta_res = predict_with_tta(
            model=model,
            image_tensor=norm_padded,
            mode=tta_mode,
            temperature=temperature,
            scales=scales,
        )
        p_tensor = tta_res["probabilities"]
        u_tensor = tta_res["uncertainty"]
        p_unpad = unpad(p_tensor, pad_spec).cpu().numpy()
        u_unpad = unpad(u_tensor, pad_spec).cpu().numpy()
        return {
            "probabilities": p_unpad.astype(np.float32),
            "uncertainty": u_unpad.astype(np.float32),
            "tiles_processed": 1,
            "resolution": (orig_h, orig_w),
        }

    # Sliding window tiled inference with Hann window blending
    stride = max(16, int(round(tile_size * (1.0 - overlap))))
    y_slices = compute_tile_slices(orig_h, tile_size, stride)
    x_slices = compute_tile_slices(orig_w, tile_size, stride)

    # Accumulator grids
    prob_accum = np.zeros((orig_h, orig_w), dtype=np.float64)
    uncert_accum = np.zeros((orig_h, orig_w), dtype=np.float64)
    weight_accum = np.zeros((orig_h, orig_w), dtype=np.float64)

    hann_win = generate_hann_window(tile_size, tile_size)

    # Collect all tile coordinates
    tile_coords = []
    for ys in y_slices:
        for xs in x_slices:
            tile_coords.append((ys, xs))

    model.eval()

    # Process tiles in batches
    for start_idx in range(0, len(tile_coords), batch_size):
        chunk = tile_coords[start_idx : start_idx + batch_size]
        for ys, xs in chunk:
            tile_tensor = img_tensor[:, :, ys, xs]
            th, tw = tile_tensor.shape[-2:]

            # If tile at corner/border has non-standard shape, adjust window
            if th != tile_size or tw != tile_size:
                padded_tile, pad_spec = pad_to_stride(tile_tensor, stride=32, square=True)
                cur_win = generate_hann_window(th, tw)
            else:
                padded_tile, pad_spec = pad_to_stride(tile_tensor, stride=32, square=True)
                cur_win = hann_win

            norm_tile = normalize_fn(padded_tile) if normalize_fn else padded_tile
            norm_tile = norm_tile.to(device)

            tta_res = predict_with_tta(
                model=model,
                image_tensor=norm_tile,
                mode=tta_mode,
                temperature=temperature,
                scales=scales,
            )

            p_tile = unpad(tta_res["probabilities"], pad_spec).cpu().numpy().astype(np.float64)
            u_tile = unpad(tta_res["uncertainty"], pad_spec).cpu().numpy().astype(np.float64)

            # Ensure matching shapes before accumulation
            cur_win_matched = cur_win[:th, :tw].astype(np.float64)
            p_tile_matched = p_tile[:th, :tw]
            u_tile_matched = u_tile[:th, :tw]

            prob_accum[ys, xs] += p_tile_matched * cur_win_matched
            uncert_accum[ys, xs] += u_tile_matched * cur_win_matched
            weight_accum[ys, xs] += cur_win_matched

    # Normalize accumulated probabilities by weights
    weight_accum = np.maximum(weight_accum, 1e-8)
    final_probs = prob_accum / weight_accum
    final_uncert = uncert_accum / weight_accum

    return {
        "probabilities": np.clip(final_probs, 0.0, 1.0).astype(np.float32),
        "uncertainty": final_uncert.astype(np.float32),
        "tiles_processed": len(tile_coords),
        "resolution": (orig_h, orig_w),
    }
