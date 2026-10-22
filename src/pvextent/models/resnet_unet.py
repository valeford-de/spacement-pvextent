# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""U-Net/ResNet18 with an explicitly prepared, hash-pinned timm encoder.

Model construction never downloads weights. Upstream safetensors provide only
the classification encoder; the segmentation decoder/head start randomly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from torch import Tensor, nn

MODEL_NAME = "unet-resnet18-v1"
MAX_WEIGHT_BYTES = 64 * 1024 * 1024
REVISION = "491b427b45c94c7fb0e78b5474cc919aff584bbf"
HUB_URL = "https://huggingface.co/timm/resnet18.a1_in1k"
RESOLVE_URL = f"{HUB_URL}/resolve/{REVISION}"

WEIGHT_SPEC = {
    "model_id": "timm/resnet18.a1_in1k", "publisher": "PyTorch Image Models (timm)",
    "revision": REVISION, "filename": "model.safetensors",
    "url": f"{RESOLVE_URL}/model.safetensors", "size": 46807446,
    "sha256": "80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612",
    "license": "Apache-2.0", "license_basis": "Publisher model card license: apache-2.0",
    "model_card_url": f"{HUB_URL}/blob/{REVISION}/README.md",
    "config_url": f"{HUB_URL}/blob/{REVISION}/config.json",
    "pretraining_dataset": "ImageNet-1k", "format": "safetensors",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
    "input_range": [0.0, 1.0], "channel_order": "RGB",
    "segmentation_preprocessing": "float32 RGB [0,1], channel mean/std; no resize or crop",
}
MODEL_CARD_SPEC = {
    "filename": "model-card.md", "url": f"{RESOLVE_URL}/README.md", "size": 38416,
    "sha256": "e96d7547dfaff6f0a0d74209819c461ec2d66556a9d6b059f1c2a50d413da5bb",
}
CONFIG_SPEC = {
    "filename": "config.json", "url": f"{RESOLVE_URL}/config.json", "size": 755,
    "sha256": "a15887d36094642dbffa237f2c7b66101d7f8ab704c45981b9fafb107cd99d28",
}


class EncoderWeightsError(ValueError):
    """The pinned local encoder artifact is unavailable, invalid or incompatible."""


def _verified_bytes(path: Path, spec: dict) -> bytes:
    if (path.is_symlink() or not path.is_file() or not 0 < spec["size"] <= MAX_WEIGHT_BYTES
            or path.stat().st_size != spec["size"]):
        raise EncoderWeightsError(f"Expected pinned regular file {path} ({spec['size']} bytes).")
    with path.open("rb") as stream:
        payload = stream.read(spec["size"] + 1)
    if len(payload) != spec["size"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise EncoderWeightsError(f"Pinned encoder source checksum mismatch: {path}. Existing files are not replaced.")
    return payload


def _cache_file(cache: Path, spec: dict, download: bool) -> Path:
    path = cache / spec["filename"]
    if os.path.lexists(path):
        _verified_bytes(path, spec)
        return path
    if not download:
        raise EncoderWeightsError(f"Missing encoder source file {path}; prepare it with download=True first.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache, prefix=".encoder-download-", delete=False) as stream:
            temporary = Path(stream.name)
            request = Request(spec["url"], headers={"User-Agent": "Spacement-PVExtent/0.1 encoder-weights"})
            with urlopen(request, timeout=60) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > spec["size"] or total > MAX_WEIGHT_BYTES:
                        raise EncoderWeightsError("Encoder download exceeds its pinned size.")
                    stream.write(block)
        _verified_bytes(temporary, spec)
        os.link(temporary, path)  # Fail atomically if another process created it.
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_encoder_weights(cache_dir: Path | str, *, download: bool = False) -> dict[str, Any]:
    """Verify local cache files; fetch only fixed publisher URLs when requested.

    The cache contains model.safetensors, the pinned model card and config.
    Existing mismatched files are never overwritten, including with download=True.
    """
    try:
        cache = Path(cache_dir).resolve()
        if download:
            cache.mkdir(parents=True, exist_ok=True)
        files = [_cache_file(cache, spec, download) for spec in (WEIGHT_SPEC, MODEL_CARD_SPEC, CONFIG_SPEC)]
        provenance = {**WEIGHT_SPEC, "path": str(files[0]), "model_card_path": str(files[1]),
                      "config_path": str(files[2]), "verified": True,
                      "source_files": [{key: spec[key] for key in ("filename", "url", "size", "sha256")}
                                       for spec in (WEIGHT_SPEC, MODEL_CARD_SPEC, CONFIG_SPEC)]}
        return provenance
    except EncoderWeightsError:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise EncoderWeightsError(f"Cannot prepare encoder cache: {error}") from error


def normalize_rgb(images: Tensor) -> Tensor:
    """Normalize float32 NCHW RGB [0,1], preserving pixels, shape and gradients."""
    import torch

    if (not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1] != 3
            or not all(images.shape) or images.dtype != torch.float32):
        raise ValueError("Expected a nonempty float32 RGB batch [N,3,H,W].")
    if not bool(torch.isfinite(images).all()) or not bool(((images >= 0) & (images <= 1)).all()):
        raise ValueError("RGB values must be finite and within [0,1] before normalization.")
    mean = images.new_tensor(WEIGHT_SPEC["mean"]).reshape(1, 3, 1, 1)
    std = images.new_tensor(WEIGHT_SPEC["std"]).reshape(1, 3, 1, 1)
    return (images - mean) / std


def build_resnet_unet(
    weights_path: Path | str | None = None,
    in_channels: int = 3,
) -> nn.Module:
    """Build the fixed architecture offline, loading only verified encoder tensors.

    Supports both 3-band (RGB) and 4-band (RGB+NIR) multispectral backbones.
    """
    try:
        import torch
        import segmentation_models_pytorch as smp
        from safetensors.torch import load
    except ImportError as error:
        raise EncoderWeightsError("Baseline dependencies missing. Run: uv sync --locked --extra baseline") from error
    try:
        # Explicit None prevents both Torchvision and Hugging Face auto-downloads.
        model = smp.Unet(
            encoder_name="resnet18", encoder_depth=5, encoder_weights=None,
            decoder_use_norm="batchnorm", decoder_channels=(256, 128, 64, 32, 16),
            decoder_attention_type=None, decoder_interpolation="nearest",
            in_channels=in_channels, classes=1, activation=None, aux_params=None,
        ).to(dtype=torch.float32)
        if weights_path is not None:
            payload = _verified_bytes(Path(weights_path), WEIGHT_SPEC)
            state = load(payload)
            classifier = {"fc.weight", "fc.bias"}
            if not classifier.issubset(state) or state["fc.weight"].shape != (1000, 512) or state["fc.bias"].shape != (1000,):
                raise EncoderWeightsError("Expected the pinned ImageNet-1k classification head before encoder conversion.")
            encoder_state = {key: value for key, value in state.items() if key not in classifier}
            if in_channels == 4 and "conv1.weight" in encoder_state:
                w3 = encoder_state["conv1.weight"]
                w_nir = 0.5 * (w3[:, 0:1, :, :] + w3[:, 1:2, :, :])
                encoder_state["conv1.weight"] = torch.cat([w3, w_nir], dim=1)
            expected = model.encoder.state_dict()
            if set(encoder_state) != set(expected):
                raise EncoderWeightsError("Pretrained encoder keys differ from the fixed ResNet18 architecture.")
            if any(encoder_state[key].shape != target.shape or encoder_state[key].dtype != target.dtype
                   for key, target in expected.items()):
                raise EncoderWeightsError("Pretrained encoder tensor shapes or dtypes do not match ResNet18.")
            if any(tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
                raise EncoderWeightsError("Pretrained source contains non-finite tensors.")
            model.encoder.load_state_dict(encoder_state, strict=True)
        return model
    except EncoderWeightsError:
        raise
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise EncoderWeightsError(f"Cannot construct the pinned ResNet18 U-Net: {error}") from error
