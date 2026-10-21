# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""SegFormer (MiT-B2) with an explicitly prepared, hash-pinned ImageNet encoder.

Model construction never downloads weights implicitly. Pretrained safetensors
provide the multi-scale vision transformer encoder; the SegFormer MLP decoder
starts randomly.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, TYPE_CHECKING
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from torch import Tensor, nn

from .resnet_unet import EncoderWeightsError

MODEL_NAME = "segformer-mit-b2-v1"
MAX_WEIGHT_BYTES = 128 * 1024 * 1024
REVISION = "868ab6f13871dcf8c3d9f90ee4519403475b65ef"
HUB_URL = "https://huggingface.co/smp-hub/mit_b2.imagenet"
RESOLVE_URL = f"{HUB_URL}/resolve/{REVISION}"

WEIGHT_SPEC = {
    "model_id": "smp-hub/mit_b2.imagenet",
    "publisher": "smp-hub (MixVisionTransformer / SegFormer)",
    "revision": REVISION,
    "filename": "model.safetensors",
    "url": f"{RESOLVE_URL}/model.safetensors",
    "size": 98867640,
    "sha256": "ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95",
    "license": "Apache-2.0",
    "license_basis": "Publisher model card license: apache-2.0",
    "model_card_url": f"{HUB_URL}/blob/{REVISION}/README.md",
    "config_url": f"{HUB_URL}/blob/{REVISION}/config.json",
    "pretraining_dataset": "ImageNet-1k",
    "format": "safetensors",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "input_range": [0.0, 1.0],
    "channel_order": "RGB",
    "segmentation_preprocessing": "float32 RGB [0,1], channel mean/std; no resize or crop",
}
MODEL_CARD_SPEC = {
    "filename": "model-card.md",
    "url": f"{RESOLVE_URL}/README.md",
    "size": 913,
    "sha256": "67c6a7e115357064edc87ad0d3eef210b234d3ce4a69ab43927c357d7eeca7da",
}
CONFIG_SPEC = {
    "filename": "config.json",
    "url": f"{RESOLVE_URL}/config.json",
    "size": 135,
    "sha256": "13c350b0a57ad271642f941fedd790491f9fa47c70e3aaddfa5cd18de212540d",
}


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
            with urlopen(request, timeout=120) as response:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > spec["size"] or total > MAX_WEIGHT_BYTES:
                        raise EncoderWeightsError("Encoder download exceeds its pinned size.")
                    stream.write(block)
        _verified_bytes(temporary, spec)
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_segformer_encoder_weights(cache_dir: Path | str, *, download: bool = False) -> dict[str, Any]:
    """Verify local cache files; fetch only fixed publisher URLs when requested."""
    try:
        cache = Path(cache_dir).resolve()
        if download:
            cache.mkdir(parents=True, exist_ok=True)
        files = [_cache_file(cache, spec, download) for spec in (WEIGHT_SPEC, MODEL_CARD_SPEC, CONFIG_SPEC)]
        provenance = {
            **WEIGHT_SPEC,
            "path": str(files[0]),
            "model_card_path": str(files[1]),
            "config_path": str(files[2]),
            "verified": True,
            "source_files": [
                {key: spec[key] for key in ("filename", "url", "size", "sha256")}
                for spec in (WEIGHT_SPEC, MODEL_CARD_SPEC, CONFIG_SPEC)
            ],
        }
        return provenance
    except EncoderWeightsError:
        raise
    except (OSError, ValueError, RuntimeError) as error:
        raise EncoderWeightsError(f"Cannot prepare SegFormer encoder cache: {error}") from error


def build_segformer_mit_b2(weights_path: Path | str | None = None) -> nn.Module:
    """Build the SegFormer (MiT-B2) architecture offline, loading verified encoder tensors."""
    try:
        import torch
        import segmentation_models_pytorch as smp
        from safetensors.torch import load
    except ImportError as error:
        raise EncoderWeightsError("Baseline dependencies missing. Run: uv sync --locked --extra baseline") from error
    try:
        model = smp.create_model(
            "segformer",
            encoder_name="mit_b2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        ).to(dtype=torch.float32)
        if weights_path is not None:
            payload = _verified_bytes(Path(weights_path), WEIGHT_SPEC)
            state = load(payload)
            classifier = {"head.weight", "head.bias"}
            if not classifier.issubset(state):
                raise EncoderWeightsError("Expected ImageNet-1k classification head in SegFormer encoder source.")
            encoder_state = {key: value for key, value in state.items() if key not in classifier}
            expected = model.encoder.state_dict()
            if set(encoder_state) != set(expected):
                raise EncoderWeightsError("Pretrained encoder keys differ from fixed MiT-B2 architecture.")
            if any(encoder_state[key].shape != target.shape or encoder_state[key].dtype != target.dtype
                   for key, target in expected.items()):
                raise EncoderWeightsError("Pretrained encoder tensor shapes or dtypes do not match MiT-B2.")
            if any(tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()) for tensor in state.values()):
                raise EncoderWeightsError("Pretrained source contains non-finite tensors.")
            model.encoder.load_state_dict(encoder_state)
        return model
    except EncoderWeightsError:
        raise
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise EncoderWeightsError(f"Cannot construct SegFormer MiT-B2: {error}") from error
