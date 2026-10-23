# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Official PVExtent Model Releases & Specifications.

Exposes the official model releases:
- PVExtent Fjell v1.0 (Recommended for current NRW imagery: fewest false alarms)
- PVExtent Vidda v1.0 (Standard Model: High-Throughput & Edge Inference)
- PVExtent Tinden v1.0 (High-End Model: Ensemble Precision)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelReleaseSpec:
    name: str
    tier: str
    version: str = "1.0.0"
    release_date: str = "2026-10-22"
    architecture: str = ""
    description: str = ""
    benchmark_iou: float = 0.0
    benchmark_dice: float = 0.0
    benchmark_accuracy: float = 0.0
    benchmark_recall: float = 0.0
    benchmark_specificity: float = 0.0
    calibration_ece: float = 0.0
    recommended_hardware: str = ""
    default_tile_size: int = 400
    default_overlap: float = 0.50
    default_tta: str = "none"
    default_scales: tuple[float, ...] = (1.0,)
    temperature: float = 1.0
    supports_3d_elevation: bool = True
    supports_4d_temporal: bool = True
    supports_multispectral_nir: bool = False
    config_file: str = ""


VIDDA_RELEASE_SPEC = ModelReleaseSpec(
    name="PVExtent Vidda",
    tier="Standard",
    version="1.0.0",
    release_date="2026-10-22",
    architecture="DeepLabV3+ ResNet50 ASPP (Photometric Invariance)",
    description=(
        "Streamlined, high-throughput model optimized for large-scale municipal screening, "
        "rapid cadastral surveys, edge/laptop deployment, and real-time interactive mapping."
    ),
    benchmark_iou=0.8816,
    benchmark_dice=0.9371,
    benchmark_accuracy=0.9754,
    benchmark_recall=0.9353,
    calibration_ece=0.0047,
    recommended_hardware="Apple Silicon (M1/M2/M3), NVIDIA CUDA (>=4GB), or modern multi-core CPU",
    default_tile_size=400,
    default_overlap=0.25,
    default_tta="none",
    default_scales=(1.0,),
    temperature=1.0,
    supports_3d_elevation=True,
    supports_4d_temporal=True,
    supports_multispectral_nir=False,
    config_file="configs/pvextent-vidda-v1.0.toml",
)

FJELL_RELEASE_SPEC = ModelReleaseSpec(
    name="PVExtent Fjell",
    tier="Standard (NRW hard negatives)",
    version="1.0.0",
    release_date="2026-10-23",
    architecture="DeepLabV3+ ResNet50 ASPP (V7: Vidda recipe plus NRW training data v1)",
    description=(
        "Single model trained with reviewed hard negatives from current Geobasis NRW "
        "orthophotos. On NRW test set v1 it has the fewest false alarms on PV-free roofs "
        "(6.5%) but misses about a third of PV roofs; see docs/research/nrw-train-v1-results.md."
    ),
    benchmark_iou=0.8883,
    benchmark_dice=0.9408,
    benchmark_accuracy=0.9769,
    benchmark_recall=0.9371,
    recommended_hardware="Apple Silicon (M1/M2/M3), NVIDIA CUDA (>=4GB), or modern multi-core CPU",
    default_tile_size=400,
    default_overlap=0.50,
    default_tta="none",
    default_scales=(1.0,),
    temperature=1.0,
    supports_3d_elevation=True,
    supports_4d_temporal=True,
    supports_multispectral_nir=False,
    config_file="configs/pvextent-fjell-v1.0.toml",
)

TINDEN_RELEASE_SPEC = ModelReleaseSpec(
    name="PVExtent Tinden",
    tier="High-End (Flagship Precision)",
    version="1.0.0",
    release_date="2026-10-22",
    architecture=(
        "Heterogeneous Production Ensemble: DeepLabV3+ ResNet50 ASPP (Photometric V5) + "
        "DeepLabV3+ ResNet50 ASPP (NRW Bayesian Self-Trained V6)"
    ),
    description=(
        "High-precision flagship pipeline designed for utility-scale solar asset accounting, "
        "official energy grid cadastre extraction, and regional aerial surveys. "
        "Integrates Dihedral D4 8-fold rotational TTA, multi-scale pyramid averaging, "
        "Bayesian aleatoric/epistemic uncertainty decomposition, 3D DSM roof fitting, "
        "4D longitudinal multi-temporal change detection, and 4-band NIR spectral vetoing."
    ),
    benchmark_iou=0.8915,
    benchmark_dice=0.9426,
    benchmark_accuracy=0.9774,
    benchmark_recall=0.9493,
    calibration_ece=0.0047,
    recommended_hardware="Apple Silicon (M1/M2/M3 Pro/Max/Ultra) or NVIDIA CUDA (>=8GB VRAM)",
    default_tile_size=400,
    default_overlap=0.50,
    default_tta="d4",
    default_scales=(1.0,),
    temperature=0.4068,
    supports_3d_elevation=True,
    supports_4d_temporal=True,
    supports_multispectral_nir=True,
    config_file="configs/pvextent-tinden-v1.0.toml",
)

MODEL_RELEASES = {
    "fjell": FJELL_RELEASE_SPEC,
    "pvextent-fjell": FJELL_RELEASE_SPEC,
    "vidda": VIDDA_RELEASE_SPEC,
    "pvextent-vidda": VIDDA_RELEASE_SPEC,
    "tinden": TINDEN_RELEASE_SPEC,
    "pvextent-tinden": TINDEN_RELEASE_SPEC,
}


def get_model_spec(name: str) -> ModelReleaseSpec:
    """Retrieve the official model release specification by key ('fjell', 'vidda' or 'tinden')."""
    key = name.strip().lower()
    if key in MODEL_RELEASES:
        return MODEL_RELEASES[key]
    raise KeyError(f"Unknown model release: '{name}'. Available releases: {list(MODEL_RELEASES.keys())}")


def list_model_releases() -> list[ModelReleaseSpec]:
    """Return unique official model releases."""
    return [FJELL_RELEASE_SPEC, VIDDA_RELEASE_SPEC, TINDEN_RELEASE_SPEC]
