# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Model construction, architecture adapters, and official v1.0 release specifications."""

from .releases import (
    MODEL_RELEASES,
    TINDEN_RELEASE_SPEC,
    VIDDA_RELEASE_SPEC,
    ModelReleaseSpec,
    get_model_spec,
    list_model_releases,
)

__all__ = [
    "MODEL_RELEASES",
    "ModelReleaseSpec",
    "TINDEN_RELEASE_SPEC",
    "VIDDA_RELEASE_SPEC",
    "get_model_spec",
    "list_model_releases",
]
