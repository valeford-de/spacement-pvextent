# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unlabelled NRW training-review pools, separate from reserved evaluation data."""

from __future__ import annotations

from pathlib import Path

from .negative_holdout import NegativeHoldoutError, _fetch_review_candidates
from .pv4ger import DatasetImportError


class TrainingCandidatesError(DatasetImportError):
    """Training candidates cannot be acquired under the fixed exclusion rules."""


def fetch_training_candidates(exclude_dirs: list[Path | str], output_dir: Path | str, *,
                              protocol_path: Path | str) -> dict:
    """Acquire at most 24 native crops with all-255, unreviewed placeholders.

    Supply the complete existing training snapshot and every reserved external
    pool as exclusions. Identity checks include every row; the geographic buffer
    uses validation, test and unassigned rows, including every reserved pilot
    case. Existing training centres alone do not exclude useful training regions.
    Their coordinates and manifest declarations are read; exclusion raster paths
    are never resolved or opened. Candidate acquisition
    alone does not authorize labels, train assignment or evaluation claims.
    """
    try:
        return _fetch_review_candidates(
            exclude_dirs, output_dir, protocol_path=protocol_path,
            dataset_id="nrw-training-candidates", spatial_group_prefix="nrw-training-",
            purpose="experimental-training-review-pool",
            distance_reference_splits=("val", "test", "unassigned"),
        )
    except NegativeHoldoutError as error:
        raise TrainingCandidatesError(f"Training candidate acquisition failed: {error}") from error
