# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Lazy native-pixel train/validation views for segmentation baselines.

Construction validates all snapshot declarations, without opening raster files.
Only the chosen train or validation split can be decoded by this dataset.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .training import MAX_TRAIN_SIDE, TrainingDataError, _raster, _read_manifest
from .validation import MAX_RECORDS


class SegmentationDataset(Dataset):
    """All eligible records in one split, in deterministic sample-ID order.

    ``max_samples`` optionally takes a prefix of that order. ``seed`` is kept
    as provenance; epoch shuffling belongs to the caller's seeded DataLoader.
    No resizing, augmentation or conversion of ignored labels to supervision
    occurs. Image and mask hashes are checked whenever an item is decoded.
    """

    def __init__(self, directory: Path | str, *, split: str = "train",
                 max_samples: int | None = None, seed: int = 20261020):
        if split not in ("train", "val"):
            raise TrainingDataError("SegmentationDataset supports only 'train' or 'val'; test remains held out.")
        if max_samples is not None and (type(max_samples) is not int or not 1 <= max_samples <= MAX_RECORDS):
            raise TrainingDataError(f"max_samples must be null or an integer between 1 and {MAX_RECORDS}.")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TrainingDataError("seed must be a nonnegative integer smaller than 2**63.")
        try:
            self.directory = Path(directory).resolve(strict=True)
            if not self.directory.is_dir():
                raise NotADirectoryError
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingDataError("Dataset directory is missing or inaccessible.") from error
        self.split, self.seed = split, seed
        self.metadata, declarations, self.dataset_fingerprint = _read_manifest(self.directory)
        self.snapshot_declarations = dict(Counter(record["split"] for record in declarations))
        eligible = sorted((record for record in declarations if record["split"] == split
                           and record["annotation_status"] in ("upstream", "reviewed")),
                          key=lambda record: record["id"])
        self.records = eligible if max_samples is None else eligible[:max_samples]
        if not self.records:
            raise TrainingDataError(f"No eligible upstream/reviewed records in {split!r} split.")
        dimensions = {(record["width"], record["height"]) for record in self.records}
        if len(dimensions) != 1:
            raise TrainingDataError("Selected records must have homogeneous dimensions; no resizing is performed.")
        self.dimensions = dimensions.pop()  # (width, height), as in the snapshot and Pillow.
        if max(self.dimensions) > MAX_TRAIN_SIDE:
            raise TrainingDataError(f"Selected raster dimensions exceed {MAX_TRAIN_SIDE} pixels per side.")
        gsd = {float(record["gsd_m"]) for record in self.records}
        if len(gsd) != 1:
            raise TrainingDataError("Selected records must have one consistent gsd_m value.")
        self.gsd = self.gsd_m = gsd.pop()
        selection = [{key: record[key] for key in ("id", "image_sha256", "mask_sha256")} for record in self.records]
        self.selection_fingerprint = hashlib.sha256(
            json.dumps(selection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record["split"] != self.split or record["annotation_status"] not in ("upstream", "reviewed"):
            raise TrainingDataError("Only declared upstream/reviewed records from the selected split may be opened.")
        mask = _raster(self.directory, record, "mask")
        width, height = mask.size
        labels = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(1, height, width)
        valid = labels.ne(255)
        if not bool(valid.any()):
            raise TrainingDataError(f"{record['id']}: an assigned mask must contain supervised pixels; all pixels are ignored.")
        image = _raster(self.directory, record, "image")
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        image_tensor = pixels.reshape(height, width, 3).permute(2, 0, 1).to(torch.float32).div_(255)
        target = torch.where(valid, labels, 0).to(torch.float32)
        return {"image": image_tensor, "target": target, "valid": valid, "id": record["id"]}
