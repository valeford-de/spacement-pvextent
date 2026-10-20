# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Small, deterministic training-only views of a PVExtent dataset snapshot.

Metadata checks cover the complete manifest. Pixel access is restricted to
eligible train records; validation and test imagery is never opened here.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
from typing import Any
import warnings

from PIL import Image
import torch
from torch.utils.data import Dataset

from .validation import (
    MAX_FILE_BYTES, MAX_LINE_BYTES, MAX_MANIFEST_BYTES,
    MAX_METADATA_BYTES, MAX_RECORDS, ValidationReport, _invalid_constant,
    _metadata, _overlap, _path, _read, _record, _unique_object,
)

MAX_TRAIN_SAMPLES = 64
MAX_TRAIN_SIDE = 1024


class TrainingDataError(ValueError):
    """A dataset or masked training input is not safe to use for supervision."""


def _require_ok(report: ValidationReport) -> None:
    if not report.ok:
        error = report.errors[0]
        raise TrainingDataError(f"{error.location}: {error.code}: {error.message}")


def _read_manifest(directory: Path) -> tuple[dict[str, Any], list[dict], str]:
    """Reuse contract checks, without resolving or opening raster paths."""
    report = ValidationReport()
    payloads = []
    for name, limit in (("dataset.toml", MAX_METADATA_BYTES), ("manifest.jsonl", MAX_MANIFEST_BYTES)):
        path = _path(directory, name, report, name)
        _require_ok(report)
        payload = _read(path, limit, report, name)
        _require_ok(report)
        payloads.append(payload)
    metadata_bytes, manifest_bytes = payloads
    metadata = _metadata(metadata_bytes, report)
    _require_ok(report)
    records = []
    seen_ids = set()
    images, groups, buildings, raster_paths = {}, {}, {}, {}
    for line_number, line in enumerate(BytesIO(manifest_bytes), 1):
        where = f"manifest.jsonl:{line_number}"
        if line_number > MAX_RECORDS or len(line) > MAX_LINE_BYTES:
            raise TrainingDataError(f"{where}: manifest record/line bound exceeded.")
        try:
            record = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object,
                                parse_constant=_invalid_constant)
        except (UnicodeError, ValueError, RecursionError) as error:
            raise TrainingDataError(f"{where}: expected strict UTF-8 JSON with unique keys and finite numbers.") from error
        _record(record, metadata["sources"], report, where)
        _require_ok(report)
        for role in ("image", "mask"):
            value = record[role]
            # Purely lexical checks here: val/test paths must not be opened,
            # resolved, read, or required to exist for a training smoke run.
            if ("\\" in value or ":" in value or PurePosixPath(value).is_absolute()
                    or any(part in ("", ".", "..") for part in value.split("/"))):
                raise TrainingDataError(f"{where}: unsafe {role} path.")
            suffix = PurePosixPath(value).suffix.lower()
            if suffix not in ((".png",) if role == "mask" else (".png", ".jpg", ".jpeg")):
                raise TrainingDataError(f"{where}: unsupported {role} format.")
            _overlap(value, record["split"], raster_paths, "raster path", report, where)
        if record["id"] in seen_ids:
            raise TrainingDataError(f"{where}: duplicate sample ID {record['id']!r}.")
        seen_ids.add(record["id"])
        _overlap(record["image_sha256"], record["split"], images, "image content", report, where)
        _overlap(record["spatial_group"], record["split"], groups, "spatial group", report, where)
        for building in record["building_ids"]:
            _overlap(building, record["split"], buildings, "building ID", report, where)
        _require_ok(report)
        records.append(record)
    if not records:
        raise TrainingDataError("Dataset manifest is empty.")
    return metadata, records, hashlib.sha256(metadata_bytes + b"\0" + manifest_bytes).hexdigest()


def _raster(directory: Path, record: dict, role: str) -> Image.Image:
    """Validate and decode one selected train raster from its hashed bytes."""
    report = ValidationReport()
    path = _path(directory, record[role], report, f"{record['id']}:{role}")
    _require_ok(report)
    payload = _read(path, MAX_FILE_BYTES, report, f"{record['id']}:{role}")
    _require_ok(report)
    if hashlib.sha256(payload).hexdigest() != record[f"{role}_sha256"]:
        raise TrainingDataError(f"{record['id']}: {role} hash mismatch.")
    expected_format = "PNG" if PurePosixPath(record[role]).suffix.lower() == ".png" else "JPEG"
    if expected_format == "PNG" and (len(payload) < 33 or payload[12:16] != b"IHDR" or payload[24] != 8):
        raise TrainingDataError(f"{record['id']}: {role} must use encoded 8-bit PNG channels.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload), formats=[expected_format]) as image:
                if image.size != (record["width"], record["height"]):
                    raise TrainingDataError(f"{record['id']}: {role} dimensions differ from the manifest.")
                if max(image.size) > MAX_TRAIN_SIDE:
                    raise TrainingDataError(f"{record['id']}: training raster dimensions exceed {MAX_TRAIN_SIDE}.")
                mode = "RGB" if role == "image" else "L"
                if image.mode != mode or getattr(image, "n_frames", 1) != 1:
                    raise TrainingDataError(f"{record['id']}: expected a single-frame {mode} {role}.")
                if image.getexif().get(274, 1) != 1:
                    raise TrainingDataError(f"{record['id']}: {role} orientation must already be normalized.")
                image.load()
                if role == "mask":
                    if any(count and label not in (0, 1, 255) for label, count in enumerate(image.histogram())):
                        raise TrainingDataError(f"{record['id']}: mask labels must be 0, 1 or 255.")
                return image.copy()
    except TrainingDataError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise TrainingDataError(f"{record['id']}: cannot decode canonical {role}.") from error


class TrainingDataset(Dataset):
    """A bounded deterministic subset of reviewed or upstream train records.

    All-zero masks are valid negative supervision. All-255 masks are skipped
    before imagery is opened. Other splits are checked only as declarations.
    No resizing, augmentation, normalization statistics or label changes occur.
    """

    def __init__(self, directory: Path | str, *, max_samples: int = 8, seed: int = 20261020):
        if type(max_samples) is not int or not 1 <= max_samples <= MAX_TRAIN_SAMPLES:
            raise TrainingDataError(f"max_samples must be an integer between 1 and {MAX_TRAIN_SAMPLES}.")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TrainingDataError("seed must be a nonnegative integer smaller than 2**63.")
        try:
            self.directory = Path(directory).resolve(strict=True)
            if not self.directory.is_dir():
                raise NotADirectoryError
        except (OSError, RuntimeError, ValueError) as error:
            raise TrainingDataError("Dataset directory is missing or inaccessible.") from error
        self.metadata, declarations, self.dataset_fingerprint = _read_manifest(self.directory)
        ranked = sorted((record for record in declarations if record["split"] == "train"
                         and record["annotation_status"] in ("upstream", "reviewed")),
                        key=lambda record: (hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest(), record["id"]))
        self.records = []
        for record in ranked:
            if record["width"] > MAX_TRAIN_SIDE or record["height"] > MAX_TRAIN_SIDE:
                raise TrainingDataError(f"{record['id']}: training dimensions exceed {MAX_TRAIN_SIDE}.")
            mask = _raster(self.directory, record, "mask")
            histogram = mask.histogram()
            if histogram[0] + histogram[1] == 0:
                continue
            _raster(self.directory, record, "image")
            self.records.append(record)
            if len(self.records) == max_samples:
                break
        if not self.records:
            raise TrainingDataError("No eligible train records with upstream/reviewed supervised pixels.")
        if len({(record["width"], record["height"]) for record in self.records}) != 1:
            raise TrainingDataError("Selected training images must have homogeneous dimensions; no resizing is performed.")
        if len({float(record["gsd_m"]) for record in self.records}) != 1:
            raise TrainingDataError("Selected training records must have one consistent gsd_m value.")
        selection = [{key: record[key] for key in ("id", "image_sha256", "mask_sha256")} for record in self.records]
        self.selection_fingerprint = hashlib.sha256(
            json.dumps(selection, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        if record["split"] != "train" or record["annotation_status"] not in ("upstream", "reviewed"):
            raise TrainingDataError("Only reviewed/upstream train records may be opened for training.")
        image = _raster(self.directory, record, "image")
        mask = _raster(self.directory, record, "mask")
        # Copies avoid non-writable NumPy buffers and add no NumPy dependency.
        width, height = image.size
        pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        image_tensor = pixels.reshape(height, width, 3).permute(2, 0, 1).to(torch.float32).div_(255)
        labels = torch.frombuffer(bytearray(mask.tobytes()), dtype=torch.uint8).reshape(1, height, width)
        valid = labels.ne(255)
        if not bool(valid.any()):
            raise TrainingDataError(f"{record['id']}: selected mask no longer contains supervised pixels.")
        target = torch.where(valid, labels, 0).to(torch.float32)
        return {"image": image_tensor, "target": target, "valid": valid, "id": record["id"]}
