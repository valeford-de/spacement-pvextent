# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Small shared helpers for immutable, locally validated dataset snapshots."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import shutil
from typing import Any

from .pv4ger import DatasetImportError, _sha256
from .validation import ValidationReport, validate_dataset


def read_snapshot(source: Path) -> tuple[list[dict], ValidationReport]:
    report = validate_dataset(source)
    if not report.ok:
        issue = report.errors[0]
        raise DatasetImportError(f"Source dataset is invalid: {issue.code}: {issue.message}")
    manifest = (source / "manifest.jsonl").read_bytes()
    if (hashlib.sha256(manifest).hexdigest() != report.stats["manifest_sha256"]
            or _sha256(source / "dataset.toml") != report.stats["metadata_sha256"]):
        raise DatasetImportError("Source metadata changed during validation.")
    return [json.loads(line) for line in manifest.decode("utf-8").splitlines() if line.strip()], report


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetImportError(f"Duplicate coordinate field: {key}")
        result[key] = value
    return result


def load_centers(source: Path, records: list[dict]) -> tuple[dict[str, tuple[float, float]], str]:
    path = source / "provenance" / "samples.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
        raise DatasetImportError("Expected bounded provenance/samples.jsonl with a centre for every sample.")
    centers = {}
    payload = path.read_bytes()
    for line in payload.decode("utf-8").splitlines():
        item = json.loads(line, object_pairs_hook=_unique_pairs)
        if not isinstance(item, dict):
            raise DatasetImportError("Coordinate records must be JSON objects.")
        key, point = item.get("id"), item.get("approximate_center")
        if (not isinstance(key, str) or key in centers or item.get("center_crs") != "EPSG:4326"
                or not isinstance(point, list) or len(point) != 2
                or any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in point)):
            raise DatasetImportError("Invalid, duplicate or unsupported coordinate record.")
        lon, lat = point
        # This first policy uses the UTM 32N metric projection around Germany.
        # Fail rather than silently apply it to distant countries or polar data.
        if not (3 <= lon <= 15 and 45 <= lat <= 60):
            raise DatasetImportError("This split policy supports centres near Germany (3..15 E, 45..60 N).")
        centers[key] = (lon, lat)
    if set(centers) != {row["id"] for row in records}:
        raise DatasetImportError("Coordinate records must match every canonical sample ID exactly.")
    return centers, hashlib.sha256(payload).hexdigest()


def copy_snapshot(source: Path, staging: Path, records: list[dict]) -> None:
    """Copy only validated pixels, metadata and bounded regular provenance files."""
    shutil.copyfile(source / "dataset.toml", staging / "dataset.toml")
    copied = set()
    for record in records:
        for name in (record["image"], record["mask"]):
            if name not in copied:
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / name, target)
                copied.add(name)
    provenance = source / "provenance"
    if provenance.is_symlink():
        raise DatasetImportError("Provenance directory must not be a symbolic link.")
    total = 0
    if provenance.is_dir():
        for path in sorted(provenance.rglob("*")):
            if path.is_symlink():
                raise DatasetImportError("Provenance links are not copied.")
            if path.is_dir():
                continue
            size = path.stat().st_size
            total += size
            if not path.is_file() or size > 16 * 1024 * 1024 or total > 64 * 1024 * 1024:
                raise DatasetImportError("Provenance files exceed the snapshot copy limits.")
            target = staging / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
