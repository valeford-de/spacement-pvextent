# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Offline checks for the PVExtent 1.0 dataset contract.

Validation checks encoded data and declared metadata. It cannot establish
annotation correctness, geographic independence, or permission to use a source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
import tomllib
from typing import Any
from urllib.parse import urlsplit
import warnings

from PIL import Image

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
MAX_PIXELS = 16_777_216
MAX_RECORDS = 100_000
MAX_ISSUES = 200
SPLITS = ("unassigned", "train", "val", "test")
ASSIGNED_SPLITS = frozenset(SPLITS[1:])
STATUSES = frozenset(("unreviewed", "upstream", "reviewed"))
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DATE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
RECORD_FIELDS = frozenset((
    "id", "image", "mask", "image_sha256", "mask_sha256", "width", "height",
    "gsd_m", "source_id", "capture_date", "spatial_group", "building_ids",
    "split", "annotation_status", "georeference",
))
SOURCE_FIELDS = frozenset((
    "name", "url", "version", "image_license", "annotation_license", "attribution",
))


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    location: str


@dataclass
class ValidationReport:
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=lambda: {
        "records": 0, "images_checked": 0, "masks_checked": 0,
        "pv_pixels": 0, "background_pixels": 0, "ignored_pixels": 0,
        "splits": dict.fromkeys(SPLITS, 0), "stopped_early": False,
    })
    warning_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str, location: str) -> None:
        if len(self.errors) < MAX_ISSUES:
            self.errors.append(ValidationIssue(code, message, location))

    def warn(self, code: str, message: str, location: str) -> None:
        self.warning_count += 1
        if len(self.warnings) < MAX_ISSUES:
            self.warnings.append(ValidationIssue(code, message, location))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": [asdict(issue) for issue in self.errors],
            "warnings": [asdict(issue) for issue in self.warnings],
            "error_count": len(self.errors),
            "warning_count": self.warning_count,
            "warnings_truncated": self.warning_count > len(self.warnings),
            "stats": self.stats,
        }


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip()) and len(value) <= 8192


def _number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and IDENTIFIER.fullmatch(value) is not None


def _keys(value: dict, expected: frozenset[str], report: ValidationReport, where: str) -> bool:
    missing = expected - value.keys()
    extra = value.keys() - expected
    if missing:
        report.error("missing_fields", f"Missing fields: {', '.join(sorted(missing))}.", where)
    if extra:
        report.error("unknown_fields", f"Unknown fields: {', '.join(sorted(extra))}.", where)
    return not missing and not extra


def _path(root: Path, value: Any, report: ValidationReport, where: str) -> Path | None:
    if (not _text(value) or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute()
            or any(part in ("", ".", "..") for part in value.split("/"))):
        report.error("unsafe_path", "Expected a relative POSIX file path without traversal.", where)
        return None
    try:
        resolved = root.joinpath(*PurePosixPath(value).parts).resolve(strict=True)
        if not resolved.is_relative_to(root):
            report.error("unsafe_path", "File resolves outside the dataset directory.", where)
        elif not resolved.is_file():
            report.error("not_a_file", "Expected a regular file.", where)
        else:
            return resolved
    except (OSError, RuntimeError, ValueError):
        report.error("missing_file", "File is missing or cannot be resolved.", where)
    return None


def _read(path: Path, limit: int, report: ValidationReport, where: str) -> bytes | None:
    try:
        if path.stat().st_size > limit:
            report.error("file_too_large", f"File exceeds {limit} bytes.", where)
            return None
        with path.open("rb") as handle:
            payload = handle.read(limit + 1)
        if len(payload) > limit:
            report.error("file_too_large", f"File exceeds {limit} bytes.", where)
            return None
        return payload
    except OSError:
        report.error("unreadable_file", "File cannot be read.", where)
        return None


def _metadata(payload: bytes, report: ValidationReport) -> dict | None:
    where = "dataset.toml"
    try:
        value = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError, ValueError, RecursionError):
        report.error("invalid_toml", "Expected UTF-8 TOML metadata.", where)
        return None
    _keys(value, frozenset(("schema_version", "dataset_id", "dataset_version", "sources")), report, where)
    if value.get("schema_version") != "1.0":
        report.error("schema_version", "Supported schema_version is '1.0'.", where)
    if not _identifier(value.get("dataset_id")):
        report.error("dataset_id", "Expected a dataset identifier of 1–128 ASCII letters, digits, '.', '_' or '-'.", where)
    if not _text(value.get("dataset_version")):
        report.error("dataset_version", "A dataset version is required.", where)
    sources = value.get("sources")
    if not isinstance(sources, dict) or not sources:
        report.error("sources", "At least one source must be declared.", where)
    else:
        for source_id, source in sources.items():
            source_where = f"dataset.toml:sources.{source_id!r}"
            if not _identifier(source_id) or not isinstance(source, dict):
                report.error("source", "Expected a source identifier and a source table.", source_where)
                continue
            _keys(source, SOURCE_FIELDS, report, source_where)
            for key in SOURCE_FIELDS:
                if not _text(source.get(key)):
                    report.error("source_metadata", f"Non-empty {key!r} is required.", source_where)
            if _text(source.get("url")):
                try:
                    url = urlsplit(source["url"])
                    valid = url.scheme in ("http", "https") and bool(url.hostname)
                except ValueError:
                    valid = False
                if not valid:
                    report.error("source_url", "Source URL must be HTTP or HTTPS.", source_where)
    return value if report.ok else None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _record(value: Any, sources: dict, report: ValidationReport, where: str) -> bool:
    before = len(report.errors)
    if not isinstance(value, dict):
        report.error("record_type", "Expected one JSON object per line.", where)
        return False
    if not _keys(value, RECORD_FIELDS, report, where):
        return False
    if not _identifier(value["id"]):
        report.error("sample_id", "Expected a sample identifier of 1–128 ASCII letters, digits, '.', '_' or '-'.", where)
    for key in ("image", "mask", "spatial_group"):
        if not _text(value[key]):
            report.error("record_field", f"Non-empty {key!r} is required.", where)
    for key in ("image_sha256", "mask_sha256"):
        if not isinstance(value[key], str) or not SHA256.fullmatch(value[key]):
            report.error("sha256", f"{key} must be a lowercase SHA-256 digest.", where)
    for key in ("width", "height"):
        if type(value[key]) is not int or value[key] <= 0:
            report.error("dimensions", f"{key} must be a positive integer.", where)
    if all(type(value[key]) is int and value[key] > 0 for key in ("width", "height")):
        if value["width"] * value["height"] > MAX_PIXELS:
            report.error("pixel_limit", f"Image exceeds {MAX_PIXELS} pixels.", where)
    if not _number(value["gsd_m"]) or value["gsd_m"] <= 0:
        report.error("gsd", "gsd_m must be a positive finite number in metres per pixel.", where)
    if not isinstance(value["source_id"], str) or value["source_id"] not in sources:
        report.error("source_id", "source_id must refer to a declared source.", where)
    capture_date = value["capture_date"]
    if capture_date is None:
        report.warn("unknown_capture_date", "Capture date is unknown; do not infer one.", where)
    else:
        try:
            if not isinstance(capture_date, str) or not DATE.fullmatch(capture_date):
                raise ValueError
            date.fromisoformat(capture_date)
        except ValueError:
            report.error("capture_date", "capture_date must be YYYY-MM-DD or null.", where)
    buildings = value["building_ids"]
    if (not isinstance(buildings, list) or any(not _text(item) for item in buildings)
            or (isinstance(buildings, list) and all(isinstance(item, str) for item in buildings)
                and len(set(buildings)) != len(buildings))):
        report.error("building_ids", "building_ids must be a list of distinct, non-empty strings.", where)
    if not isinstance(value["split"], str) or value["split"] not in SPLITS:
        report.error("split", "split must be unassigned, train, val or test.", where)
    status = value["annotation_status"]
    if not isinstance(status, str) or status not in STATUSES:
        report.error("annotation_status", "annotation_status must be unreviewed, upstream or reviewed.", where)
    elif status == "unreviewed":
        if value["split"] != "unassigned":
            report.error("unreviewed_labels", "Unreviewed labels cannot enter an assigned split.", where)
        else:
            report.warn("unreviewed_labels", "Labels are unreviewed and are not ready for supervised use.", where)
    elif status == "upstream":
        report.warn("upstream_labels", "Imported curated labels do not imply local human review.", where)
    georef = value["georeference"]
    if georef is None:
        report.warn("unknown_georeference", "Georeference is unknown; geographic independence cannot be inferred.", where)
    elif not isinstance(georef, dict):
        report.error("georeference", "georeference must be an object or null.", where)
    elif _keys(georef, frozenset(("crs", "transform")), report, where):
        transform = georef["transform"]
        if not _text(georef["crs"]):
            report.error("crs", "A non-empty CRS identifier or WKT string is required.", where)
        if not isinstance(transform, list) or len(transform) != 6 or not all(_number(n) for n in transform):
            report.error("transform", "transform must contain six finite numbers [a,b,c,d,e,f].", where)
        else:
            try:
                determinant = transform[0] * transform[4] - transform[1] * transform[3]
                valid = _number(determinant) and determinant != 0
            except OverflowError:
                valid = False
            if not valid:
                report.error("transform", "Affine transform must be finite and non-singular.", where)
    return len(report.errors) == before


def _raster(root: Path, sample: dict, role: str, report: ValidationReport, where: str) -> None:
    path = _path(root, sample[role], report, f"{where}:{role}")
    if path is None:
        return
    suffix = PurePosixPath(sample[role]).suffix.lower()
    expected_format = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}.get(suffix)
    if expected_format is None or (role == "mask" and expected_format != "PNG"):
        report.error("file_format", f"{role} must be {'PNG' if role == 'mask' else 'PNG or JPEG'}.", where)
        return
    payload = _read(path, MAX_FILE_BYTES, report, f"{where}:{role}")
    if payload is None:
        return
    if hashlib.sha256(payload).hexdigest() != sample[f"{role}_sha256"]:
        report.error("hash_mismatch", f"{role} SHA-256 does not match the manifest.", where)
        return
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload), formats=[expected_format]) as raster:
                # Pillow can expose a 16-bit RGB PNG as mode RGB, or expand
                # low-bit grayscale PNGs to L. Check the encoded IHDR depth.
                if expected_format == "PNG" and (
                    len(payload) < 33 or payload[12:16] != b"IHDR" or payload[24] != 8
                ):
                    report.error("bit_depth", f"{role} PNG must be encoded with 8-bit channels.", where)
                    return
                if raster.width * raster.height > MAX_PIXELS:
                    report.error("pixel_limit", f"{role} exceeds {MAX_PIXELS} pixels.", where)
                    return
                if raster.size != (sample["width"], sample["height"]):
                    report.error("dimension_mismatch", f"{role} dimensions do not match the manifest.", where)
                    return
                expected_mode = "L" if role == "mask" else "RGB"
                if raster.mode != expected_mode:
                    report.error("raster_mode", f"{role} must have mode {expected_mode}; found {raster.mode}.", where)
                    return
                if getattr(raster, "n_frames", 1) != 1:
                    report.error("animated_image", "Animated or multi-frame images are not supported.", where)
                    return
                if raster.getexif().get(274, 1) != 1:
                    report.error("exif_orientation", f"{role} orientation must already be normalized.", where)
                    return
                raster.load()
                report.stats[f"{role}s_checked"] += 1
                if role == "mask":
                    histogram = raster.histogram()
                    unexpected = [index for index, count in enumerate(histogram) if count and index not in (0, 1, 255)]
                    if unexpected:
                        report.error("mask_labels", f"Unexpected mask values: {unexpected}; expected only 0, 1, 255.", where)
                        return
                    for key, index in (("background_pixels", 0), ("pv_pixels", 1), ("ignored_pixels", 255)):
                        report.stats[key] += histogram[index]
                    if histogram[255] == sample["width"] * sample["height"]:
                        if sample["split"] in ASSIGNED_SPLITS:
                            report.error("all_ignored", "An assigned mask must contain supervised pixels.", where)
                        else:
                            report.warn("all_ignored", "Mask is entirely ignored and cannot provide supervision.", where)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombWarning, Image.DecompressionBombError):
        report.error("image_decode", f"{role} is not a readable, supported image.", where)


def _overlap(value: str, split: str, seen: dict, kind: str, report: ValidationReport, where: str) -> None:
    if split not in ASSIGNED_SPLITS:
        return
    previous = seen.get(value)
    if previous is not None and previous != split:
        report.error("split_leakage", f"The same {kind} occurs in both {previous!r} and {split!r}.", where)
    else:
        seen[value] = split


def validate_dataset(directory: Path | str) -> ValidationReport:
    """Validate metadata, files, mask values and declared split boundaries offline."""
    report = ValidationReport()
    try:
        root = Path(directory).resolve(strict=True)
        if not root.is_dir():
            raise NotADirectoryError
    except (OSError, RuntimeError, ValueError):
        report.error("dataset_directory", "Dataset directory is missing or inaccessible.", "dataset")
        return report
    metadata_path = _path(root, "dataset.toml", report, "dataset.toml")
    if metadata_path is None:
        return report
    metadata_bytes = _read(metadata_path, MAX_METADATA_BYTES, report, "dataset.toml")
    if metadata_bytes is None:
        return report
    metadata = _metadata(metadata_bytes, report)
    if metadata is None:
        return report
    report.stats.update(dataset_id=metadata["dataset_id"], dataset_version=metadata["dataset_version"])
    manifest_path = _path(root, "manifest.jsonl", report, "manifest.jsonl")
    if manifest_path is None:
        return report
    manifest_bytes = _read(manifest_path, MAX_MANIFEST_BYTES, report, "manifest.jsonl")
    if manifest_bytes is None:
        return report
    report.stats.update(
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        dataset_fingerprint=hashlib.sha256(metadata_bytes + b"\0" + manifest_bytes).hexdigest(),
    )
    sample_ids: set[str] = set()
    image_hashes: set[str] = set()
    split_images: dict[str, str] = {}
    groups: dict[str, str] = {}
    buildings: dict[str, str] = {}
    gsd_values: set[float] = set()
    for line_number, line in enumerate(BytesIO(manifest_bytes), start=1):
        where = f"manifest.jsonl:{line_number}"
        if len(report.errors) >= MAX_ISSUES:
            report.stats["stopped_early"] = True
            break
        if line_number > MAX_RECORDS:
            report.error("record_limit", f"Manifest exceeds {MAX_RECORDS} records.", where)
            report.stats["stopped_early"] = True
            break
        if len(line) > MAX_LINE_BYTES:
            report.error("line_limit", f"Manifest line exceeds {MAX_LINE_BYTES} bytes.", where)
            report.stats["stopped_early"] = True
            break
        report.stats["records"] += 1
        try:
            sample = json.loads(line.decode("utf-8"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
        except (UnicodeError, ValueError, RecursionError):
            report.error("invalid_json", "Expected one UTF-8 JSON object with unique keys and finite numbers per line.", where)
            continue
        if not _record(sample, metadata["sources"], report, where):
            continue
        if sample["id"] in sample_ids:
            report.error("duplicate_id", "Sample ID occurs more than once.", where)
        sample_ids.add(sample["id"])
        if sample["image_sha256"] in image_hashes:
            report.warn("duplicate_image", "Image content occurs more than once in the dataset.", where)
        image_hashes.add(sample["image_sha256"])
        _overlap(sample["image_sha256"], sample["split"], split_images, "image content", report, where)
        _overlap(sample["spatial_group"], sample["split"], groups, "spatial group", report, where)
        for building in sample["building_ids"]:
            _overlap(building, sample["split"], buildings, "building ID", report, where)
        report.stats["splits"][sample["split"]] += 1
        gsd_values.add(float(sample["gsd_m"]))
        _raster(root, sample, "image", report, where)
        _raster(root, sample, "mask", report, where)
    if report.stats["records"] == 0:
        report.error("empty_manifest", "Dataset must contain at least one record.", "manifest.jsonl")
    report.stats["gsd_m_values"] = sorted(gsd_values)
    return report
