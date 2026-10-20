# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Append reviewed negatives while preserving an existing geographic holdout."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from .pv4ger import DatasetImportError, _publish, _sha256, _write_json
from .snapshots import load_centers
from .validation import ValidationReport, _path, validate_dataset


class TrainingAugmentationError(DatasetImportError):
    """A curated collection cannot be added without violating the holdout."""


def _metadata_text(base: dict, collection: dict, dataset_version: str) -> str:
    sources = dict(base["sources"])
    for key, entry in collection["sources"].items():
        if key in sources and sources[key] != entry:
            raise TrainingAugmentationError("Conflicting source metadata identifiers.")
        sources[key] = entry
    lines = ['schema_version = "1.0"', 'dataset_id = "pvextent-training-augmented"',
             'dataset_version = ' + json.dumps(dataset_version)]
    for key, fields in sorted(sources.items()):
        lines.append('\n[sources.' + json.dumps(key) + ']')
        lines.extend(name + ' = ' + json.dumps(value, ensure_ascii=True) for name, value in fields.items())
    return '\n'.join(lines) + '\n'


def _link(source: Path, row: dict, role: str, stage: Path, relative: str) -> None:
    """Link immutable source bytes without opening or decoding held-out rasters."""
    report = ValidationReport()
    path = _path(source, row[role], report, row["id"])
    if not report.ok or path is None or not path.is_file():
        raise TrainingAugmentationError("Cannot link an inherited raster safely.")
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not os.path.samefile(path, target):
            raise TrainingAugmentationError("Conflicting inherited raster paths.")
        return
    os.link(path, target)
    if not os.path.samefile(path, target):
        raise TrainingAugmentationError("Inherited raster link verification failed.")


def _provenance(source: Path, destination: Path) -> None:
    directory = source / "provenance"
    if directory.is_symlink():
        raise TrainingAugmentationError("Provenance directory cannot be a link.")
    total = 0
    if directory.is_dir():
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise TrainingAugmentationError("Provenance links are not copied.")
            if path.is_dir():
                continue
            total += path.stat().st_size
            if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024 or total > 64 * 1024 * 1024:
                raise TrainingAugmentationError("Provenance bounds exceeded.")
            target = destination / path.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def augment_training_snapshot(base_dir: Path | str, negative_dir: Path | str, output_dir: Path | str,
                              *, dataset_version: str, buffer_m: float = 1000.0) -> dict:
    """Add supervised reviewed negatives to train; leave val/test records unchanged.

    The output inherits pixel files as hard links. Treat both snapshots as
    immutable: future edits require replacement files in another new snapshot.
    """
    stage = None
    try:
        from pyproj import Transformer
        from .training import _read_manifest

        if (not isinstance(dataset_version, str) or not dataset_version.strip() or len(dataset_version) > 128
                or isinstance(buffer_m, bool) or not isinstance(buffer_m, (int, float))
                or not math.isfinite(buffer_m) or buffer_m < 1000):
            raise TrainingAugmentationError("Use a nonempty version and a buffer of at least 1000 metres.")
        base, negative, output = Path(base_dir).resolve(), Path(negative_dir).resolve(), Path(output_dir).absolute()
        if os.path.lexists(output):
            raise TrainingAugmentationError(f"Output already exists: {output}")
        output = output.resolve()
        if any(left.is_relative_to(right) or right.is_relative_to(left)
               for left, right in ((base, negative), (base, output), (negative, output))):
            raise TrainingAugmentationError("Base, negative collection and output must be separate directories.")
        base_meta, base_rows, base_fingerprint = _read_manifest(base)
        negative_meta, negative_rows, negative_fingerprint = _read_manifest(negative)
        curation_evidence = negative / "curation-report.json"
        if os.path.lexists(curation_evidence):
            from .curation import _bounded_bytes
            from .validation import _unique_object, _invalid_constant
            evidence = json.loads(_bounded_bytes(curation_evidence, 2 * 1024 * 1024),
                                  object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
            if not isinstance(evidence, dict) or not isinstance(evidence.get("sources"), list):
                raise TrainingAugmentationError("Invalid curation lineage evidence.")
            if any(isinstance(source, dict) and source.get("dataset_id") in
                   ("nrw-negative-holdout-candidates", "pvextent-negative-holdout", "pvextent-positive-holdout") for source in evidence["sources"]):
                raise TrainingAugmentationError("Reserved holdout lineage cannot enter training.")
        if (len(negative_rows) > 64 or any(row["split"] != "unassigned" for row in negative_rows)
                or not all(any(row["split"] == split for row in base_rows) for split in ("train", "val", "test"))):
            raise TrainingAugmentationError("Expected a frozen base split and at most 64 unassigned curated records.")
        base_by_id = {row["id"]: row for row in base_rows}
        holdouts = [row for row in base_rows if row["split"] in ("val", "test")]
        held_groups, held_images = {row["spatial_group"] for row in holdouts}, {row["image_sha256"] for row in holdouts}
        held_buildings = {key for row in holdouts for key in row["building_ids"]}
        for row in negative_rows:
            if (row["image_sha256"] in held_images or row["spatial_group"] in held_groups
                    or held_buildings.intersection(row["building_ids"])):
                raise TrainingAugmentationError("Curated declarations overlap the frozen holdout; no candidate pixels were opened.")
            old = base_by_id.get(row["id"])
            if old is not None:
                if old["split"] != "unassigned":
                    raise TrainingAugmentationError("Curated records cannot replace assigned train, validation or test records.")
                unchanged = ("image_sha256", "width", "height", "gsd_m", "capture_date", "spatial_group", "building_ids", "georeference")
                if any(old[key] != row[key] for key in unchanged):
                    raise TrainingAugmentationError("Curated sample identity or spatial declarations differ from the frozen parent.")
        validation = validate_dataset(negative)  # Only the separate curated collection, never base holdout pixels.
        if not validation.ok or validation.stats["dataset_fingerprint"] != negative_fingerprint:
            raise TrainingAugmentationError("Curated negative collection is invalid or changed.")
        proposed = {}
        assignments = base / "provenance" / "split-assignments.jsonl"
        if assignments.exists():
            if assignments.is_symlink() or not assignments.is_file() or assignments.stat().st_size > 16 * 1024 * 1024:
                raise TrainingAugmentationError("Invalid parent split provenance.")
            for line in assignments.read_text(encoding="utf-8").splitlines():
                item = json.loads(line)
                if item["id"] in proposed or item["id"] not in base_by_id or item["proposed_split"] not in ("train", "val", "test"):
                    raise TrainingAugmentationError("Invalid or duplicate parent split assignment.")
                proposed[item["id"]] = item["proposed_split"]
            if set(proposed) != set(base_by_id):
                raise TrainingAugmentationError("Parent split provenance must cover all records.")
        centers, coordinate_hash = load_centers(base, base_rows)
        project = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
        points = {key: project.transform(*value, errcheck=True) for key, value in centers.items()}
        for row in negative_rows:
            if row["id"] in points:
                continue
            geo = row["georeference"]
            if geo is None:
                raise TrainingAugmentationError("New candidates need a georeference for the holdout-distance audit.")
            a, b, c, d, e, f = geo["transform"]
            x, y = a * row["width"] / 2 + b * row["height"] / 2 + c, d * row["width"] / 2 + e * row["height"] / 2 + f
            transformer = Transformer.from_crs(geo["crs"], "EPSG:25832", always_xy=True)
            point = transformer.transform(x, y, errcheck=True)
            if not all(math.isfinite(value) for value in point):
                raise TrainingAugmentationError("Invalid new candidate centre.")
            points[row["id"]] = point
            inverse = Transformer.from_crs("EPSG:25832", "EPSG:4326", always_xy=True)
            centers[row["id"]] = inverse.transform(*point, errcheck=True)
        decisions, replacements = [], {}
        for row in negative_rows:
            with Image.open(negative / row["mask"]) as mask:
                histogram = mask.histogram()
            if histogram[1] or (row["annotation_status"] == "reviewed" and histogram[0] < 256):
                raise TrainingAugmentationError("Reviewed negatives require at least 256 background pixels and no PV labels.")
            distance = min(math.dist(points[row["id"]], points[other["id"]]) for other in holdouts)
            reason = ("not_reviewed" if row["annotation_status"] != "reviewed" else
                      "reserved_geographic_block" if row["id"] in proposed and proposed[row["id"]] != "train" else
                      "geographic_buffer" if distance < buffer_m else
                      "heldout_identity_overlap" if row["spatial_group"] in held_groups or row["image_sha256"] in held_images
                      or held_buildings.intersection(row["building_ids"]) else None)
            key = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
            replacement = {**row, "split": "train" if reason is None else "unassigned",
                           "image": base_by_id[row["id"]]["image"] if row["id"] in base_by_id else f"images/curated/{key}.png",
                           "mask": f"masks/curated/{key}.png"}
            replacements[row["id"]] = replacement
            decisions.append({"id": row["id"], "split": replacement["split"], "exclusion": reason,
                              "nearest_heldout_center_m": distance, "background_pixels": histogram[0], "ignore_pixels": histogram[255]})
        merged = sorted([replacements.get(row["id"], row) for row in base_rows]
                        + [row for key, row in replacements.items() if key not in base_by_id], key=lambda row: row["id"])
        if any(next(row for row in merged if row["id"] == old["id"]) != old for old in holdouts):
            raise TrainingAugmentationError("Frozen holdout declarations changed.")
        inherited_paths = {row["image"] for row in base_rows}
        inherited_paths.update(row["mask"] for row in base_rows if row["id"] not in replacements)
        new_paths = []
        for row in negative_rows:
            replacement = replacements[row["id"]]
            new_paths.append(replacement["mask"])
            if row["id"] not in base_by_id:
                new_paths.append(replacement["image"])
        if len(set(new_paths)) != len(new_paths) or inherited_paths.intersection(new_paths):
            raise TrainingAugmentationError("Curated output paths collide with inherited immutable rasters.")
        output.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-augmentation-", dir=output.parent))
        (stage / "dataset.toml").write_text(_metadata_text(base_meta, negative_meta, dataset_version), encoding="utf-8")
        for row in base_rows:
            _link(base, row, "image", stage, row["image"])
            if row["id"] not in replacements:
                _link(base, row, "mask", stage, row["mask"])
        for row in negative_rows:
            replacement = replacements[row["id"]]
            if row["id"] not in base_by_id:
                _link(negative, row, "image", stage, replacement["image"])
            target = stage / replacement["mask"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source_mask = negative / row["mask"]
            if source_mask.stat().st_size > 64 * 1024 * 1024:
                raise TrainingAugmentationError("Curated mask exceeds its byte bound.")
            with source_mask.open("rb") as stream:
                payload = stream.read(64 * 1024 * 1024 + 1)
            if len(payload) > 64 * 1024 * 1024 or hashlib.sha256(payload).hexdigest() != row["mask_sha256"]:
                raise TrainingAugmentationError("Curated mask changed while building the snapshot.")
            with target.open("xb") as stream:
                stream.write(payload)
        provenance = stage / "provenance"
        _provenance(base, provenance / "parent")
        _provenance(negative, provenance / "curation")
        provenance.mkdir(exist_ok=True)
        shutil.copyfile(base / "dataset.toml", provenance / "parent-dataset.toml")
        shutil.copyfile(base / "manifest.jsonl", provenance / "parent-manifest.jsonl")
        shutil.copyfile(negative / "dataset.toml", provenance / "curated-dataset.toml")
        _write_json(provenance / "augmentation-decisions.json", decisions)
        reservations = [{"id": row["id"], "proposed_split": proposed.get(row["id"],
            base_by_id.get(row["id"], row)["split"] if base_by_id.get(row["id"], row)["split"] in ("train", "val", "test") else "train"),
            "split": row["split"]} for row in merged]
        (provenance / "split-assignments.jsonl").write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in reservations), encoding="utf-8")
        (provenance / "samples.jsonl").write_text(''.join(json.dumps({"id": row["id"], "approximate_center": list(centers[row["id"]]),
            "center_crs": "EPSG:4326"}, sort_keys=True) + '\n' for row in merged), encoding="utf-8")
        (stage / "manifest.jsonl").write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in merged), encoding="utf-8")
        _, checked, fingerprint = _read_manifest(stage)
        if checked != merged or _read_manifest(base)[2] != base_fingerprint or _read_manifest(negative)[2] != negative_fingerprint:
            raise TrainingAugmentationError("Source or output declarations changed during augmentation.")
        report = {"ok": True, "dataset_fingerprint": fingerprint, "parent_dataset_fingerprint": base_fingerprint,
                  "curated_dataset_fingerprint": negative_fingerprint, "parent_coordinates_sha256": coordinate_hash,
                  "reservation_provenance_sha256": _sha256(provenance / "split-assignments.jsonl"),
                  "records": len(merged), "splits": dict(Counter(row["split"] for row in merged)),
                  "reviewed_negatives_added_to_train": sum(row["split"] == "train" for row in decisions),
                  "curated_records": len(negative_rows), "buffer_m": buffer_m,
                  "minimum_added_train_to_holdout_center_m": min((row["nearest_heldout_center_m"] for row in decisions if row["split"] == "train"), default=None),
                  "heldout_records_unchanged": True, "heldout_pixels_opened": False,
                  "inherited_pixel_storage": "Hard links; all source and output pixel files must remain immutable.",
                  "validation_scope": "Complete metadata/manifest checks and curated pixel validation; inherited holdout pixel files linked without reading or decoding.",
                  "limitations": ["Experimental AI-reviewed negative annotations are not human-reviewed benchmark truth.",
                                  "Distances use source centres; reviewed negative enrichment does not estimate real-world prevalence."]}
        _write_json(stage / "augmentation-report.json", report)
        _publish(stage, output)
        stage = None
        return report
    except TrainingAugmentationError:
        raise
    except ImportError as error:
        raise TrainingAugmentationError("Install the train and geo extras for snapshot augmentation.") from error
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, OverflowError) as error:
        raise TrainingAugmentationError(f"Training augmentation failed: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage)
