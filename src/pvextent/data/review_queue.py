# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Prepare bounded, blinded review tasks without accessing any raster paths."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .pv4ger import DatasetImportError, _publish

MAX_QUEUE_SAMPLES = 2000
MAX_BATCH_SIZE = 50
MAX_QUEUE_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_QUEUE_METADATA_BYTES = 1024 * 1024


class TrainingReviewQueueError(DatasetImportError):
    """A pending training snapshot cannot safely become a review queue."""


def _absolute(path: Path | str) -> Path:
    # Normalize directory traversal lexically; never resolve a raster path.
    return Path(os.path.abspath(path))


def _no_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise TrainingReviewQueueError("Queue and dataset paths must not contain symlinks.")


def _source_bounds(source: Path) -> None:
    for name, limit in (("dataset.toml", MAX_QUEUE_METADATA_BYTES),
                        ("manifest.jsonl", MAX_QUEUE_MANIFEST_BYTES)):
        path = source / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise TrainingReviewQueueError(f"Expected a regular, bounded, non-symlink {name}.")


def _write_json(path: Path, document: dict) -> str:
    payload = (json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def export_training_review_queue(dataset_dir: Path | str, output_dir: Path | str, *,
                                 batch_size: int = 25, seed: int = 20261021) -> dict:
    """Export two independent pending review lanes, never labels or ground truth.

    Only the canonical metadata and manifest are read. Absolute image references
    are formed lexically, including when those images do not exist locally.
    """
    stage = None
    try:
        from .training import _read_manifest

        if type(batch_size) is not int or not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise TrainingReviewQueueError(f"batch_size must be an integer between 1 and {MAX_BATCH_SIZE}.")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise TrainingReviewQueueError("seed must be a nonnegative integer smaller than 2**63.")
        source, output = _absolute(dataset_dir), _absolute(output_dir)
        _no_symlinks(source)
        _no_symlinks(output)
        if not source.is_dir():
            raise TrainingReviewQueueError("Dataset directory is missing or inaccessible.")
        if source == output or source.is_relative_to(output) or output.is_relative_to(source):
            raise TrainingReviewQueueError("Dataset and queue directories must not overlap as ancestors.")
        if os.path.lexists(output):
            raise TrainingReviewQueueError("Queue output already exists; refusing to replace it.")
        _source_bounds(source)
        metadata, records, fingerprint = _read_manifest(source)
        if metadata["dataset_id"] != "nrw-training-bulk":
            raise TrainingReviewQueueError("Only the nrw-training-bulk dataset may enter this queue.")
        if not 1 <= len(records) <= MAX_QUEUE_SAMPLES:
            raise TrainingReviewQueueError(f"Queue requires 1..{MAX_QUEUE_SAMPLES} samples.")
        for record in records:
            if (record["split"] != "unassigned" or record["annotation_status"] != "unreviewed"
                    or record["width"] != 320 or record["height"] != 320 or record["gsd_m"] != .1):
                raise TrainingReviewQueueError("Queue requires unassigned, unreviewed native 320x320 samples at 0.1 m GSD.")
        ranked = sorted(records, key=lambda record: (
            hashlib.sha256(f"{seed}:{record['id']}".encode()).hexdigest(), record["id"]))
        batches = [ranked[start:start + batch_size] for start in range(0, len(ranked), batch_size)]
        output.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(output.parent)
        stage = Path(tempfile.mkdtemp(prefix=f".{output.name}-review-queue-", dir=output.parent))
        index = {
            "schema_version": "1.0", "purpose": "pending-blinded-training-review",
            "dataset_fingerprint": fingerprint, "sample_count": len(ranked),
            "batch_size": batch_size, "batch_count_per_lane": len(batches),
            "review_template_count": 2 * len(batches), "pending_only": True,
            "raster_paths_accessed": False, "labels_created": False,
            "model_predictions_used": False, "canonical_dataset_modified": False,
            "lanes": [],
        }
        for lane in ("A", "B"):
            directory = stage / f"lane-{lane.lower()}"
            directory.mkdir()
            lane_entry = {"id": lane, "batches": []}
            for number, batch in enumerate(batches, 1):
                batch_id = f"batch-{number:04d}"
                manifest_path = directory / f"{batch_id}.manifest.json"
                template_path = directory / f"{batch_id}.review.json"
                # This allowlist deliberately excludes provenance, geographic
                # groups, building IDs, masks, selection reasons and predictions.
                samples = [{
                    "id": row["id"], "image_sha256": row["image_sha256"],
                    "image": str(source / row["image"]), "width": row["width"],
                    "height": row["height"], "gsd_m": row["gsd_m"],
                    "capture_date": row["capture_date"],
                } for row in batch]
                manifest_hash = _write_json(manifest_path, {
                    "schema_version": "1.0", "dataset_fingerprint": fingerprint,
                    "batch_id": batch_id, "lane": lane, "samples": samples,
                })
                template_hash = _write_json(template_path, {
                    "review_schema_version": "1.0", "reviewer": f"PENDING-REVIEWER-{lane}-{batch_id}",
                    "reviewer_type": "ai", "model_predictions_used": False,
                    "samples": [{
                        "id": row["id"], "image_sha256": row["image_sha256"],
                        "full_image_inspected": False, "decision": "defer",
                        "positive_polygons": [], "background_polygons": [],
                        "notes": "Pending independent full-image inspection.",
                    } for row in batch],
                })
                lane_entry["batches"].append({
                    "id": batch_id, "sample_ids": [row["id"] for row in batch],
                    "manifest": str(manifest_path.relative_to(stage)),
                    "manifest_sha256": manifest_hash,
                    "review_template": str(template_path.relative_to(stage)),
                    "review_template_sha256": template_hash,
                })
            index["lanes"].append(lane_entry)
        _write_json(stage / "queue.json", index)
        (stage / "README.md").write_text(
            "# Pending training review queue\n\n"
            "This queue contains proposals only: no reviewed labels or ground truth. "
            "Canonical datasets remain unchanged. No image or mask was accessed during export.\n\n"
            "1. Assign lanes A and B to two distinct AI reviewers. Replace each PENDING reviewer "
            "placeholder with the actual reviewer identity. Keep both reviews blind to each "
            "other, acquisition provenance, existing labels and model predictions.\n"
            "2. Check each manifest image hash and inspect the complete original image at its "
            "native 320x320 resolution before setting full_image_inspected to true. "
            "Do not infer inspection from a filename, coordinate or a model prediction.\n"
            "3. Draw native-pixel polygons around visible PV modules; exclude gaps between "
            "modules and uncertain boundaries. Mark only clearly visible background in "
            "background_polygons. Leave uncertain areas unlabelled and defer ambiguous samples.\n"
            "4. Use approve-positive only for inspected positive proposals; decision=defer "
            "remains available. This positive-proposal schema cannot approve all-negative "
            "images: defer inspected clearly negative images here and route them through "
            "the separate existing negative curation workflow. Empty positive polygons "
            "never establish PV absence. The initial templates intentionally fail curation because "
            "full_image_inspected is false. Preserve IDs and hashes.\n"
            "5. Compare both independent reviews in a separate consensus/curation step before "
            "any experimental training. Queue export itself does not release supervision. "
            "The original test partition and frozen external holdouts remain reserved; "
            "never use their imagery or labels for these reviews or training.\n\n"
            "queue.json binds the source metadata/manifest fingerprint and exact batch bytes "
            "by SHA-256. Each opaque sample ID occurs exactly once in each lane. "
            "Image references are local paths, not copies.\n",
            encoding="utf-8",
        )
        _source_bounds(source)
        if _read_manifest(source)[2] != fingerprint:
            raise TrainingReviewQueueError("Dataset metadata or manifest changed while exporting the queue.")
        _no_symlinks(source)
        _no_symlinks(output)
        _publish(stage, output)
        stage = None
        return {**index, "output_dir": str(output)}
    except TrainingReviewQueueError:
        raise
    except (OSError, RuntimeError, ValueError, ImportError) as error:
        raise TrainingReviewQueueError(f"Cannot export training review queue: {error}") from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
