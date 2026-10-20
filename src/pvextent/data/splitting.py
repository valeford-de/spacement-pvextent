# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Deterministic spatial blocks with a measured buffer between assigned splits.

This creates an internal holdout for known source centres. It cannot establish
independence from published models, missing building IDs or acquisition campaigns.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from PIL import Image

from .pv4ger import DatasetImportError, _publish, _sha256, _write_json
from .snapshots import copy_snapshot, load_centers, read_snapshot
from .validation import validate_dataset

POLICY = "geographic-block-buffer-v1"
METRIC_CRS = "EPSG:25832"


def _positive_number(value: Any) -> bool:
    try:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
    except OverflowError:
        return False


def _block_split(cell: tuple[int, int], seed: int, ratios: tuple[float, ...]) -> str:
    token = f"{POLICY}:{seed}:{cell[0]}:{cell[1]}".encode()
    value = int.from_bytes(hashlib.sha256(token).digest()[:8], "big") / 2 ** 64
    return "train" if value < ratios[0] else "val" if value < ratios[0] + ratios[1] else "test"


def _components(records: list[dict]) -> list[list[dict]]:
    parent = {row["id"]: row["id"] for row in records}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    seen = {}
    for row in sorted(records, key=lambda row: row["id"]):
        tokens = [("group", row["spatial_group"]), ("image", row["image_sha256"])]
        tokens.extend(("building", item) for item in row["building_ids"])
        for token in tokens:
            if token in seen:
                parent[find(row["id"])] = find(seen[token])
            else:
                seen[token] = row["id"]
    result = defaultdict(list)
    for row in sorted(records, key=lambda row: row["id"]):
        result[find(row["id"])].append(row)
    return list(result.values())


def _proximity_index(points: list[tuple[float, float]], distance: float) -> dict:
    result = defaultdict(list)
    for point in points:
        result[(math.floor(point[0] / distance), math.floor(point[1] / distance))].append(point)
    return result


def _near(point: tuple[float, float], index: dict, distance: float) -> bool:
    x, y = math.floor(point[0] / distance), math.floor(point[1] / distance)
    return any(math.dist(point, other) < distance
               for dx in (-1, 0, 1) for dy in (-1, 0, 1)
               for other in index.get((x + dx, y + dy), ()))


def _minimum_distances(records: list[dict], points: dict) -> dict[str, float]:
    by_split = {split: [points[row["id"]] for row in records if row["split"] == split]
                for split in ("train", "val", "test")}
    return {f"{left}-{right}": min(math.dist(a, b) for a in by_split[left] for b in by_split[right])
            for left, right in (("train", "val"), ("train", "test"), ("val", "test"))}


def split_dataset(
    source_dir: Path | str, output_dir: Path | str, *, seed: int = 20261020,
    block_size_m: float = 10000.0, buffer_m: float = 1000.0,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, Any]:
    """Freeze a new validated snapshot, preserving pixel bytes and source records."""
    if (isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2 ** 32 - 1
            or any(not _positive_number(value) for value in (block_size_m, buffer_m))
            or block_size_m < 100 or buffer_m < 50 or buffer_m >= block_size_m):
        raise DatasetImportError("Use a uint32 seed, block size >=100 m and buffer >=50 m smaller than the block.")
    if (not isinstance(ratios, (list, tuple)) or len(ratios) != 3
            or any(not _positive_number(value) for value in ratios)
            or not math.isclose(sum(ratios), 1.0, abs_tol=1e-9)):
        raise DatasetImportError("Provide three positive train/val/test fractions summing to 1.")
    try:
        from pyproj import Transformer
    except ImportError as error:
        raise DatasetImportError("Geographic dependencies missing. Run: uv sync --locked --extra geo") from error
    source, output = Path(source_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise DatasetImportError(f"Output already exists: {output}")
    output = output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise DatasetImportError("Source and output dataset must be separate directories.")
    staging = None
    try:
        records, source_validation = read_snapshot(source)
        if any(row["split"] != "unassigned" for row in records):
            raise DatasetImportError("Source already contains assigned splits; refusing to reshuffle a frozen snapshot.")
        centers, coordinate_hash = load_centers(source, records)
        transform = Transformer.from_crs("EPSG:4326", METRIC_CRS, always_xy=True)
        points = {key: transform.transform(*point, errcheck=True) for key, point in centers.items()}
        groups = _components(records)
        decisions, eligible = {}, {}
        for index, members in enumerate(groups):
            cell = (math.floor(sum(points[row["id"]][0] for row in members) / len(members) / block_size_m),
                    math.floor(sum(points[row["id"]][1] for row in members) / len(members) / block_size_m))
            reason = None
            for row in members:
                with Image.open(source / row["mask"]) as image:
                    histogram = image.histogram()
                if row["annotation_status"] == "unreviewed":
                    reason = "unreviewed"
                    break
                if histogram[0] + histogram[1] == 0:
                    reason = "no_supervised_pixels"
                    break
                if histogram[1] == 0 and row["annotation_status"] != "reviewed":
                    reason = "zero_mask_requires_review"
                    break
            decisions[index] = {"cell": cell, "exclusion": reason,
                                "proposed_split": _block_split(cell, seed, ratios),
                                "split": "unassigned" if reason else _block_split(cell, seed, ratios)}
            if reason is None:
                eligible[index] = members
        # Holdout priority is fixed: test, then val, then train. We conservatively
        # buffer training against tentative validation too, even if later excluded.
        holdout_points = [points[row["id"]] for index, members in eligible.items()
                          if decisions[index]["split"] in ("val", "test") for row in members]
        test_points = [points[row["id"]] for index, members in eligible.items()
                       if decisions[index]["split"] == "test" for row in members]
        holdout_index = _proximity_index(holdout_points, buffer_m)
        test_index = _proximity_index(test_points, buffer_m)
        for index, members in eligible.items():
            split = decisions[index]["split"]
            nearby = holdout_index if split == "train" else test_index if split == "val" else None
            if nearby is not None and any(_near(points[row["id"]], nearby, buffer_m) for row in members):
                decisions[index].update(split="unassigned", exclusion="geographic_buffer")
        assignment_rows, output_rows = [], []
        for index, members in enumerate(groups):
            decision = decisions[index]
            x, y = decision["cell"]
            for row in members:
                output_rows.append({**row, "split": decision["split"],
                                    "spatial_group": f"pvextent:geo-block-v1:{x}:{y}"})
                assignment_rows.append({"id": row["id"], "source_spatial_group": row["spatial_group"],
                                        **decision})
        output_rows.sort(key=lambda row: row["id"])
        counts = Counter(row["split"] for row in output_rows)
        if any(counts[split] == 0 for split in ("train", "val", "test")):
            raise DatasetImportError("Policy leaves an empty train/val/test split; use more geographic areas or adjust the explicit policy.")
        distances = _minimum_distances(output_rows, points)
        if min(distances.values()) < buffer_m:
            raise DatasetImportError("Geographic buffer audit failed.")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}-split-"))
        copy_snapshot(source, staging, records)
        provenance = staging / "provenance"
        provenance.mkdir(exist_ok=True)
        if (provenance / "split-input-manifest.jsonl").exists():
            raise DatasetImportError("Input already contains split provenance; refusing to replace it.")
        shutil.copyfile(source / "manifest.jsonl", provenance / "split-input-manifest.jsonl")
        if (_sha256(staging / "dataset.toml") != source_validation.stats["metadata_sha256"]
                or _sha256(provenance / "split-input-manifest.jsonl") != source_validation.stats["manifest_sha256"]
                or _sha256(provenance / "samples.jsonl") != coordinate_hash):
            raise DatasetImportError("Source metadata or coordinates changed while preparing the split snapshot.")
        source_reports = {}
        for name in ("import-report.json", "validation-report.json"):
            path = source / name
            if path.exists():
                if path.is_symlink() or not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
                    raise DatasetImportError("Source reports must be bounded regular files.")
                payload = path.read_bytes()
                target = provenance / ("split-input-" + name)
                if target.exists():
                    raise DatasetImportError(f"Refusing to replace existing split provenance: {target.name}")
                target.write_bytes(payload)
                source_reports[name] = hashlib.sha256(payload).hexdigest()
        with (provenance / "split-assignments.jsonl").open("w", encoding="utf-8") as stream:
            for row in sorted(assignment_rows, key=lambda row: row["id"]):
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        with (staging / "manifest.jsonl").open("w", encoding="utf-8") as stream:
            for row in output_rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        validation = validate_dataset(staging)
        if not validation.ok:
            raise DatasetImportError(f"Split snapshot failed validation: {validation.errors[0].message}")
        _write_json(staging / "validation-report.json", validation.to_dict())
        report = {
            "ok": True, "policy": POLICY, "metric_crs": METRIC_CRS, "seed": seed,
            "block_size_m": block_size_m, "buffer_m": buffer_m,
            "target_fractions": dict(zip(("train", "val", "test"), ratios)),
            "assignment_method": "SHA-256 of policy, seed and component-mean metric cell; fractions are expectations, not exact counts.",
            "source_dataset_fingerprint": source_validation.stats["dataset_fingerprint"],
            "source_coordinates_sha256": coordinate_hash,
            "source_reports_sha256": source_reports,
            "dataset_fingerprint": validation.stats["dataset_fingerprint"],
            "records": len(output_rows), "splits": {key: counts[key] for key in ("train", "val", "test", "unassigned")},
            "components": len(groups), "geographic_blocks": len({item["cell"] for item in decisions.values()}),
            "excluded_counts": dict(Counter(row["exclusion"] for row in assignment_rows if row["exclusion"])),
            "minimum_cross_split_center_distance_m": distances,
            "validation": {"ok": validation.ok, "error_count": len(validation.errors), "warning_count": validation.warning_count},
            "limitations": ["Internal source holdout, not independent of published model training data.",
                            "Distances use approximate filename centres in EPSG:25832, not surveyed building boundaries.",
                            "Source annotation status is preserved; upstream labels are not locally reviewed.",
                            "All-zero upstream masks remain unassigned until review; representative hard negatives are still needed.",
                            "A frozen test partition must not be used for threshold, architecture or seed selection."],
        }
        _write_json(staging / "split-report.json", report)
        _publish(staging, output)
        staging = None
        return report
    except DatasetImportError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, OverflowError) as error:
        raise DatasetImportError(f"Geographic splitting failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
