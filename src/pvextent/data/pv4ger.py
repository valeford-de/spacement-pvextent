# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Import the pinned GEO-Bench m-pv4ger-seg v0.9.1 release.

Read pixel arrays directly; never deserialize upstream Python/pickle metadata.
Publisher partitions are provenance, not evidence of a geographic holdout.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import ctypes
from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
from io import BytesIO
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any, Callable
from urllib.request import Request, urlopen
import zipfile

from PIL import Image

from .validation import validate_dataset

SOURCE_URL = "https://zenodo.org/records/8277054"
DOWNLOAD_URL = "https://zenodo.org/api/records/8277054/files/{name}/content"
EXPECTED_SPLIT_COUNTS = {"train": 3221, "valid": 403, "test": 403}
GROUP_DISTANCE_M = 100.0
EARTH_RADIUS_M = 6_371_008.8
MAX_HDF5_BYTES = 2 * 1024 * 1024
COORDINATE_ID = re.compile(r"[0-9]+(?:\.[0-9]+)?,[0-9]+(?:\.[0-9]+)?\Z")


@dataclass(frozen=True)
class SourceFile:
    name: str
    size: int
    sha256: str


SOURCE_FILES = (
    SourceFile("data.zip", 316013928,
               "70438a20797bba6eecd905358bf26ace7738709a38848f8a58145f6b0448f82d"),
    SourceFile("default_partition.json", 129229,
               "b0068e4a7516f29b47155dfb674d2003603af687779d91df29a2c17d334890f6"),
    SourceFile("LICENSE", 105,
               "bdebe1f65b5d5a86435e11edd275dbcb5b61d4fcad90672ff1728cd6a2839e84"),
    SourceFile("README", 678,
               "79bfe9f379d2e3b0ce7c72aca98d24ddc1548c2f5510b1c56d6af21acec7df31"),
)

DATASET_METADATA = '''schema_version = "1.0"
dataset_id = "pv4ger-seg"
dataset_version = "geobench-v0.9.1-import-v1"

[sources.pv4ger-seg]
name = "GEO-Bench m-pv4ger-seg / 3D-PV-Locator"
url = "https://zenodo.org/records/8277054"
version = "v0.9.1"
image_license = "dl-de-zero-2.0"
annotation_license = "MIT"
attribution = "Imagery: GeoBasis NRW/openNRW. Annotations: Mayer et al. (2022), 3D-PV-Locator, copyright (c) 2020 Kevin. Adaptation: GEO-Bench, Lacoste et al. See provenance/SOURCES.md and upstream LICENSE."
'''

SOURCE_NOTES = '''# Source provenance

GEO-Bench m-pv4ger-seg v0.9.1: https://zenodo.org/records/8277054
Original: Mayer et al. (2022), 3D-PV-Locator,
https://doi.org/10.1016/j.apenergy.2021.118469
https://github.com/kdmayer/3D-PV-Locator
GEO-Bench: https://github.com/ServiceNow/geo-bench

The bundled LICENSE is retained verbatim as upstream-LICENSE. It links the
original MIT license (copyright (c) 2020 Kevin), included as original-MIT-LICENSE:
https://github.com/kdmayer/3D-PV-Locator/blob/master/LICENSE
The publisher declares MIT for this work. The original authors identify
openNRW as the image source; the GeoBasis NRW terms dated March 2020 state
Datenlizenz Deutschland - Zero 2.0 for its geobasis data and services:
https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf
https://www.govdata.de/dl-de/zero-2-0
dataset.toml therefore records imagery and annotation licenses separately.
PVExtent's Apache-2.0 code license does not relicense these data.

Conversion reads Red, Green, Blue and label arrays without executing pickle.
RGB and 0/1 labels are preserved at 320 x 320 pixels and 0.1 m/pixel.
There is no resizing, augmentation or local human annotation review.
Capture dates and building IDs are unknown. Georeference is null: upstream
metadata is pickled and its converter derives an approximate transform from
filename coordinates rather than preserving an original orthophoto transform.
Filename centers are retained separately as approximate EPSG:4326 coordinates.

default_partition.json and source_splits.json retain ALL publisher assignments.
Canonical splits are unassigned. Overlapping patches can cross publisher
splits. Connected groups of centers within 100 m are computed on the full
partition before any --limit selection. These proximity groups are a heuristic,
not verified building IDs or proof of independence. A later split/review stage
must establish suitable geographic holdouts and hard negatives. This data may
also overlap published models' training data.
'''


class DatasetImportError(ValueError):
    """An import failed without publishing a partial dataset."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: SourceFile) -> None:
    if path.is_symlink() or not path.is_file():
        raise DatasetImportError(f"Missing regular source file: {path}. Use --download to fetch it.")
    if path.stat().st_size != expected.size or _sha256(path) != expected.sha256:
        raise DatasetImportError(f"Source checksum/size mismatch: {path}. Existing files are not replaced.")


def _prepare_source(root: Path, download: bool, progress: Callable[[str], None]) -> None:
    if download:
        root.mkdir(parents=True, exist_ok=True)
    for expected in SOURCE_FILES:
        path = root / expected.name
        if download and not os.path.lexists(path):
            progress(f"Downloading {expected.name} ({expected.size:,} bytes)")
            request = Request(DOWNLOAD_URL.format(name=expected.name),
                              headers={"User-Agent": "Spacement-PVExtent/0.1"})
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=root, prefix=".download-", delete=False) as stream:
                    temporary = Path(stream.name)
                    with urlopen(request, timeout=60) as response:
                        total = 0
                        while block := response.read(1024 * 1024):
                            total += len(block)
                            if total > expected.size:
                                raise DatasetImportError(f"Download exceeds pinned size: {expected.name}")
                            stream.write(block)
                _verify(temporary, expected)
                # A concurrent process must not cause us to replace an existing cache file.
                os.link(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        _verify(path, expected)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DatasetImportError(f"Duplicate partition key: {key}")
        result[key] = value
    return result


def _partition(path: Path) -> tuple[dict[str, str], dict[str, tuple[float, float]]]:
    partition = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(partition, dict) or set(partition) != set(EXPECTED_SPLIT_COUNTS):
        raise DatasetImportError("Unexpected source partition keys.")
    splits, coordinates = {}, {}
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        ids = partition[split]
        if not isinstance(ids, list) or len(ids) != expected_count:
            raise DatasetImportError(f"Unexpected source partition count for {split}.")
        for sample in ids:
            if not isinstance(sample, str) or not COORDINATE_ID.fullmatch(sample) or sample in splits:
                raise DatasetImportError("Invalid or duplicate source sample identifier.")
            lon, lat = map(float, sample.split(","))
            # The pinned converter normalized coordinate order for NRW filenames.
            if not (5 <= lon <= 10 and 50 <= lat <= 53):
                raise DatasetImportError(f"Source coordinate outside expected NRW region: {sample}")
            splits[sample], coordinates[sample] = split, (lon, lat)
    return splits, coordinates


def _spatial_groups(coordinates: dict[str, tuple[float, float]]) -> tuple[dict[str, str], dict]:
    """Connected proximity components; 3-D chord buckets avoid map grid boundaries."""
    parent = {key: key for key in coordinates}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    buckets = defaultdict(list)
    chord_limit = 2 * EARTH_RADIUS_M * math.sin(GROUP_DISTANCE_M / (2 * EARTH_RADIUS_M))
    for key, (lon, lat) in sorted(coordinates.items()):
        lon, lat = math.radians(lon), math.radians(lat)
        point = (EARTH_RADIUS_M * math.cos(lat) * math.cos(lon),
                 EARTH_RADIUS_M * math.cos(lat) * math.sin(lon),
                 EARTH_RADIUS_M * math.sin(lat))
        cell = tuple(math.floor(value / chord_limit) for value in point)
        for offset in itertools.product((-1, 0, 1), repeat=3):
            neighbor = tuple(a + b for a, b in zip(cell, offset))
            for other, other_point in buckets[neighbor]:
                if sum((a - b) ** 2 for a, b in zip(point, other_point)) <= chord_limit ** 2:
                    parent[find(key)] = find(other)
        buckets[cell].append((key, point))
    components = defaultdict(list)
    for key in sorted(coordinates):
        components[find(key)].append(key)
    groups = {}
    for members in components.values():
        identifier = hashlib.sha256("\n".join(members).encode()).hexdigest()[:16]
        groups.update(dict.fromkeys(members, f"pv4ger-seg:proximity100m:{identifier}"))
    return groups, {"method": "connected-center-distance", "distance_m": GROUP_DISTANCE_M,
                    "groups": len(components), "largest_group": max(map(len, components.values()))}


def _archive_members(archive: zipfile.ZipFile, splits: dict[str, str]) -> dict[str, zipfile.ZipInfo]:
    members = {}
    expected = {sample + ".hdf5" for sample in splits}
    for entry in archive.infolist():
        mode = entry.external_attr >> 16
        if (entry.filename not in expected or entry.filename in members
                or entry.is_dir() or stat.S_ISLNK(mode) or entry.flag_bits & 1
                or not 0 < entry.file_size <= MAX_HDF5_BYTES
                or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise DatasetImportError(f"Unexpected or unsafe archive entry: {entry.filename!r}")
        members[entry.filename] = entry
    if set(members) != expected:
        raise DatasetImportError("Archive membership differs from the source partition.")
    return members


def _read_sample(payload: bytes, sample: str, h5py: Any, np: Any) -> tuple[Any, Any]:
    with h5py.File(BytesIO(payload), "r") as source:
        names = ("Red", "Green", "Blue", "label")
        if set(source.keys()) != set(names):
            raise DatasetImportError(f"{sample}: expected Red, Green, Blue, label arrays.")
        arrays = []
        for name in names:
            if not isinstance(source.get(name, getlink=True), h5py.HardLink):
                raise DatasetImportError(f"{sample}: external/soft HDF5 links are not supported.")
            data = source[name]
            if (not isinstance(data, h5py.Dataset) or data.shape != (320, 320)
                    or data.dtype != np.dtype("uint8") or data.is_virtual or data.external
                    or data.id.get_create_plist().get_nfilters()):
                raise DatasetImportError(f"{sample}: expected native, inline, unfiltered 320x320 uint8 {name}.")
            arrays.append(data[:])
        mask = arrays[-1]
        if not np.isin(mask, (0, 1)).all():
            raise DatasetImportError(f"{sample}: upstream labels must be 0/1; 255 is not converted to PV.")
        return np.stack(arrays[:3], axis=-1), mask


def _write_json(path: Path, document: Any) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _publish(staging: Path, output: Path) -> None:
    """Atomically publish without replacing even an empty concurrent directory."""
    if sys.platform == "win32":
        os.rename(staging, output)  # Windows rename fails when the target exists.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename = library.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        arguments = (os.fsencode(staging), os.fsencode(output), 0x4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
        arguments = (-100, os.fsencode(staging), -100, os.fsencode(output), 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise DatasetImportError("Atomic no-replace publishing requires macOS, Linux/glibc or Windows.")
    rename.restype = ctypes.c_int
    if rename(*arguments) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(output))


def import_pv4ger_seg(
    source_dir: Path | str, output_dir: Path | str, *, download: bool = False,
    limit: int | None = None, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Create a validated snapshot; never overwrite an existing output directory.

    ``limit`` selects sorted source IDs for a structural smoke run, not a split.
    Source files are pinned even in offline mode. All outputs remain unassigned.
    """
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise DatasetImportError("--limit must be a positive integer.")
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise DatasetImportError("Importer dependencies missing. Run: uv sync --locked --extra import") from error
    source, output = Path(source_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise DatasetImportError(f"Output already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise DatasetImportError("Source cache and output dataset must be separate directories.")
    notify = progress or (lambda message: None)
    staging = None
    try:
        _prepare_source(source, download, notify)
        splits, coordinates = _partition(source / "default_partition.json")
        groups, grouping = _spatial_groups(coordinates)
        selected = sorted(splits)[:limit]
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-import-", dir=output.parent))
        for folder in ("images", "masks", "provenance"):
            (staging / folder).mkdir()
        (staging / "dataset.toml").write_text(DATASET_METADATA, encoding="utf-8")
        provenance = staging / "provenance"
        for name, target in (("LICENSE", "upstream-LICENSE"), ("README", "upstream-README"),
                             ("default_partition.json", "default_partition.json")):
            shutil.copyfile(source / name, provenance / target)
        (provenance / "SOURCES.md").write_text(SOURCE_NOTES, encoding="utf-8")
        original_license = files("pvextent.data").joinpath("licenses/3d-pv-locator-MIT.txt").read_bytes()
        (provenance / "original-MIT-LICENSE").write_bytes(original_license)
        _write_json(provenance / "source_splits.json", splits)
        counts = Counter()
        with (zipfile.ZipFile(source / "data.zip") as archive,
              (staging / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
              (provenance / "samples.jsonl").open("w", encoding="utf-8") as audit):
            members = _archive_members(archive, splits)
            for index, sample in enumerate(selected, 1):
                rgb, mask = _read_sample(archive.read(members[sample + ".hdf5"]), sample, h5py, np)
                identifier = "pv4ger-" + hashlib.sha256(sample.encode()).hexdigest()[:24]
                image_path, mask_path = f"images/{identifier}.png", f"masks/{identifier}.png"
                Image.fromarray(rgb).save(staging / image_path)
                Image.fromarray(mask).save(staging / mask_path)
                record = {
                    "id": identifier, "image": image_path, "mask": mask_path,
                    "image_sha256": _sha256(staging / image_path),
                    "mask_sha256": _sha256(staging / mask_path),
                    "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "pv4ger-seg",
                    "capture_date": None, "georeference": None, "spatial_group": groups[sample],
                    "building_ids": [], "split": "unassigned", "annotation_status": "upstream",
                }
                manifest.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                lon, lat = coordinates[sample]
                audit.write(json.dumps({"id": identifier, "source_id": sample,
                                        "upstream_split": splits[sample], "center_crs": "EPSG:4326",
                                        "approximate_center": [lon, lat]}, sort_keys=True) + "\n")
                positive = int(np.count_nonzero(mask))
                counts["pv_pixels"] += positive
                counts["positive_samples" if positive else "negative_samples"] += 1
                if index == 1 or index % 500 == 0 or index == len(selected):
                    notify(f"Converted {index:,}/{len(selected):,} samples")
        notify("Validating imported images, masks and provenance")
        validation = validate_dataset(staging)
        if not validation.ok:
            first = validation.errors[0]
            raise DatasetImportError(f"Imported dataset failed validation: {first.code}: {first.message}")
        _write_json(staging / "validation-report.json", validation.to_dict())
        cross_split = defaultdict(set)
        for sample, split in splits.items():
            cross_split[groups[sample]].add(split)
        report = {
            "ok": True, "importer": "pv4ger-seg-v1", "source_url": SOURCE_URL,
            "source_version": "v0.9.1", "source_files": [asdict(item) for item in SOURCE_FILES],
            "source_split_counts": dict(Counter(splits.values())),
            "records": len(selected), "limited": len(selected) < len(splits),
            "split": "unassigned", "annotation_status": "upstream",
            "positive_samples": counts["positive_samples"], "negative_samples": counts["negative_samples"],
            "sample_count_basis": "Nonzero versus all-zero source masks; not verified image-level PV presence.",
            "pv_pixels": counts["pv_pixels"], "background_pixels": len(selected) * 320 * 320 - counts["pv_pixels"],
            "spatial_grouping": {**grouping, "publisher_cross_split_groups":
                                 sum(len(value) > 1 for value in cross_split.values())},
            "dataset_fingerprint": validation.stats["dataset_fingerprint"],
            "validation": {"ok": validation.ok, "error_count": len(validation.errors),
                           "warning_count": validation.warning_count},
            "limitations": ["Publisher splits are retained only as provenance; all records are unassigned.",
                            "100 m proximity groups do not establish geographic or building independence.",
                            "Capture dates, building identifiers and original georeferences are unknown.",
                            "All-zero source masks are not locally reviewed hard negatives.",
                            "Labels are upstream annotations without local review; published-model overlap is possible."],
        }
        _write_json(staging / "import-report.json", report)
        if os.path.lexists(output):
            raise DatasetImportError(f"Output appeared during import; refusing to replace it: {output}")
        _publish(staging, output)
        staging = None
        return report
    except DatasetImportError:
        raise
    except (OSError, ValueError, KeyError, RuntimeError, zipfile.BadZipFile) as error:
        raise DatasetImportError(f"PV4GER-SEG import failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
