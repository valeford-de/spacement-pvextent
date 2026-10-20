# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Importer checks with tiny, generated HDF5 archives; no real imagery in Git."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from pvextent.data import validate_dataset

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = None
    np = None


@unittest.skipIf(h5py is None or np is None, "Install the import extra for HDF5 importer tests")
class PV4GERImportTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.data import pv4ger

        self.module = pv4ger
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "dataset"
        self.source.mkdir()
        self.partition = {
            "train": ["6.000000,51.000000"],
            "valid": ["6.000500,51.000000"],
            "test": ["6.100000,51.000000"],
        }
        self.samples = {
            identifier: self.hdf5(index)
            for index, identifier in enumerate(
                identifier for members in self.partition.values() for identifier in members
            )
        }
        self.write_source()
        self.refresh_expected_files()
        self.count_patch = patch.object(self.module, "EXPECTED_SPLIT_COUNTS", {"train": 1, "valid": 1, "test": 1})
        self.count_patch.start()
        self.addCleanup(self.count_patch.stop)

    @staticmethod
    def hdf5(index: int = 0, *, label_value: int = 1, shape: tuple[int, int] = (320, 320)) -> bytes:
        buffer = io.BytesIO()
        with h5py.File(buffer, "w") as sample:
            for name, value in (("Red", 10 + index), ("Green", 60 + index), ("Blue", 140 + index)):
                sample.create_dataset(name, data=np.full(shape, value, dtype=np.uint8))
            label = np.zeros(shape, dtype=np.uint8)
            label[3, 4] = label_value
            sample.create_dataset("label", data=label)
            # Pixel import must not deserialize the upstream Python metadata.
            sample.attrs["pickle"] = np.void(b"this is deliberately not a Python pickle")
        return buffer.getvalue()

    def write_source(self, *, extra_entries: dict[str, bytes] | None = None) -> None:
        (self.source / "default_partition.json").write_text(json.dumps(self.partition), encoding="utf-8")
        (self.source / "LICENSE").write_text("Synthetic fixture license statement\n", encoding="utf-8")
        (self.source / "README").write_text("Synthetic HDF5 fixture, not a real dataset\n", encoding="utf-8")
        with zipfile.ZipFile(self.source / "data.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for identifier, data in self.samples.items():
                archive.writestr(identifier + ".hdf5", data)
            for name, data in (extra_entries or {}).items():
                archive.writestr(name, data)

    def refresh_expected_files(self) -> None:
        if hasattr(self, "files_patch"):
            self.files_patch.stop()
        files = []
        for name in ("data.zip", "default_partition.json", "LICENSE", "README"):
            path = self.source / name
            files.append(self.module.SourceFile(name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
        self.files_patch = patch.object(self.module, "SOURCE_FILES", tuple(files))
        self.files_patch.start()
        self.addCleanup(self.files_patch.stop)

    def import_dataset(self, **kwargs: object) -> dict[str, object]:
        return self.module.import_pv4ger_seg(self.source, self.output, **kwargs)

    def records(self, output: Path | None = None) -> list[dict[str, object]]:
        return [json.loads(line) for line in ((output or self.output) / "manifest.jsonl").read_text().splitlines()]

    def assert_import_failure(self, **kwargs: object) -> None:
        with self.assertRaises(self.module.DatasetImportError):
            self.import_dataset(**kwargs)
        self.assertFalse(self.output.exists(), "Failed imports must not publish a partial dataset")
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), ["source"])

    def test_real_conversion_keeps_rgb_order_binary_masks_and_truthful_provenance(self) -> None:
        report = self.import_dataset()
        self.assertIsInstance(report, dict)
        json.dumps(report, allow_nan=False)
        validation = validate_dataset(self.output)
        self.assertTrue(validation.ok, validation.to_dict())
        records = self.records()
        self.assertEqual(len(records), 3)
        self.assertEqual(validation.stats["pv_pixels"], 3)
        self.assertEqual(validation.stats["ignored_pixels"], 0)
        colors = set()
        for record in records:
            with Image.open(self.output / str(record["image"])) as image:
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(image.size, (320, 320))
                colors.add(image.getpixel((0, 0)))
            with Image.open(self.output / str(record["mask"])) as mask:
                self.assertEqual(mask.mode, "L")
                self.assertEqual(mask.getpixel((4, 3)), 1)
                self.assertEqual(mask.getpixel((3, 4)), 0)
                self.assertEqual(set(mask.get_flattened_data()), {0, 1})
            self.assertEqual(record["gsd_m"], 0.1)
            self.assertEqual(record["split"], "unassigned")
            self.assertEqual(record["annotation_status"], "upstream")
            self.assertIsNone(record["capture_date"])
            self.assertIsNone(record["georeference"])
            self.assertEqual(record["building_ids"], [])
        self.assertEqual(colors, {(10, 60, 140), (11, 61, 141), (12, 62, 142)})
        upstream = json.loads((self.output / "provenance" / "source_splits.json").read_text())
        expected = {identifier: split for split, members in self.partition.items() for identifier in members}
        self.assertEqual(upstream, expected)
        self.assertEqual((self.output / "provenance" / "upstream-LICENSE").read_bytes(), (self.source / "LICENSE").read_bytes())
        original_license = (self.output / "provenance" / "original-MIT-LICENSE").read_text()
        self.assertIn("Copyright (c) 2020 Kevin", original_license)
        self.assertIn("Permission is hereby granted, free of charge", original_license)

    def test_nearby_upstream_splits_share_a_group_and_limit_does_not_change_grouping(self) -> None:
        self.import_dataset()
        full = self.records()
        by_color = {}
        for record in full:
            with Image.open(self.output / str(record["image"])) as image:
                by_color[image.getpixel((0, 0))] = record
        self.assertEqual(by_color[(10, 60, 140)]["spatial_group"], by_color[(11, 61, 141)]["spatial_group"])
        self.assertNotEqual(by_color[(10, 60, 140)]["spatial_group"], by_color[(12, 62, 142)]["spatial_group"])
        limited_output = self.root / "limited"
        self.module.import_pv4ger_seg(self.source, limited_output, limit=1)
        limited = self.records(limited_output)
        self.assertEqual(len(limited), 1)
        matching_full = next(record for record in full if record["id"] == limited[0]["id"])
        self.assertEqual(limited[0]["spatial_group"], matching_full["spatial_group"])

    def test_proximity_groups_include_transitive_neighbours(self) -> None:
        self.partition = {
            "train": ["6.000000,51.000000"],
            "valid": ["6.001200,51.000000"],
            "test": ["6.002400,51.000000"],
        }
        self.samples = {
            identifier: self.hdf5(index)
            for index, identifier in enumerate(
                identifier for members in self.partition.values() for identifier in members
            )
        }
        self.write_source()
        self.refresh_expected_files()
        self.import_dataset()
        # Endpoints are ~168 m apart, connected through the middle sample.
        self.assertEqual(len({record["spatial_group"] for record in self.records()}), 1)

    def test_checksum_mismatch_is_rejected_before_hdf5_reading(self) -> None:
        with (self.source / "data.zip").open("ab") as archive:
            archive.write(b"tampered")
        self.assert_import_failure()

    def test_missing_source_never_downloads_without_explicit_flag(self) -> None:
        (self.source / "data.zip").unlink()
        with patch.object(self.module, "urlopen", side_effect=AssertionError("Network access was not requested")):
            self.assert_import_failure()

    def test_cli_download_fetches_verified_files_and_imports_real_pixels(self) -> None:
        from pvextent.cli import main

        payloads = {file.name: (self.source / file.name).read_bytes() for file in self.module.SOURCE_FILES}
        expected_urls = {
            self.module.DOWNLOAD_URL.format(name=name): payload
            for name, payload in payloads.items()
        }
        for name in payloads:
            (self.source / name).unlink()
        self.source.rmdir()

        def response(request: object, *, timeout: int) -> io.BytesIO:
            self.assertGreater(timeout, 0)
            self.assertIn(request.full_url, expected_urls)
            return io.BytesIO(expected_urls[request.full_url])

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(self.module, "urlopen", side_effect=response) as network:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = main([
                    "data", "import-pv4ger-seg", str(self.source), str(self.output),
                    "--download", "--json",
                ])
        self.assertEqual(status, 0, stderr.getvalue())
        self.assertEqual(network.call_count, len(payloads))
        self.assertEqual(
            {call.args[0].full_url for call in network.call_args_list}, set(expected_urls)
        )
        report = json.loads(stdout.getvalue())
        self.assertTrue(report["ok"])
        self.assertEqual(report["records"], 3)
        self.assertIn("Downloading", stderr.getvalue())
        self.assertEqual({path.name for path in self.source.iterdir()}, set(payloads))
        for name, payload in payloads.items():
            self.assertEqual((self.source / name).read_bytes(), payload)
        validation = validate_dataset(self.output)
        self.assertTrue(validation.ok, validation.to_dict())
        self.assertEqual(validation.stats["pv_pixels"], 3)

    def test_download_checksum_failure_removes_partial_cache_file(self) -> None:
        archive = self.source / "data.zip"
        payload = bytearray(archive.read_bytes())
        payload[0] ^= 1
        archive.unlink()
        with patch.object(self.module, "urlopen", return_value=io.BytesIO(payload)) as network:
            self.assert_import_failure(download=True)
        network.assert_called_once()
        self.assertFalse(archive.exists())
        self.assertFalse(list(self.source.glob(".download-*")))
        self.assertEqual(
            {path.name for path in self.source.iterdir()}, {"default_partition.json", "LICENSE", "README"}
        )

    def test_download_reuses_existing_valid_cache_without_writing_or_network(self) -> None:
        before = {
            path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in self.source.iterdir()
        }
        with patch.object(self.module, "urlopen", side_effect=AssertionError("Verified cache must be reused")) as network:
            self.import_dataset(download=True)
        network.assert_not_called()
        after = {
            path.name: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in self.source.iterdir()
        }
        self.assertEqual(after, before)
        self.assertTrue(validate_dataset(self.output).ok)

    def test_download_never_replaces_an_existing_invalid_cache_file(self) -> None:
        archive = self.source / "data.zip"
        payload = bytearray(archive.read_bytes())
        payload[0] ^= 1
        archive.write_bytes(payload)
        with patch.object(self.module, "urlopen", side_effect=AssertionError("Existing files must not be replaced")) as network:
            self.assert_import_failure(download=True)
        network.assert_not_called()
        self.assertEqual(archive.read_bytes(), payload)
        self.assertFalse(list(self.source.glob(".download-*")))

    def test_atomic_publish_refuses_to_replace_an_existing_empty_directory(self) -> None:
        staging = self.root / "staging"
        staging.mkdir()
        marker = staging / "keep.txt"
        marker.write_text("unpublished data", encoding="utf-8")
        self.output.mkdir()
        original_identity = (self.output.stat().st_dev, self.output.stat().st_ino)
        with self.assertRaises(OSError):
            self.module._publish(staging, self.output)
        self.assertEqual((self.output.stat().st_dev, self.output.stat().st_ino), original_identity)
        self.assertEqual(list(self.output.iterdir()), [])
        self.assertEqual(marker.read_text(), "unpublished data")

    def test_255_is_rejected_instead_of_becoming_foreground_or_ignore(self) -> None:
        identifier = self.partition["train"][0]
        self.samples[identifier] = self.hdf5(label_value=255)
        self.write_source()
        self.refresh_expected_files()
        self.assert_import_failure()

    def test_wrong_dimensions_are_rejected_without_publishing_partial_data(self) -> None:
        identifier = self.partition["test"][0]
        self.samples[identifier] = self.hdf5(shape=(319, 320))
        self.write_source()
        self.refresh_expected_files()
        self.assert_import_failure()

    def test_wrong_pixel_dtype_is_rejected(self) -> None:
        identifier = self.partition["train"][0]
        buffer = io.BytesIO(self.samples[identifier])
        with h5py.File(buffer, "r+") as sample:
            del sample["Red"]
            sample.create_dataset("Red", data=np.full((320, 320), 10, dtype=np.uint16))
        self.samples[identifier] = buffer.getvalue()
        self.write_source()
        self.refresh_expected_files()
        self.assert_import_failure()

    def test_missing_archive_sample_is_rejected(self) -> None:
        del self.samples[self.partition["test"][0]]
        self.write_source()
        self.refresh_expected_files()
        self.assert_import_failure()

    def test_duplicate_partition_identifier_is_rejected(self) -> None:
        self.partition["valid"] = list(self.partition["train"])
        self.write_source()
        self.refresh_expected_files()
        self.assert_import_failure()

    def test_archive_paths_cannot_escape_output_directory(self) -> None:
        self.write_source(extra_entries={"../escaped.hdf5": self.hdf5()})
        self.refresh_expected_files()
        self.assert_import_failure()
        self.assertFalse((self.root / "escaped.hdf5").exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep existing work", encoding="utf-8")
        with self.assertRaises(self.module.DatasetImportError):
            self.import_dataset()
        self.assertEqual(marker.read_text(), "keep existing work")
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["keep.txt"])

    def test_invalid_limit_is_rejected(self) -> None:
        for value in (0, -1, True, 0.5, "1"):
            with self.subTest(limit=value):
                self.assert_import_failure(limit=value)


if __name__ == "__main__":
    unittest.main()
