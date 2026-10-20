# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed command-line interface as a separate process."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image


class ValidationCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "cli-fixture"\n'
            'dataset_version = "1"\n\n[sources.demo]\n'
            'name = "Synthetic fixture"\nurl = "https://example.invalid/fixture"\n'
            'version = "1"\nimage_license = "CC0-1.0"\n'
            'annotation_license = "CC0-1.0"\nattribution = "Synthetic fixture"\n',
            encoding="utf-8",
        )
        image_path, mask_path = self.root / "image.png", self.root / "mask.png"
        Image.new("RGB", (8, 8), (20, 60, 140)).save(image_path)
        Image.new("L", (8, 8), 0).save(mask_path)
        record = {
            "id": "example", "image": "image.png", "mask": "mask.png",
            "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "width": 8, "height": 8, "gsd_m": 0.1, "source_id": "demo",
            "capture_date": None, "georeference": None,
            "spatial_group": "synthetic-area", "building_ids": [],
            "split": "unassigned", "annotation_status": "reviewed",
        }
        (self.root / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    def command(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(project_root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
        return subprocess.run(
            [sys.executable, "-m", "pvextent", *arguments],
            text=True, capture_output=True, check=False, timeout=30,
            cwd=project_root, env=environment,
        )

    def test_valid_dataset_with_warnings_exits_zero_and_emits_json(self) -> None:
        result = self.command("data", "validate", str(self.root), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertTrue(document["ok"])
        self.assertFalse(document["errors"])
        self.assertTrue(document["warnings"])

    def test_invalid_dataset_exits_one_with_machine_readable_errors(self) -> None:
        (self.root / "manifest.jsonl").write_text('{"invalid":\n', encoding="utf-8")
        result = self.command("data", "validate", str(self.root), "--json")
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertTrue(document["errors"])

    def test_missing_dataset_path_is_a_validation_error(self) -> None:
        result = self.command("data", "validate", str(self.root / "missing"), "--json")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)["ok"])

    def test_import_failure_exits_one_with_json_error_and_no_traceback(self) -> None:
        output = self.root / "imported"
        result = self.command(
            "data", "import-pv4ger-seg", str(self.root / "missing-cache"), str(output), "--json"
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertIsInstance(document["error"], str)
        self.assertTrue(document["error"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_import_nonpositive_limit_exits_one_with_explanation(self) -> None:
        output = self.root / "imported"
        for limit in ("0", "-1"):
            with self.subTest(limit=limit):
                result = self.command(
                    "data", "import-pv4ger-seg", str(self.root / "missing-cache"),
                    str(output), "--limit", limit, "--json",
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                document = json.loads(result.stdout)
                self.assertFalse(document["ok"])
                self.assertIn("positive integer", document["error"])
                self.assertFalse(output.exists())

    def test_import_noninteger_limit_is_a_usage_error(self) -> None:
        result = self.command(
            "data", "import-pv4ger-seg", str(self.root / "missing-cache"),
            str(self.root / "imported"), "--limit", "many", "--json",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("--limit", result.stderr)

    def test_new_data_actions_report_missing_datasets_as_json_errors(self) -> None:
        for action in ("split", "review", "fetch-nrw-candidates"):
            with self.subTest(action=action):
                output = self.root / action
                result = self.command(
                    "data", action, str(self.root / "missing-dataset"), str(output), "--json"
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                document = json.loads(result.stdout)
                self.assertFalse(document["ok"])
                self.assertTrue(document["error"])
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_candidate_limit_above_bounded_pool_is_rejected_before_fetch(self) -> None:
        output = self.root / "candidates"
        result = self.command(
            "data", "fetch-nrw-candidates", str(self.root / "missing-dataset"),
            str(output), "--limit", "25", "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertIn("between 1 and 24", document["error"])
        self.assertFalse(output.exists())

    def test_bad_command_line_exits_two(self) -> None:
        result = self.command("data", "validate")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr)


if __name__ == "__main__":
    unittest.main()
