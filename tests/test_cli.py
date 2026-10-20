# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed command-line interface as a separate process."""

from __future__ import annotations

import hashlib
import importlib.util
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

    def test_training_missing_dataset_exits_one_with_json_error(self) -> None:
        output = self.root / "train-output"
        result = self.command(
            "train", "smoke", str(self.root / "missing-dataset"), str(output),
            "--device", "cpu", "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertTrue(document["error"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_training_nonpositive_steps_are_a_clean_error(self) -> None:
        output = self.root / "train-output"
        result = self.command(
            "train", "smoke", str(self.root / "missing-dataset"), str(output),
            "--steps", "0", "--device", "cpu", "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertTrue(document["error"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_training_invalid_device_is_a_usage_error(self) -> None:
        result = self.command(
            "train", "smoke", str(self.root / "missing-dataset"), str(self.root / "train-output"),
            "--device", "invented", "--json",
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("--device", result.stderr)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Install the train extra for CLI optimization")
    def test_training_command_performs_cpu_updates_and_emits_a_json_report(self) -> None:
        dataset = self.root / "training-dataset"
        dataset.mkdir()
        (dataset / "dataset.toml").write_bytes((self.root / "dataset.toml").read_bytes())
        image_path, mask_path = dataset / "image.png", dataset / "mask.png"
        Image.new("RGB", (32, 32), (20, 60, 140)).save(image_path)
        label = Image.new("L", (32, 32), 0)
        label.putpixel((10, 10), 1)
        label.save(mask_path)
        record = json.loads((self.root / "manifest.jsonl").read_text())
        record.update(
            split="train", width=32, height=32,
            image_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
            mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
        )
        (dataset / "manifest.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
        output = self.root / "training-run"
        result = self.command(
            "train", "smoke", str(dataset), str(output), "--steps", "2", "--samples", "1",
            "--batch-size", "1", "--seed", "7", "--lr", "0.001", "--device", "cpu", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        document = json.loads(result.stdout)
        self.assertTrue(document["ok"])
        self.assertEqual(document["device"], "cpu")
        self.assertGreater(document["parameter_update_l2"], 0)
        self.assertTrue((output / "checkpoint.pt").is_file())

    def test_baseline_missing_dataset_returns_json_without_loading_encoder_weights(self) -> None:
        config = Path(__file__).resolve().parents[1] / "configs" / "baseline-resnet18.toml"
        output = self.root / "baseline-run"
        cache = self.root / "encoder-cache"
        result = self.command(
            "train", "baseline", str(self.root / "missing-dataset"), str(output),
            "--config", str(config), "--weights-dir", str(cache), "--device", "cpu", "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertTrue(document["error"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())
        self.assertFalse(cache.exists())

    def test_baseline_missing_config_is_a_clean_json_error(self) -> None:
        output = self.root / "baseline-run"
        result = self.command(
            "train", "baseline", str(self.root), str(output),
            "--config", str(self.root / "missing.toml"),
            "--weights-dir", str(self.root / "encoder-cache"), "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        document = json.loads(result.stdout)
        self.assertFalse(document["ok"])
        self.assertIn("configuration", document["error"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_baseline_requires_configuration_and_weight_cache_options(self) -> None:
        result = self.command("train", "baseline", str(self.root), str(self.root / "baseline-run"), "--json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("--config", result.stderr)
        self.assertIn("--weights-dir", result.stderr)

    def test_validation_comparison_missing_reports_returns_clean_json(self) -> None:
        output = self.root / "comparison"
        result = self.command(
            "evaluate", "compare-validations", str(self.root / "reference"),
            str(self.root / "candidate"), str(output), "--json",
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertFalse(json.loads(result.stdout)["ok"])
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(output.exists())

    def test_negative_holdout_missing_inputs_fail_without_publishing(self) -> None:
        output = self.root / "negative-holdout"
        commands = [
            ("data", "fetch-training-candidates", str(output), "--exclude", str(self.root / "missing"),
             "--protocol", str(self.root / "missing.json"), "--json"),
            ("data", "fetch-negative-holdout", str(output), "--exclude", str(self.root / "missing"),
             "--protocol", str(self.root / "missing.json"), "--json"),
            ("data", "freeze-negative-holdout", str(self.root / "curated"), str(self.root / "candidates"),
             str(output), "--exclude", str(self.root / "missing"), "--protocol", str(self.root / "missing.json"),
             "--acquisition-protocol", str(self.root / "missing-acquisition.json"), "--json"),
            ("evaluate", "negative-holdout", str(self.root / "missing"), str(self.root / "ref.pt"),
             str(self.root / "candidate.pt"), str(output), "--device", "cpu", "--json"),
        ]
        for command in commands:
            with self.subTest(action=command[1]):
                result = self.command(*command)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(json.loads(result.stdout)["ok"])
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())

    def test_bad_command_line_exits_two(self) -> None:
        result = self.command("data", "validate")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr)

    def test_positive_holdout_missing_inputs_fail_without_publishing(self) -> None:
        output = self.root / "positive-holdout"
        commands = [
            ("data", "freeze-positive-holdout", str(self.root / "candidates"),
             str(self.root / "negative"), str(output), "--exclude", str(self.root / "missing"),
             "--review", str(self.root / "a.json"), "--review", str(self.root / "b.json"),
             "--protocol", str(self.root / "missing.json"), "--json"),
            ("evaluate", "positive-holdout", str(self.root / "missing"), str(self.root / "ref.pt"),
             str(self.root / "candidate.pt"), str(output), "--device", "cpu", "--json"),
        ]
        for command in commands:
            with self.subTest(command=command[:2]):
                result = self.command(*command)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(json.loads(result.stdout)["ok"])
                self.assertNotIn("Traceback", result.stderr)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
