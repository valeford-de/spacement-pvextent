# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Frozen-checkpoint validation analysis with real synthetic optimizer state."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

# Import the fixture module, not its TestCase class into this module's globals:
# unittest discovery would otherwise run the imported baseline cases twice.
import test_baseline_training as baseline_fixtures

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(baseline_fixtures.BASELINE_AVAILABLE, "Install the baseline extra for validation analysis tests")
class ValidationAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.evaluation import validation

        self.module = validation
        self.fixture = baseline_fixtures.BaselineTrainingTests()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.setUp()
        self.fixture.write_config(epochs=1)
        self.fixture.train()
        self.fixture.prepare.reset_mock()
        self.fixture.factory.reset_mock()
        self.output = self.fixture.root / "analysis"
        self.checkpoint_path = self.fixture.output / "best.pt"

    def analyze(self, **kwargs):
        return self.module.analyze_validation(
            self.fixture.dataset, self.checkpoint_path, self.output, device="cpu", top_k=1, **kwargs
        )

    def assert_failure(self) -> None:
        source = self.fixture.snapshot(self.fixture.dataset)
        checkpoint = self.fixture.snapshot(self.fixture.output)
        entries = {path.name for path in self.fixture.root.iterdir()}
        with self.assertRaises(self.module.ValidationAnalysisError):
            self.analyze()
        self.assertFalse(self.output.exists())
        self.assertEqual(self.fixture.snapshot(self.fixture.dataset), source)
        self.assertEqual(self.fixture.snapshot(self.fixture.output), checkpoint)
        self.assertEqual({path.name for path in self.fixture.root.iterdir()}, entries)

    def test_analysis_uses_only_val_and_preserves_checkpoint_model_state_and_sources(self) -> None:
        source = self.fixture.snapshot(self.fixture.dataset)
        checkpoint = self.fixture.snapshot(self.fixture.output)
        models = []

        class ObservedModel(baseline_fixtures.TinyBatchNormModel):
            def __init__(self):
                super().__init__()
                self.observations = []
                self.before = None

            def forward(self, image):
                if self.before is None:
                    self.before = {name: value.detach().clone() for name, value in self.state_dict().items()}
                self.observations.append((self.training, torch.is_grad_enabled()))
                return super().forward(image)

        def factory(*args, **kwargs):
            model = ObservedModel()
            models.append(model)
            return model

        self.fixture.factory.side_effect = factory
        forbidden = {self.fixture.dataset / record[role]
                     for record in self.fixture.records if record["split"] != "val"
                     for role in ("image", "mask")}
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError(f"Non-validation pixel access: {path}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(torch, "load", wraps=torch.load) as loads:
            report = self.analyze()
        self.assertTrue(report["ok"])
        self.assertEqual(report["samples"], 2)
        self.assertFalse(report["test_used"])
        self.assertEqual(report["threshold"], 0.5)
        self.assertTrue(report["same_device_and_libraries"])
        self.assertTrue(report["matches_checkpoint_validation_counts"])
        summary = report["summary"]
        self.assertEqual((summary["tp"], summary["fp"], summary["fn"], summary["tn"]), (110, 1090, 0, 0))
        self.assertEqual(summary["valid_pixels"], 1200)
        self.assertAlmostEqual(summary["iou"], 110 / 1200)
        self.assertAlmostEqual(summary["macro_iou"], (0.5 + 0.01) / 2)
        self.assertEqual(summary["error_pixels"], summary["boundary_error_pixels"]
                         + summary["interior_error_pixels"] + summary["unclassified_error_pixels"])
        self.assertEqual(report["rankings"]["lowest-iou"], ["fixture-5"])
        self.assertEqual(report["rankings"]["most-fp-pixels"], ["fixture-5"])
        self.assertEqual(self.fixture.snapshot(self.fixture.dataset), source)
        self.assertEqual(self.fixture.snapshot(self.fixture.output), checkpoint)
        self.fixture.prepare.assert_not_called()
        self.assertEqual(len(models), 1)
        for model in models:
            self.assertTrue(model.observations)
            self.assertTrue(all(not training and not gradient for training, gradient in model.observations))
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, model.before[name]), name)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        for call in loads.call_args_list:
            self.assertIs(call.kwargs["weights_only"], True)
        rows = json.loads((self.output / "samples.json").read_text())
        self.assertEqual({row["id"] for row in rows}, {"fixture-4", "fixture-5"})
        self.assertTrue(all(row["split"] == "val" for row in rows))
        self.assertEqual(report["samples_sha256"], hashlib.sha256((self.output / "samples.json").read_bytes()).hexdigest())
        self.assertTrue((self.output / "index.html").is_file())
        ignored_panel = self.output / next(row["views"]["errors"] for row in rows if row["id"] == "fixture-4")
        with Image.open(ignored_panel) as image:
            self.assertEqual(image.getpixel((319, 319)), (245, 170, 50))
        json.dumps(report, allow_nan=False)

    def test_bad_checkpoint_hash_fails_before_torch_load(self) -> None:
        payload = bytearray(self.checkpoint_path.read_bytes())
        payload[-1] ^= 1
        self.checkpoint_path.write_bytes(payload)
        with patch.object(torch, "load", side_effect=AssertionError("Unverified checkpoint must not load")) as loads:
            self.assert_failure()
        loads.assert_not_called()

    def test_changed_source_fingerprint_is_rejected_before_model_construction(self) -> None:
        self.fixture.records[0]["capture_date"] = "2025-06-21"
        self.fixture.write_manifest()
        self.assert_failure()
        self.fixture.factory.assert_not_called()

    def test_existing_analysis_output_is_never_replaced(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing review work", encoding="utf-8")
        with self.assertRaises(self.module.ValidationAnalysisError):
            self.analyze()
        self.assertEqual(marker.read_text(), "existing review work")
        self.assertEqual({path.name for path in self.output.iterdir()}, {"keep.txt"})
        self.fixture.factory.assert_not_called()

    def test_gallery_failure_cleans_predictions_and_staging_without_changing_inputs(self) -> None:
        with patch.object(self.module, "_views", side_effect=OSError("synthetic PNG write failure")):
            self.assert_failure()

    def test_recorded_validation_count_mismatch_rejects_partial_analysis(self) -> None:
        document = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        document["validation"]["tp"] += 1
        torch.save(document, self.checkpoint_path)
        self.checkpoint_path.with_suffix(".pt.sha256").write_text(
            hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest() + "\n", encoding="ascii"
        )
        self.assert_failure()

    def test_full_resume_checkpoint_cannot_replace_selected_best_weights_bundle(self) -> None:
        self.checkpoint_path = self.fixture.output / "last.pt"
        self.assert_failure()
        self.fixture.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
