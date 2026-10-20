# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unlabelled candidate inference keeps supervision and review status unchanged."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    import torch
except ImportError:
    torch = None


BASELINE_AVAILABLE = torch is not None and all(importlib.util.find_spec(name) is not None for name in (
    "segmentation_models_pytorch", "torchvision", "timm", "safetensors"))


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for candidate inference tests")
class CandidatePredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.inference import candidates
        from pvextent.models import resnet_unet
        self.module, self.models = candidates, resnet_unet
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.source, self.output = self.base / "source", self.base / "predictions"
        self.checkpoint = self.base / "model" / "best.pt"
        self.checkpoint.parent.mkdir()
        for folder in ("images", "masks", "provenance"):
            (self.source / folder).mkdir(parents=True)
        (self.source / "dataset.toml").write_text(
            'schema_version="1.0"\ndataset_id="candidate-fixture"\ndataset_version="1"\n'
            '[sources.synthetic]\nname="Synthetic candidates"\nurl="https://example.invalid/candidates"\n'
            'version="1"\nimage_license="CC0-1.0"\nannotation_license="not-annotated-ignore-placeholder"\n'
            'attribution="Synthetic fixture"\n', encoding="utf-8")
        (self.source / "provenance" / "SOURCES.md").write_text("Synthetic source notice", encoding="utf-8")
        self.records = []
        for index in range(2):
            identifier = f"candidate-{index}"
            image, mask = f"images/{identifier}.png", f"masks/{identifier}.png"
            Image.new("RGB", (32, 32), (255 * index, 60, 140)).save(self.source / image)
            Image.new("L", (32, 32), 255).save(self.source / mask)
            self.records.append({
                "id": identifier, "image": image, "mask": mask,
                "image_sha256": self.digest(self.source / image), "mask_sha256": self.digest(self.source / mask),
                "width": 32, "height": 32, "gsd_m": 0.1, "source_id": "synthetic", "capture_date": None,
                "spatial_group": "candidate-area", "building_ids": [], "split": "unassigned",
                "annotation_status": "unreviewed", "georeference": None,
            })
        self.flush()
        model = self.tiny_model()
        with torch.no_grad():
            model.weight.zero_()
            model.weight[0, 0, 0, 0] = 1
            model.bias.zero_()
        self.document = {
            "baseline_weights_version": "1.0", "epoch": 3,
            "config": {"model": resnet_unet.MODEL_NAME, "encoder": resnet_unet.WEIGHT_SPEC,
                       "seed": 20261020, "threshold": 0.5, "dataset_fingerprint": "0" * 64,
                       "preprocessing": "native-float32-rgb-imagenet-mean-std-v1"},
            "model_state_dict": model.state_dict(), "validation": {"split": "val"},
        }
        torch.save(self.document, self.checkpoint)
        self.checkpoint.with_suffix(".pt.sha256").write_text(self.digest(self.checkpoint) + "\n", encoding="ascii")
        context = patch.object(resnet_unet, "build_resnet_unet", side_effect=lambda *args, **kwargs: self.tiny_model())
        self.factory = context.start()
        self.addCleanup(context.stop)

    @staticmethod
    def tiny_model():
        return torch.nn.Conv2d(3, 1, kernel_size=1)

    @staticmethod
    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def flush(self) -> None:
        (self.source / "manifest.jsonl").write_text("".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")

    def export(self, output: Path | None = None):
        return self.module.export_candidate_predictions(self.source, self.checkpoint, output or self.output, device="cpu")

    def test_real_inference_preserves_source_and_never_converts_predictions_into_labels(self) -> None:
        before = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        checkpoint_before = self.checkpoint.read_bytes()
        report = self.export()
        self.assertEqual(report["records"], 2)
        self.assertFalse(report["evaluation_performed"])
        self.assertFalse(report["domain_transfer_evaluated"])
        self.assertEqual(report["image_dimensions"], [32, 32])
        self.assertEqual(report["source_gsd_m"], 0.1)
        self.assertEqual(set(report["package_versions"]), {"torch", "torchvision", "segmentation-models-pytorch", "timm", "safetensors"})
        self.assertIsNone(report["package_versions_match_checkpoint"])
        self.assertEqual([sample["predicted_pv_pixels"] for sample in report["samples"]], [0, 1024])
        self.assertEqual([sample["predicted_pv_fraction"] for sample in report["samples"]], [0.0, 1.0])
        for key in ("tp", "fp", "fn", "tn", "accuracy", "iou", "dice", "precision", "recall", "metrics"):
            self.assertNotIn(key, report)
        document = json.loads((self.output / "review-manifest.json").read_text())
        self.assertFalse(document["predictions_are_annotations"])
        self.assertTrue(all(sample["decision"] == "pending" for sample in document["samples"]))
        for sample, expected in zip(document["samples"], (0, 1)):
            self.assertEqual(sample["source_record"]["annotation_status"], "unreviewed")
            self.assertEqual(sample["source_record"]["split"], "unassigned")
            with Image.open(self.output / sample["prediction"]) as prediction:
                self.assertEqual(prediction.mode, "L")
                self.assertEqual(set(prediction.get_flattened_data()), {expected})
            with Image.open(self.output / sample["mask"]) as mask:
                self.assertEqual(set(mask.get_flattened_data()), {255})
            self.assertTrue((self.output / sample["overlay"]).is_file())
        after = {path.relative_to(self.source): path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(checkpoint_before, self.checkpoint.read_bytes())
        self.assertEqual((self.output / "provenance" / "SOURCES.md").read_text(), "Synthetic source notice")
        html = (self.output / "index.html").read_text()
        self.assertEqual(html.count("<details>"), 2)
        self.assertNotIn("<details open", html)
        self.assertIn("Human review: <strong>pending</strong>", html)

    def test_test_or_assigned_declarations_are_rejected_before_any_pixel_read(self) -> None:
        self.records[0].update(split="test", annotation_status="upstream")
        self.flush()
        with patch.object(self.module, "validate_dataset", side_effect=AssertionError("Pixels must not be validated/opened")):
            with self.assertRaisesRegex(self.module.CandidatePredictionError, "unassigned"):
                self.export()
        self.factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_supervised_or_all_zero_masks_are_rejected_before_model_construction(self) -> None:
        for value in (0, 1):
            with self.subTest(label=value):
                record = self.records[0]
                Image.new("L", (32, 32), value).save(self.source / record["mask"])
                record["mask_sha256"] = self.digest(self.source / record["mask"])
                self.flush()
                with self.assertRaisesRegex(self.module.CandidatePredictionError, "255 ignore"):
                    self.export()
                self.factory.assert_not_called()

    def test_overlarge_candidate_count_is_rejected_before_pixel_validation(self) -> None:
        template = self.records[0]
        self.records = [{**template, "id": f"candidate-{index}"} for index in range(25)]
        self.flush()
        with patch.object(self.module, "validate_dataset", side_effect=AssertionError("unexpected pixels")):
            with self.assertRaisesRegex(self.module.CandidatePredictionError, "24"):
                self.export()
        self.factory.assert_not_called()

    def test_image_hash_failure_is_not_inferred_or_published(self) -> None:
        (self.source / self.records[0]["image"]).write_bytes(b"changed image bytes")
        with self.assertRaisesRegex(self.module.CandidatePredictionError, "invalid"):
            self.export()
        self.factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_checkpoint_hash_failure_precedes_model_construction(self) -> None:
        self.checkpoint.with_suffix(".pt.sha256").write_text("f" * 64 + "\n", encoding="ascii")
        with self.assertRaisesRegex(self.module.CandidatePredictionError, "SHA-256 mismatch"):
            self.export()
        self.factory.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_existing_or_overlapping_output_is_rejected_without_replacement(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(self.module.CandidatePredictionError):
            self.export()
        self.assertEqual(marker.read_text(), "keep")
        for output in (self.source / "nested", self.checkpoint.parent / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(self.module.CandidatePredictionError, "disjoint"):
                self.export(output)
            self.assertFalse(output.exists())

    def test_failed_publication_cleans_partial_bundle(self) -> None:
        with patch.object(self.module, "_publish", side_effect=OSError("synthetic publish failure")):
            with self.assertRaisesRegex(self.module.CandidatePredictionError, "publish failure"):
                self.export()
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.base.glob(".predictions-candidates-*")), [])

    def test_common_source_gsd_is_preserved_without_assuming_training_gsd(self) -> None:
        for record in self.records:
            record["gsd_m"] = 0.2
        self.flush()
        report = self.export()
        self.assertEqual(report["source_gsd_m"], 0.2)
        self.assertIsNone(report["training_gsd_m"])
        self.assertFalse(report["domain_transfer_evaluated"])


if __name__ == "__main__":
    unittest.main()
