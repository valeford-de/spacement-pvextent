# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic frozen negatives: only approved background can become error counts."""

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


BASELINE_AVAILABLE = torch is not None and importlib.util.find_spec("segmentation_models_pytorch") is not None


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for negative evaluation")
class NegativeEvaluationTests(unittest.TestCase):
    def setUp(self):
        from pvextent.evaluation import negative
        from pvextent.models.resnet_unet import MODEL_NAME, WEIGHT_SPEC

        self.module = negative
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.output = self.root / "holdout", self.root / "evaluation"
        (self.dataset / "provenance").mkdir(parents=True)
        (self.dataset / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "pvextent-negative-holdout"\n'
            'dataset_version = "test-v1"\n[sources.synthetic]\n'
            'name = "Synthetic"\nurl = "https://example.invalid/fixture"\nversion = "1"\n'
            'image_license = "CC0-1.0"\nannotation_license = "CC0-1.0"\nattribution = "Synthetic"\n'
        )
        self.records = []
        for index in range(3):
            image_path, mask_path = self.dataset / f"image-{index}.png", self.dataset / f"mask-{index}.png"
            Image.new("RGB", (320, 320), (30 + index, 50, 90)).save(image_path)
            mask = Image.new("L", (320, 320), 255)
            mask.paste(0, (0, 0, 160, 320))
            mask.save(mask_path)
            self.records.append({
                "id": f"sample-{index}", "image": image_path.name, "mask": mask_path.name,
                "image_sha256": self.sha(image_path), "mask_sha256": self.sha(mask_path),
                "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "synthetic",
                "capture_date": "2025-01-01", "georeference": None, "spatial_group": f"region-{index}",
                "building_ids": [], "split": "test" if index < 2 else "unassigned",
                "annotation_status": "reviewed" if index < 2 else "unreviewed",
            })
        # These source rasters are intentionally absent: metadata parsing alone
        # must never open an excluded unassigned crop.
        for role in ("image", "mask"):
            (self.dataset / self.records[2][role]).unlink()
        self.checkpoints, self.documents = {}, {}
        self.config = {"model": MODEL_NAME, "encoder": WEIGHT_SPEC,
                       "preprocessing": "native-float32-rgb-imagenet-mean-std-v1",
                       "threshold": 0.5, "seed": 20261020, "device": "cpu", "package_versions": {"fixture": "1"}}
        for role, cutoff in (("reference", 160.0), ("candidate", 80.0)):
            path = self.root / role / "best.pt"
            path.parent.mkdir()
            path.write_bytes(role.encode())
            self.checkpoints[role] = path
            self.documents[role] = {"config": dict(self.config), "epoch": 3,
                                    "model_state_dict": {"cutoff": torch.tensor(cutoff)}}
        self.protocol = {"purpose": "experimental-ai-negative-holdout", "schema_version": "1.0",
                         "frozen_before_predictions": True, "model_predictions_used_for_labels": False,
                         "human_review_claimed": False, "minimum_background_fraction": 0.5, "threshold": 0.5,
                         "checkpoint_sha256": {role: self.sha(path) for role, path in self.checkpoints.items()}}
        self.acquisition = {"schema_version": "1.0", "regions": [{"id": "fixture"}]}
        self.freeze = {"ok": True, "purpose": "experimental-ai-negative-holdout",
                       "frozen_before_predictions": True, "model_predictions_used_for_labels": False,
                       "human_review_claimed": False, "minimum_background_fraction": 0.5,
                       "reviewed_evaluation_ids": ["sample-0", "sample-1"], "training_use_allowed": False,
                       "checkpoint_sha256": dict(self.protocol["checkpoint_sha256"])}
        self.write_evidence()
        self.models = []
        owner = self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cutoff = torch.nn.Parameter(torch.tensor(160.0))
                self.observations = []
                self.before = None

            def forward(self, images):
                if self.before is None:
                    self.before = self.cutoff.detach().clone()
                self.observations.append((self.training, torch.is_grad_enabled()))
                columns = torch.arange(320, device=images.device).remainder(160)
                return (columns.lt(self.cutoff).to(torch.float32).mul(2).sub(1)
                        .reshape(1, 1, 1, 320).expand(len(images), 1, 320, 320))

        def factory(weights):
            self.assertIsNone(weights)
            model = Model()
            owner.models.append(model)
            return model

        def checkpoint(path, torch_module, *, resume):
            self.assertFalse(resume)
            role = Path(path).parent.name
            return self.documents[role], self.sha(path)

        self.factory = factory
        for patcher in (
            patch("pvextent.models.resnet_unet.build_resnet_unet", side_effect=factory),
            patch("pvextent.models.resnet_unet.prepare_encoder_weights", side_effect=AssertionError("No weights download")),
            patch("pvextent.training.baseline._read_checkpoint", side_effect=checkpoint),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    @staticmethod
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def write_evidence(self):
        (self.dataset / "manifest.jsonl").write_text(''.join(json.dumps(row) + '\n' for row in self.records))
        provenance = self.dataset / "provenance"
        acquisition = provenance / "acquisition-protocol.json"
        acquisition.write_text(json.dumps(self.acquisition))
        self.protocol["acquisition_protocol_sha256"] = self.sha(acquisition)
        protocol = provenance / "protocol.json"
        protocol.write_text(json.dumps(self.protocol))
        metadata, manifest = ((self.dataset / name).read_bytes() for name in ("dataset.toml", "manifest.jsonl"))
        self.freeze.update(dataset_fingerprint=hashlib.sha256(metadata + b'\0' + manifest).hexdigest(),
                           protocol_sha256=self.sha(protocol), acquisition_protocol_sha256=self.sha(acquisition))
        (provenance / "negative-holdout-freeze.json").write_text(json.dumps(self.freeze))

    def evaluate(self):
        return self.module.evaluate_negative_holdout(
            self.dataset, self.checkpoints["reference"], self.checkpoints["candidate"], self.output, device="cpu"
        )

    def assert_failure(self):
        before = {p.relative_to(self.dataset): p.read_bytes() for p in self.dataset.rglob('*') if p.is_file()}
        with self.assertRaises(self.module.NegativeEvaluationError):
            self.evaluate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.evaluation-negative-evaluation-*')))
        self.assertEqual(before, {p.relative_to(self.dataset): p.read_bytes() for p in self.dataset.rglob('*') if p.is_file()})

    def test_only_supervised_background_is_scored_and_models_do_not_train(self):
        original = Path.open
        forbidden = {self.dataset / self.records[2][role] for role in ("image", "mask")}

        def guard(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError("Excluded crop pixel access")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guard):
            report = self.evaluate()
        self.assertTrue(report["ok"])
        self.assertEqual(report["samples"], 2)
        a, b = (report["results"][role]["totals"] for role in ("reference", "candidate"))
        self.assertEqual(a["supervised_background_pixels"], 102400)
        self.assertEqual(a["false_positive_pixels"], 102400)
        self.assertEqual(a["true_negative_pixels"], 0)
        self.assertEqual(a["predicted_pv_pixels_in_ignored_region_unscored"], 102400)
        self.assertEqual(b["false_positive_pixels"], 51200)
        self.assertEqual(b["true_negative_pixels"], 51200)
        self.assertEqual(b["pixel_false_positive_rate"], 0.5)
        self.assertEqual(report["delta"]["pixel_false_positive_rate_percentage_points"], -50.0)
        self.assertEqual(report["supervised_background_coverage"], 0.5)
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["old_test_pixels_opened"])
        for model in self.models:
            self.assertEqual(model.observations, [(False, False)])
            self.assertTrue(torch.equal(model.cutoff, model.before))
            self.assertIsNone(model.cutoff.grad)
        rows = json.loads((self.output / 'samples.json').read_text())
        self.assertEqual(report["samples_sha256"], self.sha(self.output / 'samples.json'))
        for row in rows:
            for role in ("reference", "candidate"):
                with Image.open(self.output / row[role]["prediction"]) as image:
                    self.assertEqual(image.size, (320, 320))
                    self.assertEqual(image.mode, "L")
                    self.assertLessEqual(set(image.get_flattened_data()), {0, 1})
        self.assertNotIn('"recall"', json.dumps(report))
        self.assertNotIn('"iou"', json.dumps(report))

    def test_old_dataset_identity_rejected_before_any_raster_or_checkpoint(self):
        path = self.dataset / 'dataset.toml'
        path.write_text(path.read_text().replace('pvextent-negative-holdout', 'pv4ger-seg'))
        self.write_evidence()
        with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")), \
             patch('pvextent.training.baseline._read_checkpoint', side_effect=AssertionError("No checkpoint access")):
            self.assert_failure()

    def test_freeze_flags_ids_and_fingerprint_must_match_before_pixels(self):
        for key, value in (("frozen_before_predictions", False), ("dataset_fingerprint", "a" * 64),
                           ("reviewed_evaluation_ids", ["sample-0"]), ("human_review_claimed", True)):
            with self.subTest(key=key):
                self.write_evidence()
                freeze_path = self.dataset / 'provenance/negative-holdout-freeze.json'
                data = json.loads(freeze_path.read_text())
                data[key] = value
                freeze_path.write_text(json.dumps(data))
                with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
                    self.assert_failure()

    def test_changed_protocol_bytes_are_rejected_before_pixels(self):
        path = self.dataset / 'provenance/protocol.json'
        path.write_text(path.read_text() + ' ')
        with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
            self.assert_failure()

    def test_pv_labels_and_insufficient_coverage_are_rejected_before_models(self):
        mask_path = self.dataset / self.records[0]['mask']
        for label in (1, 255):
            with self.subTest(label=label):
                mask = Image.new('L', (320, 320), 255)
                mask.paste(0, (0, 0, 160, 320))
                mask.putpixel((0, 0), label)
                mask.save(mask_path)
                self.records[0]['mask_sha256'] = self.sha(mask_path)
                self.write_evidence()
                self.assert_failure()
                self.assertFalse(self.models)

    def test_explicit_five_percent_gate_accepts_exactly_5120_pixels_without_scoring_ignore(self):
        self.freeze['minimum_background_fraction'] = 0.05
        self.protocol['minimum_background_fraction'] = 0.05
        for row in self.records[:2]:
            path = self.dataset / row['mask']
            mask = Image.new('L', (320, 320), 255)
            mask.paste(0, (0, 0, 16, 320))
            mask.save(path)
            row['mask_sha256'] = self.sha(path)
        self.write_evidence()
        report = self.evaluate()
        self.assertEqual(report['minimum_background_fraction'], 0.05)
        self.assertEqual(report['supervised_background_coverage'], 0.05)
        for role in ('reference', 'candidate'):
            totals = report['results'][role]['totals']
            self.assertEqual(totals['supervised_background_pixels'], 10240)
            self.assertEqual(totals['false_positive_pixels'], 10240)
            self.assertEqual(totals['true_negative_pixels'], 0)
            self.assertEqual(totals['ignored_pixels'], 194560)
            self.assertEqual(totals['pixel_false_positive_rate'], 1.0)
        self.assertEqual(report['results']['candidate']['totals']['predicted_pv_pixels_in_ignored_region_unscored'], 92160)

    def test_explicit_five_percent_gate_rejects_5119_pixels_before_models(self):
        self.freeze['minimum_background_fraction'] = 0.05
        self.protocol['minimum_background_fraction'] = 0.05
        row = self.records[0]
        path = self.dataset / row['mask']
        mask = Image.new('L', (320, 320), 255)
        mask.paste(0, (0, 0, 16, 320))
        mask.putpixel((0, 0), 255)
        mask.save(path)
        row['mask_sha256'] = self.sha(path)
        self.write_evidence()
        self.assert_failure()
        self.assertFalse(self.models)

    def test_coverage_gate_requires_exact_freeze_protocol_agreement(self):
        self.freeze['minimum_background_fraction'] = 0.5
        self.protocol['minimum_background_fraction'] = 0.05
        self.write_evidence()
        with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
            self.assert_failure()

    def test_only_registered_coverage_gate_choices_are_accepted(self):
        for value in (0.1, 0.01, True, None):
            with self.subTest(value=value):
                self.freeze['minimum_background_fraction'] = value
                self.protocol['minimum_background_fraction'] = value
                self.write_evidence()
                with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
                    self.assert_failure()

    def test_changed_mask_bytes_are_not_decoded_as_current_evidence(self):
        Image.new('L', (320, 320), 0).save(self.dataset / self.records[0]['mask'])
        self.assert_failure()
        self.assertFalse(self.models)

    def test_frozen_checkpoint_identity_and_model_contract_are_required(self):
        self.freeze['checkpoint_sha256']['candidate'] = '0' * 64
        self.protocol['checkpoint_sha256']['candidate'] = '0' * 64
        self.write_evidence()
        with patch('pvextent.training.baseline._read_checkpoint', side_effect=AssertionError("Unfrozen weights must not load")):
            self.assert_failure()
        self.assertFalse(self.models)

    def test_wrong_preprocessing_is_rejected(self):
        self.documents['candidate']['config']['preprocessing'] = 'resized'
        self.assert_failure()
        self.assertFalse(self.models)

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('keep')
        with self.assertRaises(self.module.NegativeEvaluationError):
            self.evaluate()
        self.assertEqual(marker.read_text(), 'keep')
        self.assertFalse(self.models)

    def test_nonfinite_model_output_does_not_publish_partial_evaluation(self):
        original = self.documents['candidate']['model_state_dict']['cutoff']
        self.documents['candidate']['model_state_dict']['cutoff'] = torch.tensor(float('nan'))
        # The synthetic model must explicitly produce NaN; comparison with a
        # NaN cutoff alone would produce finite negative logits.
        class BadModel(torch.nn.Module):
            def forward(self, image):
                return torch.full((len(image), 1, 320, 320), float('nan'))

            def load_state_dict(self, *args, **kwargs):
                return None

        with patch('pvextent.models.resnet_unet.build_resnet_unet', return_value=BadModel()):
            self.assert_failure()
        self.documents['candidate']['model_state_dict']['cutoff'] = original

    def test_protocol_change_during_inference_prevents_publication(self):
        path = self.dataset / 'provenance/protocol.json'

        def factory(weights):
            model = self.factory(weights)
            path.write_text(path.read_text() + ' ')
            return model

        with patch('pvextent.models.resnet_unet.build_resnet_unet', side_effect=factory):
            with self.assertRaisesRegex(self.module.NegativeEvaluationError, 'evidence changed'):
                self.evaluate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.evaluation-negative-evaluation-*')))


if __name__ == '__main__':
    unittest.main()
