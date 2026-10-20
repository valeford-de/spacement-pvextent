# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic partial positives: PV, background and ignore never exchange roles."""

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


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for positive evaluation")
class PositiveEvaluationTests(unittest.TestCase):
    def setUp(self):
        from pvextent.evaluation import positive
        from pvextent.models.resnet_unet import MODEL_NAME, WEIGHT_SPEC

        self.module = positive
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset, self.output = self.root / "holdout", self.root / "evaluation"
        (self.dataset / "provenance").mkdir(parents=True)
        (self.dataset / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "pvextent-positive-holdout"\n'
            'dataset_version = "test-v1"\n[sources.synthetic]\n'
            'name = "Synthetic"\nurl = "https://example.invalid/fixture"\nversion = "1"\n'
            'image_license = "CC0-1.0"\nannotation_license = "CC0-1.0"\nattribution = "Synthetic"\n'
        )
        self.records = []
        for index in range(13):
            image_path, mask_path = self.dataset / f"image-{index}.png", self.dataset / f"mask-{index}.png"
            Image.new("RGB", (320, 320), (30 + index, 50, 90)).save(image_path)
            mask = Image.new("L", (320, 320), 255)
            mask.paste(1, (0, 0, 10, 320))
            mask.paste(0, (10, 0, 20, 320))
            mask.save(mask_path)
            self.records.append({
                "id": f"sample-{index}", "image": image_path.name, "mask": mask_path.name,
                "image_sha256": self.sha(image_path), "mask_sha256": self.sha(mask_path),
                "width": 320, "height": 320, "gsd_m": 0.1, "source_id": "synthetic",
                "capture_date": "2025-01-01", "georeference": None, "spatial_group": f"region-{index}",
                "building_ids": [], "split": "test" if index < 2 else "unassigned",
                "annotation_status": "reviewed" if index < 2 else "unreviewed",
            })
            if index >= 2:
                image_path.unlink()
                mask_path.unlink()
        self.checkpoints, self.documents = {}, {}
        self.config = {"model": MODEL_NAME, "encoder": WEIGHT_SPEC,
                       "preprocessing": "native-float32-rgb-imagenet-mean-std-v1",
                       "threshold": 0.5, "seed": 20261020, "device": "cpu", "package_versions": {"fixture": "1"}}
        for role, cutoff in (("reference", 5.0), ("candidate", 15.0)):
            path = self.root / role / "best.pt"
            path.parent.mkdir()
            path.write_bytes(role.encode())
            path.with_suffix('.pt.sha256').write_text(self.sha(path) + '\n')
            self.checkpoints[role] = path
            self.documents[role] = {"config": dict(self.config), "epoch": 3,
                                    "model_state_dict": {"cutoff": torch.tensor(cutoff)}}
        self.protocol = {"purpose": "experimental-ai-positive-holdout", "schema_version": "1.0",
                         "frozen_before_predictions": True, "model_predictions_used_for_labels": False,
                         "human_review_claimed": False, "minimum_positive_pixels": 100, "threshold": 0.5,
                         "training_use_allowed": False, "selected_ids": [row["id"] for row in self.records],
                         "selected_image_sha256": {row["id"]: row["image_sha256"] for row in self.records},
                         "checkpoint_sha256": {role: self.sha(path) for role, path in self.checkpoints.items()}}
        self.freeze = {"ok": True, "purpose": "experimental-ai-positive-holdout",
                       "frozen_before_predictions": True, "model_predictions_used_for_labels": False,
                       "human_review_claimed": False, "minimum_positive_pixels": 100,
                       "selected_ids": list(self.protocol["selected_ids"]),
                       "reviewed_evaluation_ids": ["sample-0", "sample-1"], "training_use_allowed": False,
                       "checkpoint_sha256": dict(self.protocol["checkpoint_sha256"])}
        self.write_evidence()
        self.models = []
        owner = self

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cutoff = torch.nn.Parameter(torch.tensor(5.0))
                self.observations, self.before = [], None

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
            return self.documents[Path(path).parent.name], self.sha(path)

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
        protocol = provenance / "protocol.json"
        protocol.write_text(json.dumps(self.protocol))
        metadata, manifest = ((self.dataset / name).read_bytes() for name in ("dataset.toml", "manifest.jsonl"))
        self.freeze.update(dataset_fingerprint=hashlib.sha256(metadata + b'\0' + manifest).hexdigest(),
                           protocol_sha256=self.sha(protocol))
        (provenance / "positive-holdout-freeze.json").write_text(json.dumps(self.freeze))

    def replace_mask(self, row, mask):
        path = self.dataset / row['mask']
        mask.save(path)
        row['mask_sha256'] = self.sha(path)
        self.write_evidence()

    def evaluate(self):
        return self.module.evaluate_positive_holdout(
            self.dataset, self.checkpoints["reference"], self.checkpoints["candidate"], self.output, device="cpu"
        )

    def assert_failure(self):
        before = {p.relative_to(self.dataset): p.read_bytes() for p in self.dataset.rglob('*') if p.is_file()}
        with self.assertRaises(self.module.PositiveEvaluationError):
            self.evaluate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.evaluation-positive-evaluation-*')))
        self.assertEqual(before, {p.relative_to(self.dataset): p.read_bytes() for p in self.dataset.rglob('*') if p.is_file()})

    def test_pv_bg_and_ignore_are_separate_and_models_do_not_train(self):
        original = Path.open
        forbidden = {self.dataset / row[role] for row in self.records[2:] for role in ("image", "mask")}

        def guard(path, *args, **kwargs):
            if path in forbidden:
                raise AssertionError("Excluded crop pixel access")
            return original(path, *args, **kwargs)

        with patch.object(Path, "open", guard):
            report = self.evaluate()
        self.assertTrue(report['ok'])
        self.assertEqual(report['samples'], 2)
        left, right = (report['results'][role]['totals'] for role in ('reference', 'candidate'))
        self.assertEqual((left['true_positive_pixels'], left['false_negative_pixels']), (3200, 3200))
        self.assertEqual((left['false_positive_pixels'], left['true_negative_pixels']), (0, 6400))
        self.assertEqual((right['true_positive_pixels'], right['false_negative_pixels']), (6400, 0))
        self.assertEqual((right['false_positive_pixels'], right['true_negative_pixels']), (3200, 3200))
        self.assertEqual(left['predicted_pv_pixels_in_ignored_region_unscored'], 3200)
        self.assertEqual(right['predicted_pv_pixels_in_ignored_region_unscored'], 9600)
        self.assertEqual(left['pixel_recall_on_consensus_pv'], 0.5)
        self.assertEqual(left['missed_consensus_pv_pixel_fraction'], 0.5)
        self.assertEqual(report['delta']['pixel_recall_on_consensus_pv_percentage_points'], 50)
        self.assertEqual(report['supervised_positive_coverage'], 1 / 32)
        self.assertEqual(report['supervised_background_coverage'], 1 / 32)
        self.assertEqual(report['ignored_coverage'], 30 / 32)
        for key in ('training_performed', 'old_test_pixels_opened', 'negative_holdout_pixels_opened', 'human_review_claimed'):
            self.assertFalse(report[key])
        for model in self.models:
            self.assertEqual(model.observations, [(False, False)])
            self.assertTrue(torch.equal(model.cutoff, model.before))
            self.assertIsNone(model.cutoff.grad)
        self.assertEqual(report['samples_sha256'], self.sha(self.output / 'samples.json'))
        for row in json.loads((self.output / 'samples.json').read_text()):
            for role in ('reference', 'candidate'):
                path = self.output / row[role]['prediction']
                self.assertEqual(row[role]['prediction_sha256'], self.sha(path))
                with Image.open(path) as image:
                    self.assertEqual(image.size, (320, 320))
                    self.assertEqual(image.mode, 'L')
                    self.assertLessEqual(set(image.get_flattened_data()), {0, 1})
        self.assertEqual(json.loads((self.output / 'positive-evaluation-report.json').read_text()), report)
        self.assertNotIn('"iou"', json.dumps(report))
        self.assertNotIn('"precision"', json.dumps(report))

    def test_exactly_100_pv_pixels_and_no_background_are_valid(self):
        for row in self.records[:2]:
            mask = Image.new('L', (320, 320), 255)
            mask.paste(1, (0, 0, 5, 20))
            self.replace_mask(row, mask)
        report = self.evaluate()
        for role in ('reference', 'candidate'):
            counts = report['results'][role]['totals']
            self.assertEqual(counts['supervised_positive_pixels'], 200)
            self.assertEqual(counts['true_positive_pixels'], 200)
            self.assertEqual(counts['supervised_background_pixels'], 0)
            self.assertIsNone(counts['background_pixel_false_positive_rate'])
        self.assertIsNone(report['delta']['background_pixel_false_positive_rate'])

    def test_99_pv_pixels_or_noncanonical_label_are_rejected_before_models(self):
        for label in (255, 2):
            with self.subTest(label=label):
                mask = Image.new('L', (320, 320), 255)
                mask.paste(1, (0, 0, 5, 20))
                mask.putpixel((0, 0), label)
                self.replace_mask(self.records[0], mask)
                self.assert_failure()
                self.assertFalse(self.models)

    def test_crop_total_miss_means_zero_tp_on_confirmed_pv_only(self):
        self.documents['reference']['model_state_dict']['cutoff'] = torch.tensor(0.0)
        report = self.evaluate()
        self.assertEqual(report['results']['reference']['totals']['crops_with_zero_true_positive_pixels'], 2)
        self.assertEqual(report['results']['reference']['totals']['pixel_recall_on_consensus_pv'], 0)
        self.assertEqual(report['delta']['crops_with_zero_true_positive_pixels'], -2)

    def test_old_or_negative_dataset_rejected_before_pixels_and_weights(self):
        path = self.dataset / 'dataset.toml'
        original = path.read_text()
        for dataset_id in ('pv4ger-seg', 'pvextent-negative-holdout'):
            with self.subTest(dataset_id=dataset_id):
                path.write_text(original.replace('pvextent-positive-holdout', dataset_id))
                self.write_evidence()
                with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")), \
                     patch('pvextent.training.baseline._read_checkpoint', side_effect=AssertionError("No checkpoint access")):
                    self.assert_failure()

    def test_freeze_identity_flags_and_selection_guard_before_pixels(self):
        for key, value in (("frozen_before_predictions", False), ("dataset_fingerprint", 'a' * 64),
                           ("reviewed_evaluation_ids", ['sample-0']), ("human_review_claimed", True),
                           ("training_use_allowed", True), ("selected_ids", ['sample-0']),
                           ("minimum_positive_pixels", 99)):
            with self.subTest(key=key):
                self.write_evidence()
                path = self.dataset / 'provenance/positive-holdout-freeze.json'
                freeze = json.loads(path.read_text())
                freeze[key] = value
                path.write_text(json.dumps(freeze))
                with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
                    self.assert_failure()

    def test_protocol_and_mask_bytes_must_match_freeze(self):
        path = self.dataset / 'provenance/protocol.json'
        path.write_text(path.read_text() + ' ')
        with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
            self.assert_failure()
        self.write_evidence()
        Image.new('L', (320, 320), 1).save(self.dataset / self.records[0]['mask'])
        self.assert_failure()
        self.assertFalse(self.models)

    def test_protocol_selected_image_hashes_must_match_before_pixels(self):
        self.protocol['selected_image_sha256']['sample-2'] = '0' * 64
        self.write_evidence()
        with patch('pvextent.data.training._raster', side_effect=AssertionError("No raster access")):
            self.assert_failure()

    def test_both_frozen_checkpoint_hashes_are_checked_before_loading_either(self):
        self.freeze['checkpoint_sha256']['candidate'] = '0' * 64
        self.protocol['checkpoint_sha256']['candidate'] = '0' * 64
        self.write_evidence()
        with patch('pvextent.training.baseline._read_checkpoint', side_effect=AssertionError("Unfrozen weights must not load")):
            self.assert_failure()

    def test_checkpoint_model_contract_cannot_change(self):
        for key, value in (('preprocessing', 'resized'), ('threshold', 0.6), ('encoder', 'other'), ('seed', 123)):
            with self.subTest(key=key):
                self.documents['candidate']['config'] = dict(self.config, **{key: value})
                self.assert_failure()
                self.assertFalse(self.models)

    def test_existing_or_overlapping_output_is_preserved(self):
        self.output.mkdir()
        marker = self.output / 'keep'
        marker.write_text('keep')
        with self.assertRaises(self.module.PositiveEvaluationError):
            self.evaluate()
        self.assertEqual(marker.read_text(), 'keep')
        self.output = self.dataset / 'new-output'
        self.assert_failure()
        self.assertFalse(self.models)

    def test_nonfinite_output_cleans_stage_without_publication(self):
        class BadModel(torch.nn.Module):
            def forward(self, images):
                return torch.full((len(images), 1, 320, 320), float('nan'))

            def load_state_dict(self, *args, **kwargs):
                return None

        with patch('pvextent.models.resnet_unet.build_resnet_unet', return_value=BadModel()):
            self.assert_failure()

    def test_protocol_change_during_inference_prevents_publication(self):
        path = self.dataset / 'provenance/protocol.json'

        def factory(weights):
            model = self.factory(weights)
            path.write_text(path.read_text() + ' ')
            return model

        with patch('pvextent.models.resnet_unet.build_resnet_unet', side_effect=factory):
            with self.assertRaisesRegex(self.module.PositiveEvaluationError, 'evidence changed'):
                self.evaluate()
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.evaluation-positive-evaluation-*')))

    def test_mask_or_companion_change_during_inference_prevents_publication(self):
        paths = [self.dataset / self.records[0]['mask'], self.checkpoints['reference'].with_suffix('.pt.sha256')]
        for path in paths:
            with self.subTest(path=path.name):
                original = path.read_bytes()

                def factory(weights):
                    model = self.factory(weights)
                    path.write_bytes(original + b' ')
                    return model

                with patch('pvextent.models.resnet_unet.build_resnet_unet', side_effect=factory):
                    with self.assertRaises(self.module.PositiveEvaluationError):
                        self.evaluate()
                self.assertFalse(self.output.exists())
                self.assertFalse(list(self.root.glob('.evaluation-positive-evaluation-*')))
                path.write_bytes(original)


if __name__ == '__main__':
    unittest.main()
