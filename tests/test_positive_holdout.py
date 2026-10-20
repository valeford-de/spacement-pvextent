# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic-only positive holdout tests; previous test pixels are inaccessible."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from pvextent.data import positive_holdout as freeze
from pvextent.data.validation import validate_dataset

METADATA = '''schema_version = "1.0"
dataset_id = "synthetic"
dataset_version = "1"
[sources.synthetic]
name = "Generated pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "not-annotated-ignore-placeholder"
attribution = "Synthetic regression fixture"
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def manifest(root, rows):
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows))


def fingerprint(root):
    return digest((root / 'dataset.toml').read_bytes() + b'\0' + (root / 'manifest.jsonl').read_bytes())


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def box(left, top, right, bottom):
    return [[[left, top], [right, top], [right, bottom], [left, bottom]]]


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('pyproj'), 'Install train and geo extras')
class PositiveHoldoutTests(unittest.TestCase):
    def setUp(self):
        from pyproj import Transformer
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.candidates, self.negative, self.reference, self.output = [self.root / name for name in ('candidates', 'negative', 'reference', 'frozen')]
        for directory in (self.candidates, self.negative, self.reference):
            (directory / 'provenance').mkdir(parents=True)
            (directory / 'dataset.toml').write_text(METADATA)
        (self.negative / 'dataset.toml').write_text(METADATA.replace('dataset_id = "synthetic"', 'dataset_id = "pvextent-negative-holdout"'))
        self.reference_rows, self.reference_centers = [], []
        # 421 old test records and every other partition have no raster files.
        for index, split in enumerate(['train', 'val', 'unassigned'] + ['test'] * 421):
            identifier = f'old-{index}'
            self.reference_rows.append(self.row(identifier, split))
            self.reference_centers.append({'id': identifier, 'center_crs': 'EPSG:4326', 'approximate_center': [6 + index * .00001, 50.5]})
        self.rows, self.centers = [], []
        inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        for index in range(20):
            identifier = f'positive-{index}' if index < 13 else f'prior-negative-{index}'
            row = self.row(identifier)
            x, y = 392080 + index * 100, 5711600
            row['georeference'] = {'crs': 'EPSG:25832', 'transform': [.1, 0, x - 16, 0, -.1, y + 16]}
            if index < 13:
                path = self.candidates / row['image']
                path.parent.mkdir(exist_ok=True)
                Image.new('RGB', (320, 320), (30 + index * 10, 70, 150)).save(path)
                row['image_sha256'] = digest(path.read_bytes())
            # No candidate masks, prior seven images, or negative holdout rasters exist.
            self.rows.append(row)
            self.centers.append({'id': identifier, 'center_crs': 'EPSG:4326', 'approximate_center': list(inverse.transform(x, y))})
        self.negative_rows = [{**row, 'split': 'unassigned' if i < 13 else 'test',
                               'annotation_status': 'unreviewed' if i < 13 else 'reviewed'} for i, row in enumerate(self.rows)]
        self.protocol_path = self.root / 'protocol.json'
        self.review_paths = [self.root / 'review-a.json', self.root / 'review-b.json']
        self.acquisition = {'schema_version': '1.0', 'min_reference_distance_m': 5000 + 32 * 2**.5}
        write_json(self.candidates / 'provenance/protocol.json', self.acquisition)
        (self.candidates / 'provenance/source-license.txt').write_text('CC0 generated image fixture; retained notice.')
        self.protocol = {'schema_version': '1.0', 'purpose': freeze.PURPOSE, 'dataset_version': 'positive-test-1',
                         'checkpoint_sha256': {'reference': 'a' * 64, 'candidate': 'b' * 64},
                         'threshold': .5, 'minimum_positive_pixels': 100, 'selected_ids': sorted(row['id'] for row in self.rows[:13]),
                         'frozen_before_predictions': True, 'model_predictions_used_for_labels': False,
                         'human_review_claimed': False, 'training_use_allowed': False}
        self.reviews = []
        for index in range(2):
            samples = []
            for number, row in enumerate(self.rows[:13]):
                positive = box(5 + index * 5, 5 + index * 5, 24 + index * 5, 24 + index * 5)
                if number == 1: positive = box(0, 0, 9, 9)
                if number == 2: positive = box(0, 0, 10, 8)
                samples.append({'id': row['id'], 'image_sha256': row['image_sha256'], 'full_image_inspected': True,
                    'decision': 'approve-positive' if number <= 2 else 'defer', 'positive_polygons': positive,
                    'background_polygons': box(100 + index * 10, 100 + index * 10, 119 + index * 10, 119 + index * 10),
                    'notes': 'Synthetic full-image visual-review fixture.'})
            self.reviews.append({'review_schema_version': '1.0', 'reviewer': 'reviewer-' + str(index), 'reviewer_type': 'ai',
                                 'model_predictions_used': False, 'samples': samples})
        self.refresh()
        self.write_reviews()

    def row(self, identifier, split='unassigned'):
        return {'id': identifier, 'image': f'images/{identifier}.png', 'mask': f'masks/{identifier}.png',
                'image_sha256': digest(('image-' + identifier).encode()), 'mask_sha256': digest(('mask-' + identifier).encode()),
                'width': 320, 'height': 320, 'gsd_m': .1, 'capture_date': None, 'source_id': 'synthetic',
                'spatial_group': 'group-' + identifier, 'building_ids': ['synthetic:' + identifier],
                'split': split, 'annotation_status': 'unreviewed' if split == 'unassigned' else 'upstream', 'georeference': None}

    def refresh(self):
        for directory, rows, centers in ((self.candidates, self.rows, self.centers),
                                        (self.reference, self.reference_rows, self.reference_centers)):
            manifest(directory, rows)
            (directory / 'provenance/samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in centers))
        manifest(self.negative, self.negative_rows)
        self.candidate_report = {'ok': True, 'dataset_fingerprint': fingerprint(self.candidates),
            'protocol_sha256': digest((self.candidates / 'provenance/protocol.json').read_bytes()),
            'references': [{'dataset_fingerprint': fingerprint(self.reference),
                'coordinate_sha256': digest((self.reference / 'provenance/samples.jsonl').read_bytes())}]}
        write_json(self.candidates / 'candidate-report.json', self.candidate_report)
        write_json(self.negative / 'provenance/negative-holdout-freeze.json', {
            'ok': True, 'purpose': 'experimental-ai-negative-holdout', 'dataset_fingerprint': fingerprint(self.negative),
            'candidate_dataset_fingerprint': fingerprint(self.candidates), 'checkpoint_sha256': self.protocol['checkpoint_sha256'],
            'reviewed_evaluation_ids': [row['id'] for row in self.negative_rows if row['split'] == 'test'],
            'model_predictions_used_for_labels': False, 'training_use_allowed': False})
        self.protocol.update(candidate_dataset_fingerprint=fingerprint(self.candidates),
                             negative_holdout_dataset_fingerprint=fingerprint(self.negative),
                             reference_dataset_fingerprints=[fingerprint(self.reference)],
                             selected_image_sha256={row['id']: row['image_sha256'] for row in self.rows[:13]})
        write_json(self.protocol_path, self.protocol)

    def write_reviews(self):
        for path, review in zip(self.review_paths, self.reviews): write_json(path, review)

    def call(self, **kwargs):
        options = {'review_paths': self.review_paths, 'protocol_path': self.protocol_path}
        options.update(kwargs)
        return freeze.freeze_positive_holdout(self.candidates, self.negative, [self.reference], self.output, **options)

    def rejects(self, pattern='.', **kwargs):
        with self.assertRaisesRegex(freeze.PositiveHoldoutError, pattern): self.call(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.frozen-positive-*')))

    def test_consensus_100_pixel_gate_defer_and_exact_source_bytes(self):
        before = {directory: tree_bytes(directory) for directory in (self.candidates, self.negative, self.reference)}
        report = self.call()
        self.assertEqual(report['selected_ids'], sorted(self.protocol['selected_ids']))
        self.assertEqual(report['reviewed_evaluation_ids'], ['positive-0', 'positive-1'])
        self.assertEqual(len(report['selected_ids']), 13)
        self.assertFalse(report['training_use_allowed'])
        self.assertTrue(validate_dataset(self.output).ok)
        rows = {row['id']: row for row in map(json.loads, (self.output / 'manifest.jsonl').read_text().splitlines())}
        with Image.open(self.output / rows['positive-0']['mask']) as mask:
            self.assertEqual(mask.getpixel((10, 10)), 1)
            self.assertEqual(mask.getpixel((5, 5)), 255)
            self.assertEqual(mask.getpixel((110, 110)), 0)
            self.assertEqual(mask.getpixel((100, 100)), 255)
        for identifier in ('positive-2', 'positive-3'):
            self.assertEqual(rows[identifier]['split'], 'unassigned')
            with Image.open(self.output / rows[identifier]['mask']) as mask: self.assertEqual(mask.histogram()[255], 320 * 320)
        for row in rows.values():
            self.assertEqual((self.output / row['image']).read_bytes(), (self.candidates / row['image']).read_bytes())
        for directory, value in before.items(): self.assertEqual(tree_bytes(directory), value)
        for path, evidence in zip(self.review_paths, report['reviewers']):
            self.assertEqual((self.output / evidence['file']).read_bytes(), path.read_bytes())
            self.assertEqual(evidence['sha256'], digest(path.read_bytes()))
        self.assertEqual((self.output / 'provenance/protocol.json').read_bytes(), self.protocol_path.read_bytes())
        self.assertEqual(len((self.output / 'provenance/samples.jsonl').read_text().splitlines()), 13)
        self.assertIn('annotation_license = "Apache-2.0"', (self.output / 'dataset.toml').read_text())
        self.assertIn('image_license = "CC0-1.0"', (self.output / 'dataset.toml').read_text())
        self.assertEqual((self.output / 'provenance/acquisition/source-license.txt').read_text(), 'CC0 generated image fixture; retained notice.')

    def test_previous_seven_and_all_421_test_rasters_are_never_opened_or_resolved(self):
        forbidden = {self.reference / row[role] for row in self.reference_rows for role in ('image', 'mask')}
        forbidden |= {self.negative / row[role] for row in self.negative_rows for role in ('image', 'mask')}
        forbidden |= {self.candidates / row[role] for row in self.rows for role in ('image', 'mask') if role == 'mask' or row['id'].startswith('prior-')}
        original_open, original_resolve, original_image, original_validate = Path.open, Path.resolve, Image.open, freeze.validate_dataset
        def guard_open(path, *args, **kwargs):
            if path in forbidden: raise AssertionError('Previous or unselected raster opened')
            return original_open(path, *args, **kwargs)
        def guard_resolve(path, *args, **kwargs):
            if path in forbidden: raise AssertionError('Previous or unselected raster resolved')
            return original_resolve(path, *args, **kwargs)
        def guard_image(path, *args, **kwargs):
            if isinstance(path, (str, Path)) and Path(path) in forbidden: raise AssertionError('Previous raster decoded')
            return original_image(path, *args, **kwargs)
        def guard_validate(path):
            if Path(path) in (self.candidates, self.negative, self.reference): raise AssertionError('Input snapshot validated')
            return original_validate(path)
        with patch.object(Path, 'open', guard_open), patch.object(Path, 'resolve', guard_resolve), \
             patch.object(Image, 'open', guard_image), patch.object(freeze, 'validate_dataset', side_effect=guard_validate):
            self.assertTrue(self.call()['ok'])

    def test_intra_review_pv_background_conflict_is_rejected_even_when_deferred(self):
        self.reviews[0]['samples'][3]['background_polygons'] = box(5, 5, 24, 24)
        self.write_reviews()
        self.rejects('overlap')

    def test_cross_review_class_disagreement_remains_ignored(self):
        sample = self.reviews[1]['samples'][0]
        sample['background_polygons'].extend(box(5, 5, 9, 9))
        self.write_reviews()
        self.call()
        with Image.open(self.output / 'masks/positive-0.png') as mask: self.assertEqual(mask.getpixel((5, 5)), 255)

    def test_distinct_reviewers_hashes_and_full_selection_are_required(self):
        original = copy.deepcopy(self.reviews)
        mutations = [lambda: self.reviews[1].update(reviewer=' REVIEWER-0 '),
                     lambda: self.reviews[0]['samples'][0].update(image_sha256='f' * 64),
                     lambda: self.reviews[0]['samples'][0].update(full_image_inspected=False),
                     lambda: self.reviews[0]['samples'].pop(),
                     lambda: self.reviews[0]['samples'][0].update(extra='unknown')]
        for mutation in mutations:
            self.reviews = copy.deepcopy(original)
            mutation(); self.write_reviews()
            self.rejects('review|Review|fields|IDs')

    def test_duplicate_json_nan_and_invalid_geometry_are_rejected(self):
        original = self.review_paths[0].read_bytes()
        for payload in (b'{"reviewer":1,"reviewer":2}', b'{"reviewer":NaN}'):
            self.review_paths[0].write_bytes(payload)
            self.rejects()
        self.review_paths[0].write_bytes(original)
        for polygon in ([[[0, 0], [320, 0], [0, 9]]], [[[0, 0], [1.1, 2], [0, 9]]], [[[0, 0], [9, 9], [0, 9], [9, 0]]]):
            self.reviews[0]['samples'][0]['positive_polygons'] = polygon
            self.write_reviews(); self.rejects('polygon|Polygon|pixel')

    def test_reference_identity_overlap_precedes_pixel_access(self):
        original = copy.deepcopy(self.reference_rows)
        for field in ('id', 'image_sha256', 'building_ids', 'spatial_group'):
            self.reference_rows = copy.deepcopy(original)
            self.reference_rows[0][field] = copy.deepcopy(self.rows[0][field])
            old_id = self.reference_centers[0]['id']
            self.reference_centers[0]['id'] = self.reference_rows[0]['id']
            self.refresh()
            with patch.object(freeze, '_image_payload', side_effect=AssertionError('Pixels opened before metadata guard')):
                self.rejects('overlap')
            self.reference_centers[0]['id'] = old_id

    def test_distance_checks_every_partition_and_coordinate_hash_is_bound(self):
        original = copy.deepcopy(self.reference_centers)
        for index in (0, 1, 2, 3, 423):
            self.reference_centers = copy.deepcopy(original)
            self.reference_centers[index]['approximate_center'] = self.centers[0]['approximate_center']
            self.refresh()
            with patch.object(freeze, '_image_payload', side_effect=AssertionError('Premature image access')):
                self.rejects('buffer')
        self.reference_centers = original
        self.refresh()
        self.reference_centers[0]['approximate_center'][0] += .00001
        (self.reference / 'provenance/samples.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in self.reference_centers))
        self.rejects('coordinates|basis')

    def test_affine_and_candidate_provenance_disagreement_is_rejected(self):
        self.centers[0]['approximate_center'][0] += .0001
        self.refresh()
        self.rejects('centre')

    def test_selected_ids_and_prior_negative_identity_are_frozen(self):
        self.protocol['selected_ids'][0] = self.rows[-1]['id']
        write_json(self.protocol_path, self.protocol)
        self.rejects('complement')
        self.protocol['selected_ids'] = sorted(row['id'] for row in self.rows[:13])
        self.negative_rows[0]['building_ids'] = ['changed']
        self.refresh()
        self.rejects('identity')

    def test_selected_image_hash_map_is_bound_before_pixels(self):
        self.protocol['selected_image_sha256']['positive-0'] = 'f' * 64
        write_json(self.protocol_path, self.protocol)
        with patch.object(freeze, '_image_payload', side_effect=AssertionError('Premature pixel access')):
            self.rejects('complement')

    def test_freeze_requires_the_evaluators_order_and_complete_hash_map(self):
        original = dict(self.protocol)
        for field, value in (('selected_ids', list(reversed(self.protocol['selected_ids']))),
                             ('selected_image_sha256', None)):
            with self.subTest(field=field):
                self.protocol = {**original, field: value}
                write_json(self.protocol_path, self.protocol)
                with patch.object(freeze, '_image_payload', side_effect=AssertionError('Premature pixel access')):
                    self.rejects('complement')
        self.protocol = original

    def test_protocol_flags_and_prediction_use_are_rejected(self):
        for key, value in (('training_use_allowed', True), ('minimum_positive_pixels', 99), ('threshold', .6)):
            original = self.protocol[key]
            self.protocol[key] = value; write_json(self.protocol_path, self.protocol)
            self.rejects('protocol')
            self.protocol[key] = original

    def test_no_eligible_masks_leave_no_output_or_staging(self):
        for review in self.reviews:
            for sample in review['samples']: sample['decision'] = 'defer'
        self.write_reviews(); self.rejects('No masks')

    def test_changed_review_during_final_validation_is_rejected_atomically(self):
        original_validate = freeze.validate_dataset
        def mutate(path):
            result = original_validate(path)
            self.review_paths[0].write_text(self.review_paths[0].read_text() + ' ')
            return result
        with patch.object(freeze, 'validate_dataset', side_effect=mutate): self.rejects('changed')

    def test_no_overwrite_and_disjoint_output(self):
        self.output.mkdir(); (self.output / 'sentinel').write_text('preserve')
        with self.assertRaises(freeze.PositiveHoldoutError): self.call()
        self.assertEqual((self.output / 'sentinel').read_text(), 'preserve')
        with self.assertRaises(freeze.PositiveHoldoutError):
            freeze.freeze_positive_holdout(self.candidates, self.negative, [self.reference], self.candidates / 'inside',
                                           review_paths=self.review_paths, protocol_path=self.protocol_path)
        self.assertFalse((self.candidates / 'inside').exists())


if __name__ == '__main__':
    unittest.main()
