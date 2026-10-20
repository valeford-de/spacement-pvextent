# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Freeze only geographically separate, evidenced partial negatives."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from pvextent.data import negative_freeze as freeze
from pvextent.data.curation import curate_negative_collection
from pvextent.data.validation import validate_dataset

METADATA = '''schema_version = "1.0"
dataset_id = "freeze-fixture"
dataset_version = "1"
[sources.synthetic]
name = "Generated fixture pixels"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "not-annotated-ignore-placeholder"
attribution = "Synthetic test fixture"
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n', encoding='utf-8')


def manifest(root, rows):
    (root / 'manifest.jsonl').write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')


def fingerprint(root):
    return digest((root / 'dataset.toml').read_bytes() + b'\0' + (root / 'manifest.jsonl').read_bytes())


def tree_bytes(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('pyproj'), 'Install train and geo extras')
class NegativeFreezeTests(unittest.TestCase):
    def setUp(self):
        from pyproj import Transformer
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.reference, self.candidates = self.root / 'reference', self.root / 'candidates'
        self.curated, self.output = self.root / 'curated', self.root / 'frozen'
        for root in (self.reference, self.candidates):
            (root / 'provenance').mkdir(parents=True)
            (root / 'dataset.toml').write_text(METADATA, encoding='utf-8')
        self.reference_rows = []
        self.reference_centers = []
        for index, split in enumerate(('train', 'val', 'test', 'unassigned')):
            self.reference_rows.append(self.row('old-' + split, index, split))
            self.reference_centers.append({'id': 'old-' + split, 'center_crs': 'EPSG:4326',
                                           'approximate_center': [6.0 + index * .01, 50.5]})
        # Every old raster is intentionally absent. Metadata-only exclusions
        # must succeed even for the inaccessible existing test split.
        manifest(self.reference, self.reference_rows)
        self.write_reference_centers()
        inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        self.candidate_rows, self.candidate_centers = [], []
        (self.candidates / 'images').mkdir()
        (self.candidates / 'masks').mkdir()
        for index, identifier in enumerate(('accepted', 'boundary', 'small', 'deferred')):
            row = self.row(identifier, index + 20, 'unassigned')
            x, y = 392080 + index * 400, 5711600
            row.update(width=320, height=320, capture_date='2025-03-07',
                       annotation_status='unreviewed',
                       georeference={'crs': 'EPSG:25832', 'transform': [.1, 0, x-16, 0, -.1, y+16]})
            Image.new('RGB', (320, 320), (30 + index * 40, 70, 150)).save(self.candidates / row['image'])
            Image.new('L', (320, 320), 255).save(self.candidates / row['mask'])
            for role in ('image', 'mask'):
                row[role + '_sha256'] = digest((self.candidates / row[role]).read_bytes())
            self.candidate_rows.append(row)
            self.candidate_centers.append({'id': identifier, 'center_crs': 'EPSG:4326',
                                           'approximate_center': list(inverse.transform(x, y))})
        manifest(self.candidates, self.candidate_rows)
        self.write_candidate_centers()
        self.acquisition_path, self.protocol_path = self.root / 'acquisition.json', self.root / 'evaluation.json'
        self.acquisition = {'schema_version': '1.0', 'dataset_version': 'separate-candidates-1',
            'regions': [{'id': 'fixture-region', 'bbox_crs84': [7.435, 51.545, 7.455, 51.565], 'limit': 4}],
            'min_reference_distance_m': 5000 + 32 * 2**.5, 'min_candidate_distance_m': 200}
        write_json(self.acquisition_path, self.acquisition)
        self.candidate_report = {'ok': True, 'records': 4, 'dataset_fingerprint': fingerprint(self.candidates),
                                 'protocol_sha256': digest(self.acquisition_path.read_bytes()),
                                 'references': self.reference_evidence()}
        write_json(self.candidates / 'candidate-report.json', self.candidate_report)
        (self.candidates / 'provenance/protocol.json').write_bytes(self.acquisition_path.read_bytes())
        self.review_paths = [self.root / 'review-a.json', self.root / 'review-b.json']
        for index, path in enumerate(self.review_paths):
            samples = []
            for row in self.candidate_rows:
                right = {'accepted': 319, 'boundary': 159, 'small': 158, 'deferred': None}[row['id']]
                samples.append({'id': row['id'], 'image_sha256': row['image_sha256'], 'full_image_inspected': True,
                    'decision': 'defer' if right is None else 'approve-negative',
                    'background_polygons': [] if right is None else [[[0, 0], [right, 0], [right, 319], [0, 319]]],
                    'notes': 'Synthetic full-image inspection for a regression test.'})
            write_json(path, {'review_schema_version': '1.0', 'reviewer': 'ai-review-' + str(index),
                             'reviewer_type': 'ai', 'model_predictions_used': False, 'samples': samples})
        curate_negative_collection([self.candidates], self.review_paths, self.curated,
                                   dataset_version='two-reviewed-1', allow_ai_review=True)
        self.protocol = {'schema_version': '1.0', 'purpose': 'experimental-ai-negative-holdout',
            'dataset_version': 'frozen-negative-1', 'acquisition_protocol_sha256': digest(self.acquisition_path.read_bytes()),
            'minimum_background_fraction': .5, 'threshold': .5, 'model_predictions_used_for_labels': False,
            'frozen_before_predictions': True, 'human_review_claimed': False,
            'checkpoint_sha256': {'reference': 'a'*64, 'candidate': 'b'*64},
            'reference_dataset_fingerprints': [fingerprint(self.reference)]}
        write_json(self.protocol_path, self.protocol)

    def row(self, identifier, index, split):
        return {'id': identifier, 'image': 'images/' + identifier + '.png', 'mask': 'masks/' + identifier + '.png',
                'image_sha256': digest(('old-image-' + str(index)).encode()),
                'mask_sha256': digest(('old-mask-' + str(index)).encode()),
                'width': 320, 'height': 320, 'gsd_m': .1, 'capture_date': None,
                'source_id': 'synthetic', 'spatial_group': 'group-' + identifier,
                'building_ids': ['synthetic:' + identifier], 'split': split,
                'annotation_status': 'unreviewed' if split == 'unassigned' else 'upstream', 'georeference': None}

    def write_reference_centers(self):
        (self.reference / 'provenance/samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.reference_centers))

    def write_candidate_centers(self):
        (self.candidates / 'provenance/samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.candidate_centers))

    def reference_evidence(self):
        return [{'dataset_fingerprint': fingerprint(self.reference),
                 'coordinate_sha256': digest((self.reference / 'provenance/samples.jsonl').read_bytes()),
                 'records': len(self.reference_rows)}]

    def refresh_acquisition_reference_evidence(self):
        self.candidate_report['references'] = self.reference_evidence()
        write_json(self.candidates / 'candidate-report.json',self.candidate_report)

    def call(self, **kwargs):
        options = {'protocol_path': self.protocol_path, 'acquisition_protocol_path': self.acquisition_path}
        options.update(kwargs)
        return freeze.freeze_negative_holdout(self.curated, self.candidates, [self.reference], self.output, **options)

    def fail(self, pattern='.', **kwargs):
        with self.assertRaisesRegex(freeze.NegativeFreezeError, pattern):
            self.call(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob('.frozen-freeze-*')), [])

    def curated_rows(self):
        return [json.loads(line) for line in (self.curated / 'manifest.jsonl').read_text().splitlines()]

    def update_curated(self, rows):
        manifest(self.curated, rows)
        path = self.curated / 'curation-report.json'
        report = json.loads(path.read_text())
        report['dataset_fingerprint'] = fingerprint(self.curated)
        write_json(path, report)

    def update_reference(self):
        manifest(self.reference, self.reference_rows)
        self.write_reference_centers()
        self.refresh_acquisition_reference_evidence()
        self.protocol['reference_dataset_fingerprints'] = [fingerprint(self.reference)]
        write_json(self.protocol_path, self.protocol)

    def test_exact_coverage_gate_defer_ignore_and_source_immutability(self):
        before = {str(p): tree_bytes(p) for p in (self.reference, self.candidates, self.curated)}
        report = self.call()
        self.assertEqual(report['evaluation_samples'], 2)
        self.assertEqual(report['reviewed_evaluation_ids'], ['accepted', 'boundary'])
        self.assertFalse(report['old_reference_pixels_opened'])
        self.assertFalse(report['training_use_allowed'])
        rows = {r['id']: r for r in map(json.loads, (self.output / 'manifest.jsonl').read_text().splitlines())}
        self.assertEqual({i: r['split'] for i,r in rows.items()},
                         {'accepted':'test','boundary':'test','small':'unassigned','deferred':'unassigned'})
        self.assertTrue(validate_dataset(self.output).ok)
        for source in (self.reference, self.candidates, self.curated):
            self.assertEqual(tree_bytes(source), before[str(source)])
        for row in rows.values():
            for role in ('image','mask'):
                self.assertEqual((self.output / row[role]).read_bytes(), (self.curated / row[role]).read_bytes())
                self.assertFalse((self.output / row[role]).samefile(self.curated / row[role]))
        self.assertEqual((self.output / 'provenance/protocol.json').read_bytes(), self.protocol_path.read_bytes())
        self.assertEqual((self.output / 'provenance/acquisition-protocol.json').read_bytes(), self.acquisition_path.read_bytes())

    def test_reference_rasters_are_never_resolved_opened_or_validated(self):
        original_open, original_resolve, original_image_open = Path.open, Path.resolve, Image.open
        original_validate = freeze.validate_dataset
        forbidden = {self.reference / row[role] for row in self.reference_rows for role in ('image','mask')}
        def guard_open(path, *args, **kwargs):
            if path in forbidden: raise AssertionError('Old reference raster opened')
            return original_open(path,*args,**kwargs)
        def guard_resolve(path, *args, **kwargs):
            if path in forbidden: raise AssertionError('Old reference raster resolved')
            return original_resolve(path,*args,**kwargs)
        def guard_image(path, *args, **kwargs):
            if isinstance(path,(str,Path)) and Path(path) in forbidden: raise AssertionError('Old reference image decoded')
            return original_image_open(path,*args,**kwargs)
        def guard_validate(path):
            if Path(path) == self.reference: raise AssertionError('Old reference dataset validated')
            return original_validate(path)
        with patch.object(Path,'open',guard_open), patch.object(Path,'resolve',guard_resolve), \
             patch.object(Image,'open',guard_image), patch.object(freeze,'validate_dataset',side_effect=guard_validate):
            self.assertTrue(self.call()['ok'])

    def test_identity_overlaps_fail_before_any_new_pixel_validation(self):
        for field in ('id','image_sha256','building_ids','spatial_group'):
            with self.subTest(field=field):
                backup = copy.deepcopy(self.reference_rows)
                old_centers = copy.deepcopy(self.reference_centers)
                self.reference_rows[0][field] = copy.deepcopy(self.candidate_rows[0][field])
                if field == 'id': self.reference_centers[0]['id'] = self.candidate_rows[0]['id']
                self.update_reference()
                with patch.object(freeze,'validate_dataset',side_effect=AssertionError('Pixels opened before identity guard')):
                    self.fail('overlap')
                self.reference_rows, self.reference_centers = backup, old_centers
                self.update_reference()

    def test_every_reference_partition_is_in_the_geographic_exclusion(self):
        for index in range(4):
            with self.subTest(split=self.reference_rows[index]['split']):
                original = self.reference_centers[index]['approximate_center']
                self.reference_centers[index]['approximate_center'] = self.candidate_centers[0]['approximate_center']
                self.write_reference_centers()
                self.refresh_acquisition_reference_evidence()
                with patch.object(freeze,'validate_dataset',side_effect=AssertionError('Pixel read before distance guard')):
                    self.fail('buffer|distance')
                self.reference_centers[index]['approximate_center'] = original
                self.write_reference_centers()
                self.refresh_acquisition_reference_evidence()

    def test_changed_original_image_identity_and_candidate_ids_are_rejected(self):
        rows = self.curated_rows()
        rows[0]['image_sha256'] = 'f'*64
        self.update_curated(rows)
        self.fail('identity|imagery')
        rows = self.curated_rows()[1:]
        self.update_curated(rows)
        self.fail('all|candidate')

    def test_pv_pixels_are_never_accepted_as_negative_background(self):
        rows = self.curated_rows()
        image = Image.open(self.curated / rows[0]['mask']).copy()
        image.putpixel((5,5),1)
        image.save(self.curated / rows[0]['mask'])
        rows[0]['mask_sha256'] = digest((self.curated / rows[0]['mask']).read_bytes())
        self.update_curated(rows)
        self.fail('negative|mask|curat|review')

    def test_same_histogram_mask_tampering_cannot_replace_review_consensus(self):
        rows = self.curated_rows()
        row = next(r for r in rows if r['id'] == 'boundary')
        image = Image.open(self.curated / row['mask']).copy()
        self.assertEqual((image.getpixel((5,5)),image.getpixel((200,5))),(0,255))
        image.putpixel((5,5),255)
        image.putpixel((200,5),0)
        image.save(self.curated / row['mask'])
        row['mask_sha256'] = digest((self.curated / row['mask']).read_bytes())
        self.update_curated(rows)
        self.fail('consensus|review|mask')

    def test_retained_review_bytes_must_match_their_recorded_hashes(self):
        report = json.loads((self.curated / 'curation-report.json').read_text())
        path = self.curated / report['reviewers'][0]['file']
        review = json.loads(path.read_text())
        review['samples'][0]['notes'] = 'Changed evidence after recorded review hash.'
        write_json(path,review)
        self.fail('evidence|hash|review')

    def test_frozen_protocol_checkpoints_and_reference_inventory_are_required(self):
        original = copy.deepcopy(self.protocol)
        variants = [dict(checkpoint_sha256={'reference':'a'*64}), dict(threshold=True),
                    dict(minimum_background_fraction=.4),dict(frozen_before_predictions=False),
                    dict(model_predictions_used_for_labels=True),dict(human_review_claimed=True),
                    dict(reference_dataset_fingerprints=['c'*64]),dict(acquisition_protocol_sha256='0'*64)]
        for changes in variants:
            with self.subTest(changes=changes):
                write_json(self.protocol_path,{**original,**changes})
                self.fail()

    def test_existing_or_overlapping_outputs_are_not_modified(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'; sentinel.write_text('keep')
        with self.assertRaises(freeze.NegativeFreezeError): self.call()
        self.assertEqual(sentinel.read_text(),'keep')
        sentinel.unlink();self.output.rmdir()
        self.output = self.reference / 'nested'
        self.fail('separate')

    def test_failed_publication_cleans_staging_and_preserves_sources(self):
        before = tree_bytes(self.curated)
        with patch.object(freeze,'_publish',side_effect=OSError('synthetic publish failure')):
            self.fail('publish')
        self.assertEqual(tree_bytes(self.curated),before)

    def test_reference_coordinate_changes_during_freeze_are_detected(self):
        real_validate = freeze.validate_dataset
        def mutate_after_final_validation(path):
            result = real_validate(path)
            if Path(path) != self.curated:
                coordinates = self.reference / 'provenance/samples.jsonl'
                coordinates.write_text(' '+coordinates.read_text())
            return result
        with patch.object(freeze,'validate_dataset',side_effect=mutate_after_final_validation):
            self.fail('coordinate|changed')

    def test_reference_coordinate_change_since_acquisition_fails_before_pixels(self):
        original_fingerprint = fingerprint(self.reference)
        self.reference_centers[0]['approximate_center'] = [6.001,50.501]
        self.write_reference_centers()
        self.assertEqual(fingerprint(self.reference),original_fingerprint)
        self.assertNotEqual(self.reference_evidence(),self.candidate_report['references'])
        with patch.object(freeze,'validate_dataset',side_effect=AssertionError('Pixels read before acquisition coordinate binding')):
            self.fail('coordinate|acquisition|reference')

    def test_explicit_five_percent_protocol_accepts_exactly_5120_background_pixels(self):
        for path in self.review_paths:
            review = json.loads(path.read_text())
            for sample in review['samples']:
                if sample['id'] == 'deferred': continue
                right = {'accepted':15,'boundary':14,'small':9}[sample['id']]
                sample['background_polygons'] = [[[0,0],[right,0],[right,319],[0,319]]]
            write_json(path,review)
        shutil.rmtree(self.curated)
        curate_negative_collection([self.candidates],self.review_paths,self.curated,
                                   dataset_version='small-reviewed-2',allow_ai_review=True)
        original_protocol = self.protocol_path.read_bytes()
        self.fail('coverage|background|gate')
        v2_path = self.root / 'evaluation-v2.json'
        write_json(v2_path,{**self.protocol,'dataset_version':'frozen-negative-2',
                            'minimum_background_fraction':.05})
        report = self.call(protocol_path=v2_path)
        self.assertEqual(report['minimum_background_fraction'],.05)
        self.assertEqual(report['reviewed_evaluation_ids'],['accepted'])
        self.assertEqual(report['evaluation_samples'],1)
        decisions = {row['id']:row for row in report['decisions']}
        self.assertEqual(decisions['accepted']['background_pixels'],5120)
        self.assertTrue(decisions['accepted']['eligible'])
        self.assertEqual(decisions['boundary']['background_pixels'],4800)
        self.assertFalse(decisions['boundary']['eligible'])
        self.assertEqual(self.protocol_path.read_bytes(),original_protocol)

    def test_missing_or_duplicate_review_evidence_cannot_claim_two_passes(self):
        path = self.curated / 'curation-report.json'
        original = json.loads(path.read_text())
        variants = [[], [original['reviewers'][0]] * 2,
                    [{**original['reviewers'][0],'reviewer':''}, {**original['reviewers'][1],'reviewer':' '}],
                    [{**original['reviewers'][0],'reviewer':'Same'}, {**original['reviewers'][1],'reviewer':'same'}]]
        for reviewers in variants:
            with self.subTest(reviewers=reviewers):
                write_json(path,{**original,'reviewers':reviewers})
                self.fail('review|curat')
        write_json(path,original)
        (self.curated / original['reviewers'][0]['file']).unlink()
        self.fail('review|curat|evidence|file')

    def test_candidate_report_must_match_the_exact_acquisition_inventory(self):
        self.candidate_report['dataset_fingerprint'] = 'f'*64
        write_json(self.candidates / 'candidate-report.json',self.candidate_report)
        self.fail('candidate|acquisition|fingerprint')

    def test_false_coordinate_provenance_cannot_move_nearby_raster_away(self):
        # Both reported centres can be moved together without changing image
        # metadata. The affine grid remains the actual geographical anchor.
        from pyproj import Transformer
        inverse = Transformer.from_crs(25832,4326,always_xy=True)
        actual = self.candidate_centers[0]['approximate_center']
        self.reference_centers[0]['approximate_center'] = actual
        self.write_reference_centers()
        self.refresh_acquisition_reference_evidence()
        self.candidate_centers[0]['approximate_center'] = list(inverse.transform(450000,5800000))
        self.write_candidate_centers()
        self.fail('coordinate|centre|center|georefer|buffer|distance')


if __name__ == '__main__':
    unittest.main()
