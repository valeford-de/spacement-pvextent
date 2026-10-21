# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic conservative consensus and metadata-only inherited holdout guards."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    from pyproj import Transformer
    from pvextent.data.bulk_curation import (
        BulkCurationError, MIN_REFERENCE_DISTANCE_M, augment_bulk_training, curate_bulk_training, export_bulk_mask_proposals,
    )
    from pvextent.data.review_queue import export_training_review_queue
    from pvextent.data.training import _read_manifest
except ImportError:
    Transformer = None


SOURCE_FIELDS = {'name': 'Synthetic fixture', 'url': 'https://example.org/fixture', 'version': '1',
                 'image_license': 'CC0-1.0', 'annotation_license': 'CC0-1.0',
                 'attribution': 'Synthetic pixels and metadata only'}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def write_rows(path, values):
    path.write_text(''.join(json.dumps(v, sort_keys=True) + '\n' for v in values))


@unittest.skipIf(Transformer is None, 'Install train and geo extras for bulk-curation tests')
class BulkCurationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base, self.source, self.pilot = [self.root / p for p in ('base', 'source', 'pilot')]
        self.queue, self.reviews, self.curated, self.augmented = [self.root / p for p in ('queue', 'reviews', 'curated', 'augmented')]
        self.protocol_path = self.root / 'protocol.json'
        self.inverse = Transformer.from_crs(25832, 4326, always_xy=True)
        self.points = {'parent': (392000, 5700000), 'base-val': (400000, 5720000),
                       'base-test': (402000, 5720000), 'base-pending': (404000, 5720000),
                       'pilot': (406000, 5720000), 'new-positive': (392000, 5700000),
                       'new-negative': (392200, 5700000), 'new-disagree': (392400, 5700000),
                       'new-empty': (392600, 5700000)}
        for directory, dataset_id, source_id in ((self.base, 'fixture-base', 'pv4ger-seg'),
                                                  (self.source, 'nrw-training-bulk', 'nrw-dop'),
                                                  (self.pilot, 'fixture-pilot', 'pv4ger-seg')):
            for name in ('images', 'masks', 'provenance'):
                (directory / name).mkdir(parents=True)
            self.metadata(directory, dataset_id, source_id)
        self.base_rows = [self.row(key, split=split, source_id='pv4ger-seg', side=32)
                          for key, split in (('parent', 'train'), ('base-val', 'val'), ('base-test', 'test'), ('base-pending', 'unassigned'))]
        for row in self.base_rows:
            self.pixels(self.base, row, inherited=row['split'] != 'train')
        self.pilot_rows = [self.row('pilot', split='test', source_id='pv4ger-seg', side=32)]
        self.pixels(self.pilot, self.pilot_rows[0], inherited=True)
        self.source_rows = [self.row(key) for key in ('new-positive', 'new-negative', 'new-disagree', 'new-empty')]
        self.source_rows[0]['building_ids'] = ['building:parent']
        for row in self.source_rows:
            self.pixels(self.source, row)
        for directory, rows in ((self.base, self.base_rows), (self.source, self.source_rows), (self.pilot, self.pilot_rows)):
            write_rows(directory / 'manifest.jsonl', rows)
            write_rows(directory / 'provenance/samples.jsonl', [self.coordinate(r['id']) for r in rows])
        base_fp = _read_manifest(self.base)[2]
        pilot_fp = _read_manifest(self.pilot)[2]
        source_fp = _read_manifest(self.source)[2]
        references = [{'dataset_fingerprint': _read_manifest(directory)[2],
                       'coordinate_sha256': digest((directory / 'provenance/samples.jsonl').read_bytes()),
                       'role': role} for directory, role in ((self.base, 'training'), (self.pilot, 'protected'))]
        write_json(self.source / 'provenance/protocol.json', {'synthetic': True})
        acquisition_protocol_hash = digest((self.source / 'provenance/protocol.json').read_bytes())
        binding = {'schema_version': '1.0', 'protocol_sha256': acquisition_protocol_hash, 'references': references}
        write_json(self.source / 'provenance/reference-binding.json', binding)
        self.candidates = [{**{'id': r['id'], 'kind': 'known' if i == 0 else 'geometry',
                              'parent_training_id': 'parent' if i == 0 else None,
                              'center_epsg25832': list(self.points[r['id']]), 'building_ids': r['building_ids']}}
                           for i, r in enumerate(self.source_rows)]
        plan = {'binding': binding, 'known': self.candidates[:1], 'geometry': self.candidates[1:]}
        write_json(self.source / 'provenance/selection-plan.json', plan)
        accepted = [{'id': c['id'], 'candidate': c, 'accepted': True} for c in self.candidates]
        write_json(self.source / 'provenance/acquisition-decisions.json', accepted)
        write_json(self.source / 'candidate-report.json', {
            'ok': True, 'complete': True, 'dataset_id': 'nrw-training-bulk', 'dataset_fingerprint': source_fp,
            'protocol_sha256': acquisition_protocol_hash,
            'selection_plan_sha256': digest((self.source / 'provenance/selection-plan.json').read_bytes()),
            'model_predictions_used': False, 'original_masks_transferred': False, 'references': references})
        (self.source / 'provenance/NOTICE').write_text('Preserved synthetic source notice\n')
        (self.source / 'provenance/responses').mkdir()
        (self.source / 'provenance/responses/example.bin').write_bytes(b'synthetic raw acquisition response')
        export_training_review_queue(self.source, self.queue, batch_size=2)
        self.make_reviews()
        self.protocol = {'schema_version': '1.0', 'review_schema_version': '1.1',
            'purpose': 'experimental-ai-bulk-training', 'dataset_version': 'training-fixture-v3',
            'curated_dataset_version': 'curated-fixture-v1', 'source_dataset_fingerprint': source_fp,
            'queue_sha256': digest((self.queue / 'queue.json').read_bytes()), 'base_dataset_fingerprint': base_fp,
            'protected_dataset_fingerprints': [pilot_fp], 'minimum_positive_pixels': 100,
            'minimum_positive_background_pixels': 256, 'minimum_negative_background_pixels': 1024,
            'min_reference_distance_m': MIN_REFERENCE_DISTANCE_M, 'min_candidate_distance_m': 80,
            'model_predictions_used': False, 'human_review_claimed': False, 'training_run_allowed': False,
            'base_supervised_train_records': 1}
        write_json(self.protocol_path, self.protocol)

    def metadata(self, directory, dataset_id, source_id):
        lines = ['schema_version = "1.0"', 'dataset_id = ' + json.dumps(dataset_id), 'dataset_version = "fixture-1"',
                 '[sources.' + json.dumps(source_id) + ']']
        lines.extend(k + ' = ' + json.dumps(v) for k, v in SOURCE_FIELDS.items())
        (directory / 'dataset.toml').write_text('\n'.join(lines) + '\n')

    def row(self, key, *, split='unassigned', source_id='nrw-dop', side=320):
        x, y = self.points[key]
        return {'id': key, 'image': f'images/{key}.png', 'mask': f'masks/{key}.png',
                'image_sha256': '0' * 64, 'mask_sha256': '0' * 64, 'width': side, 'height': side,
                'gsd_m': .1, 'source_id': source_id, 'capture_date': '2026-01-01',
                'spatial_group': 'group-' + key, 'building_ids': ['building:' + key], 'split': split,
                'annotation_status': 'upstream' if split != 'unassigned' else 'unreviewed',
                'georeference': {'crs': 'EPSG:25832', 'transform': [.1, 0, x - side * .05, 0, -.1, y + side * .05]}}

    def pixels(self, directory, row, inherited=False):
        if inherited:
            for role in ('image', 'mask'):
                (directory / row[role]).write_bytes(('Undecodable reserved ' + row['id'] + role).encode())
        else:
            colour = tuple(hashlib.sha256(row['id'].encode()).digest()[:3])
            Image.new('RGB', (row['width'], row['height']), colour).save(directory / row['image'])
            Image.new('L', (row['width'], row['height']), 255 if row['split'] == 'unassigned' else 0).save(directory / row['mask'])
        for role in ('image', 'mask'):
            row[role + '_sha256'] = digest((directory / row[role]).read_bytes())

    def coordinate(self, key):
        return {'id': key, 'approximate_center': list(self.inverse.transform(*self.points[key])), 'center_crs': 'EPSG:4326'}

    @staticmethod
    def rectangle(left, top, right, bottom):
        return [[left, top], [right, top], [right, bottom], [left, bottom]]

    def make_reviews(self):
        for lane in ('a', 'b'):
            (self.reviews / f'lane-{lane}').mkdir(parents=True, exist_ok=True)
            for path in (self.queue / f'lane-{lane}').glob('*.review.json'):
                review = json.loads(path.read_bytes())
                review.update(review_schema_version='1.1', reviewer=f'Codex synthetic independent visual AI role {lane.upper()}')
                for sample in review['samples']:
                    sample.update(full_image_inspected=True, notes='Synthetic fixture review.')
                    if sample['id'] in ('new-positive', 'new-empty'):
                        sample.update(decision='approve-positive',
                                      positive_polygons=[] if sample['id'] == 'new-empty' else [self.rectangle(10 if lane == 'a' else 11, 10, 20, 19)],
                                      background_polygons=[self.rectangle(0, 30, 15, 45)])
                    elif sample['id'] == 'new-negative':
                        sample.update(decision='approve-negative', positive_polygons=[],
                                      background_polygons=[self.rectangle(0, 0, 31, 31)])
                    else:
                        sample.update(decision='defer' if lane == 'b' else 'approve-negative',
                                      background_polygons=[self.rectangle(0, 0, 319, 319)])
                write_json(self.reviews / f'lane-{lane}' / path.name, review)

    def change_review(self, key, mutate, lane='a'):
        for path in (self.reviews / ('lane-' + lane)).glob('*.review.json'):
            review = json.loads(path.read_bytes())
            for sample in review['samples']:
                if sample['id'] == key:
                    mutate(sample)
                    write_json(path, review)
                    return
        self.fail('Missing synthetic sample')

    def curate(self, **kwargs):
        return curate_bulk_training(self.source, self.queue, self.reviews, self.curated,
            dataset_version='curated-fixture-v1', protocol_path=self.protocol_path, allow_ai_review=True, **kwargs)

    def augment(self):
        return augment_bulk_training(self.base, self.curated, [self.pilot], self.augmented,
                                     dataset_version='training-fixture-v3', protocol_path=self.protocol_path)

    def test_consensus_partial_counts_and_immutable_provenance(self):
        queue_before = {p: p.read_bytes() for p in self.queue.rglob('*') if p.is_file()}
        original_images = {r['id']: (self.source / r['image']).read_bytes() for r in self.source_rows}
        report = self.curate()
        self.assertEqual((report['positive_records'], report['negative_records'], report['deferred_records']), (1, 1, 2))
        self.assertEqual((report['pv_pixels'], report['background_pixels']), (100, 1280))
        self.assertEqual(report['ignored_pixels'], 4 * 320 * 320 - 1380)
        rows = _read_manifest(self.curated)[1]
        for row in rows:
            self.assertEqual(row['split'], 'unassigned')
            self.assertEqual((self.curated / row['image']).read_bytes(), original_images[row['id']])
            with Image.open(self.curated / row['mask']) as mask:
                histogram = mask.histogram()
                self.assertEqual(sum(histogram), 320 * 320)
                self.assertTrue(all(not c or label in (0, 1, 255) for label, c in enumerate(histogram)))
                if row['id'] in ('new-disagree', 'new-empty'):
                    self.assertEqual(histogram[255], 320 * 320)
                    self.assertEqual(row['annotation_status'], 'unreviewed')
        self.assertEqual(queue_before, {p: p.read_bytes() for p in queue_before})
        self.assertEqual((self.curated / 'provenance/acquisition/provenance/responses/example.bin').read_bytes(), b'synthetic raw acquisition response')
        with self.assertRaisesRegex(BulkCurationError, 'already exists'):
            self.curate()

    def test_augmentation_never_opens_or_resolves_any_inherited_or_pilot_raster(self):
        self.curate()
        watched = {self.base / r[role] for r in self.base_rows for role in ('image', 'mask')}
        watched |= {self.pilot / r[role] for r in self.pilot_rows for role in ('image', 'mask')}
        original_open, original_resolve = Path.open, Path.resolve
        def guarded_open(path, *args, **kwargs):
            if path in watched:
                self.fail('Inherited or pilot raster was opened')
            return original_open(path, *args, **kwargs)
        def guarded_resolve(path, *args, **kwargs):
            if path in watched:
                self.fail('Inherited or pilot raster was resolved')
            return original_resolve(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open), patch.object(Path, 'resolve', guarded_resolve):
            report = self.augment()
        self.assertEqual(report['splits'], {'train': 3, 'val': 1, 'test': 1, 'unassigned': 3})
        self.assertEqual(report['effective_supervised_train_records'], 3)
        self.assertFalse(report['heldout_pixels_opened'])
        _, merged, _ = _read_manifest(self.augmented)
        for original in self.base_rows:
            self.assertEqual(next(r for r in merged if r['id'] == original['id']), original)
            for role in ('image', 'mask'):
                self.assertTrue(os.path.samefile(self.base / original[role], self.augmented / original[role]))
        known = next(r for r in merged if r['id'] == 'new-positive')
        self.assertEqual(known['spatial_group'], self.base_rows[0]['spatial_group'])
        with self.assertRaisesRegex(BulkCurationError, 'already exists'):
            self.augment()

    def test_explicit_ai_authorization_required(self):
        with self.assertRaisesRegex(BulkCurationError, 'authorization'):
            curate_bulk_training(self.source, self.queue, self.reviews, self.curated,
                                 dataset_version='curated-fixture-v1', protocol_path=self.protocol_path)

    def test_review_rejects_hash_inspection_overlap_negative_pv_and_duplicates(self):
        mutations = [lambda s: s.update(image_sha256='f' * 64), lambda s: s.update(full_image_inspected=False),
                     lambda s: s.update(background_polygons=[self.rectangle(10, 10, 20, 20)]),
                     lambda s: s.update(decision='approve-negative')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.make_reviews()
                self.change_review('new-positive', mutate)
                with self.assertRaises(BulkCurationError):
                    self.curate()
                self.assertFalse(self.curated.exists())
        self.make_reviews()
        path = next((self.reviews / 'lane-a').glob('*.review.json'))
        review = json.loads(path.read_bytes())
        review['samples'][1] = copy.deepcopy(review['samples'][0])
        write_json(path, review)
        with self.assertRaises(BulkCurationError):
            self.curate()

    def test_queue_template_and_batch_bytes_are_bound(self):
        path = next((self.queue / 'lane-a').glob('*.review.json'))
        path.write_bytes(path.read_bytes() + b' ')
        with self.assertRaisesRegex(BulkCurationError, 'bytes changed'):
            self.curate()
        self.assertFalse(self.curated.exists())

    def test_independent_lane_roles_and_placeholders(self):
        for kind in ('same', 'mixed', 'placeholder'):
            self.make_reviews()
            for number, path in enumerate(sorted((self.reviews / 'lane-b').glob('*.review.json'))):
                review = json.loads(path.read_bytes())
                if kind == 'same':
                    review['reviewer'] = 'Codex synthetic independent visual AI role A'
                elif kind == 'mixed' and number == 1:
                    review['reviewer'] = 'Codex other independent visual AI role C'
                elif kind == 'placeholder':
                    review['reviewer'] = 'PENDING-REVIEWER-B'
                write_json(path, review)
            with self.subTest(kind=kind), self.assertRaises(BulkCurationError):
                self.curate()

    def test_protocol_rejects_predictions_unsafe_paths_and_symlinks(self):
        self.protocol['model_predictions_used'] = True
        write_json(self.protocol_path, self.protocol)
        with self.assertRaises(BulkCurationError):
            self.curate()
        self.protocol['model_predictions_used'] = False
        write_json(self.protocol_path, self.protocol)
        with self.assertRaisesRegex(BulkCurationError, 'ancestors'):
            curate_bulk_training(self.source, self.queue, self.reviews, self.source / 'bad-output',
                dataset_version='curated-fixture-v1', protocol_path=self.protocol_path, allow_ai_review=True)
        path = next((self.reviews / 'lane-a').glob('*.review.json'))
        other = self.root / 'linked-review.json'
        path.rename(other)
        path.symlink_to(other)
        with self.assertRaisesRegex(BulkCurationError, 'symbolic'):
            self.curate()

    def test_metadata_hash_mismatch_rejected_with_missing_holdout_rasters(self):
        self.curate()
        for row in self.base_rows:
            if row['split'] != 'train':
                for role in ('image', 'mask'):
                    (self.base / row[role]).unlink()
        self.protocol['base_dataset_fingerprint'] = 'f' * 64
        write_json(self.protocol_path, self.protocol)
        with self.assertRaisesRegex(BulkCurationError, 'Frozen base/source'):
            self.augment()
        self.assertFalse(self.augmented.exists())

    def test_tampered_curated_mask_cannot_be_released(self):
        self.curate()
        rows = _read_manifest(self.curated)[1]
        row = next(r for r in rows if r['id'] == 'new-negative')
        Image.new('L', (320, 320), 0).save(self.curated / row['mask'])
        row['mask_sha256'] = digest((self.curated / row['mask']).read_bytes())
        write_rows(self.curated / 'manifest.jsonl', rows)
        report_path = self.curated / 'curation-report.json'
        report = json.loads(report_path.read_bytes())
        report['dataset_fingerprint'] = _read_manifest(self.curated)[2]
        write_json(report_path, report)
        with self.assertRaisesRegex(BulkCurationError, 'conservative independent'):
            self.augment()
        self.assertFalse(self.augmented.exists())

    def test_source_after_check_and_no_replace_publication(self):
        from pvextent.data import bulk_curation
        original_publish = bulk_curation._publish
        def raced_publish(stage, output):
            output.mkdir()
            (output / 'winner').write_bytes(b'concurrent writer')
            return original_publish(stage, output)
        with patch.object(bulk_curation, '_publish', raced_publish), self.assertRaises(BulkCurationError):
            self.curate()
        self.assertEqual((self.curated / 'winner').read_bytes(), b'concurrent writer')
        self.assertEqual(list(self.root.glob('.curated-bulk-curation-*')), [])


    def add_base_train(self, key, point, *, buildings=None):
        self.points[key] = point
        row = self.row(key, split='train', source_id='pv4ger-seg', side=32)
        if buildings is not None:
            row['building_ids'] = buildings
        self.pixels(self.base, row)
        self.base_rows.append(row)
        write_rows(self.base / 'manifest.jsonl', self.base_rows)
        write_rows(self.base / 'provenance/samples.jsonl', [self.coordinate(r['id']) for r in self.base_rows])
        self.protocol['base_dataset_fingerprint'] = _read_manifest(self.base)[2]
        self.protocol['base_supervised_train_records'] += 1
        write_json(self.protocol_path, self.protocol)
        binding_path = self.source / 'provenance/reference-binding.json'
        binding = json.loads(binding_path.read_bytes())
        binding['references'][0].update(dataset_fingerprint=self.protocol['base_dataset_fingerprint'],
            coordinate_sha256=digest((self.base / 'provenance/samples.jsonl').read_bytes()))
        write_json(binding_path, binding)
        plan_path = self.source / 'provenance/selection-plan.json'
        plan = json.loads(plan_path.read_bytes())
        plan['binding'] = binding
        write_json(plan_path, plan)
        report_path = self.source / 'candidate-report.json'
        report = json.loads(report_path.read_bytes())
        report.update(references=binding['references'], selection_plan_sha256=digest(plan_path.read_bytes()))
        write_json(report_path, report)

    def test_false_known_train_parent_coordinate_rejected(self):
        self.add_base_train('parent-far', (393000, 5700000), buildings=['building:parent'])
        self.curate()
        retained = self.curated / 'provenance/acquisition'
        plan_path = retained / 'provenance/selection-plan.json'
        plan = json.loads(plan_path.read_bytes())
        plan['known'][0]['parent_training_id'] = 'parent-far'
        write_json(plan_path, plan)
        decisions_path = retained / 'provenance/acquisition-decisions.json'
        decisions = json.loads(decisions_path.read_bytes())
        next(r for r in decisions if r.get('id') == 'new-positive')['candidate'] = plan['known'][0]
        write_json(decisions_path, decisions)
        report_path = retained / 'candidate-report.json'
        report = json.loads(report_path.read_bytes())
        report['selection_plan_sha256'] = digest(plan_path.read_bytes())
        write_json(report_path, report)
        with self.assertRaisesRegex(BulkCurationError, 'Known TRAIN parent coordinate'):
            self.augment()
        self.assertFalse(self.augmented.exists())

    def test_geometry_spacing_rechecked_against_every_base_train_center(self):
        self.add_base_train('parent-near', (392210, 5700000))
        self.curate()
        with self.assertRaisesRegex(BulkCurationError, 'spacing against a base TRAIN'):
            self.augment()
        self.assertFalse(self.augmented.exists())

    def test_full_negative_is_explicit_and_empty_negative_stays_ignored(self):
        for lane in ('a', 'b'):
            self.change_review('new-negative', lambda sample: sample.update(
                background_polygons=[self.rectangle(0, 0, 319, 319)]), lane)
            self.change_review('new-empty', lambda sample: sample.update(
                decision='approve-negative', positive_polygons=[], background_polygons=[]), lane)
        report = self.curate()
        self.assertEqual(report['background_pixels'], 102400 + 256)
        rows = _read_manifest(self.curated)[1]
        with Image.open(self.curated / next(r for r in rows if r['id'] == 'new-negative')['mask']) as mask:
            self.assertEqual(mask.histogram()[0], 102400)
        with Image.open(self.curated / next(r for r in rows if r['id'] == 'new-empty')['mask']) as mask:
            self.assertEqual(mask.histogram()[255], 102400)

    def test_source_change_after_validation_prevents_publication(self):
        from pvextent.data import bulk_curation
        original_validate = bulk_curation.validate_dataset
        def changed_after_validation(directory):
            result = original_validate(directory)
            path = self.queue / 'queue.json'
            path.write_bytes(path.read_bytes() + b' ')
            return result
        with patch.object(bulk_curation, 'validate_dataset', changed_after_validation):
            with self.assertRaisesRegex(BulkCurationError, 'evidence changed'):
                self.curate()
        self.assertFalse(self.curated.exists())
        self.assertEqual(list(self.root.glob('.curated-bulk-curation-*')), [])


    def enable_v2(self):
        self.package = self.root / 'checks-v2'
        self.protocol.update(schema_version='2.0', review_schema_version='1.2', proposal_schema_version='1.0',
                             proposal_author='Codex synthetic independent visual AI role A',
                             review_method='two-independent-checks-of-shared-ai-visual-proposals')
        write_json(self.protocol_path, self.protocol)

    def export_proposals(self):
        return export_bulk_mask_proposals(self.source, self.queue, self.reviews / 'lane-a', self.package,
                                          protocol_path=self.protocol_path)

    def test_v2_incremental_proposals_are_immutable_and_prediction_blind(self):
        self.enable_v2()
        second = self.reviews / 'lane-a/batch-0002.review.json'
        withheld = second.with_suffix('.pending')
        second.rename(withheld)
        watched = {self.source / row[role] for row in self.source_rows for role in ('image', 'mask')}
        original_open = Path.open
        def no_rasters(path, *args, **kwargs):
            if path in watched:
                self.fail('Proposal export opened a source raster')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', no_rasters):
            first = self.export_proposals()
        self.assertEqual((first['ready_batches'], first['total_batches'], first['complete']), (1, 2, False))
        self.assertFalse((self.package / 'proposals/index.json').exists())
        proposal_path = self.package / 'proposals/batch-0001.proposal.json'
        fixed = (proposal_path.read_bytes(), proposal_path.stat().st_mtime_ns)
        proposal = json.loads(fixed[0])
        self.assertEqual(set(proposal), {'proposal_schema_version', 'batch_id', 'dataset_fingerprint',
                                        'queue_manifest_sha256', 'draft_sha256', 'samples'})
        for sample in proposal['samples']:
            self.assertEqual(set(sample), {'id', 'image_sha256', 'positive_polygons', 'background_polygons'})
        withheld.rename(second)
        complete = self.export_proposals()
        self.assertTrue(complete['complete'])
        self.assertEqual(complete['ready_samples'], 4)
        self.assertEqual((proposal_path.read_bytes(), proposal_path.stat().st_mtime_ns), fixed)
        self.assertTrue((self.package / 'proposals/index.json').is_file())
        self.assertEqual(self.export_proposals(), complete)
        proposal_path.write_bytes(fixed[0] + b' ')
        with self.assertRaisesRegex(BulkCurationError, 'immutable proposal-package bytes differ'):
            self.export_proposals()


    def make_checks(self):
        for lane, role in (('a', 'C'), ('b', 'D')):
            for path in sorted((self.package / 'proposals').glob('*.proposal.json')):
                proposal = json.loads(path.read_bytes())
                review = {'review_schema_version': '1.2', 'reviewer': f'Codex synthetic independent checker {role}',
                          'reviewer_type': 'ai', 'model_predictions_used': False,
                          'proposal_sha256': digest(path.read_bytes()), 'samples': []}
                for sample in proposal['samples']:
                    key = sample['id']
                    decision = ('approve-positive' if key in ('new-positive', 'new-empty') else
                                'defer' if key == 'new-disagree' and lane == 'b' else 'approve-negative')
                    review['samples'].append({'id': key, 'image_sha256': sample['image_sha256'],
                        'full_image_inspected': True, 'proposal_overlay_inspected': True, 'decision': decision,
                        'approved_positive_polygons': list(range(len(sample['positive_polygons']))) if decision == 'approve-positive' else [],
                        'approved_background_polygons': list(range(len(sample['background_polygons']))) if decision != 'defer' else [],
                        'notes': 'Synthetic full-image and frozen-region check.'})
                write_json(self.package / ('lane-' + lane) / (proposal['batch_id'] + '.review.json'), review)

    def change_check(self, key, mutate, lane='a', *, top=None):
        for path in (self.package / ('lane-' + lane)).glob('*.review.json'):
            check = json.loads(path.read_bytes())
            if top is not None:
                top(check)
                write_json(path, check)
                continue
            for sample in check['samples']:
                if sample['id'] == key:
                    mutate(sample)
                    write_json(path, check)
                    return

    def curate_v2(self):
        return curate_bulk_training(self.source, self.queue, self.package, self.curated,
            dataset_version='curated-fixture-v1', protocol_path=self.protocol_path, allow_ai_review=True)

    def test_v2_common_proposal_checks_and_augmentation(self):
        self.enable_v2()
        self.export_proposals()
        self.make_checks()
        report = self.curate_v2()
        self.assertEqual((report['positive_records'], report['negative_records'], report['deferred_records']), (1, 1, 2))
        self.assertEqual((report['pv_pixels'], report['background_pixels']), (110, 1280))
        self.assertIn('correlated proposal omissions', report['shared_proposal_bias'])
        self.assertIn('Two independent checks of common AI visual proposals', (self.curated / 'dataset.toml').read_text())
        self.assertEqual((self.curated / 'provenance/completed-reviews/source-drafts/batch-0001.review.json').read_bytes(),
                         (self.reviews / 'lane-a/batch-0001.review.json').read_bytes())
        watched = {self.base / r[role] for r in self.base_rows for role in ('image', 'mask')}
        watched |= {self.pilot / r[role] for r in self.pilot_rows for role in ('image', 'mask')}
        original_open, original_resolve = Path.open, Path.resolve
        def guarded_open(path, *args, **kwargs):
            if path in watched:
                self.fail('V2 inherited or pilot raster was opened')
            return original_open(path, *args, **kwargs)
        def guarded_resolve(path, *args, **kwargs):
            if path in watched:
                self.fail('V2 inherited or pilot raster was resolved')
            return original_resolve(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open), patch.object(Path, 'resolve', guarded_resolve):
            augmented = self.augment()
        self.assertEqual(augmented['effective_supervised_train_records'], 3)
        self.assertEqual(augmented['pv_pixels_added'], 110)
        self.assertEqual(augmented['proposal_index_sha256'], report['proposal_index_sha256'])
        self.assertEqual(augmented['review_method'], self.protocol['review_method'])

    def test_v2_requires_joint_whole_polygon_indices_not_incidental_pixel_overlap(self):
        self.enable_v2()
        self.change_review('new-positive', lambda s: s.update(positive_polygons=[
            self.rectangle(10, 10, 29, 29), self.rectangle(15, 10, 34, 29)]))
        self.export_proposals()
        self.make_checks()
        self.change_check('new-positive', lambda s: s.update(approved_positive_polygons=[0]), 'a')
        self.change_check('new-positive', lambda s: s.update(approved_positive_polygons=[1]), 'b')
        report = self.curate_v2()
        self.assertEqual(report['positive_records'], 0)
        decision = next(d for d in report['decisions'] if d['id'] == 'new-positive')
        self.assertEqual(decision['approved_positive_polygons'], [])
        self.assertEqual(decision['positive_pixels'], 0)
        row = next(r for r in _read_manifest(self.curated)[1] if r['id'] == 'new-positive')
        with Image.open(self.curated / row['mask']) as mask:
            self.assertEqual(mask.histogram()[255], 102400)

    def test_v2_checker_rejects_bad_indices_false_inspection_predictions_and_roles(self):
        self.enable_v2()
        self.export_proposals()
        mutations = [lambda s: s.update(approved_positive_polygons=[True]),
                     lambda s: s.update(approved_positive_polygons=[1]),
                     lambda s: s.update(approved_positive_polygons=[0, 0]),
                     lambda s: s.update(full_image_inspected=False),
                     lambda s: s.update(proposal_overlay_inspected=False),
                     lambda s: s.update(decision='defer'),
                     lambda s: s.update(decision='approve-negative')]
        for mutate in mutations:
            self.make_checks()
            self.change_check('new-positive', mutate)
            with self.subTest(mutate=mutate), self.assertRaises(BulkCurationError):
                self.curate_v2()
            self.assertFalse(self.curated.exists())
        top_mutations = [lambda r: r.update(model_predictions_used=True),
                         lambda r: r.update(proposal_sha256='f' * 64),
                         lambda r: r.update(reviewer=self.protocol['proposal_author']),
                         lambda r: r.update(review_schema_version='1.1')]
        for mutate in top_mutations:
            self.make_checks()
            self.change_check(None, None, top=mutate)
            with self.subTest(top_mutate=mutate), self.assertRaises(BulkCurationError):
                self.curate_v2()
        self.make_checks()
        self.change_check(None, None, lane='b', top=lambda r: r.update(reviewer='Codex synthetic independent checker C'))
        with self.assertRaisesRegex(BulkCurationError, 'distinct independent'):
            self.curate_v2()

    def test_v2_deferred_drafts_export_no_regions_and_complete_index_is_required(self):
        self.enable_v2()
        self.change_review('new-positive', lambda s: s.update(decision='defer'))
        self.export_proposals()
        self.make_checks()
        index_path = self.package / 'proposals/index.json'
        index_bytes = index_path.read_bytes()
        index_path.unlink()
        with self.assertRaises(BulkCurationError):
            self.curate_v2()
        index_path.write_bytes(index_bytes)
        proposals = [json.loads(path.read_bytes()) for path in (self.package / 'proposals').glob('*.proposal.json')]
        proposed = next(s for p in proposals for s in p['samples'] if s['id'] == 'new-positive')
        self.assertEqual(proposed['positive_polygons'], [])
        self.assertEqual(proposed['background_polygons'], [])
        report = self.curate_v2()
        self.assertEqual(report['positive_records'], 0)
        self.assertEqual(next(d for d in report['decisions'] if d['id'] == 'new-positive')['ignored_pixels'], 102400)

    def test_v2_rejects_unbound_drafts_and_changed_proposal_bytes(self):
        self.enable_v2()
        self.export_proposals()
        self.make_checks()
        extra = self.package / 'source-drafts/unindexed.review.json'
        extra.write_bytes(b'{}')
        with self.assertRaisesRegex(BulkCurationError, 'unbound or mixed'):
            self.curate_v2()
        extra.unlink()
        proposal_path = next((self.package / 'proposals').glob('*.proposal.json'))
        proposal_path.write_bytes(proposal_path.read_bytes() + b' ')
        with self.assertRaisesRegex(BulkCurationError, 'proposal bytes changed'):
            self.curate_v2()

    def test_v2_augmentation_rederives_checker_approved_mask(self):
        self.enable_v2()
        self.export_proposals()
        self.make_checks()
        self.curate_v2()
        rows = _read_manifest(self.curated)[1]
        row = next(r for r in rows if r['id'] == 'new-positive')
        Image.new('L', (320, 320), 1).save(self.curated / row['mask'])
        row['mask_sha256'] = digest((self.curated / row['mask']).read_bytes())
        write_rows(self.curated / 'manifest.jsonl', rows)
        report_path = self.curated / 'curation-report.json'
        report = json.loads(report_path.read_bytes())
        report['dataset_fingerprint'] = _read_manifest(self.curated)[2]
        write_json(report_path, report)
        with self.assertRaisesRegex(BulkCurationError, 'conservative independent'):
            self.augment()
        self.assertFalse(self.augmented.exists())


if __name__ == '__main__':
    unittest.main()
