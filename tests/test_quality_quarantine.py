# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic quality quarantine checks with inaccessible inherited raster bytes."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
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

from pvextent.cli import main
from pvextent.data.quality_quarantine import QualityQuarantineError, quarantine_training_quality
from pvextent.data.training import _read_manifest


SOURCE_FIELDS = {
    'name': 'Synthetic quality fixture', 'url': 'https://example.invalid/fixture',
    'version': '1', 'image_license': 'CC0-1.0', 'annotation_license': 'CC0-1.0',
    'attribution': 'Synthetic fixture; no third-party imagery.',
}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows), encoding='utf-8')


class QualityQuarantineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.base = self.root / 'base'
        self.source = self.root / 'source'
        self.output = self.root / 'output'
        self.protocol_path = self.root / 'protocol.json'
        self.evidence_path = self.root / 'findings.json'
        self.version = 'quality-fixture-v4'
        for directory in (self.base, self.source):
            for child in ('images', 'masks', 'provenance'):
                (directory / child).mkdir(parents=True)
        self.metadata(self.base, 'fixture-v2', ['fixture-base'])
        self.metadata(self.source, 'fixture-v3', ['fixture-base', 'ai-bulk-fixture'])
        self.base_rows = [self.row(key, split, 'fixture-base',
                                   'upstream' if split != 'unassigned' else 'unreviewed')
                          for key, split in (('base-train', 'train'), ('base-val', 'val'),
                                             ('base-test', 'test'), ('base-deferred', 'unassigned'))]
        for row in self.base_rows:
            self.unreadable_rasters(self.base, row)
            for role in ('image', 'mask'):
                os.link(self.base / row[role], self.source / row[role])
        self.new_rows = [self.row(key, split, 'ai-bulk-fixture', status)
                         for key, split, status in (('new-positive', 'train', 'reviewed'),
                                                    ('new-negative', 'train', 'reviewed'),
                                                    ('new-deferred', 'unassigned', 'unreviewed'))]
        for row in self.new_rows:
            self.unreadable_rasters(self.source, row)
        self.positive = self.new_rows[0]
        image = Image.new('L', (8, 8), 255)
        image.putdata([1] * 4 + [0] * 8 + [255] * 52)
        image.save(self.source / self.positive['mask'])
        self.positive['mask_sha256'] = digest((self.source / self.positive['mask']).read_bytes())
        self.source_rows = copy.deepcopy([*self.base_rows, *self.new_rows])
        self.coordinates = [
            {'id': row['id'], 'approximate_center': [7 + index * .01, 51], 'center_crs': 'EPSG:4326'}
            for index, row in enumerate(self.source_rows)
        ]
        write_rows(self.base / 'manifest.jsonl', self.base_rows)
        write_rows(self.source / 'manifest.jsonl', self.source_rows)
        write_rows(self.base / 'provenance/samples.jsonl', self.coordinates[:len(self.base_rows)])
        write_rows(self.source / 'provenance/samples.jsonl', self.coordinates)
        (self.source / 'provenance/parent').mkdir()
        (self.source / 'provenance/parent/frozen-coordinate-evidence.txt').write_text('Exact inherited coordinate evidence.\n')
        (self.source / 'NOTICE').write_text('Synthetic immutable source notice.\n')
        self.decisions = [
            {'id': 'new-positive', 'classification': 'positive', 'eligible': True,
             'positive_pixels': 4, 'background_pixels': 8, 'ignored_pixels': 52, 'split': 'train',
             'parent_training_id': 'base-train', 'kind': 'known', 'spatial_group': 'group-new-positive'},
            {'id': 'new-negative', 'classification': 'negative', 'eligible': True,
             'positive_pixels': 0, 'background_pixels': 16, 'ignored_pixels': 48, 'split': 'train',
             'parent_training_id': None, 'kind': 'geometry', 'spatial_group': 'group-new-negative'},
            {'id': 'new-deferred', 'classification': 'deferred', 'eligible': False,
             'positive_pixels': 0, 'background_pixels': 0, 'ignored_pixels': 64, 'split': 'unassigned',
             'parent_training_id': None, 'kind': 'geometry', 'spatial_group': 'group-new-deferred'},
        ]
        write_json(self.source / 'provenance/augmentation-decisions.json', self.decisions)
        self.source_fp = _read_manifest(self.source)[2]
        self.base_fp = _read_manifest(self.base)[2]
        self.parent_report = {
            'ok': True, 'purpose': 'experimental-ai-bulk-training',
            'dataset_fingerprint': self.source_fp, 'dataset_version': 'fixture-v3',
            'parent_dataset_fingerprint': self.base_fp,
            'records': len(self.source_rows), 'splits': dict(Counter(row['split'] for row in self.source_rows)),
            'curated_records': 3, 'qualified_records_added_to_train': 2,
            'positive_records_added_to_train': 1, 'negative_records_added_to_train': 1,
            'deferred_records_added_unassigned': 1,
            'previously_verified_base_supervised_train_records': 1,
            'effective_supervised_train_records': 3,
            'pv_pixels_added': 4, 'background_pixels_added': 24, 'ignored_pixels_added': 164,
            'all_inherited_records_unchanged': True, 'heldout_records_unchanged': True,
            'heldout_pixels_opened': False, 'protected_pixels_opened': False,
            'inherited_raster_paths_resolved': False,
            'model_predictions_used': False, 'human_review_claimed': False, 'training_run_allowed': False,
        }
        write_json(self.source / 'augmentation-report.json', self.parent_report)
        write_json(self.evidence_path, {
            'reviewer': 'Synthetic additional AI quality checker',
            'findings': [{'id': 'new-positive', 'decision': 'unsafe', 'reason': 'Synthetic uncertainty.'}],
            'human_review_claimed': False, 'model_predictions_used': False,
        })
        self.protocol = {
            'schema_version': '1.0', 'purpose': 'experimental-ai-training-quality-quarantine',
            'source_dataset_fingerprint': self.source_fp, 'base_dataset_fingerprint': self.base_fp,
            'dataset_version': self.version, 'reviewer': 'Synthetic additional AI quality checker',
            'reviewer_type': 'ai', 'evidence_sha256': digest(self.evidence_path.read_bytes()),
            'model_predictions_used': False, 'human_review_claimed': False, 'training_run_allowed': False,
            'exclusions': [{'id': 'new-positive', 'reason': 'Synthetic PV uncertainty; remove all supervision.'}],
        }
        self.save_protocol()

    @staticmethod
    def metadata(directory, version, sources):
        lines = ['schema_version = "1.0"', 'dataset_id = "pvextent-training-augmented"',
                 'dataset_version = ' + json.dumps(version)]
        for source_id in sources:
            lines.append('[sources.' + json.dumps(source_id) + ']')
            lines.extend(key + ' = ' + json.dumps(value) for key, value in SOURCE_FIELDS.items())
        (directory / 'dataset.toml').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    @staticmethod
    def row(key, split, source_id, status):
        return {'id': key, 'image': f'images/{key}.png', 'mask': f'masks/{key}.png',
                'image_sha256': '0' * 64, 'mask_sha256': '0' * 64, 'width': 8, 'height': 8,
                'gsd_m': .1, 'source_id': source_id, 'capture_date': '2026-01-01',
                'spatial_group': 'group-' + key, 'building_ids': ['building-' + key],
                'split': split, 'annotation_status': status,
                'georeference': {'crs': 'EPSG:25832', 'transform': [.1, 0, 392000, 0, -.1, 5700000]}}

    @staticmethod
    def unreadable_rasters(directory, row):
        for role in ('image', 'mask'):
            payload = ('Forbidden to decode synthetic ' + row['id'] + ':' + role).encode()
            (directory / row[role]).write_bytes(payload)
            row[role + '_sha256'] = digest(payload)

    def save_protocol(self):
        write_json(self.protocol_path, self.protocol)

    def quarantine(self):
        return quarantine_training_quality(self.source, self.base, self.output,
                                           protocol_path=self.protocol_path, evidence_path=self.evidence_path,
                                           dataset_version=self.version)

    @contextmanager
    def prohibit_unchanged_pixel_access(self):
        forbidden = {self.base / row[role] for row in self.base_rows for role in ('image', 'mask')}
        selected = {item['id'] for item in self.protocol['exclusions']}
        allowed_masks = {row['id'] for row in self.source_rows if row['id'] in selected
                         and row['source_id'].startswith('ai-bulk-')
                         and (row['split'], row['annotation_status']) == ('train', 'reviewed')}
        forbidden.update(self.source / row[role] for row in self.source_rows for role in ('image', 'mask')
                         if row['id'] not in allowed_masks or role != 'mask')
        protected_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in forbidden}
        original_open, original_resolve = Path.open, Path.resolve
        def aliases_forbidden(path):
            try:
                info = path.stat()
            except OSError:
                return False
            return (info.st_dev, info.st_ino) in protected_inodes
        def guarded_open(path, *args, **kwargs):
            if path in forbidden or aliases_forbidden(path):
                self.fail('Unchanged raster was opened: ' + str(path))
            return original_open(path, *args, **kwargs)
        def guarded_resolve(path, *args, **kwargs):
            if path in forbidden or aliases_forbidden(path):
                self.fail('Unchanged raster was resolved: ' + str(path))
            return original_resolve(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded_open), patch.object(Path, 'resolve', guarded_resolve):
            yield

    def test_exclusion_removes_all_supervision_and_preserves_every_other_record_and_inode(self):
        source_meta, original_rows, source_fp = _read_manifest(self.source)
        source_manifest = (self.source / 'manifest.jsonl').read_bytes()
        parent_report_bytes = (self.source / 'augmentation-report.json').read_bytes()
        source_coordinates = (self.source / 'provenance/samples.jsonl').read_bytes()
        with self.prohibit_unchanged_pixel_access():
            report = self.quarantine()
        metadata, rows, fingerprint = _read_manifest(self.output)
        self.assertEqual(metadata, {**source_meta, 'dataset_version': self.version})
        self.assertEqual(report['dataset_fingerprint'], fingerprint)
        self.assertEqual(report['source_dataset_fingerprint'], source_fp)
        self.assertEqual(report['base_dataset_fingerprint'], self.base_fp)
        self.assertEqual(report['records'], len(rows))
        self.assertEqual(report['splits'], {'train': 2, 'val': 1, 'test': 1, 'unassigned': 3})
        self.assertEqual(report['qualified_records_added_to_train'], 1)
        self.assertEqual(report['effective_supervised_train_records'], 2)
        self.assertEqual(report['positive_records_added_to_train'], 0)
        self.assertEqual(report['negative_records_added_to_train'], 1)
        self.assertEqual(report['deferred_records_added_unassigned'], 2)
        self.assertEqual((report['pv_pixels_added'], report['background_pixels_added'], report['ignored_pixels_added']),
                         (0, 16, 176))
        self.assertEqual(report['evidence_sha256'], self.protocol['evidence_sha256'])
        self.assertEqual(report['protocol_sha256'], digest(self.protocol_path.read_bytes()))
        for claim in ('model_predictions_used', 'human_review_claimed', 'training_run_allowed'):
            self.assertIs(report[claim], False)
        by_id = {row['id']: row for row in rows}
        for original in original_rows:
            actual = by_id[original['id']]
            if original['id'] == 'new-positive':
                self.assertEqual({key for key in original if actual[key] != original[key]},
                                 {'mask', 'mask_sha256', 'annotation_status', 'split'})
                self.assertEqual(actual['split'], 'unassigned')
                self.assertEqual(actual['annotation_status'], 'unreviewed')
                mask_path = self.output / actual['mask']
                self.assertEqual(digest(mask_path.read_bytes()), actual['mask_sha256'])
                with Image.open(mask_path) as mask:
                    self.assertEqual(mask.mode, 'L')
                    self.assertEqual(mask.size, (8, 8))
                    self.assertEqual(mask.getextrema(), (255, 255))
                self.assertFalse(os.path.samefile(self.source / original['mask'], mask_path))
                roles = ('image',)
            else:
                self.assertEqual(actual, original)
                roles = ('image', 'mask')
            for role in roles:
                self.assertTrue(os.path.samefile(self.source / original[role], self.output / actual[role]))
        self.assertEqual((self.output / 'provenance/samples.jsonl').read_bytes(), source_coordinates)
        self.assertEqual((self.output / 'provenance/parent/frozen-coordinate-evidence.txt').read_bytes(),
                         (self.source / 'provenance/parent/frozen-coordinate-evidence.txt').read_bytes())
        self.assertEqual((self.output / 'provenance/parent-augmentation-report.json').read_bytes(), parent_report_bytes)
        self.assertFalse((self.output / 'augmentation-report.json').exists())
        self.assertEqual(json.loads((self.output / 'quality-quarantine-report.json').read_bytes()), report)
        self.assertEqual((self.source / 'manifest.jsonl').read_bytes(), source_manifest)
        self.assertEqual(_read_manifest(self.source)[2], source_fp)

    def test_multiple_positive_and_negative_exclusions_remove_supervision_cumulatively(self):
        negative = self.source_rows[len(self.base_rows) + 1]
        mask = Image.new('L', (8, 8), 255)
        mask.putdata([0] * 16 + [255] * 48)
        mask.save(self.source / negative['mask'])
        negative['mask_sha256'] = digest((self.source / negative['mask']).read_bytes())
        write_rows(self.source / 'manifest.jsonl', self.source_rows)
        self.protocol['exclusions'].append({'id': 'new-negative', 'reason': 'Synthetic background uncertainty.'})
        self.rebind_changed_source_metadata()
        with self.prohibit_unchanged_pixel_access():
            report = self.quarantine()
        self.assertEqual(report['splits'], {'train': 1, 'val': 1, 'test': 1, 'unassigned': 4})
        self.assertEqual(report['qualified_records_added_to_train'], 0)
        self.assertEqual(report['effective_supervised_train_records'], 1)
        self.assertEqual(report['positive_records_added_to_train'], 0)
        self.assertEqual(report['negative_records_added_to_train'], 0)
        self.assertEqual(report['deferred_records_added_unassigned'], 3)
        self.assertEqual((report['pv_pixels_added'], report['background_pixels_added'], report['ignored_pixels_added']),
                         (0, 0, 192))
        rows = {row['id']: row for row in _read_manifest(self.output)[1]}
        for key in ('new-positive', 'new-negative'):
            self.assertEqual(rows[key]['split'], 'unassigned')
            self.assertEqual(rows[key]['annotation_status'], 'unreviewed')
            with Image.open(self.output / rows[key]['mask']) as current:
                self.assertEqual(current.getextrema(), (255, 255))
        self.assertEqual(rows['new-deferred'], self.source_rows[-1])

    def test_base_train_validation_and_test_cannot_be_excluded_without_pixel_access(self):
        for key in ('base-train', 'base-val', 'base-test', 'base-deferred'):
            with self.subTest(identifier=key):
                self.protocol['exclusions'] = [{'id': key, 'reason': 'Invalid frozen-base exclusion.'}]
                self.save_protocol()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_duplicate_foreign_and_deferred_ids_are_rejected(self):
        exclusions = [
            [self.protocol['exclusions'][0], self.protocol['exclusions'][0]],
            [{'id': 'foreign-id', 'reason': 'Unknown sample.'}],
            [{'id': 'new-deferred', 'reason': 'Already unassigned.'}],
        ]
        for values in exclusions:
            with self.subTest(exclusions=values):
                self.protocol['exclusions'] = values
                self.save_protocol()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_protocol_requires_exact_schema_ai_identity_false_claims_and_bounded_nonempty_exclusions(self):
        original = copy.deepcopy(self.protocol)
        mutations = [
            lambda value: value.update(extra='undeclared'),
            lambda value: value.update(schema_version='2.0'),
            lambda value: value.update(purpose='arbitrary'),
            lambda value: value.update(reviewer='unknown'),
            lambda value: value.update(reviewer_type='human'),
            lambda value: value.update(model_predictions_used=True),
            lambda value: value.update(human_review_claimed=True),
            lambda value: value.update(training_run_allowed=True),
            lambda value: value.update(exclusions=[]),
            lambda value: value.update(exclusions=[{'id': '', 'reason': 'Empty ID.'}]),
            lambda value: value.update(exclusions=[{'id': 'x' * 129, 'reason': 'Long ID.'}]),
            lambda value: value.update(exclusions=[{'id': 'new-positive', 'reason': ''}]),
            lambda value: value.update(exclusions=[{'id': 'new-positive', 'reason': 'x' * 8193}]),
            lambda value: value.update(exclusions=[{'id': 'new-positive', 'reason': 'Test', 'extra': True}]),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                self.protocol = copy.deepcopy(original)
                mutate(self.protocol)
                self.save_protocol()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_evidence_hash_drift_and_missing_evidence_are_rejected(self):
        original = self.evidence_path.read_bytes()
        self.evidence_path.write_bytes(original + b' ')
        with self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())
        self.evidence_path.unlink()
        with self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_source_base_fingerprint_drift_is_rejected(self):
        for field in ('source_dataset_fingerprint', 'base_dataset_fingerprint'):
            original = self.protocol[field]
            with self.subTest(field=field):
                self.protocol[field] = 'f' * 64
                self.save_protocol()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())
            self.protocol[field] = original

    def test_changed_base_record_is_rejected_even_when_source_fingerprint_is_rebound(self):
        self.source_rows[0]['capture_date'] = '2026-02-01'
        write_rows(self.source / 'manifest.jsonl', self.source_rows)
        fingerprint = _read_manifest(self.source)[2]
        self.protocol['source_dataset_fingerprint'] = fingerprint
        self.parent_report['dataset_fingerprint'] = fingerprint
        self.save_protocol()
        write_json(self.source / 'augmentation-report.json', self.parent_report)
        with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_selected_mask_hash_drift_is_rejected_without_opening_any_other_raster(self):
        with (self.source / self.positive['mask']).open('ab') as stream:
            stream.write(b'drift')
        with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_parent_report_fingerprint_counts_and_claims_must_be_bound(self):
        original = copy.deepcopy(self.parent_report)
        mutations = [
            lambda value: value.update(dataset_fingerprint='f' * 64),
            lambda value: value.update(parent_dataset_fingerprint='f' * 64),
            lambda value: value.update(effective_supervised_train_records=4),
            lambda value: value.update(qualified_records_added_to_train=3),
            lambda value: value.update(positive_records_added_to_train=2),
            lambda value: value.update(pv_pixels_added=-1),
            lambda value: value.update(training_run_allowed=True),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed = copy.deepcopy(original)
                mutate(changed)
                write_json(self.source / 'augmentation-report.json', changed)
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def rebind_changed_source_metadata(self):
        fingerprint = _read_manifest(self.source)[2]
        self.protocol['source_dataset_fingerprint'] = fingerprint
        self.parent_report['dataset_fingerprint'] = fingerprint
        self.save_protocol()
        write_json(self.source / 'augmentation-report.json', self.parent_report)

    def test_excluded_mask_cannot_alias_any_inherited_or_unchanged_raster_inode(self):
        selected = self.source / self.positive['mask']
        aliases = [(self.base, self.base_rows[0], 'mask'), (self.base, self.base_rows[1], 'image'),
                   (self.source, self.new_rows[1], 'mask'), (self.source, self.positive, 'image')]
        for directory, row, role in aliases:
            with self.subTest(alias=row['id'] + ':' + role):
                selected.unlink()
                os.link(directory / row[role], selected)
                self.source_rows[len(self.base_rows)]['mask_sha256'] = row[role + '_sha256']
                write_rows(self.source / 'manifest.jsonl', self.source_rows)
                self.rebind_changed_source_metadata()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_excluded_mask_cannot_reuse_a_frozen_base_raster_path(self):
        self.source_rows[len(self.base_rows)].update(mask=self.base_rows[0]['mask'],
                                                   mask_sha256=self.base_rows[0]['mask_sha256'])
        write_rows(self.source / 'manifest.jsonl', self.source_rows)
        self.rebind_changed_source_metadata()
        with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_source_cannot_change_frozen_base_source_declarations(self):
        path = self.source / 'dataset.toml'
        path.write_text(path.read_text().replace('version = "1"', 'version = "changed"', 1))
        self.rebind_changed_source_metadata()
        with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_selected_mask_mode_dimensions_labels_and_decision_counts_must_match(self):
        variants = [('RGB', (8, 8), (0, 0, 0)), ('L', (9, 8), 1), ('L', (8, 8), 2),
                    ('L', (8, 8), 255), ('L', (8, 8), 1)]
        for mode, size, value in variants:
            with self.subTest(mode=mode, size=size, value=value):
                Image.new(mode, size, value).save(self.source / self.positive['mask'])
                self.source_rows[len(self.base_rows)]['mask_sha256'] = digest(
                    (self.source / self.positive['mask']).read_bytes())
                write_rows(self.source / 'manifest.jsonl', self.source_rows)
                self.rebind_changed_source_metadata()
                with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_duplicate_protocol_and_evidence_json_keys_are_rejected(self):
        self.protocol_path.write_text(self.protocol_path.read_text().replace(
            '"schema_version": "1.0"', '"schema_version": "1.0", "schema_version": "1.0"'))
        with self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())
        self.evidence_path.write_bytes(b'{"findings": [], "findings": []}\n')
        self.protocol['evidence_sha256'] = digest(self.evidence_path.read_bytes())
        self.save_protocol()
        with self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertFalse(self.output.exists())

    def test_evidence_must_be_a_strict_json_object_even_when_its_hash_is_bound(self):
        for payload in (b'[]', b'{"nonfinite": NaN}', b'{"broken":'):
            with self.subTest(payload=payload):
                self.evidence_path.write_bytes(payload)
                self.protocol['evidence_sha256'] = digest(payload)
                self.save_protocol()
                with self.assertRaises(QualityQuarantineError):
                    self.quarantine()
                self.assertFalse(self.output.exists())

    def test_output_cannot_overwrite_an_existing_directory(self):
        self.output.mkdir()
        marker = self.output / 'keep.txt'
        marker.write_bytes(b'Existing output remains untouched.')
        with self.prohibit_unchanged_pixel_access(), self.assertRaises(QualityQuarantineError):
            self.quarantine()
        self.assertEqual(marker.read_bytes(), b'Existing output remains untouched.')
        self.assertEqual(list(self.output.iterdir()), [marker])

    def test_cli_accepts_frozen_inputs_and_emits_current_json_report(self):
        stream = io.StringIO()
        with self.prohibit_unchanged_pixel_access(), patch('sys.stdout', stream):
            code = main(['data', 'quarantine-training-quality', str(self.source), str(self.base), str(self.output),
                         '--protocol', str(self.protocol_path), '--evidence', str(self.evidence_path),
                         '--dataset-version', self.version, '--json'])
        self.assertEqual(code, 0)
        report = json.loads(stream.getvalue())
        self.assertTrue(report['ok'])
        self.assertEqual(report['effective_supervised_train_records'], 2)


if __name__ == '__main__':
    unittest.main()
