# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Synthetic metadata-only queue checks; no imagery or masks are created."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvextent.data import review_queue as queue
from pvextent.data.pv4ger import DatasetImportError


METADATA = '''schema_version = "1.0"
dataset_id = "nrw-training-bulk"
dataset_version = "1"
[sources.hidden_origin]
name = "SECRET acquisition origin"
url = "https://example.org/synthetic"
version = "1"
image_license = "CC0-1.0"
annotation_license = "not-annotated-ignore-placeholder"
attribution = "SECRET known-PV seed or geometry provenance"
'''


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def tree_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Install train extras')
class ReviewQueueTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source, self.output = self.root / 'source', self.root / 'queue'
        self.source.mkdir()
        (self.source / 'dataset.toml').write_text(METADATA, encoding='utf-8')
        self.rows = [self.row(index) for index in range(53)]
        self.refresh()

    def row(self, index):
        identifier = f'nrw-bulk-{index:012x}'
        return {
            'id': identifier, 'image': f'images/{identifier}.png',
            'mask': f'masks/{identifier}.png',
            'image_sha256': digest(f'image-{index}'.encode()),
            'mask_sha256': digest(f'mask-{index}'.encode()),
            'width': 320, 'height': 320, 'gsd_m': .1, 'capture_date': None,
            'source_id': 'hidden_origin', 'spatial_group': f'SECRET-group-{index}',
            'building_ids': [f'SECRET-building-{index}'],
            'split': 'unassigned', 'annotation_status': 'unreviewed',
            'georeference': {'crs': 'EPSG:25832', 'transform': [.1, 0, 1, 0, -.1, 2]},
        }

    def refresh(self):
        (self.source / 'manifest.jsonl').write_text(''.join(
            json.dumps(row, sort_keys=True) + '\n' for row in self.rows), encoding='utf-8')

    def call(self, **kwargs):
        return queue.export_training_review_queue(self.source, self.output, **kwargs)

    def rejects(self, pattern='.', **kwargs):
        with self.assertRaisesRegex(queue.TrainingReviewQueueError, pattern):
            self.call(**kwargs)
        self.assertFalse(self.output.exists())
        self.assertFalse(list(self.root.glob('.queue-review-queue-*')))

    def test_deterministic_shuffle_exact_coverage_and_bound_batch_bytes(self):
        before = tree_bytes(self.source)
        report = self.call()
        self.assertEqual(report['sample_count'], 53)
        self.assertEqual(report['batch_count_per_lane'], 3)
        self.assertEqual(report['review_template_count'], 6)
        self.assertTrue(report['pending_only'])
        self.assertFalse(report['raster_paths_accessed'])
        self.assertFalse(report['labels_created'])
        self.assertFalse(report['canonical_dataset_modified'])
        expected_ids = {row['id'] for row in self.rows}
        fingerprint = digest((self.source / 'dataset.toml').read_bytes() + b'\0'
                             + (self.source / 'manifest.jsonl').read_bytes())
        self.assertEqual(report['dataset_fingerprint'], fingerprint)
        index = json.loads((self.output / 'queue.json').read_bytes())
        self.assertEqual(index, {key: value for key, value in report.items() if key != 'output_dir'})
        for lane in index['lanes']:
            ids = [identifier for batch in lane['batches'] for identifier in batch['sample_ids']]
            self.assertEqual(len(ids), len(expected_ids))
            self.assertEqual(set(ids), expected_ids)
            self.assertNotEqual(ids, sorted(ids))
            self.assertEqual([len(batch['sample_ids']) for batch in lane['batches']], [25, 25, 3])
            for batch in lane['batches']:
                for file_key, sha_key in (('manifest', 'manifest_sha256'),
                                           ('review_template', 'review_template_sha256')):
                    self.assertEqual(digest((self.output / batch[file_key]).read_bytes()), batch[sha_key])
        second = self.root / 'second'
        queue.export_training_review_queue(self.source, second)
        self.assertEqual(tree_bytes(self.output), tree_bytes(second))
        third = self.root / 'third'
        changed = queue.export_training_review_queue(self.source, third, seed=123)
        self.assertNotEqual(index['lanes'][0]['batches'][0]['sample_ids'],
                            changed['lanes'][0]['batches'][0]['sample_ids'])
        self.assertEqual(tree_bytes(self.source), before)

    def test_manifests_and_templates_are_blinded_allowlists(self):
        report = self.call()
        reviewers = []
        for lane in report['lanes']:
            for batch in lane['batches']:
                manifest = json.loads((self.output / batch['manifest']).read_bytes())
                self.assertEqual(manifest['dataset_fingerprint'], report['dataset_fingerprint'])
                self.assertEqual(set(manifest), {'schema_version', 'dataset_fingerprint', 'batch_id', 'lane', 'samples'})
                for sample in manifest['samples']:
                    self.assertEqual(set(sample), {'id', 'image_sha256', 'image', 'width', 'height', 'gsd_m', 'capture_date'})
                    self.assertTrue(Path(sample['image']).is_absolute())
                    self.assertEqual(sample['image'], str(self.source / 'images' / f"{sample['id']}.png"))
                review = json.loads((self.output / batch['review_template']).read_bytes())
                self.assertEqual(set(review), {'review_schema_version', 'reviewer', 'reviewer_type', 'model_predictions_used', 'samples'})
                self.assertEqual(review['review_schema_version'], '1.0')
                self.assertEqual(review['reviewer_type'], 'ai')
                self.assertIs(review['model_predictions_used'], False)
                reviewers.append(review['reviewer'])
                self.assertEqual([sample['id'] for sample in review['samples']], batch['sample_ids'])
                for sample in review['samples']:
                    self.assertEqual(set(sample), {'id', 'image_sha256', 'full_image_inspected', 'decision',
                                                  'positive_polygons', 'background_polygons', 'notes'})
                    self.assertIs(sample['full_image_inspected'], False)
                    self.assertEqual(sample['decision'], 'defer')
                    self.assertEqual(sample['positive_polygons'], [])
                    self.assertEqual(sample['background_polygons'], [])
                    self.assertTrue(sample['notes'].startswith('Pending'))
        self.assertEqual(len(reviewers), len(set(reviewers)))
        for relative, payload in tree_bytes(self.output).items():
            if relative.endswith('.json'):
                self.assertNotIn(b'SECRET', payload)
                self.assertNotIn(b'hidden_origin', payload)
                self.assertNotIn(b'seed', payload)
                self.assertNotIn(b'georeference', payload)
                self.assertNotIn(b'mask_sha256', payload)

    def test_unfilled_template_fails_real_review_parser_before_rasterizing(self):
        from pvextent.data.positive_holdout import PositiveHoldoutError, _review
        report = self.call(batch_size=1)
        batch = report['lanes'][0]['batches'][0]
        originals = {row['id']: row for row in self.rows if row['id'] in batch['sample_ids']}
        with patch('PIL.Image.new', side_effect=AssertionError('must reject before polygon rasterization')):
            with self.assertRaisesRegex(PositiveHoldoutError, 'full-image inspection'):
                _review((self.output / batch['review_template']).read_bytes(), originals)

    def test_export_never_resolves_opens_stats_or_decodes_raster_paths(self):
        from pvextent.data import training
        original_resolve, original_open, original_stat = Path.resolve, Path.open, Path.stat

        def guard(path):
            if 'images' in path.parts or 'masks' in path.parts:
                raise AssertionError('raster path touched')

        def resolve(path, *args, **kwargs):
            guard(path)
            return original_resolve(path, *args, **kwargs)

        def open_path(path, *args, **kwargs):
            guard(path)
            return original_open(path, *args, **kwargs)

        def stat(path, *args, **kwargs):
            guard(path)
            return original_stat(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve), patch.object(Path, 'open', open_path), \
                patch.object(Path, 'stat', stat), \
                patch.object(training, '_raster', side_effect=AssertionError('no pixels')), \
                patch('PIL.Image.open', side_effect=AssertionError('no decode')), \
                patch('pvextent.data.validation.validate_dataset', side_effect=AssertionError('no full validation')):
            self.assertEqual(self.call()['sample_count'], 53)
        self.assertFalse((self.source / 'images').exists())
        self.assertFalse((self.source / 'masks').exists())

    def test_batch_and_seed_bounds(self):
        for value in (0, 51, True, 1.5, '25'):
            with self.subTest(batch_size=value): self.rejects('batch_size', batch_size=value)
        for value in (-1, 2**63, True, 1.5, '0'):
            with self.subTest(seed=value): self.rejects('seed', seed=value)
        report = self.call(batch_size=50, seed=0)
        self.assertEqual(report['batch_count_per_lane'], 2)

    def test_wrong_dataset_identifier(self):
        (self.source / 'dataset.toml').write_text(METADATA.replace('nrw-training-bulk', 'another-dataset'))
        self.rejects('nrw-training-bulk')

    def test_rejects_nonpending_or_non_native_rows(self):
        for key, value in (('split', 'train'), ('annotation_status', 'reviewed'),
                           ('width', 321), ('height', 321), ('gsd_m', .2)):
            with self.subTest(field=key):
                original = copy.deepcopy(self.rows)
                self.rows[0][key] = value
                self.refresh()
                self.rejects('unassigned|unreviewed|320x320|split')
                self.rows = original
                self.refresh()

    def test_record_count_bounds(self):
        self.rows = []
        self.refresh()
        self.rejects('empty')
        self.rows = [self.row(index) for index in range(2001)]
        self.refresh()
        self.rejects('1..2000')

    def test_maximum_size_is_bounded_and_readme_explains_negative_routing(self):
        self.rows = [self.row(index) for index in range(2000)]
        self.refresh()
        report = self.call(batch_size=50)
        self.assertEqual(report['sample_count'], 2000)
        self.assertEqual(report['batch_count_per_lane'], 40)
        self.assertEqual(report['review_template_count'], 80)
        readme = (self.output / 'README.md').read_text()
        self.assertIn('separate existing negative curation workflow', readme)
        self.assertIn('Empty positive polygons never establish PV absence', readme)
        self.assertIn('original test partition', readme)

    def test_manifest_contract_rejects_duplicate_id_and_unsafe_raster_name(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.refresh()
        self.rejects('duplicate')
        self.rows.pop()
        self.rows[0]['image'] = '../secret.png'
        self.refresh()
        self.rejects('unsafe')

    def test_metadata_and_manifest_file_bounds(self):
        for name, limit in (('dataset.toml', queue.MAX_QUEUE_METADATA_BYTES),
                            ('manifest.jsonl', queue.MAX_QUEUE_MANIFEST_BYTES)):
            with self.subTest(file=name):
                path = self.source / name
                original = path.read_bytes()
                path.write_bytes(b' ' * (limit + 1))
                self.rejects('bounded')
                path.write_bytes(original)

    def test_existing_output_is_never_replaced(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('keep')
        with self.assertRaisesRegex(queue.TrainingReviewQueueError, 'already exists'):
            self.call()
        self.assertEqual(sentinel.read_text(), 'keep')
        self.assertFalse(list(self.root.glob('.queue-review-queue-*')))

    def test_refuses_directory_and_metadata_symlinks(self):
        source_alias = self.root / 'source-alias'
        source_alias.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(queue.TrainingReviewQueueError, 'symlink'):
            queue.export_training_review_queue(source_alias, self.output)
        self.output.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(queue.TrainingReviewQueueError, 'symlink'):
            self.call()
        self.output.unlink()
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(queue.TrainingReviewQueueError, 'symlink'):
            queue.export_training_review_queue(self.source, alias / 'queue')
        metadata = self.source / 'dataset.toml'
        backup = self.source / 'original.toml'
        metadata.rename(backup)
        metadata.symlink_to(backup)
        self.rejects('symlink')

    def test_refuses_same_or_ancestor_output(self):
        for output in (self.source, self.source / 'queue', self.root):
            with self.subTest(output=output):
                with self.assertRaisesRegex(queue.TrainingReviewQueueError, 'overlap'):
                    queue.export_training_review_queue(self.source, output)

    def test_source_change_aborts_and_removes_staging(self):
        real_write = queue._write_json
        mutated = False

        def change_source(path, document):
            nonlocal mutated
            result = real_write(path, document)
            if not mutated:
                mutated = True
                self.rows[0]['capture_date'] = '2026-01-01'
                self.refresh()
            return result

        with patch.object(queue, '_write_json', change_source):
            self.rejects('changed')

    def test_concurrent_output_creation_is_not_replaced_and_stage_is_removed(self):
        real_publish = queue._publish

        def concurrent_publish(stage, output):
            output.mkdir()
            (output / 'keep.txt').write_text('concurrent')
            real_publish(stage, output)

        with patch.object(queue, '_publish', concurrent_publish):
            with self.assertRaises(queue.TrainingReviewQueueError):
                self.call()
        self.assertEqual((self.output / 'keep.txt').read_text(), 'concurrent')
        self.assertFalse(list(self.root.glob('.queue-review-queue-*')))

    def test_error_is_dataset_import_error(self):
        self.assertTrue(issubclass(queue.TrainingReviewQueueError, DatasetImportError))


if __name__ == '__main__':
    unittest.main()
