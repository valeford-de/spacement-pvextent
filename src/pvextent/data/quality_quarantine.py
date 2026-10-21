# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Remove explicitly disputed new bulk supervision in an immutable snapshot.

The frozen base and every nonexcluded record are metadata-only inputs. Only
explicitly excluded new TRAIN masks are read; imagery is never opened. A
quarantine replaces the entire mask with ignore, never with background.
"""

from __future__ import annotations

from collections import Counter
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import warnings

from PIL import Image

from .bulk_curation import (
    BulkCurationError, MAX_JSON_BYTES, MAX_PROVENANCE_BYTES, PURPOSE as AUGMENTATION_PURPOSE,
    _Evidence, _absolute, _coordinates, _digest, _json_bytes, _jsonl,
    _link_inherited, _manifest, _no_symlinks, _object, _paths, _relative, _version,
)
from .pv4ger import _publish
from .validation import IDENTIFIER, MAX_FILE_BYTES, SHA256

PURPOSE = 'experimental-ai-training-quality-quarantine'
PROTOCOL_FIELDS = {
    'schema_version', 'purpose', 'source_dataset_fingerprint', 'base_dataset_fingerprint',
    'dataset_version', 'reviewer', 'reviewer_type', 'evidence_sha256',
    'model_predictions_used', 'human_review_claimed', 'training_run_allowed', 'exclusions',
}
MAX_EXCLUSIONS = 2000


class QualityQuarantineError(BulkCurationError):
    """A quality exclusion is not bound to the frozen experimental snapshot."""


class _QualityEvidence(_Evidence):
    """Keep the inode guard active for every read, including the drift recheck."""

    def __init__(self):
        super().__init__()
        self.protected_inodes = set()

    def read(self, path, maximum=MAX_JSON_BYTES):
        if self.protected_inodes:
            path = _absolute(path)
            _no_symlinks(path)
            info = path.stat()
            if (info.st_dev, info.st_ino) in self.protected_inodes:
                raise QualityQuarantineError('Read-only evidence aliases a protected raster.')
        return super().read(path, maximum)


def _protocol(path, evidence):
    payload = evidence.read(path, 1024 * 1024)
    value = _object(payload)
    reviewer = value.get('reviewer')
    if (set(value) != PROTOCOL_FIELDS or value.get('schema_version') != '1.0'
            or value.get('purpose') != PURPOSE or value.get('reviewer_type') != 'ai'
            or any(value.get(key) is not False for key in
                   ('model_predictions_used', 'human_review_claimed', 'training_run_allowed'))
            or not isinstance(reviewer, str) or not 8 <= len(reviewer.strip()) or len(reviewer) > 256
            or re.search(r'pending|placeholder|unknown|template|\btodo\b', reviewer, re.I)):
        raise QualityQuarantineError('Expected the exact quality-quarantine protocol and an identified AI reviewer.')
    _version(value['dataset_version'])
    for name in ('source_dataset_fingerprint', 'base_dataset_fingerprint', 'evidence_sha256'):
        if not isinstance(value[name], str) or not SHA256.fullmatch(value[name]):
            raise QualityQuarantineError('Protocol fingerprint/hash is invalid.')
    exclusions = value['exclusions']
    if not isinstance(exclusions, list) or not 1 <= len(exclusions) <= MAX_EXCLUSIONS:
        raise QualityQuarantineError('Provide a bounded nonempty list of explicit exclusions.')
    seen = set()
    for item in exclusions:
        if not isinstance(item, dict) or set(item) != {'id', 'reason'}:
            raise QualityQuarantineError('Each exclusion requires exactly an ID and reason.')
        key, reason = item['id'], item['reason']
        if (not isinstance(key, str) or not IDENTIFIER.fullmatch(key) or key in seen
                or not isinstance(reason, str) or not reason.strip() or len(reason) > 8192):
            raise QualityQuarantineError('Exclusion IDs must be valid and unique, with bounded nonempty reasons.')
        seen.add(key)
    return value, payload


def _metadata(metadata, version):
    lines = ['schema_version = "1.0"', 'dataset_id = ' + json.dumps(metadata['dataset_id']),
             'dataset_version = ' + json.dumps(version)]
    for source_id, fields in metadata['sources'].items():
        lines.append('\n[sources.' + json.dumps(source_id) + ']')
        lines.extend(name + ' = ' + json.dumps(value) for name, value in fields.items())
    return ('\n'.join(lines) + '\n').encode()


def _count(value, key):
    result = value.get(key)
    if type(result) is not int or result < 0:
        raise QualityQuarantineError(f'Parent augmentation count is invalid: {key}.')
    return result


def _parent_report(source, rows, base_rows, source_fp, base_fp, source_meta, evidence):
    payload = evidence.read(source / 'augmentation-report.json')
    value = _object(payload)
    base_ids = {r['id'] for r in base_rows}
    added = [r for r in rows if r['id'] not in base_ids]
    qualified = sum(r['split'] == 'train' for r in added)
    supervised = sum(r['split'] == 'train' and r['annotation_status'] in ('upstream', 'reviewed')
                     for r in base_rows)
    if (value.get('ok') is not True or value.get('purpose') != AUGMENTATION_PURPOSE
            or value.get('dataset_fingerprint') != source_fp
            or value.get('parent_dataset_fingerprint') != base_fp
            or value.get('dataset_version') != source_meta['dataset_version']
            or any(value.get(key) is not False for key in
                   ('model_predictions_used', 'human_review_claimed', 'training_run_allowed'))
            or not added or len(added) > MAX_EXCLUSIONS
            or any(not r['source_id'].startswith('ai-bulk-')
                   or (r['split'], r['annotation_status']) not in
                   (('train', 'reviewed'), ('unassigned', 'unreviewed')) for r in added)):
        raise QualityQuarantineError('Source lacks a bound experimental bulk augmentation report.')
    expected = {'records': len(rows), 'curated_records': len(added),
                'qualified_records_added_to_train': qualified,
                'deferred_records_added_unassigned': len(added) - qualified,
                'previously_verified_base_supervised_train_records': supervised,
                'effective_supervised_train_records': supervised + qualified}
    if (any(_count(value, key) != count for key, count in expected.items())
            or not isinstance(value.get('splits'), dict)
            or any(type(count) is not int for count in value['splits'].values())
            or value['splits'] != dict(Counter(r['split'] for r in rows))
            or _count(value, 'positive_records_added_to_train')
            + _count(value, 'negative_records_added_to_train') != qualified
            or sum(_count(value, key) for key in
                   ('pv_pixels_added', 'background_pixels_added', 'ignored_pixels_added'))
            != sum(r['width'] * r['height'] for r in added)):
        raise QualityQuarantineError('Parent augmentation counts disagree with frozen source metadata.')
    return value, payload, added


def _guard_mask_inodes(source, base, rows, base_rows, exclusions):
    """Reject aliases to every protected raster before reading a selected mask."""
    protected = set()
    for directory, declarations in ((base, base_rows), (source, rows)):
        for row in declarations:
            for role in ('image', 'mask'):
                if directory == source and role == 'mask' and row['id'] in exclusions:
                    continue
                path = _relative(directory, row[role])
                _no_symlinks(path)
                info = path.stat()
                if not stat.S_ISREG(info.st_mode):
                    raise QualityQuarantineError('Inherited raster paths must identify regular files.')
                protected.add((info.st_dev, info.st_ino))
    for row in rows:
        if row['id'] not in exclusions:
            continue
        path = _relative(source, row['mask'])
        _no_symlinks(path)
        info = path.stat()
        if not stat.S_ISREG(info.st_mode) or (info.st_dev, info.st_ino) in protected:
            raise QualityQuarantineError('Excluded mask aliases an inherited or otherwise protected raster.')
    return protected


def _mask(source, row, evidence):
    """Read only one explicitly excluded mask; its image is never touched."""
    payload = evidence.read(_relative(source, row['mask']), MAX_FILE_BYTES)
    if _digest(payload) != row['mask_sha256']:
        raise QualityQuarantineError(f"{row['id']}: excluded mask hash mismatch.")
    if len(payload) < 33 or payload[12:16] != b'IHDR' or payload[24] != 8:
        raise QualityQuarantineError('Excluded masks must use encoded 8-bit PNG channels.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload), formats=['PNG']) as mask:
                if (mask.mode != 'L' or mask.size != (row['width'], row['height'])
                        or getattr(mask, 'n_frames', 1) != 1 or mask.getexif().get(274, 1) != 1):
                    raise QualityQuarantineError('Excluded mask mode, dimensions, frames or orientation differ.')
                mask.load()
                counts = mask.histogram()
                if (any(count and label not in (0, 1, 255) for label, count in enumerate(counts))
                        or counts[0] + counts[1] == 0):
                    raise QualityQuarantineError('Excluded TRAIN mask must contain valid supervised labels.')
                return {'positive_pixels': counts[1], 'background_pixels': counts[0],
                        'ignored_pixels': counts[255]}
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        if isinstance(error, QualityQuarantineError):
            raise
        raise QualityQuarantineError('Excluded mask cannot be decoded safely.') from error


def _decisions(source, rows, evidence):
    payload = evidence.read(source / 'provenance/augmentation-decisions.json')
    # Use the same strict JSON parser as every other input, including unique keys.
    from .validation import _invalid_constant, _unique_object
    values = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(values, list) or len(values) != len(rows):
        raise QualityQuarantineError('Parent augmentation decisions must cover exactly the new records.')
    by_id = {}
    for item in values:
        if not isinstance(item, dict) or item.get('id') in by_id or item.get('id') not in rows:
            raise QualityQuarantineError('Parent augmentation decision IDs are invalid or duplicated.')
        by_id[item['id']] = item
    return by_id


def _link_provenance(source, destination):
    """Retain parent provenance bytes by inode, without inspecting file contents."""
    _no_symlinks(source)
    if not source.is_dir():
        raise QualityQuarantineError('Source provenance directory is required.')
    destination.mkdir()
    total = 0
    for path in sorted(source.rglob('*')):
        _no_symlinks(path)
        info = path.stat()
        target = destination / path.relative_to(source)
        if stat.S_ISDIR(info.st_mode):
            target.mkdir(exist_ok=True)
            continue
        total += info.st_size
        if not stat.S_ISREG(info.st_mode) or total > MAX_PROVENANCE_BYTES:
            raise QualityQuarantineError('Parent provenance must contain bounded regular files.')
        target.parent.mkdir(parents=True, exist_ok=True)
        os.link(path, target, follow_symlinks=False)
        if not os.path.samefile(path, target):
            raise QualityQuarantineError('Parent provenance hard-link verification failed.')


def quarantine_training_quality(source_dir: Path | str, base_dir: Path | str,
                                output_dir: Path | str, *, protocol_path: Path | str,
                                evidence_path: Path | str, dataset_version: str) -> dict:
    """Quarantine explicit new bulk TRAIN records, preserving every other dict."""
    stage = None
    try:
        from .training import _read_manifest
        _version(dataset_version)
        inputs, output = _paths([source_dir, base_dir], output_dir)
        source, base = inputs
        protocol_at, findings_at = _absolute(protocol_path), _absolute(evidence_path)
        if any(path.suffix.lower() != '.json' or path.is_relative_to(output)
               for path in (protocol_at, findings_at)):
            raise QualityQuarantineError('Protocol and findings must be existing JSON files outside the output.')
        evidence = _QualityEvidence()
        protocol, protocol_bytes = _protocol(protocol_at, evidence)
        findings_bytes = evidence.read(findings_at)
        _object(findings_bytes)  # Findings are data only; referenced media is never followed.
        if _digest(findings_bytes) != protocol['evidence_sha256']:
            raise QualityQuarantineError('Quality evidence SHA256 differs from the frozen protocol.')
        metadata, rows, source_fp = _manifest(source, evidence)
        base_meta, base_rows, base_fp = _manifest(base, evidence)
        if (source_fp != protocol['source_dataset_fingerprint']
                or base_fp != protocol['base_dataset_fingerprint']
                or dataset_version != protocol['dataset_version']
                or dataset_version == metadata['dataset_version']):
            raise QualityQuarantineError('Source/base fingerprint or new dataset version differs from the frozen protocol.')
        originals = {r['id']: r for r in rows}
        base_ids = {r['id'] for r in base_rows}
        if (any(originals.get(r['id']) != r for r in base_rows)
                or any(metadata['sources'].get(key) != value for key, value in base_meta['sources'].items())):
            raise QualityQuarantineError('Every frozen base record and source declaration must remain exactly unchanged.')
        exclusions = {item['id']: item for item in protocol['exclusions']}
        for key in exclusions:
            row = originals.get(key)
            if (key in base_ids or row is None or not row['source_id'].startswith('ai-bulk-')
                    or row['split'] != 'train' or row['annotation_status'] != 'reviewed'):
                raise QualityQuarantineError('Only explicitly named new reviewed ai-bulk TRAIN records may be excluded; frozen base IDs are protected.')
        evidence.protected_inodes = _guard_mask_inodes(source, base, rows, base_rows, exclusions)
        parent, parent_bytes, added = _parent_report(source, rows, base_rows, source_fp, base_fp, metadata, evidence)
        decisions = _decisions(source, {r['id']: r for r in added}, evidence)
        _, coordinate_hash, _ = _coordinates(source, rows, evidence)
        removed, replacements, excluded_details = Counter(), {}, []
        used_paths = {r[role] for r in rows for role in ('image', 'mask')}
        for key, item in exclusions.items():
            row = originals[key]
            counts = _mask(source, row, evidence)
            classification = 'positive' if counts['positive_pixels'] else 'negative'
            decision = decisions[key]
            if (any(type(decision.get(name)) is not int or decision[name] != value
                    for name, value in counts.items())
                    or decision.get('classification') != classification or decision.get('eligible') is not True
                    or decision.get('split') != 'train'):
                raise QualityQuarantineError('Excluded mask counts differ from retained augmentation decisions.')
            removed.update(counts)
            removed[classification + '_records'] += 1
            buffer = io.BytesIO()
            Image.new('L', (row['width'], row['height']), 255).save(buffer, format='PNG')
            payload = buffer.getvalue()
            relative = f'masks/quality-quarantine/{_digest(key.encode())}.png'
            if relative in used_paths:
                raise QualityQuarantineError('New quarantine mask path collides with an inherited raster.')
            used_paths.add(relative)
            replacement = {**row, 'mask': relative, 'mask_sha256': _digest(payload),
                           'split': 'unassigned', 'annotation_status': 'unreviewed'}
            replacements[key] = (replacement, payload)
            excluded_details.append({**item, **counts, 'original_mask_sha256': row['mask_sha256'],
                                     'quarantined_mask_sha256': replacement['mask_sha256'],
                                     'classification_before_quarantine': classification})
        current = [replacements[r['id']][0] if r['id'] in replacements else r for r in rows]
        pixel_counts = {'pv_pixels_added': parent['pv_pixels_added'] - removed['positive_pixels'],
                        'background_pixels_added': parent['background_pixels_added'] - removed['background_pixels'],
                        'ignored_pixels_added': parent['ignored_pixels_added'] + removed['positive_pixels'] + removed['background_pixels']}
        if (min(pixel_counts.values()) < 0
                or parent['positive_records_added_to_train'] < removed['positive_records']
                or parent['negative_records_added_to_train'] < removed['negative_records']):
            raise QualityQuarantineError('Excluded counts exceed the bound parent augmentation totals.')
        output.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(output.parent)
        stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-quality-quarantine-', dir=output.parent))
        (stage / 'dataset.toml').write_bytes(_metadata(metadata, dataset_version))
        (stage / 'manifest.jsonl').write_bytes(_jsonl(current))
        checked_meta, checked, fingerprint = _read_manifest(stage)
        if checked != current or checked_meta != {**metadata, 'dataset_version': dataset_version}:
            raise QualityQuarantineError('Complete output metadata differs from the authorized replacements.')
        for row in rows:
            _link_inherited(source, row, 'image', stage, row['image'])
            if row['id'] not in replacements:
                _link_inherited(source, row, 'mask', stage, row['mask'])
            else:
                replacement, payload = replacements[row['id']]
                target = _relative(stage, replacement['mask'])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        provenance = stage / 'provenance'
        _link_provenance(source / 'provenance', provenance)
        retained = provenance / 'quality-quarantine'
        retained.mkdir()  # Refuse collisions with inherited provenance.
        (retained / 'protocol.json').write_bytes(protocol_bytes)
        (retained / 'findings.json').write_bytes(findings_bytes)
        for directory, prefix in ((source, 'source'), (base, 'base')):
            for name in ('dataset.toml', 'manifest.jsonl'):
                (retained / (prefix + '-' + name)).write_bytes(evidence.read(directory / name))
        parent_report_at = provenance / 'parent-augmentation-report.json'
        if os.path.lexists(parent_report_at):
            raise QualityQuarantineError('Parent augmentation-report provenance path already exists.')
        parent_report_at.write_bytes(parent_bytes)
        for name in ('LICENSE', 'NOTICE'):
            path = source / name
            if os.path.lexists(path):
                _no_symlinks(path)
                os.link(path, stage / name, follow_symlinks=False)
        report = {
            'ok': True, 'purpose': PURPOSE, 'schema_version': '1.0', 'dataset_version': dataset_version,
            'dataset_fingerprint': fingerprint, 'records': len(current),
            'splits': dict(Counter(r['split'] for r in current)),
            'source_dataset_fingerprint': source_fp, 'base_dataset_fingerprint': base_fp,
            'parent_dataset_fingerprint': source_fp, 'protocol_sha256': _digest(protocol_bytes),
            'evidence_sha256': _digest(findings_bytes), 'parent_augmentation_report_sha256': _digest(parent_bytes),
            'source_coordinates_sha256': coordinate_hash, 'excluded_ids': list(exclusions),
            'excluded_records': len(exclusions), 'exclusions': excluded_details,
            'curated_records': len(added),
            'qualified_records_added_to_train': parent['qualified_records_added_to_train'] - len(exclusions),
            'positive_records_added_to_train': parent['positive_records_added_to_train'] - removed['positive_records'],
            'negative_records_added_to_train': parent['negative_records_added_to_train'] - removed['negative_records'],
            'deferred_records_added_unassigned': parent['deferred_records_added_unassigned'] + len(exclusions),
            'previously_verified_base_supervised_train_records': parent['previously_verified_base_supervised_train_records'],
            'effective_supervised_train_records': parent['effective_supervised_train_records'] - len(exclusions),
            **pixel_counts, 'inherited_base_records': len(base_rows),
            'all_inherited_records_unchanged': True, 'all_nonexcluded_records_unchanged': True,
            'heldout_records_unchanged': True, 'inherited_pixels_opened': False,
            'heldout_pixels_opened': False, 'protected_pixels_opened': False,
            'inherited_raster_paths_resolved': False, 'images_opened': False,
            'inherited_pixel_storage': 'Immutable hard links',
            'inherited_supervision_verification': 'Frozen base records and prior count verified from metadata; inherited masks were not reopened.',
            'parent_augmentation_report': 'provenance/parent-augmentation-report.json',
            'pixel_count_scope': 'New bulk records only; bound parent totals minus the explicitly excluded masks, with removed supervision changed to ignore.',
            'validation_scope': 'Complete canonical metadata, exact frozen-base and nonexcluded record equality; only explicitly excluded new TRAIN masks decoded, all other rasters hard-linked unopened and unresolved.',
            'reviewer': protocol['reviewer'], 'reviewer_type': 'ai', 'ai_review_authorized': True,
            'model_predictions_used': False, 'human_review_claimed': False, 'training_run_allowed': False,
        }
        (stage / 'quality-quarantine-report.json').write_bytes(_json_bytes(report))
        evidence.check()
        _no_symlinks(output)
        _publish(stage, output)
        stage = None
        return report
    except QualityQuarantineError:
        raise
    except (BulkCurationError, ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError,
            RecursionError, OverflowError) as error:
        raise QualityQuarantineError(f'Quality quarantine failed: {error}') from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)
