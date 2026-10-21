# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Conservative independent bulk reviews and immutable training augmentation.

Only mutually approved polygon intersections become supervision. Metadata and
coordinates protect every inherited non-training record and external pilot;
inherited raster files are hard-linked without resolving or reading them.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile

from PIL import Image, ImageChops, ImageDraw

from .curation import _image_payload, _polygons, REVIEW_FIELDS
from .pv4ger import DatasetImportError, _publish
from .snapshots import load_centers
from .validation import IDENTIFIER, SHA256, _invalid_constant, _unique_object, validate_dataset

PURPOSE = "experimental-ai-bulk-training"
MIN_REFERENCE_DISTANCE_M = 5000 + 32 * math.sqrt(2)
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_PROVENANCE_BYTES = 2 * 1024 * 1024 * 1024
SAMPLE_FIELDS = {"id", "image_sha256", "full_image_inspected", "decision",
                 "positive_polygons", "background_polygons", "notes"}
PROTOCOL_FIELDS = {"schema_version", "purpose", "dataset_version", "curated_dataset_version",
                   "source_dataset_fingerprint", "queue_sha256", "base_dataset_fingerprint",
                   "protected_dataset_fingerprints", "minimum_positive_pixels",
                   "minimum_positive_background_pixels", "minimum_negative_background_pixels",
                   "min_reference_distance_m", "min_candidate_distance_m", "model_predictions_used",
                   "human_review_claimed", "training_run_allowed", "base_supervised_train_records", "review_schema_version"}
V2_METHOD = "two-independent-checks-of-shared-ai-visual-proposals"
V2_PROTOCOL_FIELDS = PROTOCOL_FIELDS | {"proposal_schema_version", "proposal_author", "review_method"}
IDENTITY = ("image_sha256", "width", "height", "gsd_m", "capture_date", "georeference", "building_ids")


class BulkCurationError(DatasetImportError):
    """Bulk evidence cannot safely release conservative training supervision."""


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl(rows) -> bytes:
    return ''.join(json.dumps(row, sort_keys=True, allow_nan=False) + '\n' for row in rows).encode()


def _object(payload: bytes):
    result = json.loads(payload, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(result, dict):
        raise BulkCurationError("Expected a strict JSON object.")
    return result


def _absolute(path) -> Path:
    return Path(os.path.abspath(path))


def _no_symlinks(path: Path) -> None:
    for component in (path, *path.parents):
        if component.is_symlink():
            raise BulkCurationError("Input and output paths must not contain symbolic links.")


def _relative(root: Path, value: str) -> Path:
    if (not isinstance(value, str) or not value or '\\' in value or ':' in value
            or PurePosixPath(value).is_absolute() or any(p in ('', '.', '..') for p in value.split('/'))):
        raise BulkCurationError("Expected a lexical relative path without traversal.")
    return root.joinpath(*PurePosixPath(value).parts)


class _Evidence:
    """Hash every bounded source byte and check it again before publication."""

    def __init__(self):
        self.watched = {}

    def read(self, path: Path, maximum=MAX_JSON_BYTES) -> bytes:
        path = _absolute(path)
        _no_symlinks(path)
        if not path.is_file() or path.stat().st_size > maximum:
            raise BulkCurationError(f"Expected a bounded regular file: {path}")
        with path.open('rb') as stream:
            payload = stream.read(maximum + 1)
        if len(payload) > maximum:
            raise BulkCurationError("Evidence exceeds its byte bound.")
        record = (maximum, _digest(payload))
        if path in self.watched and self.watched[path][1] != record[1]:
            raise BulkCurationError("Read-only input evidence changed.")
        self.watched[path] = (min(maximum, self.watched.get(path, record)[0]), record[1])
        return payload

    def check(self):
        for path, (maximum, digest) in list(self.watched.items()):
            if _digest(self.read(path, maximum)) != digest:
                raise BulkCurationError("Read-only input evidence changed before publication.")


def _paths(inputs, output):
    paths, output = [_absolute(p) for p in inputs], _absolute(output)
    for path in [*paths, output]:
        _no_symlinks(path)
    if os.path.lexists(output):
        raise BulkCurationError("Output already exists; refusing to replace it.")
    if any(not p.is_dir() for p in paths):
        raise BulkCurationError("Input directories must exist.")
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for i, a in enumerate([*paths, output]) for b in [*paths, output][i + 1:]):
        raise BulkCurationError("Input and output directories must be distinct and cannot be ancestors.")
    return paths, output


def _version(value):
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 128:
        raise BulkCurationError("Expected a nonempty dataset version of at most 128 characters.")


def _protocol(path, evidence):
    payload = evidence.read(Path(path), 64 * 1024)
    value = _object(payload)
    v2 = value.get('schema_version') == '2.0'
    if (set(value) != (V2_PROTOCOL_FIELDS if v2 else PROTOCOL_FIELDS)
            or value['schema_version'] != ('2.0' if v2 else '1.0') or value['purpose'] != PURPOSE
            or value['review_schema_version'] != ('1.2' if v2 else '1.1')
            or any(value[key] is not False for key in ('model_predictions_used', 'human_review_claimed', 'training_run_allowed'))
            or type(value['base_supervised_train_records']) is not int or value['base_supervised_train_records'] < 0):
        raise BulkCurationError("Expected the exact frozen experimental bulk-training protocol.")
    if v2 and (value['proposal_schema_version'] != '1.0' or value['review_method'] != V2_METHOD
               or not isinstance(value['proposal_author'], str) or not 8 <= len(value['proposal_author'].strip()) <= 256
               or re.search(r'pending|placeholder|unknown|template|\btodo\b', value['proposal_author'], re.I)):
        raise BulkCurationError('Expected the frozen common-proposal format, method and identified AI author.')
    for key in ('dataset_version', 'curated_dataset_version'):
        _version(value[key])
    for key in ('source_dataset_fingerprint', 'queue_sha256', 'base_dataset_fingerprint'):
        if not isinstance(value[key], str) or not SHA256.fullmatch(value[key]):
            raise BulkCurationError("Protocol fingerprint/hash is invalid.")
    protected = value['protected_dataset_fingerprints']
    if (not isinstance(protected, list) or not 1 <= len(protected) <= 8 or len(set(protected)) != len(protected)
            or any(not isinstance(v, str) or not SHA256.fullmatch(v) for v in protected)):
        raise BulkCurationError("Freeze one to eight distinct protected snapshot fingerprints.")
    for key, expected in (('minimum_positive_pixels', 100), ('minimum_positive_background_pixels', 256),
                          ('minimum_negative_background_pixels', 1024)):
        if type(value[key]) is not int or value[key] != expected:
            raise BulkCurationError("Conservative pixel gates are fixed at 100/256/1024.")
    for key, low in (('min_reference_distance_m', MIN_REFERENCE_DISTANCE_M), ('min_candidate_distance_m', 80)):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not low <= value[key] <= 100000:
            raise BulkCurationError("Frozen geographic distance is below the conservative minimum.")
    return value, payload


def _manifest(source, evidence):
    from .training import _read_manifest
    metadata_bytes = evidence.read(source / 'dataset.toml', 1024 * 1024)
    manifest_bytes = evidence.read(source / 'manifest.jsonl')
    metadata, rows, fingerprint = _read_manifest(source)
    if fingerprint != _digest(metadata_bytes + b'\0' + manifest_bytes):
        raise BulkCurationError("Dataset metadata changed while reading.")
    return metadata, rows, fingerprint


def _coordinates(source, rows, evidence):
    payload = evidence.read(source / 'provenance/samples.jsonl')
    points, digest = load_centers(source, rows)
    if digest != _digest(payload):
        raise BulkCurationError("Coordinate evidence changed while reading.")
    return points, digest, payload


def _metadata(metadata, version, *, curated=False, review_method=None):
    sources = metadata['sources']
    annotation_method = ('Two independent checks of common AI visual proposals; shared-proposal bias remains.'
                         if review_method == V2_METHOD else 'Two independent AI visual reviews.')
    lines = ['schema_version = "1.0"', 'dataset_id = ' + json.dumps('nrw-training-curated' if curated else 'pvextent-training-augmented'),
             'dataset_version = ' + json.dumps(version)]
    for key, fields in sorted(sources.items()):
        if curated:
            fields = {**fields, 'annotation_license': 'Apache-2.0',
                      'version': fields['version'] + '; AI bulk annotation ' + version,
                      'attribution': fields['attribution'].replace('All masks are unannotated ignore placeholders.', 'The raw acquisition snapshot contained unannotated ignore placeholder masks.') + ' Current curated partial masks: copyright 2026 Valeford, Apache-2.0. ' + annotation_method + ' No model predictions or human review claimed.'}
        lines.append('\n[sources.' + json.dumps('ai-bulk-' + key if curated else key) + ']')
        lines.extend(name + ' = ' + json.dumps(value) for name, value in fields.items())
    return ('\n'.join(lines) + '\n').encode()


def _copy_tree(source: Path, destination: Path, evidence: _Evidence):
    _no_symlinks(source)
    if not source.is_dir():
        raise BulkCurationError("Required source provenance directory is missing.")
    total = 0
    for path in sorted(source.rglob('*')):
        _no_symlinks(path)
        if path.is_dir():
            continue
        payload = evidence.read(path)
        total += len(payload)
        if total > MAX_PROVENANCE_BYTES:
            raise BulkCurationError("Provenance aggregate byte bound exceeded.")
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _copy_notices(source, destination, evidence):
    for name in ('LICENSE', 'NOTICE'):
        path = source / name
        if os.path.lexists(path):
            destination.mkdir(parents=True, exist_ok=True)
            (destination / name).write_bytes(evidence.read(path, 2 * 1024 * 1024))


def _review(payload, originals):
    review = _object(payload)
    reviewer = review.get('reviewer')
    if (set(review) != REVIEW_FIELDS or review['review_schema_version'] != '1.1'
            or review['reviewer_type'] != 'ai' or review['model_predictions_used'] is not False
            or not isinstance(reviewer, str) or not 8 <= len(reviewer.strip()) <= 256
            or re.search(r'pending|placeholder|unknown|template|\btodo\b', reviewer, re.I)
            or not isinstance(review['samples'], list) or len(review['samples']) != len(originals)):
        raise BulkCurationError("Expected completed schema 1.1, an identified AI role, and no predictions.")
    samples, vertices = {}, 0
    for sample in review['samples']:
        if not isinstance(sample, dict) or set(sample) != SAMPLE_FIELDS:
            raise BulkCurationError("Expected exactly the completed sample review fields.")
        identifier = sample['id']
        if (not isinstance(identifier, str) or not IDENTIFIER.fullmatch(identifier) or identifier in samples
                or identifier not in originals or sample['image_sha256'] != originals[identifier]['image_sha256']
                or sample['full_image_inspected'] is not True
                or sample['decision'] not in ('approve-positive', 'approve-negative', 'defer')
                or not isinstance(sample['notes'], str) or len(sample['notes']) > 8192):
            raise BulkCurationError("Invalid review ID, hash, inspection flag, notes or decision.")
        planes = []
        for field in ('positive_polygons', 'background_polygons'):
            polygons = _polygons(sample[field], 320, 320)
            vertices += sum(map(len, polygons))
            if vertices > 8192:
                raise BulkCurationError("Each completed batch has a maximum of 8192 vertices.")
            plane = Image.new('L', (320, 320), 0)
            draw = ImageDraw.Draw(plane)
            for polygon in polygons:
                draw.polygon(polygon, fill=255)
            planes.append(plane)
        if ImageChops.darker(*planes).getbbox() is not None:
            raise BulkCurationError("PV and background polygons overlap within a review.")
        if sample['decision'] == 'approve-negative' and sample['positive_polygons']:
            raise BulkCurationError("An approved negative requires explicitly empty PV polygons.")
        samples[identifier] = {**sample, 'planes': planes}
    if set(samples) != set(originals):
        raise BulkCurationError("A completed batch must cover its exact bound sample IDs.")
    return {**review, 'samples': samples}


def _queue(source, queue, results, originals, fingerprint, protocol, evidence, *, review_loader=None, load_reviews=True, additional_review_paths=()):
    queue_bytes = evidence.read(queue / 'queue.json', 2 * 1024 * 1024)
    index = _object(queue_bytes)
    fields = {'schema_version', 'purpose', 'dataset_fingerprint', 'sample_count', 'batch_size',
              'batch_count_per_lane', 'review_template_count', 'pending_only', 'raster_paths_accessed',
              'labels_created', 'model_predictions_used', 'canonical_dataset_modified', 'lanes'}
    if (set(index) != fields or _digest(queue_bytes) != protocol['queue_sha256']
            or index['schema_version'] != '1.0' or index['purpose'] != 'pending-blinded-training-review'
            or index['dataset_fingerprint'] != fingerprint or index['sample_count'] != len(originals)
            or index['pending_only'] is not True or any(index[k] is not False for k in
                ('raster_paths_accessed', 'labels_created', 'model_predictions_used', 'canonical_dataset_modified'))
            or type(index['batch_size']) is not int or not 1 <= index['batch_size'] <= 50
            or type(index['batch_count_per_lane']) is not int or not 1 <= index['batch_count_per_lane'] <= 2000
            or index['review_template_count'] != index['batch_count_per_lane'] * 2
            or not isinstance(index['lanes'], list) or len(index['lanes']) != 2):
        raise BulkCurationError("Queue schema, frozen hash, source fingerprint or bounds differ.")
    reviews, report, lane_ids, batch_ids, result_paths = {}, [], [], None, set()
    for lane, entry in zip(('A', 'B'), index['lanes']):
        if (not isinstance(entry, dict) or set(entry) != {'id', 'batches'} or entry['id'] != lane
                or not isinstance(entry['batches'], list) or len(entry['batches']) != index['batch_count_per_lane']):
            raise BulkCurationError("Queue must contain exact independent lanes A and B.")
        seen, ordered, identity = set(), [], None
        lane_reviews = {}
        for number, batch in enumerate(entry['batches'], 1):
            batch_id = f'batch-{number:04d}'
            prefix = f'lane-{lane.lower()}/{batch_id}'
            if (not isinstance(batch, dict) or set(batch) != {'id', 'sample_ids', 'manifest', 'manifest_sha256', 'review_template', 'review_template_sha256'}
                    or batch['id'] != batch_id or batch['manifest'] != prefix + '.manifest.json'
                    or batch['review_template'] != prefix + '.review.json' or not isinstance(batch['sample_ids'], list)
                    or not 1 <= len(batch['sample_ids']) <= index['batch_size']
                    or any(not isinstance(v, str) or v not in originals or v in seen for v in batch['sample_ids'])
                    or len(set(batch['sample_ids'])) != len(batch['sample_ids'])):
                raise BulkCurationError("Queue batch IDs, paths or sample coverage differ.")
            selected = {key: originals[key] for key in batch['sample_ids']}
            seen.update(selected)
            ordered.append(batch['sample_ids'])
            manifest_bytes = evidence.read(queue / batch['manifest'], 2 * 1024 * 1024)
            template_bytes = evidence.read(queue / batch['review_template'], 2 * 1024 * 1024)
            if _digest(manifest_bytes) != batch['manifest_sha256'] or _digest(template_bytes) != batch['review_template_sha256']:
                raise BulkCurationError("Original queue manifest or pending template bytes changed.")
            expected_samples = [{key: row[key] for key in ('id', 'image_sha256', 'width', 'height', 'gsd_m', 'capture_date')}
                                | {'image': str(source / row['image'])} for row in selected.values()]
            if _object(manifest_bytes) != {'schema_version': '1.0', 'dataset_fingerprint': fingerprint,
                                          'batch_id': batch_id, 'lane': lane, 'samples': expected_samples}:
                raise BulkCurationError("Batch manifest differs from its source identity or lane binding.")
            expected_template = {'review_schema_version': '1.0', 'reviewer': f'PENDING-REVIEWER-{lane}-{batch_id}',
                'reviewer_type': 'ai', 'model_predictions_used': False,
                'samples': [{'id': row['id'], 'image_sha256': row['image_sha256'], 'full_image_inspected': False,
                    'decision': 'defer', 'positive_polygons': [], 'background_polygons': [],
                    'notes': 'Pending independent full-image inspection.'} for row in selected.values()]}
            if _object(template_bytes) != expected_template:
                raise BulkCurationError("Queue templates must remain the original pending proposals.")
            if not load_reviews:
                continue
            path = results / (prefix + '.review.json')
            payload = evidence.read(path, 2 * 1024 * 1024)
            review = (_review(payload, selected) if review_loader is None else
                      review_loader(payload, selected, lane, batch_id, batch))
            normalized = review['reviewer'].strip().casefold()
            if identity is not None and identity != normalized:
                raise BulkCurationError("Reviewer role must remain constant within each independent lane.")
            identity = normalized
            lane_reviews.update(review['samples'])
            result_paths.add(path)
            report.append({'lane': lane, 'batch_id': batch_id, 'reviewer': review['reviewer'],
                           'reviewer_type': 'ai', 'model_predictions_used': False,
                           'file': 'provenance/completed-reviews/' + prefix + '.review.json',
                           'sha256': _digest(payload), 'manifest_sha256': batch['manifest_sha256'],
                           'review_template_sha256': batch['review_template_sha256'],
                           **({'proposal_sha256': review['proposal_sha256']} if 'proposal_sha256' in review else {})})
        if seen != set(originals) or (batch_ids is not None and batch_ids != ordered):
            raise BulkCurationError("Both lanes must cover each bound sample exactly once in the same batches.")
        batch_ids = ordered
        lane_ids.append(identity)
        reviews[lane] = lane_reviews
    if load_reviews and len(set(lane_ids)) != 2:
        raise BulkCurationError("Two distinct independent AI reviewer roles are required.")
    if load_reviews and set(results.rglob('*.review.json')) != result_paths | set(additional_review_paths):
        raise BulkCurationError("Completed review directory contains unbound or mixed batch results.")
    return reviews, report, queue_bytes


def _consensus(first, second):
    approved = {}
    if 'approved_positive_polygons' in first:
        planes = []
        for field, polygons_field in (('approved_positive_polygons', 'positive_polygons'),
                                      ('approved_background_polygons', 'background_polygons')):
            indices = sorted(set(first[field]) & set(second[field]))
            approved[field] = indices
            plane = Image.new('L', (320, 320), 0)
            draw = ImageDraw.Draw(plane)
            for index in indices:
                draw.polygon(first['proposal_polygons'][polygons_field][index], fill=255)
            planes.append(plane)
        positive, background = planes
    else:
        positive = ImageChops.darker(first['planes'][0], second['planes'][0])
        background = ImageChops.darker(first['planes'][1], second['planes'][1])
    pv, bg = positive.histogram()[255], background.histogram()[255]
    decisions = (first['decision'], second['decision'])
    kind = ('positive' if decisions == ('approve-positive', 'approve-positive') and pv >= 100 and bg >= 256
            else 'negative' if decisions == ('approve-negative', 'approve-negative') and pv == 0 and bg >= 1024
            else 'deferred')
    mask = Image.new('L', (320, 320), 255)
    if kind != 'deferred':
        mask.paste(1, mask=positive)
        mask.paste(0, mask=background)
    histogram = mask.histogram()
    return mask, {**approved, 'classification': kind, 'eligible': kind != 'deferred',
                  'review_decisions': list(decisions), 'intersection_positive_pixels': pv,
                  'intersection_background_pixels': bg, 'positive_pixels': histogram[1],
                  'background_pixels': histogram[0], 'ignored_pixels': histogram[255]}


def _acquisition(source, fingerprint, originals, evidence):
    report = _object(evidence.read(source / 'candidate-report.json'))
    protocol_payload = evidence.read(source / 'provenance/protocol.json', 64 * 1024)
    plan_payload = evidence.read(source / 'provenance/selection-plan.json')
    binding = _object(evidence.read(source / 'provenance/reference-binding.json'))
    decisions = json.loads(evidence.read(source / 'provenance/acquisition-decisions.json'),
                           object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if (report.get('ok') is not True or report.get('complete') is not True
            or report.get('dataset_id') != 'nrw-training-bulk' or report.get('dataset_fingerprint') != fingerprint
            or report.get('protocol_sha256') != _digest(protocol_payload)
            or report.get('selection_plan_sha256') != _digest(plan_payload)
            or report.get('model_predictions_used') is not False or report.get('original_masks_transferred') is not False
            or binding.get('protocol_sha256') != _digest(protocol_payload)
            or report.get('references') != binding.get('references') or not isinstance(decisions, list)):
        raise BulkCurationError("Frozen source acquisition evidence is invalid or unbound.")
    plan = _object(plan_payload)
    if plan.get('binding') != binding:
        raise BulkCurationError("Selection plan and reference binding differ.")
    candidates = {}
    for kind in ('known', 'geometry'):
        if not isinstance(plan.get(kind), list):
            raise BulkCurationError("Expected explicit known and geometry selection lists.")
        for candidate in plan[kind]:
            if (not isinstance(candidate, dict) or candidate.get('kind') != kind
                    or not isinstance(candidate.get('id'), str) or candidate['id'] in candidates):
                raise BulkCurationError("Invalid or duplicate acquisition-plan candidate.")
            candidates[candidate['id']] = candidate
    accepted = {}
    for row in decisions:
        if not isinstance(row, dict) or type(row.get('accepted')) is not bool:
            raise BulkCurationError("Invalid acquisition decision evidence.")
        if not row['accepted']:
            continue
        key = row.get('id')
        if (not isinstance(key, str) or key not in originals or key in accepted
                or row.get('candidate') != candidates.get(key)):
            raise BulkCurationError("Accepted source lineage must exactly match the frozen selection plan.")
        accepted[key] = row
    if set(accepted) != set(originals):
        raise BulkCurationError("Source acquisition decisions must cover every canonical sample exactly once.")
    return report, accepted


def curate_bulk_training(dataset_dir: Path | str, queue_dir: Path | str, review_dir: Path | str,
                         output_dir: Path | str, *, dataset_version: str, protocol_path: Path | str,
                         allow_ai_review: bool = False) -> dict:
    """Publish all bound photos with partial consensus masks, still unassigned."""
    stage = None
    try:
        if allow_ai_review is not True:
            raise BulkCurationError("AI curation requires explicit allow_ai_review=True authorization.")
        _version(dataset_version)
        (source, queue, results), output = _paths([dataset_dir, queue_dir, review_dir], output_dir)
        if _absolute(protocol_path).is_relative_to(output):
            raise BulkCurationError("Protocol must exist outside the new output.")
        evidence = _Evidence()
        protocol, protocol_bytes = _protocol(protocol_path, evidence)
        if dataset_version != protocol['curated_dataset_version']:
            raise BulkCurationError("Curated dataset version differs from the frozen protocol.")
        metadata, originals_list, fingerprint = _manifest(source, evidence)
        if (metadata['dataset_id'] != 'nrw-training-bulk' or fingerprint != protocol['source_dataset_fingerprint']
                or not 1 <= len(originals_list) <= 2000
                or any(row['split'] != 'unassigned' or row['annotation_status'] != 'unreviewed'
                       or row['width'] != 320 or row['height'] != 320 or row['gsd_m'] != .1 for row in originals_list)):
            raise BulkCurationError("Expected the frozen, native, unreviewed and unassigned bulk source.")
        originals = {row['id']: row for row in originals_list}
        _, coordinates_hash, coordinate_bytes = _coordinates(source, originals_list, evidence)
        _acquisition(source, fingerprint, originals, evidence)
        reviews, review_evidence, review_context = _bound_reviews(source, queue, results, originals, fingerprint, protocol, evidence, protocol_sha256=_digest(protocol_bytes))
        output.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(output.parent)
        stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-bulk-curation-', dir=output.parent))
        (stage / 'dataset.toml').write_bytes(_metadata(metadata, dataset_version, curated=True, review_method=protocol.get('review_method')))
        rows, decisions = [], []
        for identifier in sorted(originals):
            row = originals[identifier]
            mask, decision = _consensus(reviews['A'][identifier], reviews['B'][identifier])
            image_path, mask_path = f'images/{identifier}.png', f'masks/{identifier}.png'
            image_at = _relative(source, row['image'])
            _no_symlinks(image_at)
            payload = _image_payload(source, row)  # Only the new photo; source masks remain unopened.
            if evidence.read(image_at) != payload:
                raise BulkCurationError("Reviewed image changed while constructing output.")
            (stage / image_path).parent.mkdir(exist_ok=True)
            (stage / mask_path).parent.mkdir(exist_ok=True)
            (stage / image_path).write_bytes(payload)
            buffer = io.BytesIO()
            mask.save(buffer, format='PNG')
            mask_bytes = buffer.getvalue()
            (stage / mask_path).write_bytes(mask_bytes)
            rows.append({**row, 'image': image_path, 'mask': mask_path, 'mask_sha256': _digest(mask_bytes),
                         'source_id': 'ai-bulk-' + row['source_id'], 'split': 'unassigned',
                         'annotation_status': 'reviewed' if decision['eligible'] else 'unreviewed'})
            decisions.append({'id': identifier, **decision})
        (stage / 'manifest.jsonl').write_bytes(_jsonl(rows))
        provenance = stage / 'provenance'
        provenance.mkdir()
        (provenance / 'protocol.json').write_bytes(protocol_bytes)
        (provenance / 'samples.jsonl').write_bytes(coordinate_bytes)
        (provenance / 'original-source-records.jsonl').write_bytes(_jsonl(originals_list))
        acquisition = provenance / 'acquisition'
        acquisition.mkdir()
        for name in ('dataset.toml', 'manifest.jsonl', 'candidate-report.json'):
            (acquisition / name).write_bytes(evidence.read(source / name))
        _copy_tree(source / 'provenance', acquisition / 'provenance', evidence)
        _copy_notices(source, acquisition, evidence)
        _copy_tree(queue, provenance / 'review-queue', evidence)
        _copy_tree(results, provenance / 'completed-reviews', evidence)
        annotation_description = (
            'One AI author proposed regions; two new independent AI checkers inspected the same frozen polygons against full native RGB. '
            'Only entire polygon indices approved by both checkers can become supervision. Shared-proposal bias is explicitly retained. '
            if protocol['schema_version'] == '2.0' else
            'Two distinct identified AI roles independently inspected and drew regions on full original 320x320 photos. '
            'Only same-class polygon intersections are supervised. ')
        (provenance / 'AI-ANNOTATION-PROVENANCE.md').write_text(
            '# Experimental AI visual-review annotations\n\nCopyright 2026 Valeford. New partial masks: Apache-2.0.\n'
            + annotation_description +
            'No predictions, pseudo-labels or human review are claimed. Polygon boundaries are included; every other pixel is 255 ignore. '
            'Positive consensus requires both positive approvals, 100 PV pixels and 256 background pixels. '
            'Negative consensus requires both explicit negative approvals, no PV polygons and 1024 background pixels. '
            'Empty polygons never prove PV absence. All records remain unassigned pending geographic augmentation. '
            'Original imagery, coordinates, notices and raw acquisition/review evidence are retained.\n', encoding='utf-8')
        validation = validate_dataset(stage)  # Exclusively newly reviewed photos and newly created masks.
        if not validation.ok:
            raise BulkCurationError('Curated dataset failed validation: ' + validation.errors[0].message)
        counts = Counter(row['classification'] for row in decisions)
        report = {'ok': True, 'purpose': PURPOSE, 'dataset_id': 'nrw-training-curated', 'dataset_version': dataset_version,
                  'dataset_fingerprint': validation.stats['dataset_fingerprint'], 'records': len(rows),
                  'source_dataset_fingerprint': fingerprint, 'source_coordinates_sha256': coordinates_hash,
                  'queue_sha256': protocol['queue_sha256'], 'protocol_sha256': _digest(protocol_bytes),
                  'positive_records': counts['positive'], 'negative_records': counts['negative'],
                  'deferred_records': counts['deferred'], 'eligible_records': counts['positive'] + counts['negative'],
                  'pv_pixels': sum(d['positive_pixels'] for d in decisions),
                  'background_pixels': sum(d['background_pixels'] for d in decisions),
                  'ignored_pixels': sum(d['ignored_pixels'] for d in decisions),
                  'ai_review_authorized': True, 'human_review_claimed': False, 'model_predictions_used': False,
                  'training_run_allowed': False, 'all_records_unassigned': True, 'reviewers': review_evidence,
                  'decisions': decisions, 'metadata_sha256': validation.stats['metadata_sha256'],
                  'manifest_sha256': validation.stats['manifest_sha256'],
                  'validation': {'ok': True, 'warning_count': validation.warning_count}, **review_context}
        (stage / 'curation-report.json').write_bytes(_json_bytes(report))
        (stage / 'validation-report.json').write_bytes(_json_bytes(validation.to_dict()))
        evidence.check()
        _no_symlinks(output)
        _publish(stage, output)
        stage = None
        return report
    except BulkCurationError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError, OverflowError) as error:
        raise BulkCurationError(f'Bulk curation failed: {error}') from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _point(row, center, project):
    geo = row['georeference']
    if row['width'] != 320 or row['height'] != 320 or row['gsd_m'] != .1 or geo is None or geo['crs'] != 'EPSG:25832':
        raise BulkCurationError('New photos must preserve native 320x320 EPSG:25832 geometry at 0.1 m.')
    a, b, c, d, e, f = geo['transform']
    if any(abs(v - wanted) > 1e-9 for v, wanted in zip((a, b, d, e), (.1, 0, 0, -.1))):
        raise BulkCurationError('New photos must preserve the native affine grid.')
    point = (a * 160 + b * 160 + c, d * 160 + e * 160 + f)
    if math.dist(point, project.transform(*center, errcheck=True)) > .2:
        raise BulkCurationError('Coordinate evidence differs from the actual affine photo centre.')
    return point


def _link_inherited(source, row, role, stage, relative):
    """Do not resolve, open, hash or decode any inherited raster path."""
    path = _relative(source, row[role])
    target = _relative(stage, relative)
    _no_symlinks(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(target):
        if not os.path.samefile(path, target):
            raise BulkCurationError('Inherited raster output path collision.')
        return
    os.link(path, target, follow_symlinks=False)
    if not os.path.samefile(path, target):
        raise BulkCurationError('Inherited hard-link verification failed.')


def augment_bulk_training(base_dir: Path | str, curated_dir: Path | str,
                          protected_dirs: list[Path | str], output_dir: Path | str, *,
                          dataset_version: str, protocol_path: Path | str) -> dict:
    """Append conservative bulk labels; preserve every inherited record and pixel."""
    stage = None
    try:
        from pyproj import Transformer
        from .training import _read_manifest, _raster
        if not isinstance(protected_dirs, (list, tuple)) or not 1 <= len(protected_dirs) <= 8:
            raise BulkCurationError('Provide one to eight frozen external protected snapshots.')
        _version(dataset_version)
        inputs, output = _paths([base_dir, curated_dir, *protected_dirs], output_dir)
        base, curated, *protected = inputs
        if _absolute(protocol_path).is_relative_to(output):
            raise BulkCurationError('Protocol must exist outside the output.')
        evidence = _Evidence()
        protocol, protocol_bytes = _protocol(protocol_path, evidence)
        if dataset_version != protocol['dataset_version']:
            raise BulkCurationError('Training snapshot version differs from the frozen protocol.')
        base_meta, base_rows, base_fp = _manifest(base, evidence)
        curated_meta, new_rows, curated_fp = _manifest(curated, evidence)
        base_by_id = {row['id']: row for row in base_rows}
        source = curated / 'provenance/acquisition'
        source_meta, source_rows, source_fp = _manifest(source, evidence)
        originals = {row['id']: row for row in source_rows}
        if (base_fp != protocol['base_dataset_fingerprint'] or source_fp != protocol['source_dataset_fingerprint']
                or curated_meta['dataset_id'] != 'nrw-training-curated' or source_meta['dataset_id'] != 'nrw-training-bulk'
                or curated_meta['dataset_version'] != protocol['curated_dataset_version']
                or set(originals) != {r['id'] for r in new_rows} or not 1 <= len(new_rows) <= 2000
                or any(r['split'] != 'unassigned' for r in new_rows)
                or not all(any(r['split'] == split for r in base_rows) for split in ('train', 'val', 'test'))
                or protocol['base_supervised_train_records'] != sum(r['split'] == 'train' and r['annotation_status'] in ('upstream', 'reviewed') for r in base_rows)):
            raise BulkCurationError('Frozen base/source identity, curated coverage or inherited supervision declaration differs.')
        report_bytes = evidence.read(curated / 'curation-report.json')
        curation = _object(report_bytes)
        if (curation.get('ok') is not True or curation.get('purpose') != PURPOSE
                or curation.get('dataset_fingerprint') != curated_fp or curation.get('source_dataset_fingerprint') != source_fp
                or curation.get('protocol_sha256') != _digest(protocol_bytes)
                or curation.get('queue_sha256') != protocol['queue_sha256']
                or curation.get('model_predictions_used') is not False or curation.get('human_review_claimed') is not False
                or curation.get('ai_review_authorized') is not True):
            raise BulkCurationError('Curated snapshot lacks bound authorized review evidence.')
        acquisition, accepted = _acquisition(source, source_fp, originals, evidence)
        # Rebind retained queue to the original acquisition path recorded in its manifests.
        retained_queue = curated / 'provenance/review-queue'
        queue_index = _object(evidence.read(retained_queue / 'queue.json', 2 * 1024 * 1024))
        first_batch = queue_index.get('lanes', [{}])[0].get('batches', [{}])[0]
        first_manifest = _object(evidence.read(_relative(retained_queue, first_batch.get('manifest', 'invalid'))))
        first_sample = first_manifest.get('samples', [{}])[0]
        first_original = originals.get(first_sample.get('id'))
        if first_original is None or not isinstance(first_sample.get('image'), str):
            raise BulkCurationError('Retained queue cannot identify its original source root.')
        expected_suffix = PurePosixPath(first_original['image']).parts
        bound_image = PurePosixPath(first_sample['image'])
        if tuple(bound_image.parts[-len(expected_suffix):]) != expected_suffix or not bound_image.is_absolute():
            raise BulkCurationError('Original queue image path is unbound.')
        original_root = Path(*bound_image.parts[:-len(expected_suffix)])
        reviews, review_evidence, review_context = _bound_reviews(original_root, retained_queue, curated / 'provenance/completed-reviews',
                                            originals, source_fp, protocol, evidence, protocol_sha256=_digest(protocol_bytes))
        if (review_evidence != curation.get('reviewers')
                or any(curation.get(key) != value for key, value in review_context.items())):
            raise BulkCurationError('Retained review hashes differ from the curation report.')
        base_centers, base_coord_hash, _ = _coordinates(base, base_rows, evidence)
        new_centers, new_coord_hash, new_coordinate_bytes = _coordinates(curated, new_rows, evidence)
        _, source_coord_hash, source_coordinate_bytes = _coordinates(source, source_rows, evidence)
        if new_coordinate_bytes != source_coordinate_bytes or curation.get('source_coordinates_sha256') != source_coord_hash:
            raise BulkCurationError('Actual source EPSG:4326 coordinates were not retained unchanged.')
        project = Transformer.from_crs(4326, 25832, always_xy=True)
        base_train_points = {r['id']: project.transform(*base_centers[r['id']], errcheck=True) for r in base_rows if r['split'] == 'train'}
        protected_rows = [r for r in base_rows if r['split'] != 'train']
        protected_points = [project.transform(*base_centers[r['id']], errcheck=True) for r in protected_rows]
        protected_info = []
        reference_by_fp = {}
        for path in protected:
            _, rows, fingerprint = _manifest(path, evidence)
            centers, coordinate_hash, _ = _coordinates(path, rows, evidence)
            protected_rows.extend(rows)
            protected_points.extend(project.transform(*centers[r['id']], errcheck=True) for r in rows)
            protected_info.append({'dataset_fingerprint': fingerprint, 'coordinate_sha256': coordinate_hash,
                                   'records': len(rows)})
            reference_by_fp[fingerprint] = coordinate_hash
        if sorted(reference_by_fp) != sorted(protocol['protected_dataset_fingerprints']):
            raise BulkCurationError('Protected snapshots differ from frozen protocol fingerprints.')
        acquisition_refs = acquisition.get('references')
        if not isinstance(acquisition_refs, list) or any(not isinstance(v, dict) for v in acquisition_refs):
            raise BulkCurationError('Source acquisition reference evidence is invalid.')
        for fingerprint, coordinate_hash in [(base_fp, base_coord_hash), *reference_by_fp.items()]:
            if not any(v.get('dataset_fingerprint') == fingerprint and v.get('coordinate_sha256') == coordinate_hash for v in acquisition_refs):
                raise BulkCurationError('Acquisition and augmentation reference fingerprints/coordinates differ.')
        held_ids = {r['id'] for r in protected_rows}
        held_hashes = {r['image_sha256'] for r in protected_rows}
        held_groups = {r['spatial_group'] for r in protected_rows}
        held_buildings = {b for r in protected_rows for b in r['building_ids']}
        all_ids = set(base_by_id) | held_ids
        all_hashes = {r['image_sha256'] for r in base_rows} | held_hashes
        all_buildings = {b for r in base_rows for b in r['building_ids']} | held_buildings
        all_groups = {r['spatial_group'] for r in base_rows} | held_groups
        added_points, decisions, replacements, supervised_masks = [], [], [], {}
        seen_ids, seen_hashes, geometry_buildings = set(), set(), set()
        for row in sorted(new_rows, key=lambda r: r['id']):
            key, original = row['id'], originals[row['id']]
            if (any(row[k] != original[k] for k in IDENTITY) or row['source_id'] != 'ai-bulk-' + original['source_id']
                    or row['spatial_group'] != original['spatial_group']
                    or key in all_ids or key in seen_ids or row['image_sha256'] in all_hashes or row['image_sha256'] in seen_hashes
                    or set(row['building_ids']) & held_buildings or row['spatial_group'] in held_groups):
                raise BulkCurationError('New photo identity overlaps frozen references or differs from its acquisition.')
            seen_ids.add(key)
            seen_hashes.add(row['image_sha256'])
            candidate = accepted[key]['candidate']
            parent = candidate.get('parent_training_id')
            if candidate['kind'] == 'known':
                old = base_by_id.get(parent)
                if (old is None or old['split'] != 'train' or old['source_id'] != 'pv4ger-seg'
                        or row['building_ids'] != old['building_ids']):
                    raise BulkCurationError('Known recaptures require explicit original TRAIN parent lineage.')
                group = old['spatial_group']
            else:
                if (parent is not None or row['spatial_group'] in all_groups
                        or set(row['building_ids']) & (all_buildings | geometry_buildings)):
                    raise BulkCurationError('Geometry-only photos must have distinct building/group identities.')
                group = row['spatial_group']
                geometry_buildings.update(row['building_ids'])
            point = _point(row, new_centers[key], project)
            if candidate['kind'] == 'known' and math.dist(point, base_train_points[parent]) > .2:
                raise BulkCurationError('Known TRAIN parent coordinate differs from the actual new crop centre.')
            if candidate['kind'] == 'geometry' and any(math.dist(point, old_point) < protocol['min_candidate_distance_m'] for old_point in base_train_points.values()):
                raise BulkCurationError('Geometry-only crop violates spacing against a base TRAIN centre.')
            if candidate.get('building_ids') != row['building_ids']:
                raise BulkCurationError('Acquisition building lineage differs from the source record.')
            if (not isinstance(candidate.get('center_epsg25832'), list)
                    or len(candidate['center_epsg25832']) != 2
                    or math.dist(point, candidate['center_epsg25832']) > .2):
                raise BulkCurationError('Acquisition lineage centre differs from the canonical crop.')
            nearest = min(math.dist(point, reference) for reference in protected_points)
            spacing = min((math.dist(point, previous) for previous in added_points), default=None)
            if nearest < protocol['min_reference_distance_m'] or (spacing is not None and spacing < protocol['min_candidate_distance_m']):
                raise BulkCurationError('Bulk centres violate the protected buffer or new-photo spacing.')
            added_points.append(point)
            expected_mask, decision = _consensus(reviews['A'][key], reviews['B'][key])
            if row['annotation_status'] != ('reviewed' if decision['eligible'] else 'unreviewed'):
                raise BulkCurationError('Curated annotation status differs from independent review consensus.')
            # Only new curated masks/photos are decoded, after all reference identity declarations were checked.
            actual = _raster(curated, row, 'mask')
            if actual.tobytes() != expected_mask.tobytes():
                raise BulkCurationError('Curated mask differs from conservative independent intersections.')
            supervised_masks[key] = decision['eligible']
            token = _digest(key.encode())[:24]
            replacements.append({**row, 'split': 'train' if decision['eligible'] else 'unassigned',
                                 'spatial_group': group, 'image': f'images/bulk/{token}.png', 'mask': f'masks/bulk/{token}.png'})
            decisions.append({'id': key, **decision, 'split': replacements[-1]['split'],
                              'parent_training_id': parent, 'kind': candidate['kind'], 'spatial_group': group,
                              'nearest_protected_center_m': nearest, 'minimum_previous_new_center_m': spacing})
        validation = validate_dataset(curated)
        if not validation.ok or validation.stats['dataset_fingerprint'] != curated_fp:
            raise BulkCurationError('New curated pixels are invalid or changed.')
        merged = sorted([*base_rows, *replacements], key=lambda r: r['id'])
        sources = dict(base_meta['sources'])
        for key, value in curated_meta['sources'].items():
            if key in sources and sources[key] != value:
                raise BulkCurationError('Source metadata identifiers conflict.')
            sources[key] = value
        output.parent.mkdir(parents=True, exist_ok=True)
        _no_symlinks(output.parent)
        stage = Path(tempfile.mkdtemp(prefix=f'.{output.name}-bulk-augmentation-', dir=output.parent))
        (stage / 'dataset.toml').write_bytes(_metadata({'sources': sources}, dataset_version))
        (stage / 'manifest.jsonl').write_bytes(_jsonl(merged))
        _, checked, output_fp = _read_manifest(stage)  # Lexical-only full manifest checks before inherited raster access.
        if checked != merged or any(next(r for r in checked if r['id'] == old['id']) != old for old in base_rows):
            raise BulkCurationError('Inherited declarations changed while merging.')
        inherited_paths = {r[role] for r in base_rows for role in ('image', 'mask')}
        new_paths = [r[role] for r in replacements for role in ('image', 'mask')]
        if len(set(new_paths)) != len(new_paths) or inherited_paths.intersection(new_paths):
            raise BulkCurationError('New raster paths collide with immutable inherited rasters.')
        for row in base_rows:
            for role in ('image', 'mask'):
                _link_inherited(base, row, role, stage, row[role])
        for row in replacements:
            original = next(r for r in new_rows if r['id'] == row['id'])
            for role in ('image', 'mask'):
                payload = evidence.read(_relative(curated, original[role]))
                if _digest(payload) != original[role + '_sha256']:
                    raise BulkCurationError('Curated raster bytes changed while constructing augmentation.')
                target = _relative(stage, row[role])
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
        provenance = stage / 'provenance'
        provenance.mkdir()
        _copy_tree(base / 'provenance', provenance / 'parent', evidence)
        _copy_tree(curated / 'provenance', provenance / 'curation', evidence)
        _copy_notices(base, provenance / 'parent', evidence)
        _copy_notices(curated, provenance / 'curation', evidence)
        for parent, prefix in ((base, 'parent'), (curated, 'curated')):
            for name in ('dataset.toml', 'manifest.jsonl'):
                (provenance / (prefix + '-' + name)).write_bytes(evidence.read(parent / name))
        (provenance / 'curation-report.json').write_bytes(report_bytes)
        (provenance / 'protocol.json').write_bytes(protocol_bytes)
        # Full original base coordinate rows remain in provenance/parent; canonical centres retain correct CRS.
        coordinates = [{'id': r['id'], 'approximate_center': list((base_centers if r['id'] in base_by_id else new_centers)[r['id']]),
                        'center_crs': 'EPSG:4326'} for r in merged]
        (provenance / 'samples.jsonl').write_bytes(_jsonl(coordinates))
        (provenance / 'augmentation-decisions.json').write_bytes(_json_bytes(decisions))
        assignments_at = base / 'provenance/split-assignments.jsonl'
        proposed = {}
        if os.path.lexists(assignments_at):
            for line in evidence.read(assignments_at).splitlines():
                item = _object(line)
                key = item.get('id')
                if (key not in base_by_id or key in proposed or item.get('proposed_split') not in ('train', 'val', 'test')):
                    raise BulkCurationError('Inherited split reservation evidence is invalid.')
                proposed[key] = item['proposed_split']
            if set(proposed) != set(base_by_id):
                raise BulkCurationError('Inherited split reservation evidence must cover the base exactly.')
        reservations = [{'id': r['id'], 'proposed_split': proposed.get(r['id'], r['split'] if r['split'] != 'unassigned' else 'train'),
                         'split': r['split']} for r in merged]
        (provenance / 'split-assignments.jsonl').write_bytes(_jsonl(reservations))
        added = sum(r['split'] == 'train' for r in replacements)
        report = {'ok': True, 'purpose': PURPOSE, 'dataset_fingerprint': output_fp,
                  'dataset_version': dataset_version, 'records': len(merged),
                  'splits': dict(Counter(r['split'] for r in merged)), 'parent_dataset_fingerprint': base_fp,
                  'curated_dataset_fingerprint': curated_fp, 'source_dataset_fingerprint': source_fp,
                  'protocol_sha256': _digest(protocol_bytes), 'parent_coordinates_sha256': base_coord_hash,
                  'curated_coordinates_sha256': new_coord_hash, 'protected_snapshots': protected_info,
                  'curated_records': len(new_rows), 'qualified_records_added_to_train': added,
                  'positive_records_added_to_train': sum(d['classification'] == 'positive' for d in decisions),
                  'negative_records_added_to_train': sum(d['classification'] == 'negative' for d in decisions),
                  'deferred_records_added_unassigned': len(new_rows) - added,
                  'previously_verified_base_supervised_train_records': protocol['base_supervised_train_records'],
                  'effective_supervised_train_records': protocol['base_supervised_train_records'] + added,
                  'inherited_supervision_verification': 'Previously verified frozen count; inherited masks were not reopened.',
                  'heldout_records_unchanged': True, 'all_inherited_records_unchanged': True,
                  'heldout_pixels_opened': False, 'protected_pixels_opened': False,
                  'inherited_raster_paths_resolved': False, 'inherited_pixel_storage': 'Immutable hard links',
                  'model_predictions_used': False, 'human_review_claimed': False, 'training_run_allowed': False,
                  'minimum_reference_distance_required_m': protocol['min_reference_distance_m'],
                  'minimum_reference_distance_observed_m': min(d['nearest_protected_center_m'] for d in decisions),
                  'minimum_candidate_center_spacing_m': protocol['min_candidate_distance_m'],
                  'pv_pixels_added': sum(d['positive_pixels'] for d in decisions),
                  'background_pixels_added': sum(d['background_pixels'] for d in decisions),
                  'ignored_pixels_added': sum(d['ignored_pixels'] for d in decisions), 'reviews': review_evidence,
                  'validation_scope': 'Complete canonical metadata, coordinates, new-photo pixels and exact review consensus; inherited rasters linked without opening or resolving.',
                  **review_context}
        (stage / 'augmentation-report.json').write_bytes(_json_bytes(report))
        evidence.check()
        _no_symlinks(output)
        _publish(stage, output)
        stage = None
        return report
    except BulkCurationError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError, OverflowError) as error:
        raise BulkCurationError(f'Bulk augmentation failed: {error}') from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


def _immutable_file(path, payload, evidence):
    """Publish a complete file atomically; existing bytes must match exactly."""
    _no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        if evidence.read(path) != payload:
            raise BulkCurationError('Existing immutable proposal-package bytes differ.')
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            _publish(temporary, path)
        except FileExistsError:
            if evidence.read(path) != payload:
                raise BulkCurationError('Concurrent immutable proposal-package bytes differ.')
        if evidence.read(path) != payload:
            raise BulkCurationError('Immutable proposal-package publication failed.')
    finally:
        temporary.unlink(missing_ok=True)


def _proposal_from_draft(payload, selected, batch_id, manifest_hash, fingerprint, protocol):
    draft = _review(payload, selected)
    if draft['reviewer'] != protocol['proposal_author']:
        raise BulkCurationError('Draft author differs from the exact frozen proposal-author identity.')
    proposal = {'proposal_schema_version': '1.0', 'batch_id': batch_id,
                'dataset_fingerprint': fingerprint, 'queue_manifest_sha256': manifest_hash,
                'draft_sha256': _digest(payload), 'samples': []}
    for key in selected:
        sample = draft['samples'][key]
        proposal['samples'].append({'id': key, 'image_sha256': selected[key]['image_sha256'],
            'positive_polygons': sample['positive_polygons'] if sample['decision'] != 'defer' else [],
            'background_polygons': sample['background_polygons'] if sample['decision'] != 'defer' else []})
    return proposal


def export_bulk_mask_proposals(dataset_dir: Path | str, queue_dir: Path | str,
                               draft_dir: Path | str, package_dir: Path | str, *,
                               protocol_path: Path | str) -> dict:
    """Immutably export available author drafts without reading any raster.

    Each complete proposal and exact draft are fixed before its independent
    checks. The combined index is published only after every draft is present.
    Repeated calls validate existing bytes; they never replace published files.
    """
    stage = None
    try:
        source, queue, drafts, package = [_absolute(p) for p in (dataset_dir, queue_dir, draft_dir, package_dir)]
        paths = [source, queue, drafts, package]
        for path in paths:
            _no_symlinks(path)
        if any(not p.is_dir() for p in (source, queue, drafts)):
            raise BulkCurationError('Source dataset, queue and draft directory must exist.')
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
               for i, a in enumerate(paths) for b in paths[i + 1:]):
            raise BulkCurationError('Proposal package and inputs must be separate and cannot be ancestors.')
        if _absolute(protocol_path).is_relative_to(package):
            raise BulkCurationError('Frozen protocol must exist outside the proposal package.')
        evidence = _Evidence()
        protocol, protocol_bytes = _protocol(protocol_path, evidence)
        if protocol['schema_version'] != '2.0':
            raise BulkCurationError('Proposal export requires the frozen V2 protocol.')
        metadata, rows, fingerprint = _manifest(source, evidence)
        originals = {row['id']: row for row in rows}
        if (metadata['dataset_id'] != 'nrw-training-bulk' or fingerprint != protocol['source_dataset_fingerprint']
                or not 1 <= len(rows) <= 2000 or any(row['split'] != 'unassigned' or row['annotation_status'] != 'unreviewed'
                    or row['width'] != 320 or row['height'] != 320 or row['gsd_m'] != .1 for row in rows)):
            raise BulkCurationError('Proposal export requires the frozen unreviewed native bulk source.')
        _, _, queue_bytes = _queue(source, queue, None, originals, fingerprint, protocol, evidence, load_reviews=False)
        queue_index = _object(queue_bytes)
        batches = queue_index['lanes'][0]['batches']
        expected_drafts = {drafts / (batch['id'] + '.review.json') for batch in batches}
        if not set(drafts.glob('*.review.json')) <= expected_drafts:
            raise BulkCurationError('Draft directory contains unbound or mixed batch files.')
        if not os.path.lexists(package):
            package.parent.mkdir(parents=True, exist_ok=True)
            _no_symlinks(package.parent)
            stage = Path(tempfile.mkdtemp(prefix=f'.{package.name}-proposal-package-', dir=package.parent))
            for name in ('proposals', 'source-drafts', 'lane-a', 'lane-b'):
                (stage / name).mkdir()
            _publish(stage, package)
            stage = None
        elif not package.is_dir():
            raise BulkCurationError('Proposal package must be a regular directory.')
        for name in ('proposals', 'source-drafts', 'lane-a', 'lane-b'):
            _no_symlinks(package / name)
            (package / name).mkdir(exist_ok=True)
        exported = []
        for batch in batches:
            path = drafts / (batch['id'] + '.review.json')
            if not os.path.lexists(path):
                continue
            payload = evidence.read(path, 2 * 1024 * 1024)
            selected = {key: originals[key] for key in batch['sample_ids']}
            proposal = _proposal_from_draft(payload, selected, batch['id'], batch['manifest_sha256'], fingerprint, protocol)
            proposal_bytes = _json_bytes(proposal)
            draft_relative = 'source-drafts/' + batch['id'] + '.review.json'
            proposal_relative = 'proposals/' + batch['id'] + '.proposal.json'
            # The complete draft exists before the proposal is visible to a checker.
            _immutable_file(package / draft_relative, payload, evidence)
            if evidence.read(path, 2 * 1024 * 1024) != payload:
                raise BulkCurationError('Source draft changed before proposal publication.')
            _immutable_file(package / proposal_relative, proposal_bytes, evidence)
            exported.append({'id': batch['id'], 'sample_ids': batch['sample_ids'],
                'queue_manifest_sha256': batch['manifest_sha256'], 'proposal': proposal_relative,
                'proposal_sha256': _digest(proposal_bytes), 'draft': draft_relative, 'draft_sha256': _digest(payload)})
        expected_proposals = {package / entry['proposal'] for entry in exported}
        expected_copies = {package / entry['draft'] for entry in exported}
        if (set((package / 'proposals').glob('*.proposal.json')) != expected_proposals
                or set((package / 'source-drafts').glob('*.review.json')) != expected_copies):
            raise BulkCurationError('Proposal package contains unbound files or a previously exported source draft is missing.')
        complete = len(exported) == len(batches)
        index = {'proposal_index_schema_version': '1.0', 'dataset_fingerprint': fingerprint,
                 'queue_sha256': protocol['queue_sha256'], 'protocol_sha256': _digest(protocol_bytes),
                 'proposal_author': protocol['proposal_author'], 'review_method': V2_METHOD,
                 'sample_count': len(rows), 'batch_count': len(batches), 'batches': exported}
        index_path = package / 'proposals/index.json'
        if complete:
            evidence.check()
            _immutable_file(index_path, _json_bytes(index), evidence)
        elif os.path.lexists(index_path):
            raise BulkCurationError('A finalized proposal index cannot coexist with missing source drafts.')
        evidence.check()
        return {'ok': True, 'complete': complete, 'ready_batches': len(exported), 'total_batches': len(batches),
                'ready_samples': sum(len(entry['sample_ids']) for entry in exported),
                'proposal_index_sha256': _digest(_json_bytes(index)) if complete else None,
                'dataset_fingerprint': fingerprint, 'queue_sha256': protocol['queue_sha256'],
                'protocol_sha256': _digest(protocol_bytes), 'review_method': V2_METHOD,
                'rasters_opened': False, 'per_batch_fixed_before_checks': True, 'batches': exported}
    except BulkCurationError:
        raise
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, TypeError, RecursionError, OverflowError) as error:
        raise BulkCurationError(f'Bulk proposal export failed: {error}') from error
    finally:
        if stage is not None:
            shutil.rmtree(stage, ignore_errors=True)


CHECK_FIELDS = REVIEW_FIELDS | {'proposal_sha256'}
CHECK_SAMPLE_FIELDS = {'id', 'image_sha256', 'full_image_inspected', 'proposal_overlay_inspected', 'decision',
                       'approved_positive_polygons', 'approved_background_polygons', 'notes'}
INDEX_FIELDS = {'proposal_index_schema_version', 'dataset_fingerprint', 'queue_sha256', 'protocol_sha256',
                'proposal_author', 'review_method', 'sample_count', 'batch_count', 'batches'}
INDEX_BATCH_FIELDS = {'id', 'sample_ids', 'queue_manifest_sha256', 'proposal', 'proposal_sha256', 'draft', 'draft_sha256'}


def _checker_review(payload, selected, proposal, proposal_hash):
    review = _object(payload)
    reviewer = review.get('reviewer')
    if (set(review) != CHECK_FIELDS or review['review_schema_version'] != '1.2'
            or review['reviewer_type'] != 'ai' or review['model_predictions_used'] is not False
            or review['proposal_sha256'] != proposal_hash
            or not isinstance(reviewer, str) or not 8 <= len(reviewer.strip()) <= 256
            or re.search(r'pending|placeholder|unknown|template|\btodo\b', reviewer, re.I)
            or not isinstance(review['samples'], list) or len(review['samples']) != len(selected)):
        raise BulkCurationError('Expected completed schema 1.2 from an identified AI checker bound to exact proposal bytes.')
    proposed = {sample['id']: sample for sample in proposal['samples']}
    samples = {}
    for sample in review['samples']:
        if not isinstance(sample, dict) or set(sample) != CHECK_SAMPLE_FIELDS:
            raise BulkCurationError('Expected exactly the independent polygon-check sample fields.')
        key = sample['id']
        if (not isinstance(key, str) or key not in selected or key in samples
                or sample['image_sha256'] != selected[key]['image_sha256']
                or sample['full_image_inspected'] is not True or sample['proposal_overlay_inspected'] is not True
                or sample['decision'] not in ('approve-positive', 'approve-negative', 'defer')
                or not isinstance(sample['notes'], str) or len(sample['notes']) > 8192):
            raise BulkCurationError('Invalid check ID/hash, native-image or proposal inspection, decision or notes.')
        proposal_polygons = {}
        for approved_field, polygon_field in (('approved_positive_polygons', 'positive_polygons'),
                                               ('approved_background_polygons', 'background_polygons')):
            values = sample[approved_field]
            polygons = _polygons(proposed[key][polygon_field], 320, 320)
            if (not isinstance(values, list) or len(values) > 64
                    or any(type(index) is not int or not 0 <= index < len(polygons) for index in values)
                    or len(set(values)) != len(values)):
                raise BulkCurationError('Approved polygon indices must be distinct bounded integers in the exact proposal.')
            proposal_polygons[polygon_field] = polygons
        if (sample['decision'] == 'defer' and (sample['approved_positive_polygons'] or sample['approved_background_polygons'])):
            raise BulkCurationError('Deferred checks cannot approve any proposed polygon.')
        if sample['decision'] == 'approve-negative' and sample['approved_positive_polygons']:
            raise BulkCurationError('Explicit full-image negative checks cannot approve PV polygons.')
        samples[key] = {**sample, 'proposal_polygons': proposal_polygons}
    if set(samples) != set(selected):
        raise BulkCurationError('Each checker batch must cover its exact bound sample IDs.')
    return {**review, 'samples': samples}


def _bound_reviews(source, queue, results, originals, fingerprint, protocol, evidence, *, protocol_sha256):
    if protocol['schema_version'] == '1.0':
        reviews, reports, _ = _queue(source, queue, results, originals, fingerprint, protocol, evidence)
        return reviews, reports, {}
    _, _, queue_bytes = _queue(source, queue, None, originals, fingerprint, protocol, evidence, load_reviews=False)
    queue_index = _object(queue_bytes)
    index_bytes = evidence.read(results / 'proposals/index.json', 2 * 1024 * 1024)
    index = _object(index_bytes)
    batches = queue_index['lanes'][0]['batches']
    if (set(index) != INDEX_FIELDS or not isinstance(index['batches'], list) or len(index['batches']) != len(batches)):
        raise BulkCurationError('V2 curation requires the complete immutable proposal index for every queue batch.')
    proposals, entries, draft_paths, proposal_paths = {}, [], set(), set()
    for entry, batch in zip(index['batches'], batches):
        batch_id = batch['id']
        draft_relative = 'source-drafts/' + batch_id + '.review.json'
        proposal_relative = 'proposals/' + batch_id + '.proposal.json'
        if (not isinstance(entry, dict) or set(entry) != INDEX_BATCH_FIELDS
                or entry['id'] != batch_id or entry['sample_ids'] != batch['sample_ids']
                or entry['queue_manifest_sha256'] != batch['manifest_sha256']
                or entry['proposal'] != proposal_relative or entry['draft'] != draft_relative):
            raise BulkCurationError('Proposal index batch identity, paths or lane-A queue binding differs.')
        draft_path, proposal_path = results / draft_relative, results / proposal_relative
        draft_payload = evidence.read(draft_path, 2 * 1024 * 1024)
        proposal_payload = evidence.read(proposal_path, 2 * 1024 * 1024)
        if _digest(draft_payload) != entry['draft_sha256'] or _digest(proposal_payload) != entry['proposal_sha256']:
            raise BulkCurationError('Indexed source draft or fixed proposal bytes changed.')
        selected = {key: originals[key] for key in batch['sample_ids']}
        proposal = _proposal_from_draft(draft_payload, selected, batch_id, batch['manifest_sha256'], fingerprint, protocol)
        if proposal_payload != _json_bytes(proposal):
            raise BulkCurationError('Fixed proposal differs from the exact bound author draft, including empty deferred regions.')
        proposals[batch_id] = proposal, entry['proposal_sha256']
        entries.append({'id': batch_id, 'sample_ids': batch['sample_ids'],
            'queue_manifest_sha256': batch['manifest_sha256'], 'proposal': proposal_relative,
            'proposal_sha256': _digest(proposal_payload), 'draft': draft_relative, 'draft_sha256': _digest(draft_payload)})
        draft_paths.add(draft_path)
        proposal_paths.add(proposal_path)
    expected_index = {'proposal_index_schema_version': '1.0', 'dataset_fingerprint': fingerprint,
                      'queue_sha256': protocol['queue_sha256'], 'protocol_sha256': protocol_sha256,
                      'proposal_author': protocol['proposal_author'], 'review_method': V2_METHOD,
                      'sample_count': len(originals), 'batch_count': len(batches), 'batches': entries}
    if index_bytes != _json_bytes(expected_index):
        raise BulkCurationError('Final proposal index does not exactly bind source, protocol and every immutable batch hash.')
    if (set(results.rglob('*.proposal.json')) != proposal_paths
            or {p for p in (results / 'source-drafts').rglob('*') if not p.is_dir()} != draft_paths
            or {p for p in (results / 'proposals').rglob('*') if not p.is_dir()} != proposal_paths | {results / 'proposals/index.json'}):
        raise BulkCurationError('Proposal package contains unbound or mixed proposal/draft files.')
    def checker_loader(payload, selected, lane, batch_id, batch):
        # Both checkers inspect the SAME lane-A-bound proposal. The original
        # lane-B manifest has different header bytes and is never substituted.
        proposal, proposal_hash = proposals[batch_id]
        return _checker_review(payload, selected, proposal, proposal_hash)
    reviews, reports, _ = _queue(source, queue, results, originals, fingerprint, protocol, evidence,
                                review_loader=checker_loader, additional_review_paths=draft_paths)
    if any(report['reviewer'].strip().casefold() == protocol['proposal_author'].strip().casefold() for report in reports):
        raise BulkCurationError('Both independent checker roles must differ from the common proposal author.')
    context = {'review_method': V2_METHOD, 'proposal_index_sha256': _digest(index_bytes),
               'proposal_author': protocol['proposal_author'], 'proposal_batches': entries,
               'shared_proposal_bias': 'Two independent AI checks share one AI visual proposal set; correlated proposal omissions remain possible.',
               'per_batch_fixed_before_checks': True, 'combined_index_finalized_after_author_completion': True}
    return reviews, reports, context
