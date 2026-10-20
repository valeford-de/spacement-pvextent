# Experimental negative curation

PVExtent can create a new collection of partial background masks from two
independent AI visual reviews, then add eligible records to a new training
snapshot. This workflow has explicit authorization for experimental training.
It does not claim human review or establish benchmark ground truth.

Both reviewers inspect the complete native RGB crop without consulting model
predictions. Each draws only regions they can identify as background. The
curator intersects their approved regions; it never fills an entire crop from
an empty source mask or a likely-negative classification. Ambiguous roofs,
shadows, equipment, and possible modules stay ignored.

## Prepare the review evidence

Use canonical source snapshots and retain their exact image hashes. Only
`unassigned` source records can be curated; existing training, validation and
test assignments are protected. The two review files must identify distinct
reviewers and cover exactly the same sample identifiers, with no duplicates.

This is a dedicated JSON format, separate from the downloads produced by the
[local review gallery](label-review.md). A schema example for one sample is:

```json
{
  "review_schema_version": "1.0",
  "reviewer": "ai-visual-review-a",
  "reviewer_type": "ai",
  "model_predictions_used": false,
  "samples": [
    {
      "id": "example-crop",
      "image_sha256": "<actual 64-character source image SHA-256>",
      "full_image_inspected": true,
      "decision": "approve-negative",
      "background_polygons": [[[12, 18], [94, 18], [94, 72], [12, 72]]],
      "notes": "Describe the inspected background and any excluded ambiguity."
    }
  ]
}
```

Replace the example identifier, hash, notes and geometry with actual review
evidence. The second reviewer supplies their own independently drawn polygons
and reviewer identifier. Do not copy the first review as a second inspection.
Use `"decision": "defer"` and an empty polygon list when the crop is unresolved.
Both files require `full_image_inspected: true` and
`model_predictions_used: false`; candidate prediction exports are not input to
this workflow.

Vertices are integer native-image pixel coordinates `[x, y]`, addressing pixel
centres; polygon boundaries are included. Each polygon must be simple,
nondegenerate, and inside the image, with 3–128 vertices. Each sample accepts
up to 64 polygons; each review accepts 1–64 samples, at most 8,192 vertices in
total, and at most 2 MiB. The fields shown above are the exact required fields;
unknown fields are rejected. See the [implementation](../src/pvextent/data/curation.py)
for the complete validation rules.

## Create a separate curated collection

Install the dependencies used by curation and geographic augmentation:

```sh
uv sync --locked --extra train --extra geo
uv run --locked --extra train --extra geo pvextent data curate-negatives \
  data/negative-curation-new \
  --source data/pv4ger-seg-geographic-v1 \
  --source data/nrw-roof-candidates \
  --review data/reviews/review-a.json \
  --review data/reviews/review-b.json \
  --dataset-version ai-consensus-new \
  --allow-ai-review --json
```

The review paths are files you prepare in the preceding step. Repeat `--source`
for each source snapshot; sample identifiers must be unique across sources.
The command runs offline, refuses an existing or overlapping output, and
publishes a staged collection only after validation. Source snapshots are
unchanged.

Within each review, polygons form a union. If both decisions are
`approve-negative`, their two unions are intersected. At least **256 consensus
pixels** are required to mark the new record `annotation_status = "reviewed"`.
Only those consensus pixels become background `0`; all other pixels remain
ignore `255`. A deferred review or insufficient consensus produces an entirely
ignored, `unreviewed` mask. No PV `1` pixels are generated, and every record
remains `split = "unassigned"`.

Here `reviewed` records the explicitly authorized AI process. AI origin and the
absence of human review are retained in source attribution, review files and
provenance; this status does not certify that the whole crop is PV-free.

The output includes `curation-report.json`, `validation-report.json`, the exact
two review files, original source records and hashes, coordinate provenance
where available, and source notices under `provenance/`. New masks are copyright
2026 Valeford under Apache-2.0. Original imagery rights and upstream annotation
licenses remain separate and their notices are preserved. Keep real images,
masks and review files outside Git, including the ignored local `data/` tree.

## Add eligible negatives to a new training snapshot

```sh
uv run --locked --extra train --extra geo pvextent data augment-training \
  data/pv4ger-seg-geographic-v1 \
  data/pvextent-negative-curation-v1 \
  data/training-augmented-new \
  --dataset-version training-augmented-new --buffer-m 1000 --json
```

The base, curated collection and output must be separate directories. The
output must be new and on a filesystem that supports hard links to its inputs.
All inherited pixel files are linked without opening or decoding the base
holdout rasters; curated masks are copied to separate files. **Treat both input
and output snapshots as immutable.** Editing a hard-linked image or inherited
mask in place also changes the other snapshot. Further corrections require
replacement files in another new version.

Only `reviewed` records with at least 256 background pixels and no PV labels can
enter training. The augmenter checks image hashes, spatial groups and building
identifiers against the frozen validation/test declarations, preserves existing
identity metadata, respects reserved geographic assignments, and requires a
centre distance of at least `--buffer-m` from every holdout record. The minimum
allowed buffer is 1,000 m. Deferred records and spatial exclusions remain
unassigned; validation and test records are unchanged. Distances use source
centres and do not establish independence from a published model's training
data.

`augmentation-report.json` records source fingerprints, split counts, added
training records, distance checks and validation scope.
`provenance/augmentation-decisions.json` explains each inclusion or exclusion;
the parent split provenance and curation evidence are retained. The operation
checks inherited declarations and validates curated pixels, without claiming
that it revalidated unopened holdout pixel content.

## Executed local collections

The executed `data/pvextent-negative-curation-v1` contains 15 records:
11 with reviewed partial background masks and four still unreviewed. Its masks
contain **217,276 background pixels, 1,318,724 ignored pixels and zero PV
pixels**. Its dataset fingerprint is
`cd4073e9adb75610ed4b8be80c6fbce5b5aea16888fd3090a4b967afedaa876e`.

The resulting `data/pvextent-training-v2` contains 4,034 records:

| Split | Records |
| --- | ---: |
| Training | 3,293 |
| Validation | 278 |
| Test | 421 |
| Unassigned | 42 |

Eleven reviewed negative records entered training. The nearest added training
centre is 3,258.635 m from the holdout. The snapshot fingerprint is
`1e78b1ce00d826e22d60c50f4c2f7c1b337bc3eeb9f36303b51e53daf0e54f14`.
All 699 validation/test record declarations remain identical, with their
inherited rasters verified as hard links without reading their pixels during
augmentation. See the [curation audit](research/negative-curation-audit.md) for
the measured evidence.

Creating this dataset does not retrain the original baseline or change its
validation results. The subsequent [controlled comparison](baseline-comparison.md)
is recorded in its own [audit](research/baseline-negative-comparison-audit.md).
These selected partial negatives do not estimate
real-world prevalence or provide human-reviewed benchmark truth; no PVExtent
weights are released by this workflow.
