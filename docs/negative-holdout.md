# Experimental negative holdout

This workflow acquires a small, geographically separate review pool, freezes
two-pass AI background annotations before predictions, and evaluates the two
already fixed baseline checkpoints. It measures predicted PV on approved
background pixels. It does not create human-reviewed benchmark truth, a
roof-level false-positive rate, or a representative prevalence estimate.

The new holdout is separate from training and from the original 421 reserved
PV4GER test examples. Those original test rasters remain unopened throughout
acquisition, freezing and this evaluation. The new collection must not be
merged into training.
The standard training augmenter rejects this pool's curation lineage even when
all curated records are still unassigned; reservation covers the candidate
pool and review evidence as well as the final test selection.

## Protocol history and amendment

The committed [acquisition protocol](research/negative-holdout-acquisition-protocol-v1.json)
and [original V1 evaluation protocol](research/negative-holdout-evaluation-protocol-v1.json)
were pushed as commit `9e68cb4` before the first RGB fetch. They fixed:

- Three small geographic regions: Medebach, Bad Laasphe and Lienen, with at
  most eight candidate slots each, for a maximum of 24 crops.
- Exclusion against all 4,034 records in `data/pvextent-training-v2`, including
  its training, validation, test and unassigned records. Only declarations and
  coordinates are read; reference raster paths are not resolved or decoded.
- A minimum reference-centre distance of `5000 + 32 * sqrt(2)` metres
  (5,045.255 m), and at least 200 m between new candidate centres.
- Two separate full-image AI reviews, blind to each other and model predictions,
  with an explicit whole-crop deferral policy and at least 50% consensus
  background coverage for evaluation.
- The exact SHA-256 identities of both existing `best.pt` checkpoints, fixed
  threshold `0.5`, and no model, seed, threshold, sampling or annotation tuning
  from these holdout predictions in this experiment.

The V1 freeze admitted **zero eligible crops**. All seven mutually approved
masks were conservative partial annotations with 6,212–38,821 background pixels
each, only 6.1–37.9% of a full crop. None passed the 50% gate. No V1 frozen
output was published and no model predictions had been made. The original
protocol and local failure evidence, including
`artifacts/negative-holdout-freeze-v1.stdout.json` and its stderr log, are retained.

The [V2 evaluation protocol](research/negative-holdout-evaluation-protocol-v2.json)
was registered as commit `f6c771f` **before any model predictions**. It lowers
only the minimum consensus-background coverage to **5%: 5,120 pixels, or
51.2 m² at 0.1 m/pixel**, for an exploratory pixel diagnostic. The same twenty
images, two review files and two fixed checkpoints are retained. There is no
label expansion, replacement sampling or prediction-based selection. V2 records
its amendment rationale and the V1 protocol hash; it does not overwrite V1.
This is an amended protocol, not a completed evaluation under the original
50% rule.

The implementation accepts only `0.5` or `0.05` from a retained coverage
protocol, with no command-line option for arbitrary gate tuning. This admission
gate is separate from the models' unchanged probability threshold `0.5`.

Stable building identifiers and geometry determine candidate slots before RGB
inspection. Missing metadata, rejected imagery or later review deferrals do
not trigger replacement sampling to fill the quota. Geographic separation
applies to the supplied exclusion snapshots and their coordinates; it does not
establish independence from unknown pretraining imagery or acquisition campaigns.

## Acquire the unreviewed pool

From the repository directory:

```sh
uv sync --locked --extra train --extra geo --extra baseline
uv run --locked --extra train --extra geo pvextent data fetch-negative-holdout \
  data/nrw-negative-holdout-candidates-v1 \
  --exclude data/pvextent-training-v2 \
  --protocol docs/research/negative-holdout-acquisition-protocol-v1.json --json
```

This command makes bounded requests to the official NRW building, imagery
metadata and WMS services. `--exclude` can be repeated for additional complete
reference snapshots; the frozen evaluation protocol must identify the same
references. Keep a new output directory separate from all inputs. Existing
outputs are never overwritten.

Each accepted image is the unchanged RGB PNG response at **320 × 320 pixels,
0.1 m/pixel**, covering 32 × 32 m. One official metadata tile must support the
native resolution and acquisition date for the complete crop. Missing support
excludes the slot. Retrieval time and building-object currency are retained
separately from `capture_date`. The WMS grid is not a promise of pixel-identical
extraction from an original orthophoto file, and live service responses can change.

Every acquired mask contains only ignore `255`, with `annotation_status =
"unreviewed"` and `split = "unassigned"`. No negative label is inferred from
building geometry or a lack of mask annotations. `candidate-report.json`
records requests, exclusions, distances, reference fingerprints and the
candidate fingerprint. Original request responses, coordinate evidence,
protocol and source notices accompany the snapshot under `provenance/`.

## Review without model predictions

Two independent AI review passes inspect every complete native crop. Neither
uses the other pass's decisions, segmentation predictions, or prediction-based
rankings. Follow the [annotation guide](annotation-guide.md) and the frozen
review policy:

1. Recognizable PV anywhere in the crop, or unresolved possible roof collectors,
   means **defer the whole crop**. Do not carve an easy background region out
   of a positive or ambiguous roof to admit that crop to the negative holdout.
2. Otherwise, draw conservative background polygons. Trees, occlusion and
   unresolved non-roof areas stay ignored. Do not infer invisible surfaces.
3. Record the exact image SHA-256, complete-image inspection, reviewer identity,
   `model_predictions_used: false`, decision and native-pixel polygons in the
   dedicated [negative curation JSON format](negative-curation.md#prepare-the-review-evidence).

The two files must cover the same complete candidate-ID set. Generic gallery
decision downloads are evidence only and do not have this polygon-review
schema. A declaration of blindness records the review process; software cannot
certify the visual judgment or establish statistical independence between AI
reviewers.

Prepare a separate curated collection using the two completed review files:

```sh
uv run --locked --extra train --extra geo pvextent data curate-negatives \
  data/negative-holdout-curation-v1 \
  --source data/nrw-negative-holdout-candidates-v1 \
  --review data/reviews/negative-holdout-a.json \
  --review data/reviews/negative-holdout-b.json \
  --dataset-version negative-holdout-curation-v1 --allow-ai-review --json
```

The review paths above are local evidence files prepared by the two reviewers.
Only intersections of their approved background polygon unions become `0`;
all remaining pixels are `255`, and no PV `1` is generated. The generic curator
requires both approvals and at least 256 consensus pixels for `reviewed`
status, but keeps **every record unassigned**. That lower threshold does not
qualify a record for this holdout.

Source snapshots remain unchanged. New masks are Valeford annotations under
Apache-2.0; the original NRW imagery terms and notices remain separate. Keep
real imagery, polygons, review files and masks outside Git.

## Freeze the new evaluation split

```sh
uv run --locked --extra train --extra geo pvextent data freeze-negative-holdout \
  data/negative-holdout-curation-v1 data/nrw-negative-holdout-candidates-v1 \
  data/negative-holdout-frozen-v2 \
  --exclude data/pvextent-training-v2 \
  --protocol docs/research/negative-holdout-evaluation-protocol-v2.json \
  --acquisition-protocol docs/research/negative-holdout-acquisition-protocol-v1.json \
  --json
```

Freezing checks acquisition and reference identities, coordinate provenance,
distance gates, image hashes and both retained review files. It recomputes the
two-review polygon intersection and requires the curated mask to match exactly.
The source and curated collections must contain the same complete candidate
set, all still unassigned, with at most 24 records.

Crop centres are recomputed from their EPSG:25832 affine transforms and must
agree with retained coordinate provenance within 0.2 m. The fixed distance
gate is checked from those crop centres against every reference centre.

Under V2, only reviewed records with **at least 5,120 background pixels out of
102,400** are assigned to `test` in this new snapshot. Every other record remains
`unassigned`; it is not a training example. A deferred crop stays entirely
ignored. If no record qualifies, freezing fails. Background coverage and
annotation status do not certify that ignored areas or entire properties are
PV-free.

The freezer copies the new pixel files into a new canonical snapshot and
preserves acquisition and curation evidence. Its
`provenance/negative-holdout-freeze.json` binds the eligible IDs, masks, source
fingerprints, checkpoint identities and the ban on training use. It also
retains `protocol.json`, `acquisition-protocol.json`, source notices and a
canonical validation report. Both parent metadata files, manifests and reports
are retained under the acquisition/curation provenance directories. All input
collections remain unchanged. Freeze before producing either model's
predictions on these images.

## Evaluate both fixed checkpoints

Once review and freezing have completed successfully:

```sh
uv run --locked --extra baseline pvextent evaluate negative-holdout \
  data/negative-holdout-frozen-v2 \
  runs/baseline-resnet18-001/best.pt \
  runs/baseline-resnet18-negatives-001/best.pt \
  artifacts/negative-holdout-evaluation-v2 --device mps --json
```

The evaluator accepts only this explicit negative-holdout dataset type and its
frozen `test` selection. It verifies retained protocol/selection evidence, the
retained background-coverage gate (5% for V2) and exact checkpoint hashes,
including their SHA-256 companions. Both models are restored offline in
evaluation mode without an optimizer. Native RGB normalization and sigmoid
threshold `>= 0.5` are fixed;
there is no resizing or threshold option.

Use the original MPS device and package versions for the closest execution
context. `auto`, `cpu` and `cuda` are also supported when available; actual
versions, device and agreement with checkpoint contexts are recorded. Outputs
require a new directory disjoint from the dataset and checkpoint runs. Only
the newly frozen evaluation records are opened; deferred/unassigned pixel
files and the original PV4GER test rasters are outside this evaluation path.

| Output | Contents |
| --- | --- |
| `negative-evaluation-report.json` | Aggregate FP/TN, scored coverage, checkpoint/protocol identities, environment and candidate-minus-reference differences |
| `samples.json` | Per-crop measurements, image/mask hashes and prediction references |
| `predictions/` | Native single-channel `0/1` PNG predictions for each model |

For each model, a PV prediction on approved background is a false-positive
pixel; other approved background is true-negative. The reported pixel rate is:

```text
pixel_false_positive_rate = FP / (FP + TN)
```

Only consensus background enters that denominator. Predictions in ignored
areas are counted separately as **unscored**, never declared correct or false.
The count of crops with any false-positive pixels is a crop diagnostic, not a
roof-level rate. No recall, Dice or IoU is reported from this negative-only
collection. Rate differences are candidate minus reference; percentage-point
differences multiply that rate difference by 100.

## Executed pilot

The executed candidate snapshot contains **20 images from a maximum of 24**:
six Medebach, six Bad Laasphe and eight Lienen crops. It used three building,
23 metadata and 20 image requests, for 46 within the acquisition command.
Separate preliminary region-screening metadata requests are not included in
that count. All 20 have
officially supported acquisition dates and native 0.1 m/pixel grids. The
nearest accepted centre is **6,649.234 m** from the 4,034-record reference.
Acquisition dates are `2025-06-20` (six crops), `2025-06-21` (six),
`2026-03-18` (six) and `2026-03-19` (two).
Its dataset fingerprint is
`7a3ec2dfdc495de62778803e35435bf9a6fbe1884dda798b61574a28a2467218`.

The V2 snapshot was frozen before inference, with **seven new test crops**:
one from Medebach, three from Bad Laasphe and three from Lienen. The other
thirteen remain unassigned. The frozen dataset fingerprint is
`303f4f1429bf2a56c64e35cf2d3f218bdcb51bfe4549e66efae748820341587f`.
Affine-derived centre distances at freezing are at least 6,649.257 m across all
twenty records and **7,364.646 m among the seven eligible crops**. The small
difference from the acquisition minimum above reflects the recomputed crop
centres rather than the retained source-centre coordinates.

The seven evaluated crops supply **150,374 background pixels, or 1,503.74 m²**.
This is 20.9785% of their combined native pixels. Their remaining **566,426
pixels are ignored**. Both fixed checkpoints were evaluated on MPS at threshold
`0.5`, matching their recorded device and package versions:

| Measurement on the seven selected crops | Reference | Candidate |
| --- | ---: | ---: |
| False-positive pixels within approved background | 31 | 0 |
| True-negative pixels within approved background | 150,343 | 150,374 |
| Pixel FP / (FP + TN) | 0.0206153% | 0% |
| Crops with any such FP pixel, diagnostic only | 1 of 7 | 0 of 7 |
| PV predictions in ignored regions, **unscored** | 220 | 93 |

The ignored predictions are not false-positive or true-positive measurements.
Zero observed candidate FP pixels applies only to this small approved region;
it does not demonstrate zero false alarms on whole roofs or other imagery.
No recall, Dice or IoU follows from this negative-only evaluation.

The original datasets and checkpoint hashes are unchanged. All 699 original
validation/test declarations remain identical, and no original test pixels
were opened. No training or label adjustment followed these predictions.
Local completion was on 2026-10-21 in Europe/Berlin; protocol and acquisition
versions retain their 2026-10-20 UTC date. Full evidence is recorded in the
[holdout audit](research/negative-holdout-audit.md) and
[aggregate evidence](research/negative-holdout-v2.json).

This small geometry-selected convenience sample tests behavior on selected
AI-reviewed background regions, with possible correlated label errors. It
provides no estimate of positive-roof recall or representative prevalence and
does not establish statistical significance or general model superiority.
Preserve both baseline runs and the frozen evidence. Changing a model,
threshold, sampling rule or annotation
after inspecting these predictions requires a separate experiment and cannot
silently reuse this result as untouched evaluation evidence.
