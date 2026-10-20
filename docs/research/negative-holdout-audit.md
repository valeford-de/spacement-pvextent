# Spatially separate negative pilot, 2026-10-21

Both frozen models were evaluated on **seven geographically separate,
AI-reviewed partial negatives**. Within **150,374 approved background pixels**,
the original baseline predicted PV on **31 pixels** and the augmented candidate
on **zero**. This is a small exploratory result, not evidence of a zero real-world
error rate or a roof-level false-positive rate.

The [aggregate audit](negative-holdout-v2.json) retains measured counts,
fingerprints, protocol identities and limitations. See the
[workflow guide](../negative-holdout.md) for reproduction. Images, polygons,
individual decisions, per-crop predictions and checkpoints remain outside Git.
The local date is 21 October; acquisition, protocol commits and evaluation
timestamps are recorded in UTC on 20 October 2026.

## Selection and geographic separation

Three fixed small NRW query regions were selected using existing coordinates
and official metadata, before inspecting any new RGB images or predictions.
The [acquisition protocol](negative-holdout-acquisition-protocol-v1.json) and
initial evaluation protocol were committed as `9e68cb4` before image requests.

| Region | Acquired crops | Evaluated negatives | Image acquisition dates |
| --- | ---: | ---: | --- |
| Medebach | 6 | 1 | 2025-06-20 |
| Bad Laasphe | 6 | 3 | 2025-06-21 |
| Lienen | 8 | 3 | 2026-03-18 and 2026-03-19 |

Each region allowed at most eight crops. Geometry, stable building identifiers,
200 m candidate-centre spacing and source metadata determined the slots.
One geometry slot was unavailable and three selected slots lacked verified
native-resolution metadata. They were not replaced. Building queries were
bounded to the first 500 returned features per region; this is a convenience
sample, not a complete or representative survey.

The acquisition command made **46 requests**: three building, 23 metadata and
20 image requests, without hidden retries. Preliminary region selection used
separate bounded metadata requests, which are not included in this command's
request count. All accepted images are native-metadata-confirmed 10 cm,
320 × 320 RGB PNG responses retained byte-for-byte, without resizing. Exact
WMS rendering is not claimed to be a pixel-identical original-file extraction.

Every candidate was checked against **all 4,034 existing sample centres**,
including training, validation, the old test split and unassigned records.
Identity, image-hash, building-ID and spatial-group overlaps were excluded.
The required centre buffer was `5000 + 32√2 = 5045.2548 m`. The smallest
acquired-building-centre distance was 6,649.2345 m. Native crop centres were
then derived from their affine grids and checked against coordinate provenance
within 0.2 m. Their minimum over all 20 records was **6,649.2574 m**; among the
seven evaluated records it was **7,364.6463 m**.

Reference metadata and coordinate hashes are bound to the acquisition evidence
and checked again at freezing. Old reference rasters were never opened or
decoded. These distances concern declared centres, not surveyed footprints or
independence from every unknown pretraining source.

Sources are the [official ALKIS building collection](https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk)
and [Geobasis NRW orthophoto service](https://www.wms.nrw.de/geobasis/wms_nw_dop).
Their DL-DE-Zero-2.0 notices and exact bounded responses remain with the local
snapshot. New Valeford masks retain separate Apache-2.0 annotation provenance.

## Blind review and the explicit protocol amendment

Two separate AI passes each inspected all twenty native originals without
model predictions or the other pass's decisions. Pass A approved nine and
deferred eleven; pass B approved seven and deferred thirteen. Seven were
approved by both. Visible PV or unresolved possible roof collectors deferred
the entire crop. Background is restricted to the intersection of approved
polygons; unresolved areas remain ignore. Separate passes do not establish
statistically independent AI errors or human-reviewed benchmark truth.

The [initial evaluation protocol](negative-holdout-evaluation-protocol-v1.json)
required background supervision on at least **50% of each whole crop**.
**No mask passed.** The seven conservative intersections contained 6,212–38,821
background pixels each, only 6.07%–37.91% of a crop. The attempted freeze failed
without publishing a dataset or making model predictions. The original
protocol and failure evidence were retained.

An explicit [V2 amendment](negative-holdout-evaluation-protocol-v2.json), committed
as `f6c771f` **before the first model prediction**, changed only the coverage
criterion to **5%: 5,120 pixels or 51.2 m²** of approved background per image.
This permits a useful area of conservative background without requiring half
of an entire scene to be resolved. It does not establish complete roof labels.
The acquired images, both sealed reviews, polygons, checkpoint identities and
model threshold were unchanged. No labels were expanded and no candidates
were added to improve the outcome.

This amendment followed inspection of annotation coverage, so the result is
explicitly exploratory. It must not be represented as a successful execution
of the original preregistered 50% protocol.

## Frozen dataset and measurements

V2 contains **seven reviewed `test` records and thirteen `unassigned` records**.
The thirteen excluded cases remain all-ignore. The evaluator opens only the
seven newly selected test records. The previously reserved 421 test images
remain unopened; they are a different partition.

The seven selected masks supervise **150,374 background pixels = 1,503.74 m²**.
Their other **566,426 pixels remain ignored**. Only 20.9785% of the selected
crop pixels are scored. Mask consensus bytes, two review-file hashes, source
identities and geographic exclusions were checked before freezing. Both
original models and threshold `0.5` were already fixed before image acquisition.

| Scored-background measure | Original baseline | Augmented candidate |
| --- | ---: | ---: |
| Approved background pixels | 150,374 | 150,374 |
| Predicted PV on approved background: FP pixels | 31 | 0 |
| Predicted background on approved background: TN pixels | 150,343 | 150,374 |
| FP / approved background pixels | 0.0206153% | 0% |
| Crops with any scored FP pixels | 1 of 7 | 0 of 7 |

The pixel-rate difference is **−0.0206153 percentage points**, candidate minus
reference. A crop with any predicted pixel is a diagnostic unit, not a roof
detection event. These numbers do not supply roof-level rates or population
confidence bounds. No recall, Dice or IoU was computed on this negative-only
set.

The reference also predicted PV on 220 ignored pixels and the candidate on 93.
**Those predictions are unscored**: their true status is unknown. They are
neither counted as false positives nor used to claim an improvement.

Inference ran offline on MPS in evaluation mode with gradients disabled,
batch size two, the original preprocessing and matching checkpoint package
versions. There was no training, threshold tuning, annotation editing or model
selection from these predictions. The new test collection, its candidate
pool and its curation evidence are reserved from future training; the standard
augmentation path rejects this curation lineage.

## Verification and interpretation

All **295 tests passed**, covering native-pixel and ignore semantics, exact
coverage boundaries, retained review evidence, coordinate provenance, frozen
checkpoint hashes, protection of old raster paths, training-lineage exclusion
and atomic publication. Wheel and source-distribution builds also passed; the
packaged holdout, training-augmentation and CLI code match the source, with
real datasets, reviews and checkpoints excluded. Independent code review found
a coordinate-provenance
binding gap; it was fixed and regression-tested before the V2 freeze.

All original dataset metadata/coordinates and model artifacts retained their
recorded hashes. The 699 original validation/test declarations remained
identical. These checks do not claim to have newly read or revalidated old test
pixel contents. Two native contact sheets of masks and two of model overlays
were inspected; the gallery independently recounts its FP/ignore pixels and
matches the evaluation report. Neither labels nor outputs were edited in
response to predictions.

The candidate's zero scored FP pixels is encouraging, but seven partial
AI-reviewed examples are too few and too selected to establish broad false-
positive performance. The 13 deferred cases, uncertain structures and ignored
pixels may contain precisely the harder situations. The experiment cannot
measure missed PV because it has no supervised positive pixels.

Keep this frozen pilot as a diagnostic. Next steps are a separate positive
collection and broader, clearly reviewed negative coverage under a new fixed
evaluation protocol. Preserve this set from training and avoid selecting seeds,
thresholds or architectures repeatedly on these seven cases. The original
[internal validation tradeoff](baseline-negative-comparison-audit.md) remains:
higher candidate precision accompanied lower recall there. No model weights
or independent human-confirmed benchmark were released.
