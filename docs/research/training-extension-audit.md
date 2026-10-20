# Six-epoch continuation audit

Local date: **2026-10-21**, Europe/Berlin. UTC logs use 2026-10-20.
The [aggregate JSON](training-extension-v1.json) records measured metrics,
checkpoint identities, input hashes and the separate training-review pools.
The [baseline guide](../baseline-training.md) gives the exact continuation command.

## Training result

The augmented candidate was continued from epoch three through **epoch six**,
adding three epochs and 4,941 optimizer updates. The run uses the unchanged
3,293-image training split and 278-image internal validation split. Its final
global step is **9,882**. It adds no newly collected NRW examples or labels.

The best checkpoint is **epoch 5**, with internal pixel IoU
**0.92183825** and Dice **0.95932969**. IoU improves by
**0.280326 percentage points** over the parent best. The fixed
last epoch has lower IoU (0.91783037), so it does not replace the best.
The gain is small and does not demonstrate better external transfer.

| Internal validation measure | Parent best, epoch 3 | Continued best, epoch 5 | Fixed final epoch 6 |
| --- | ---: | ---: | ---: |
| Pixel IoU | 0.91903499 | 0.92183825 | 0.91783037 |
| Dice | 0.95780952 | 0.95932969 | 0.95715490 |
| Precision | 0.96084849 | 0.96085002 | 0.95256894 |
| Recall | 0.95478971 | 0.95781417 | 0.96178524 |
| TP pixels | 5,823,842 | 5,842,290 | 5,866,512 |
| FP pixels | 237,303 | 238,045 | 292,110 |
| FN pixels | 275,765 | 257,317 | 233,095 |
| TN pixels | 22,130,290 | 22,129,548 | 22,075,483 |

The best checkpoint is selected solely by internal validation pixel IoU. The
fixed final epoch is reported even when it is not the best. These are global
masked pixel metrics against upstream labels, not array/roof scores or an
independent external benchmark. The earlier positive pilot motivated concern
about transfer; it was not rerun to choose checkpoints, thresholds or settings
for this control. No new external performance claim follows from this run.

| Additional epoch | Validation pixel IoU | Dice | FP pixels | FN pixels |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 0.91074304 | 0.95328678 | 223,900 | 340,517 |
| 5 | 0.92183825 | 0.95932969 | 238,045 | 257,317 |
| 6 | 0.91783037 | 0.95715490 | 292,110 | 233,095 |

## Registered configuration and immutable inputs

The [duration-control protocol](training-extension-protocol-v1.json) and
[six-epoch TOML](../../configs/baseline-resnet18-continued6.toml) were committed
as `6a32373` before training. Only `epochs` changes, from three to six. AdamW,
learning rate `0.0001`, weight decay `0.0001`, batch size two, seed `20261020`,
horizontal/vertical flips, native RGB normalization and threshold `0.5` remain
unchanged. Regular resume enforces the original dataset/split fingerprints,
encoder, preprocessing, device, package versions and optimizer/RNG state.

- Dataset fingerprint:
  `1e78b1ce00d826e22d60c50f4c2f7c1b337bc3eeb9f36303b51e53daf0e54f14`.
- Parent `last.pt` SHA-256:
  `cdb958053609ee1b3cb4781f59518da9eb921ab28c5f8d2d379a6fcbd728e720`.
- New best checkpoint SHA-256: `2e04c0e009f3b76a1a0c233aa9efea064251f6aed572d8db9272c2455c83edc6`.
- New last checkpoint SHA-256: `d7ed0d58e3230c55edac19aaca23d593bf16e1b8695f399a948fd9a70a298747`.

The continuation is published under `runs/baseline-resnet18-continued-001`.
The original parent and reference checkpoints remain unchanged. Start-validation
confusion counts exactly reproduce the parent's final counts. Reloading the
new best checkpoint also exactly reproduces its recorded validation counts.
Model parameters changed during optimization; the run records a finite,
nonzero parameter-update norm. All 26 retained source/model artifacts and all
699 original validation/test record dictionaries remain unchanged.
The original 421 test rasters and the external twenty-case pilot rasters were
not opened for this work.

Training ran on Apple M2 / MPS with the same package versions as its parent.
Recorded wall time is 1376.68 seconds (about 22.94 minutes).
Acquisition and CPU tests ran concurrently, so elapsed time is not a dedicated
throughput benchmark. Training implementation bytes match the registered
commit; the acquisition refactor does not change model or trainer code.

## Separate new NRW training-review inputs

A new [training acquisition API](../nrw-candidates.md#separate-training-review-pools)
was implemented and registered as `d769751`. The
[selection plan](nrw-training-selection-plan-v1.json) and
[initial acquisition protocol](nrw-training-acquisition-protocol-v1.json)
precede RGB inspection. ID, image-hash, building and spatial-group exclusions
cover **all 4,054 records** in the full training snapshot and twenty-case pilot.
Geographic distance protects **761 val/test/unassigned centres**, including every
pilot case. New training imagery may be near existing train-only examples;
this does not relax evaluation acquisition's all-split rule.

Two combined-pool attempts encountered a WMS timeout and connection reset;
neither published a partial dataset. The
[region-batching record](nrw-training-region-batching-v1.json), committed as
`939a3c9`, retains the same regions, 24 slots, distance rules and quotas. Each
region then publishes independently. There is no replacement sampling or
prediction-based choice. Actual successful acquisitions are:

| Region | Native crops | Capture date | Minimum actual affine-centre distance to protected reference |
| --- | ---: | --- | ---: |
| Schleiden | 8 | 2025-03-07 | 11,399.725 m |
| Dülmen | 8 | 2026-04-07 | 10,296.754 m |
| Marsberg area | 6 | 2025-06-30 | 5,635.661 m |

The **22 native RGB crops** are all 320 × 320 at verified 0.1 m/pixel, retain
exact source response bytes, and have unique IDs, buildings and image hashes.
An independent metadata audit checked full-crop GFI support, reference hashes
and actual affine crop centres. Global candidate spacing is at least
**207.294 m**. Two Marsberg slots cross a returned metadata-tile boundary;
the tile reports 0.10 m but does not cover the full crop, so full-crop native
resolution/date are unconfirmed and those slots are excluded. This is not
proof that their source imagery has 20 cm resolution.

Every canonical mask remains entirely `255`, status `unreviewed`, split
`unassigned`, and `training_use_allowed: false`. There are **zero new canonical
supervised pixels**, and no new crop was added to the continuation. The source
images retain DL-DE-Zero-2.0 and their acquisition notices. Region fingerprints,
exact request counts and minimum distances are in the aggregate JSON; building
responses are bounded feature prefixes, not complete regional surveys.

Two new independent blind AI workflows each inspected all 22 originals and
approved the same seven positive crops, deferring fifteen. Their exact review
hashes are recorded in the aggregate JSON. Proposal schema, simple polygon
geometry, image hashes and within-review PV/background non-overlap passed
validation; native proposal overlays were inspected. The intersections supply
**8,938 proposed PV pixels** and **20,899 proposed background pixels** across
seven cases at the preliminary 100-PV/256-background gate. These are partial
experimental AI proposals, not human-confirmed truth or imported canonical
labels. The canonical datasets remain unchanged and wholly ignored. No model
prediction was used for these proposals.

## Validation and next work

The complete suite passed **334 tests**, including unchanged negative-acquisition
contracts, near-train eligibility, metadata-only protection of all twenty pilot
cases, strict training split rules, native metadata gates and atomic publication.
Wheel and source-distribution builds passed. Changed module bytes match the
working source, package metadata contains the final README, and real imagery,
masks, reviews, runs and checkpoints are excluded.
Real imagery, labels, review decisions, checkpoints and raw run outputs remain
outside Git. Original model/data artifacts are retained.

Longer optimization is a development control. It does not establish a remedy
for the observed external failure. The new raw images and any review proposals
still require positive-mask curation, an immutable training snapshot and a
separate registered NRW-enriched experiment. A handful of positive examples can
be overwhelmed by 3,293 uniformly sampled records; record sample and supervised
pixel shares and design sampling explicitly. Broader independent positive
evaluation is needed before relying on internal validation gains.
