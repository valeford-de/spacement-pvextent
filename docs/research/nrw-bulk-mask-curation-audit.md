# NRW bulk mask curation and final V4 audit

The final local training snapshot is `data/pvextent-training-v4`, version
`2026-10-21.training-v4`. It contains **5,034 records**: **3,775 TRAIN, 278
validation, 421 test and 560 unassigned**. Of the 1,000 additional NRW photos,
**482 contribute experimental partial supervision: 246 positive and 236 explicit
negative cases**. The other **518 are completely ignored and unassigned**.
The independent V4 technical audit completed with **PASS** on 2026-10-21 at
10:27:04 UTC. No new training, pilot inference or model change accompanied this
preparation.

The current dataset fingerprint is
`1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788`.
The [aggregate V4 record](nrw-bulk-mask-curation-v4.json) binds the selected
protocols, reports, execution records and audit artifacts by SHA-256. Raw
reviews, label geometry, pixels and complete per-file audit ledgers remain
outside Git.

## Final result and historical snapshots

| Stage | New positive | New negative | New ignored/unassigned | Combined TRAIN / val / test / unassigned |
| --- | ---: | ---: | ---: | --- |
| Initial two-check V2 curation, combined V3 snapshot | 247 | 238 | 515 | 3,778 / 278 / 421 / 557 |
| Final quality-quarantined V4 snapshot | 246 | 236 | 518 | 3,775 / 278 / 421 / 560 |

The initial curation supplied 32,369 PV and 953,250 background pixels. The final
V4 new masks contain **32,243 PV, 947,484 background and 101,420,273 ignore
pixels**. Thus 979,727 of 102,400,000 new pixels are supervised, approximately
**0.9568%**. These are sparse module-interior and background regions, not
complete PV array boundaries. Counts refer to the 1,000 new masks; inherited
base masks were not reopened to recount their pixels.

The original 4,034 base record dictionaries remain unchanged, including all
278 validation records, 421 original test records and 42 earlier unassigned
records. The base contributes its previously verified 3,293 supervised TRAIN
records; 3,293 + 482 gives the final 3,775. All 20 external pilot cases remain
protected. The earlier 22-photo NRW collection remains separate from these
1,000 photos.

| Immutable lineage stage | Dataset fingerprint |
| --- | --- |
| Raw 1,000-photo collection | `2b4409c46d14fb9c7cc4cefc13112bcdf53e2868fc805f3f542c75311672bb55` |
| Frozen 4,034-record base | `1e78b1ce00d826e22d60c50f4c2f7c1b337bc3eeb9f36303b51e53daf0e54f14` |
| Initial V2 curated new-photo dataset | `1ad9205e76bb4075bcd800c6066d8197622aafc9d66bf062eb5b2e4467d710d6` |
| Initial combined V3 snapshot | `e3279df66c913bdf502b5cae12d29ef27a673f13f1b485287dd2a453916bc5bd` |
| Final combined V4 snapshot | `1eb811eb633a476c4c1b87fa43e0fe274e630119ef195a6a8a3f5b2c13fc9788` |

The V4 root `quality-quarantine-report.json` supplies current totals. The V3
augmentation report, retained at V4's
`provenance/parent-augmentation-report.json`, describes the earlier result.
Both initial snapshots and their original reports remain unchanged.

## Source collection and scope

As recorded in the [collection audit](nrw-training-bulk-audit.md) and its
[aggregate evidence](nrw-training-bulk-v1.json), the source collection contains
600 current-source crops at known TRAIN locations and 400 geometry-selected
roof crops. It does not represent 1,000 independent new roofs. The known-location
crops remain correlated with their original training parents, and geometry
selection favors the registered roof criteria rather than a representative
benchmark sample.

Source capture dates range from 2022-06-15 to 2026-05-01: three photos are dated
2022, 331 are dated 2024, 481 are dated 2025 and 185 are dated 2026. Current-source
RGB uniqueness does not establish a different flight from the original imagery.
Native crops are 320 × 320 pixels at the recorded 0.1 m grid. The prior technical
audit measured a minimum new-center spacing of 80.068970 m and a minimum distance
to protected reference centers of 5,047.007867 m, above the required
5,045.254834 m. No holdout or pilot supervision was added to TRAIN.

## Why the method changed from V1 to V2

The [V1 protocol](nrw-bulk-mask-curation-protocol-v1.json) requested independently
drawn sparse PV-interior and background polygons. At the diagnostic checkpoint,
author A had 250 drafts and the original reviewer B had 200 drawings. Among
200 paired cases, 110 had two positive image decisions. Their median PV
intersection was four pixels; 49 had no PV intersection. Only one met the
100-PV-pixel gate and none met both positive gates. Of 40 paired negative
decisions, six met the 1,024-background-pixel gate.

A further visual check of six paired positive cases found different selections
of module rows or array interiors and arbitrary background locations, with no
systematic coordinate-transform error in those cases. This diagnosed failure of
sparse drawing intersection, not model performance or annotation accuracy. V1
produced no canonical bulk masks or training run. Original drawings, ledgers,
failed-method diagnostics and reviewer B evidence were retained unchanged.
The [registered V2 method amendment](nrw-bulk-mask-method-amendment-v2.md)
records the change before canonical curation.

## Two independent checks of shared frozen proposals

Under the [V2 curation protocol](nrw-bulk-mask-curation-protocol-v2.json), author
A prepared visual proposals for all 1,000 photos in 40 batches of 25. Two **new**
AI checker roles, C and D, each inspected all 1,000 full native RGB photos and
the exact corresponding proposal polygons. They did not receive the author's
approval decisions or notes, or the other checker's decisions. Historical lane
A/B storage names refer to checker C/D here; they do not make the proposal author
one of the two checkers.

Each batch's source and proposal hashes were fixed before its checks. The
combined proposal index was finalized after all author drafts were available.
The independent audit verifies 40 completed batches per checker, exactly 1,000
cases per checker, and the exact source, queue, proposal and raw-review hash
bindings. The source queue hash is
`6b3c65fe222867847eadb582a8cd8b3a1b7a217797c1ccfc1d2d27b5bacdce14`;
the final proposal-index hash is
`371537520f9bcd470c551ccf30b728f8c84772af7f2eb321a2224a01cfe10e07`.

A checker approved or rejected each complete proposed polygon. Uncertain or
partly incorrect polygons were rejected in full. For each class, curation
intersected the two checkers' **approved polygon index sets** and rasterized
those same whole polygons from the frozen proposal. It did not intersect two
new independently drawn masks.

The original gates were unchanged:

- Positive: both checkers approve the positive image decision, with at least
  100 jointly approved PV pixels and 256 jointly approved background pixels.
- Explicit negative: both checkers assert whole-image absence of visible PV,
  with zero approved PV pixels and at least 1,024 background pixels. A plausible
  background patch alone does not establish whole-image PV absence.
- Every other case remains completely ignored. In eligible masks, all pixels
  outside the jointly approved regions remain ignore. Mask values are PV `1`,
  background `0` and ignore `255`.

These are experimental AI visual reviews. No segmentation-model predictions or
pseudo-ground-truth supplied the proposals or decisions. Separate checker roles
and blinded decisions do not remove shared-proposal bias or correlated AI errors.

## Additional quality exclusions

After the technically correct two-check curation, a deterministic visual spot
check selected 12 eligible positive and 12 eligible negative cases, taking the
first cases in each class ordered lexicographically by SHA-256 of the UTF-8
sample ID. It inspected native RGB and overlays of the actually saved masks.
One positive region contained 126 PV-labelled pixels whose boxes could cross
panel frames or inter-row substrate.

The spot-check descriptions of two other cases considered their background
patches plausible while noting PV elsewhere in the photo. A metadata check
identified the conflict with explicit whole-image negative decisions. Subsequent
full native inspection of all 12 selected negative photos found clear PV in one
case and unresolved glass/PV ambiguity in the other.

The [quality amendment](nrw-bulk-mask-quality-amendment.md) documents these three
cases. The [first quality protocol](nrw-bulk-mask-quality-quarantine-protocol-v1.json)
registered only the disputed positive case and **was never executed**. The
[final V2 quality protocol](nrw-bulk-mask-quality-quarantine-protocol-v2.json)
added both contradictory negative cases before the first quarantine execution.

Exactly three new TRAIN records were moved to unassigned with entirely ignored
masks. Their photos remain available. This withdrew 126 PV and 5,766 background
pixels, adding 5,892 ignore pixels. It did not relabel disputed pixels as
background. Every other new mask and record dictionary, all 997 of them, stayed
unchanged. The original check decisions, raw evidence, earlier snapshots and
pixel gates were preserved.

This 24-case selected check and the subsequent negative reinspection do not
estimate an annotation error rate or certify the remaining labels. The quality
vetoes are conservative exclusions pending better adjudication.

## Independent audits and protected data

The initial V2 audit passed for source commit
`3c8924ebb9f60874bb0f91678fda7a7006d0a35b`. Its artifacts remain at
`artifacts/nrw-bulk-mask-final-independent-audit-v2/audit.json` and `audit.md`.
It reconstructed all 1,000 new masks from the jointly approved proposal indices
and checked exact equality in both the curated and combined V3 snapshots.
Its first harness attempt incorrectly required new TRAIN tile groups to be
unique; only that auditor-only assertion was removed. The complete rerun
passed with reference disjointness and all distance gates intact. The original
failed attempt and correction are retained; the auditor changed no source,
review or dataset files.

The final V4 audit passed for source commit
`69e096b56a775fa28c15708efa2fc4770889e613`. Its artifacts are
`artifacts/nrw-bulk-mask-final-independent-audit-v4/audit.json` and `audit.md`.
It freshly rasterized the original V2 approved indices independently, verified
all 1,000 source V3 masks and all 1,000 final V4 masks, and applied exactly the
three frozen complete-ignore vetoes. It also confirmed the 997 unchanged new
mask files and dictionaries, all 1,000 new RGB hash declarations, all 5,034
source image inodes, and byte-for-byte equality of all 5,034 coordinate rows.
The original V2 audit and 2,474 raw-review, queue and parent-provenance metadata
hash bindings were reverified unchanged.

Protection checks preserved all 4,034 base dictionaries and 8,068 inherited
raster inodes. The 20 external pilot cases account for 40 image/mask paths,
checked by metadata/stat only. The final audit checked 8,108 protected raster
path identities with **zero protected open, hash, decode or resolve attempts**.
The original 421 test images stayed closed. No generic combined-dataset pixel
validator or gallery ran.

The canonical curation and augmentation executions are recorded in
`artifacts/nrw-bulk-mask-curate-execution-v2.json` and
`artifacts/nrw-bulk-mask-augment-execution-v2.json`. The separate quarantine
execution is recorded in
`artifacts/nrw-bulk-mask-quality-quarantine-execution-v2.json`. Its reviewed
wrapper guarded 18,173 raster paths and 10,105 inodes, including hard-link
aliases: all base/pilot rasters, all V3 images and all unchanged V3 masks were
blocked. Only the three explicitly excluded new V3 masks were authorized for
pixel reads. Execution exited successfully with zero blocked access attempts.
All other rasters were hard-linked without opening or resolving them.

The technical PASS verifies deterministic construction, metadata preservation
and evidence binding. It does not validate the semantic correctness of every
approval, estimate annotation accuracy, or measure model quality. Preparing this
research report read and hashed only JSON, Markdown and code metadata; it did
not inspect images, masks or checkpoints.

## Verification, monitoring and next use

The recorded full test suite passed **433 tests** in
`artifacts/bulk-mask-quality-full-tests.log`. No new training run or pilot
reinference was performed; the model is unchanged. The protocols continue to
record `training_run_allowed: false`. This preparation therefore supplies an
audited experimental input snapshot, not a new model result.

The implemented [live training monitor](../live-training.md) polls every two
seconds and prefers the current V4 quality report. Future baseline runs emit
actual scalar optimization-batch and epoch telemetry. Completed run histories
are separately labelled; existing history does not imply that a new run is
active. The monitor does not start training. The
[quality-quarantine guide](../training-quality-quarantine.md) describes the
immutable exclusion command and its access boundaries.

Fuller target-source annotations and a separately reviewed evaluation set remain
necessary before claims about complete array extraction or model improvement.
The sparse supervision, shared proposals, correlated AI review, known-TRAIN
location dependence and selected quality-check coverage remain material limits.
There is no human-confirmed benchmark ground truth in this curation.

## Selected evidence bindings

The complete compact list of selected metadata/code SHA-256 bindings is in the
[aggregate V4 record](nrw-bulk-mask-curation-v4.json). The following bind the
final outcome directly; paths identify local retained artifacts.

| Artifact | SHA-256 |
| --- | --- |
| `data/pvextent-training-v4/quality-quarantine-report.json` | `0d395111f6cd66dedfb216b2b281a08abed2e0f072043bc26d1e7756d0ebabb4` |
| `docs/research/nrw-bulk-mask-quality-quarantine-protocol-v2.json` | `4f5fc0550e5a708697a7831b4d735d509edbfe1552f5d36cb5cb0a7345bb728b` |
| `artifacts/nrw-bulk-mask-quality-quarantine-v2/quality-evidence.json` | `77d305014687438623a69923b4a9ccf18c702850454ae649fbefae940f285a67` |
| `artifacts/nrw-bulk-mask-quality-quarantine-execution-v2.json` | `9ff719ba2b3656ceae390dbc2f7e3d1edf385ff9b37ea5ac1f463c99e3bfc2f3` |
| `artifacts/nrw-bulk-mask-final-independent-audit-v4/audit.json` | `a4f3b3d61f60117d697fc61f9a9a5a3a22468644debbda8aa8373321391e9887` |
| `artifacts/nrw-bulk-mask-final-independent-audit-v4/audit.md` | `ccf26ffafcb1c3a4727f366f93bd66999d28708aa1931374ec0d101ce3e33b46` |

Per-ID decisions, native geometry, raw review notes and the audits' large file
hash inventories are intentionally retained only in local evidence artifacts.
Their contents are bound by the aggregate report and audit hashes above rather
than copied into version-controlled research documentation.
