# Experimental negative curation, 2026-10-20

A first versioned collection of **eleven partially supervised negative masks**
has been created and added to a new training snapshot. Four other candidates
remain entirely ignored and unassigned. Original datasets, the pretrained
baseline checkpoint and the frozen validation/test records were preserved.

The user explicitly authorized AI-reviewed labels for experimental training
after two separate visual reviews. This collection claims **AI review, not
human approval**. The [aggregate audit](negative-curation-v1.json) records
fingerprints, review-file hashes, split counts and the training smoke result.
Real pixels, polygons, per-image decisions, masks and checkpoints remain outside
Git. See the [curation workflow](../negative-curation.md) for reproduction.

## Annotation process

Two separate AI review passes inspected all fifteen original native 320 × 320
images, checking sample identity, image hashes and unassigned source status
before pixel access. Neither pass used segmentation predictions or the other's
annotations. Both followed the [annotation guide](../annotation-guide.md).
Separate passes do not establish statistical independence; correlated AI
mistakes remain possible.

Each pass proposed conservative background polygons or deferred the crop.
Reviewer A approved eleven crops and deferred four; reviewer B approved twelve
and deferred three. A mask received background `0` only where **both** approved
polygons overlapped, and only after both approved the sample with at least 256
intersection pixels. All other pixels remained `255` ignore. No PV `1` labels
were generated. Disagreement deferred the whole crop.

| Result | Count |
| --- | ---: |
| Input crops | 15 |
| AI-reviewed partial negative masks | 11 |
| Deferred, all-ignore masks | 4 |
| Supervised background pixels | 217,276 |
| Ignore pixels | 1,318,724 |
| New supervised PV pixels | 0 |

Only 14.15% of the combined crop pixels became background supervision. The
remaining 85.85% were not silently interpreted as negatives. A partial negative
does not assert absence of PV in ignored areas, on the entire roof or on the
whole property. The final masks and overlays were also inspected on three
native-resolution contact sheets covering all fifteen crops.

The collection `pvextent-negative-curation-v1` retains every sample as
`unassigned`. Approved masks have `annotation_status = "reviewed"` with explicit
AI origin in new source entries and provenance; deferred masks remain
`unreviewed`. The collection passed canonical validation with no errors and
24 warnings, including inherited unknown source information and deferred masks.

Original imagery licenses and notices are preserved. Newly created masks are
identified as Valeford annotations under Apache-2.0, separately from source
imagery and original upstream annotations. Redistribution must include the
applicable [license](../../LICENSE) and notices alongside source attribution.

## Training snapshot and holdout protection

The curated collection was merged into the frozen geographic PV4GER source as
`pvextent-training-v2`:

| Partition | Original | New snapshot |
| --- | ---: | ---: |
| Train | 3,282 | 3,293 |
| Validation | 278 | 278 |
| Test | 421 | 421 |
| Unassigned | 46 | 42 |
| Total | 4,027 | 4,034 |

The eight PV4GER candidates replaced only previously unassigned records; seven
NRW records were added. Eleven reviewed negatives passed the training gate.
All originally reserved geographic-block assignments remain reserved across
successive augmentations. Duplicate held-out image content, spatial groups
or building identities are rejected before candidate pixel validation.

The minimum new-training-to-holdout centre distance was **3,258.64 m**, exceeding
the fixed 1,000 m buffer. Distances use source centres in EPSG:25832, not verified
building footprints. All **699 validation/test record dictionaries are exactly
unchanged**. Their raster references were checked with same-file inode checks;
the augmentation did not open, decode or evaluate held-out pixel files.

Inherited image/mask files use hard links, so snapshots must remain immutable.
Deleting a snapshot does not delete the other link, but editing linked bytes
would affect both. Future corrections must create new files in another new
snapshot. New masks are copied exclusively after hash checks, and target-path
collisions are rejected before linking inherited files.

Output validation checks the complete metadata/manifest and all curated
pixels. It does not newly decode inherited holdout rasters or certify their
annotation truth. Parent/source fingerprints and full review provenance are
retained with the snapshot.

## Executed training check

A separate negative-only subset containing the eleven newly eligible records
was prepared for the existing tiny training smoke workflow. Eight deterministically
selected examples supplied four MPS optimizer updates at batch size two.

- Fixed training-probe loss: `1.802925` → `1.756776`.
- Pre-clipping gradient norm: approximately `0.69896`–`0.72759`.
- Parameter-update L2 norm: `0.313694`.
- Checkpoint reload: exactly identical probe logits.

This verifies background/ignore supervision and the training/checkpoint path.
It does not fine-tune the pretrained PV baseline, measure generalization or
show a lower false-positive rate. Validation and test pixels were not used.
The earlier baseline score remains attached to its original dataset snapshot.

## Verification and next experiment

All **235 tests passed**, including exact polygon intersection and ignore
handling, authorization and review validation, source immutability, forbidden
held-out pixel access, geographic exclusions, preservation of block reservations
through two merges, and protection against inherited hard-link path collisions.
The wheel and source distribution also built successfully; packaged curation,
augmentation and CLI code match the source, with dataset pixels and checkpoints
excluded from the packages.

The next model experiment should compare a fixed training protocol using this
new snapshot against the existing baseline. Eleven partial negatives form a
small, selected addition; their sampling weight and effective supervision must
be recorded. Any validation comparison remains development evidence. No new
model-quality or independent benchmark result is claimed by this curation step.
