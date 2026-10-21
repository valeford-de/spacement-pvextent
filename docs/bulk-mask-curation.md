# Bulk training mask curation

The [bulk imagery collection](bulk-training-data.md) contains 1,000 native
320 × 320 NRW photos with frozen source identities and an immutable two-lane
queue. Experimental AI annotation is explicitly authorized. These labels are
partial AI supervision, not human-reviewed benchmark ground truth.

The [V2 protocol](research/nrw-bulk-mask-curation-protocol-v2.json) uses one
visual proposal author and **two new independent checks of the same frozen
polygons**. Each checker inspects the full original RGB photo and the exact
indexed overlay, without seeing the author's decisions or the other checker's
results. A checker approves or rejects each entire polygon; uncertain or partly
wrong polygons must be rejected. No model predictions supply ground truth.

V1 attempted an intersection of independently drawn sparse interiors. The
[method amendment](research/nrw-bulk-mask-method-amendment-v2.md) records why
those regions seldom overlapped and preserves that diagnostic evidence.
V2 keeps the original pixel gates. Sharing proposals makes omissions and other
correlated errors possible even after two checks; label coverage remains sparse.

## Freeze common proposals

These commands require the `train` extra because the shared manifest helpers
live in the training data module; augmentation also requires `geo`. With a
complete native-image queue and identified schema 1.1 author drafts:

```sh
uv run --locked --extra train pvextent data prepare-mask-proposals \
  data/nrw-training-bulk-v1 artifacts/nrw-training-bulk-review-v1 \
  artifacts/nrw-training-bulk-mask-reviews-v1/lane-a \
  artifacts/nrw-training-bulk-mask-checks-v2 \
  --protocol docs/research/nrw-bulk-mask-curation-protocol-v2.json --json
```

This command can be repeated as drafts arrive. Each complete proposal and exact
source draft is published atomically and cannot be replaced with different
bytes. Common proposals exclude author notes and decisions. Each batch binds
the source image hashes, original lane-A manifest hash and exact draft hash.
Both checkers use that same proposal, including its lane-A binding. The complete
index is finalized only after all forty proposal batches exist.

The two checkers write schema 1.2 files to the new package's `lane-a` and
`lane-b` directories. Each file declares its distinct AI checker identity,
the exact proposal SHA-256 and `model_predictions_used: false`. Each sample
binds its image hash, records both full-image and proposal-overlay inspection,
and supplies a decision plus distinct, zero-based approved PV/background
polygon indices. A deferred decision cannot approve any region. An explicit
negative cannot approve a PV polygon. Empty proposals do not prove PV absence.

## Create immutable partial masks

After both checkers complete every batch:

```sh
uv run --locked --extra train pvextent data curate-training-bulk \
  data/nrw-training-bulk-v1 artifacts/nrw-training-bulk-review-v1 \
  artifacts/nrw-training-bulk-mask-checks-v2 \
  data/nrw-training-bulk-curated-v2 \
  --protocol docs/research/nrw-bulk-mask-curation-protocol-v2.json \
  --dataset-version 2026-10-21.bulk-curated-v2 --allow-ai-review --json
```

Only the **same polygon indices approved by both checkers** become labels.
Positive records require two positive decisions, at least 100 PV pixels and
256 background pixels. Negative records require two explicit negative
decisions, no approved PV region and at least 1,024 background pixels.
All other records remain deferred with entirely ignored masks. Label `1`
means confirmed PV interior, `0` approved background and `255` unknown.
Sparse interior labels do not represent complete array boundaries.

The curator verifies complete queue coverage, identities, inspection flags,
immutable proposals, drafts, indices and hashes before rasterizing. All 1,000
records stay unassigned in this separate curated snapshot. Raw evidence,
coordinate provenance, imagery attribution and new mask attribution remain
alongside the canonical output outside Git. Output directories cannot be
overwritten.

## Prepare the training snapshot

```sh
uv run --locked --extra train --extra geo pvextent data augment-training-bulk \
  data/pvextent-training-v2 data/nrw-training-bulk-curated-v2 \
  data/pvextent-training-v3 \
  --protected data/nrw-negative-holdout-candidates-v1 \
  --protocol docs/research/nrw-bulk-mask-curation-protocol-v2.json \
  --dataset-version 2026-10-21.training-v3 --json
```

Augmentation re-derives every new mask from the exact review evidence. Qualified
new records enter training; deferred records stay unassigned. All 4,034 base
record dictionaries are preserved, including 278 validation, 421 test and 42
unassigned records. Inherited rasters are hard-linked without opening, hashing,
decoding or resolving them. Only the new photos and masks are pixel-validated.
Do not run a generic full-dataset pixel validator or gallery over the combined
snapshot while the original test pixels are reserved.

Metadata and projected coordinates enforce the acquisition protocol's separation
from every base non-training record and all twenty external-pilot records.
New geometry candidates must also be separated from existing training centres;
repeat crops at known training locations inherit their training parent's group.
All new centres remain at least 80 m apart. The earlier separate 22-photo
annotation exercise is not silently included.

The resulting snapshot has 5,034 total records. The measured curation and
augmentation reports determine how many of the additional 1,000 qualify as
supervision. Preparing this snapshot does not launch training, modify the
existing checkpoints, or evaluate reserved pixels. The
[live monitor](live-training.md) displays progress from actual completed files.
