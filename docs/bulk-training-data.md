# Bulk NRW training imagery

The bulk collector prepares additional photos for experimental training review.
The registered [protocol](research/nrw-training-bulk-protocol-v1.json) targets
**1,000 distinct photos**: 600 current captures at original PV4GER training
locations and 400 ALKIS geometry crops separated from original training centres.
The [selection record](research/nrw-training-bulk-selection-v1.json) binds the
reference metadata and coordinates before the first new RGB request.

The collection is now complete: **1,000 photos**, independently verified, with
forty pending review batches per lane. Capture dates span 2022–2026; current
source availability does not mean every crop is newly flown. No labels or new
model were created. See the [measured audit](research/nrw-training-bulk-audit.md).

Current captures at known training locations may show changed roofs or arrays.
They are additional photos of existing training locations, not independent new
roofs. No old mask is read or transferred. Geometry crops use the existing
small-roof rules; neither path guarantees PV presence. This is a convenience
sample for training, not a representative evaluation set.

## Acquisition

Install the `geo` and `train` extras. Run from the repository root:

```sh
uv run --locked --extra train --extra geo pvextent data fetch-training-bulk \
  data/pvextent-training-v2 data/nrw-training-bulk-v1 \
  --protected data/nrw-negative-holdout-candidates-v1 \
  --identity-only data/nrw-training-schleiden-v1 \
  --identity-only data/nrw-training-duelmen-v1 \
  --identity-only data/nrw-training-marsberg-area-v1 \
  --protocol docs/research/nrw-training-bulk-protocol-v1.json \
  --cache-dir data/cache/nrw-training-bulk-v1 --json
```

The protocol fixes quotas, seed, 30 geographic boxes, candidate bounds, distances,
concurrency and request budgets. Known locations are ranked deterministically
and interleaved across projected 10 km cells, without reading source masks or
using model predictions. The operational plan is sealed before RGB collection.
Every crop must have a dated metadata tile covering its full 32 m extent and
confirming native 0.1 m resolution. Accepted images preserve the exact 320 × 320
RGB PNG response, without resizing.

Reference IDs, encoded image hashes and building IDs prevent identity reuse.
The original non-training records and **all twenty external-pilot cases** are
protected by at least `5000 + 32√2` metres. Their raster paths are never opened.
Only training images are decoded to reject pixel-identical recaptures. The
existing 22 training-review images and new crops have at least 80 m centre
spacing; known-location candidates also have 200 m seed spacing. Geometry
candidates keep at least 80 m from original training centres.

Successful official responses enter a verified immutable cache with URL, exact
bytes, SHA-256 and original retrieval time. A persistent ledger reserves every
network attempt before requesting it. Up to four workers share a minimum 0.35 s
request interval, at most three attempts per URL and a 5,000-request run budget.
There are no additional retries inside the cache. Retrieval time and snapshot
assembly time are distinct.

On interruption, rerun the same command with the same references, protocol and
cache. Cached responses and attempt counts survive; corrupt entries fail rather
than silently downloading replacements. The final dataset is published only
when both quotas are met. `--allow-partial` explicitly permits fewer images;
it must not be described as a completed 1,000-image collection. Existing output
directories are never replaced. Per-image acquisition evidence stays outside Git.

## Review queue

```sh
uv run --locked --extra train pvextent data review-queue \
  data/nrw-training-bulk-v1 artifacts/nrw-training-bulk-review-v1 \
  --batch-size 25 --seed 20261021 --json
```

For 1,000 photos this prepares 40 batches in each of two reviewer lanes.
Each lane contains every image exactly once. Opaque IDs and a restricted image
manifest hide acquisition origin, inherited labels and model predictions.
Queue export reads metadata only. It creates **pending templates**, not reviews:
`full_image_inspected` starts false, no polygons are present and no labels change.

Assign distinct reviewers, inspect original pixels, record native-pixel PV and
clear-background polygons independently, and retain uncertain areas as ignore.
The current proposal schema approves positive cases; clearly negative cases
need the separate negative curation workflow. Empty polygons do not confirm PV
absence. Two completed reviews, consensus curation and a new immutable training
snapshot are required before a separate training experiment. AI review is
explicitly experimental, not human-confirmed benchmark truth.

All canonical masks initially contain only `255`; all samples remain unreviewed
and unassigned. A collection of 1,000 photos therefore supplies **zero additional
labelled examples** until curation. Regular checkpoint resume continues to reject
changed dataset fingerprints.
