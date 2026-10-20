# Geographic split snapshots

The first split policy creates an internal geographic holdout from an unassigned
PVExtent dataset. It uses source centre coordinates, declared geographic groups,
building identifiers and duplicate image hashes. It does not establish independence
from a published model's training data or an unseen acquisition campaign.

## Create a snapshot

```sh
uv sync --locked --extra import --extra geo
uv run --locked --extra geo pvextent data split \
  data/pv4ger-seg-v0.9.1 data/pv4ger-seg-geographic-v1 --json
```

Defaults are seed `20261020`, 10,000 m blocks, a 1,000 m centre buffer, and expected
train/validation/test fractions `0.8 0.1 0.1`. Explicit options are `--seed`,
`--block-size-m`, `--buffer-m`, and `--ratios TRAIN VAL TEST`. These parameters
and the input/output fingerprints are written to `split-report.json`.

The input must validate, have every sample `unassigned`, and provide
`provenance/samples.jsonl` with exactly one coordinate record per canonical ID:

```json
{"id":"example","center_crs":"EPSG:4326","approximate_center":[7.44,51.55]}
```

Additional provenance fields are retained. Duplicate keys or IDs, missing
coordinates, non-finite values and unsupported coordinate systems fail. This
first policy supports coordinates around Germany (3–15° E, 45–60° N), projected
into EPSG:25832 with explicit longitude/latitude axis order.

The importer already produces the required centre sidecar. These approximate
filename centres remain separate from an original orthophoto affine transform.

## Assignment and exclusion

1. Join samples connected by a declared geographic group, building ID or identical
   encoded image hash. These constraints are transitive and kept together.
2. Compute each component's mean metric centre and its 10 km cell. Components
   in the same cell receive the same proposed split from a SHA-256 hash of the
   policy version, seed and cell. Fractions are expectations across cells, not
   exact per-image quotas; do not try seeds until a test result looks favourable.
3. Keep components containing unreviewed labels or masks without supervised
   pixels unassigned. Upstream masks without PV-labelled pixels also stay
   unassigned until reviewed. This is a quarantine, not a negative label.
4. Preserve test priority. Exclude an entire training component if one centre
   is within the buffer of a tentative validation or test centre. Exclude an
   entire validation component if near test. Training exclusion is conservative:
   it also respects tentative validation points that are subsequently excluded.
5. Audit all assigned cross-split distances and validate the resulting dataset.
   All three assigned splits must remain nonempty.

Distances refer to centres in EPSG:25832, not measured roof boundaries. The
default 1 km buffer is deliberately larger than a 32 m PV4GER crop. Appropriate
buffers for other image footprints, resolutions and sources require a separate
policy decision; a valid dataset is not proof of geographic independence.

## Frozen output and provenance

The command creates a new directory and refuses an existing target or an input
already containing assigned splits. It preserves exact image and mask bytes,
annotation status and source notices. The canonical manifest records new split
and block group values. Original values remain in:

- `provenance/split-input-manifest.jsonl`
- `provenance/split-assignments.jsonl`, including proposed assignments and exclusions
- `provenance/split-input-import-report.json`, when the input has an import report
- `provenance/split-input-validation-report.json`, when available

Source coordinate and report hashes, policy parameters and both dataset
fingerprints are recorded in `split-report.json`. `validation-report.json`
contains the final structural checks. The source is not modified, and output
is published atomically only after validation.

Treat this snapshot as frozen before model/threshold selection. Label corrections
or an expanded reviewed test set need a new version, with a documented relation
to existing assignments. Do not overwrite or quietly reshuffle an existing test.

The [first measured split audit](research/geographic-split-audit.md) records
the actual counts and distances. [Local review bundles](label-review.md) support
the remaining annotation work; candidate roofs are not automatically negatives.
