# Exclude disputed training supervision

An additional quality check can withdraw experimental supervision without
changing the original imagery, reviews or snapshots. The quarantine command
creates a new immutable snapshot. Each explicitly named **new bulk TRAIN**
record becomes unreviewed and unassigned with an entirely ignored (`255`)
mask. It never turns disputed PV into background.

```sh
uv run --locked --extra train pvextent data quarantine-training-quality \
  data/pvextent-training-v3 data/pvextent-training-v2 \
  data/pvextent-training-v4 \
  --protocol docs/research/nrw-bulk-mask-quality-quarantine-protocol-v2.json \
  --evidence artifacts/nrw-bulk-mask-quality-quarantine-v2/quality-evidence.json \
  --dataset-version 2026-10-21.training-v4 --json
```

The exact protocol binds the source and original-base fingerprints, new
version, identified AI reviewer, exclusions and evidence hash. The evidence
is retained data, not executable instructions. Every frozen base record must
already exist unchanged in the source. A base training, validation, test or
unassigned record cannot be excluded through this command.

Only the explicitly excluded new masks are opened, hash-checked and decoded.
All images and other masks stay unopened and unresolved; unchanged files are
hard-linked after lexical/symlink checks. Inode checks prevent a selected mask
from aliasing any protected or unselected raster. Source provenance and
coordinates remain unchanged. The complete output manifest is checked without
running a generic pixel validator over the combined dataset.

The current `quality-quarantine-report.json` records the new counts and
fingerprint. The earlier augmentation report remains historical evidence at
`provenance/parent-augmentation-report.json`; its larger counts are not the
current training totals. The [live monitor](live-training.md) prefers the V4
quality report when present.

The [registered amendment](research/nrw-bulk-mask-quality-amendment.md) explains
why this project withdraws three cases after its original two-check curation.
The raw checker decisions and pixel gates remain unchanged. This conservative
follow-up does not establish that every remaining AI annotation is correct or
that the model improves. It does not start training.
