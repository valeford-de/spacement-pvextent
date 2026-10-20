# PV4GER-SEG import audit, 2026-10-20

The complete GEO-Bench m-pv4ger-seg v0.9.1 release was imported locally using
the [documented importer](../pv4ger-import.md). The
[machine-readable summary](pv4ger-import-v0.9.1.json) records pinned source
hashes, dependency versions, counts and the resulting dataset fingerprint.
Imagery, masks and visual diagnostics remain outside Git.

| Check | Result |
| --- | --- |
| Imported image/mask pairs | 4,027 |
| Image dimensions and GSD | 320 × 320 pixels, 0.1 m/pixel |
| Mask values | Exactly 0 and 1; no resampling |
| Nonzero masks | 4,019 |
| All-zero masks | 8 |
| PV-labelled pixels | 78,907,780 (19.14%) |
| Background-labelled pixels | 333,457,020 |
| Validation errors | 0 |
| Validation warnings | 12,081: unknown date, unknown georeference, upstream review status for each sample |
| 100 m proximity groups | 3,628; largest group has 32 samples |
| Groups spanning publisher splits | 82 |
| Canonical assigned splits | None; all samples remain unassigned |
| Converted snapshot size | Approximately 254 MB |

The source partition contains 3,221 train, 403 valid and 403 test identifiers.
Its exact bytes and the full assignment mapping are retained in the local
snapshot. The 82 cross-split proximity groups indicate nearby centres, not 82
proven overlaps. They must not be presented as independent geographic holdouts.

A visual diagnostic inspected RGB images, masks and overlays for the smallest
nonzero mask, a median nonzero mask, a fully positive mask and one all-zero
mask. The overlays align with the converted imagery. This is a conversion
spot check, not a review of the source annotations. The selected all-zero mask
includes a roof with PV-like features that require review. Source labels can
also cover array interiors and gaps more coarsely than our annotation guide.
Preserve those labels as `upstream`; do not silently claim local review or
reinterpret all-zero masks as verified PV-free images.

This source is sufficient to begin constructing a training pipeline, but it
does not supply a representative collection of negative roofs or an independent
test set. The next stage must add reviewed hard negatives and define geographic
splits before model selection. No model was trained or evaluated in this audit.
