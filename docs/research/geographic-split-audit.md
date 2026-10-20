# Geographic split audit, 2026-10-20

The complete imported PV4GER-SEG release was split using
[`geographic-block-buffer-v1`](../geographic-splits.md), seed `20261020`, 10 km
metric cells and a 1 km buffer. The seed and policy were fixed before any model
training or evaluation. The
[machine-readable report](pv4ger-geographic-v1.json) contains the fingerprints,
source coordinate hash and policy parameters. No imagery or per-sample labels
are included in this repository.

| Partition or exclusion | Samples |
| --- | ---: |
| Training | 3,282 |
| Validation | 278 |
| Test | 421 |
| Unassigned: geographic buffer | 38 |
| Unassigned: upstream zero mask pending review | 8 |
| Total | 4,027 |

The policy uses 3,628 linked source components and 360 geographic blocks.
The minimum measured centre distances are:

| Pair | Metres, EPSG:25832 |
| --- | ---: |
| Training ↔ validation | 1,012.82 |
| Training ↔ test | 1,010.49 |
| Validation ↔ test | 1,146.68 |

The final snapshot validates with no errors. Its 12,081 warnings preserve the
known missing acquisition dates, missing original georeferences and upstream
annotation status. Source pixels are unchanged. Repeating the split with the
same parameters produced the same canonical dataset fingerprint:
`597b9facb7f115448e81af3b88de7e1081d510c38108c294195de05763d841c5`.

This is a buffered internal holdout from one source. It is not a new independent
benchmark, does not exclude all possible city/campaign correlations, and cannot
prove independence from existing pretrained models. Local label review and
additional reviewed hard negatives remain outstanding. No model performance
has been measured on this test partition.
