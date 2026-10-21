# 1,000-photo NRW training collection

The registered bulk collection completed on 2026-10-21 with **1,000 additional
native 320 × 320 RGB photos at 0.1 m/pixel**. It contains 600 current-source
crops at existing PV4GER training locations and 400 distinct ALKIS roof crops.
An independent audit verified 1,000 different encoded images and decoded RGB
arrays, source evidence, distances and unchanged reference metadata.

This is an unlabelled photo collection. All 102,400,000 mask pixels remain
ignore `255`; no new labels, training snapshot, model or accuracy result was
created. The existing 22-image review pool and its proposals remain separate.

The [machine-readable audit](nrw-training-bulk-v1.json),
[registered protocol](nrw-training-bulk-protocol-v1.json),
[selection record](nrw-training-bulk-selection-v1.json), and
[workflow guide](../bulk-training-data.md) provide the aggregate evidence.
Raw photos, source responses, coordinates, decisions and pending reviews stay
outside Git.

## Selection and source evidence

The exact protocol was committed before RGB acquisition. Identity checks cover
4,076 reference records. Geographic distance protects all 761 original
validation/test/unassigned centres and external-pilot cases; every one of the
twenty external acquisition cases remains protected. Reference rasters were
never opened, except 3,293 explicitly guarded TRAIN images for RGB deduplication.
No reference mask was read or transferred. The independent auditor opened zero
reference rasters and verified all registered manifest/coordinate hashes again.

A frozen plan contains 1,600 known-location candidates and 1,053 eligible roof
candidates within 30 fixed query boxes. Known candidates are interleaved across
10 km cells using only metadata. Acceptance follows plan order. The retained
600 parents span 289 such cells; the 400 roof crops come from thirteen of the
registered boxes. Selection stops at each quota and does not use PV labels,
image appearance or predictions to choose replacements.

Each accepted crop has one dated official metadata tile covering its complete
32 m extent and confirming native 0.1 m resolution. The exact RGB response and
native grid are preserved without resizing. A total of 1,721 distinct successful
source responses retain their URLs, bytes, hashes and original retrieval times.
Images occupy 237,290,335 bytes, excluding masks and provenance copies.

Capture years in the currently available source are:

| Year | Photos |
| --- | ---: |
| 2022 | 3 |
| 2024 | 331 |
| 2025 | 481 |
| 2026 | 185 |

Dates range from 2022-06-15 to 2026-05-01. These are current-source crops, not a
claim of 1,000 newly flown scenes. Original PV4GER capture dates are unknown;
RGB uniqueness does not by itself establish a different flight. The 600
known-location crops retain their TRAIN parent lineage and are not independent
new roofs. Geometry sampling favors smaller roofs and supplies no representative
prevalence or benchmark estimate.

## Independently measured separation

| Check | Observed minimum | Required minimum |
| --- | ---: | ---: |
| New crops to 761 protected centres | 5,047.008 m | 5,045.255 m |
| Any two new crop centres | 80.069 m | 80 m |
| Known-location crop centres | 258.568 m | 200 m |
| Geometry crops to all original TRAIN centres | 83.591 m | 80 m |
| New crops to the earlier 22-image review pool | 664.412 m | 80 m |

Distinct roof IDs, encoded hashes and training-parent eligibility were checked
in addition to distances. Native source XML, full-crop containment, acquisition
dates, PNG shape/hash and all-ignore masks were independently verified.

## Execution and verification

The initial collector commit was `add5b64`. Live measurements showed batch
barriers leaving workers idle behind slow responses. After 130 accepted photos,
the process was stopped deliberately and resumed from the same immutable cache,
frozen plan, reference binding and attempt ledger using `35bf5ad`. That change
feeds four workers continuously with a quota-limited window of at most 32
pending results while retaining acceptance order. Source selection and quotas
did not change. Both committed source versions and interruption evidence are
retained locally and verified by the independent audit.

The complete ledger reserved 2,354 attempts within the 5,000-attempt budget;
each URL was limited to three attempts. The final assembly reused 692 cached
responses. A separate pre-registration preflight made thirty official requests
and zero RGB requests. Ledger creation to final assembly took 1,486.123 seconds.
Sixty-one acquisition candidates lacked acceptable native/date/coverage support,
and 117 encountered exhausted request attempts; they did not count toward 1,000.
No partial-output option was used.

All 389 synthetic tests pass. They cover cache interruption/integrity, protected
raster guards, deterministic quotas, bounded continuous scheduling, retries,
resume, no-overwrite publishing and pending review contracts. Canonical validation
passes with 2,000 expected warnings: every sample is unreviewed and all-ignore.
The independent real-data audit passed 39,969 assertions; these checks establish
integrity and separation, not annotation truth or model quality.

## Pending annotation

Two blinded reviewer lanes contain exactly forty batches of twenty-five images
each. Each of the 1,000 IDs occurs once per lane. Eighty templates remain pending,
with inspection false, no polygons and no completed reviews. Their source
fingerprint and exact batch bytes are hash-bound. Exporting them opened no raster
and changed no source data. Clearly negative cases need the separate negative
curation path; empty positive polygons do not establish PV absence.

A photo-only gallery has twenty pages of fifty native images. Its sixteen-image
contact sheet retains source pixels unchanged. The preview is a structural
quality check, not either required annotation pass.

Next, complete the two independent visual reviews, curate consensus masks, and
build a new immutable training snapshot. A separate registered experiment must
then test the effect of target-domain training. The current model and its earlier
validation/external-pilot results remain unchanged.

## Artifact identities

- Dataset: `2b4409c46d14fb9c7cc4cefc13112bcdf53e2868fc805f3f542c75311672bb55`
- Protocol: `060d08e052ae3061647e738b8157d627f99c363f77b6c8293f211af6eb7cf5a7`
- Frozen plan: `e5e880231cb4bb89ffe854cd48a7ccfc992cb0b14f06957fd15f65a2bf8280a3`
- Queue: `6b3c65fe222867847eadb582a8cd8b3a1b7a217797c1ccfc1d2d27b5bacdce14`
- Independent audit: `dd0db89047d264a05f9c7b87c9fe1c1f7c11f4144c8adeac4e0a9798f5aca29b`
