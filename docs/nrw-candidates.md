# Unreviewed NRW roof candidates

The candidate fetcher prepares a small separate roof collection for annotation.
It uses official [Geobasis NRW building features](https://ogc-api.nrw.de/lika/v1/collections/gebaeude_bauwerk)
and [current orthophotos](https://www.wms.nrw.de/geobasis/wms_nw_dop).
Building function, image appearance and missing masks are not used to infer
PV presence or absence. Every generated mask is entirely `255` (ignore).

```sh
uv sync --locked --extra geo
uv run --locked --extra geo pvextent data fetch-nrw-candidates \
  data/pv4ger-seg-v0.9.1 data/nrw-roof-candidates-dortmund-v1 --limit 24 --json

uv run --locked pvextent data review \
  data/nrw-roof-candidates-dortmund-v1 data/review-nrw-roof-candidates --json
```

Network access is explicit in `fetch-nrw-candidates`. It accepts the complete,
validated PV4GER-SEG source snapshot, whose original publisher partition is
checked against its pinned checksum. Geographic separation is checked against
**all 4,027 source centres**, including training, validation, test and exclusions.
The command refuses to overwrite an existing candidate directory.

## Bounded selection

This first sampler uses one fixed Dortmund-Nord query box:
`7.435,51.545,7.445,51.555` in longitude/latitude. One official ALKIS query returns
at most 500 features. Stable identifiers determine selection order. The sampler
checks complete small building footprints, interior crop centres, footprint
area (40–650 m²), a footprint extent that fits a 32 m crop, and suitable building
function. It excludes obvious accessories such as garages or carports.

Candidate centres are at least 80 m apart and at least `1,000 + 32√2` metres
from every pinned PV4GER centre. These are metric EPSG:25832 distances. A
conservative common geographic group keeps the entire candidate pool together
for any later split decision. Official building IDs are recorded with a source
namespace; footprints do not verify roof segments or visibility.

`--limit` accepts 1–24. A run makes one building request, at most 24 image
requests, and at most 24 metadata requests, with metadata cached by 1 km tile.
There are no hidden retries. Each request times out after 30 seconds, with a
4 MiB image limit, 2 MiB building-response limit and 64 KiB metadata limit.
If a network request fails, no partial dataset is published. Retry the explicit
command when the service is available. Fewer than the requested count can be
returned if the fixed query or metadata rules exclude candidates; the report
records this rather than relaxing the selection silently.

## Resolution, dates and georeferencing

Each GetMap request uses EPSG:25832, a 32 m square snapped to a 0.1 m grid,
and a 320 × 320 RGB PNG. Exact response bytes are stored as the image; no
resizing or sharpening is applied. The affine transform follows the exact
requested pixel grid: `[0.1,0,xmin,0,-0.1,ymax]`.

Requested/rendered pixel spacing alone does not establish native source detail.
The sampler accepts a crop only when the official metadata layer reports
native `Bodenauflösung = 0.10` and its tile covers the entire crop. A single
valid `Bildflugdatum` for that tile supplies `capture_date`. An unavailable or
ambiguous acquisition date remains null; a metadata response from a neighbouring
tile is not reused as evidence for this crop. Building `aktualit` and retrieval
time are kept separately and never substituted for an acquisition date.

Services are mutable. `provenance/` retains the exact bounded building response,
metadata responses, request URLs, byte hashes, retrieval time and sample records.
`candidate-report.json` records selection exclusions, reference hashes, observed
distances, request counts and validation results. Re-running a live service can
produce a different snapshot even with the same selection rules.

## Annotation and use

The first [recorded fetch](research/nrw-candidates-dortmund-v1.json) accepted
seven images from 235 returned building features. All seven have metadata
reporting 10 cm native resolution and acquisition on 2025-03-07. The nearest
PV4GER centre is 2,337.89 m away. Two metadata tile requests failed in that run;
the resulting unverified candidates were excluded. Several inspected images
show industrial structures and equipment, illustrating the geographic and
building-type bias of this small pool.

All samples remain `unreviewed` and `unassigned`. Masks have zero supervised
pixels. The pool can contain PV, dark roofs, skylights, equipment or ambiguous
surfaces; **none is a confirmed hard negative yet**. Use the
[local review workflow](label-review.md), then explicitly curate masks according
to the [annotation guide](annotation-guide.md). No model prediction or automatic
mask fill establishes ground truth.

This is a geographic convenience sample, not a representative prevalence study,
an independent benchmark or proof of separation from unknown model training.
Additional regions, building types, acquisition campaigns and reviewed labels
are required before evaluating generalisation.

The official building collection and imagery terms identify
[Datenlizenz Deutschland – Zero 2.0](https://www.govdata.de/dl-de/zero-2-0).
Preserve Geobasis NRW provenance and the
[imagery terms dated March 2020](https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf).
The placeholder annotation license is recorded as
`not-annotated-ignore-placeholder`; it does not claim a completed dataset license
for future human annotations. PVExtent's code license does not replace source
data terms.
