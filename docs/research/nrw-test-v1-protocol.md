# NRW test set v1 protocol

**Status**: frozen before annotation, 2026-10-22.
**Purpose**: measure how PVExtent models perform on current Geobasis NRW
orthophotos, which the PV4GER benchmark does not cover.

## Sampling

- Built by [`scripts/build_nrw_test_pool.py`](../../scripts/build_nrw_test_pool.py),
  seed `20261022`, into the untracked `data/nrw-test-v1`.
- Imagery: current `nw_dop_rgb` WMS layer, 320 × 320 px at 0.1 m/px, centred on
  one ALKIS building footprint (from the ALKIS-vereinfacht WFS) per crop, using the
  footprint filter of `pvextent.data.nrw._roof` (40–650 m², fits the crop).
- Regions: the ten districts in the script's list that lie farthest from every
  location already used anywhere in the project.
- Separation: every crop centre lies at least 1,045 m (1 km plus the crop
  diagonal) from all 4,027 PV4GER locations and every georeferenced record in
  `data/*/manifest.jsonl`, the same rule the project uses for its holdouts.
- Strata per region: up to 8 buildings that OpenStreetMap tags with solar
  generation (`osm_solar`), filled up with randomly drawn buildings to 33 crops
  (`random`); crop centres are at least 80 m apart. The `random` stratum gives an
  unbiased PV prevalence and false-positive denominator; `osm_solar` enriches PV.
- Annotation order is shuffled, and the annotator never sees the stratum.
- Realized pool: 321 crops from Hamm, Arnsberg, Detmold, Minden,
  Bielefeld-Sennestadt, Bonn-Beuel, Meschede, Rheine, Paderborn and
  Münster-Gievenbeck; flights from 2024 (154), 2025 (119) and 2026 (24), plus 24
  without tile metadata. OpenStreetMap tags rooftop solar on almost none of these
  buildings, so only 3 crops are `osm_solar` and 318 are `random`: the set is in
  effect a random sample at natural PV prevalence.

## Annotation

- One human annotator draws polygons on an iPad with Apple Pencil in a dedicated
  labeling page; no model predictions or suggestions are shown (blind labeling).
- Classes: `pv` for each contiguous photovoltaic module field including module
  frames; `ignore` for anything the annotator cannot decide. Solar thermal
  collectors, skylights, glass roofs and conservatories are not PV.
- Every crop ends as `done` with zero or more polygons, optionally flagged
  `unsure`. Masks are rasterized from the polygons: PV `1`, ignore `255`,
  everything else background `0`.

## Evaluation

The test set is used only for final evaluation of frozen models; nothing is tuned,
selected or thresholded on it. Reported per model, at the model's documented
threshold:

- Pixel IoU, precision and recall over all crops, ignore pixels excluded.
- Crop-level detection: share of PV crops with at least one predicted PV pixel
  overlapping the reference, and share of PV-free crops with any predicted PV
  component of at least 1 m² (false-alarm rate).
- Each metric separately for the `random` and `osm_solar` strata, so the enriched
  stratum cannot mask performance at natural prevalence.
- Crops flagged `unsure` are reported with and without inclusion.

## Known limitations

- A single annotator; no inter-annotator agreement is measured in v1.
- The footprint filter excludes buildings larger than about 650 m², so large
  commercial roofs are under-represented.
- OpenStreetMap coverage of rooftop PV is sparse and uneven, so the enriched
  stratum is small in some regions.
