# Import PV4GER-SEG

The PV4GER-SEG importer converts the segmentation portion of GEO-Bench into
PVExtent's [dataset format](data-format.md). It uses the pinned
[m-pv4ger-seg v0.9.1 release](https://zenodo.org/records/8277054), containing
4,027 RGB crops and their segmentation masks. It does not download the rest of
GEO-Bench.

## Run the import

Install the optional HDF5 and NumPy dependencies:

```sh
uv sync --locked --extra import
```

Download the four required source files and import the complete dataset:

```sh
uv run --locked --extra import pvextent data import-pv4ger-seg \
  data/raw/pv4ger-seg-v0.9.1 data/pv4ger-seg-v0.9.1 --download

uv run --locked pvextent data validate data/pv4ger-seg-v0.9.1 --json
```

The first directory is a source cache; the second is the new canonical dataset.
Keep both outside Git. The repository's `data/` directory is ignored.
`--download` permits retrieval of missing source files from the pinned Zenodo
record. Without it, the importer works offline and requires the complete source
cache. Existing files must pass the pinned size and SHA-256 checks.

The archive download is approximately 316 MB. PNG conversion and the source
archive need approximately 570 MB together for the full release. The importer refuses an existing output
directory, prepares the result in a temporary sibling directory, and publishes
it only after dataset validation succeeds.
Atomic publication without replacing an existing target is supported on macOS,
Linux with glibc, and Windows; unsupported platforms fail without publishing.

For a small format and dependency smoke test, use a separate output directory:

```sh
uv run --locked --extra import pvextent data import-pv4ger-seg \
  data/raw/pv4ger-seg-v0.9.1 data/pv4ger-seg-smoke --limit 8 --json
```

`--limit` takes the first identifiers in deterministic sorted order. It still
requires the complete source archive and partition. A limited import is not a
representative evaluation set.

## Pixel conversion and provenance

The [official converter](https://github.com/ServiceNow/geo-bench/blob/main/make_benchmark/dataset_converters/pv4ger.py)
stores one HDF5 file per sample. The importer reads the `Red`, `Green`, `Blue`,
and `label` arrays directly. RGB bands become an 8-bit RGB PNG; the mask becomes
a single-channel 8-bit PNG. Native 320 × 320 dimensions and 0.1 m/pixel resolution
are retained, with no resizing or augmentation.

GEO-Bench already normalized the source labels to `0` for background and `1`
for PV. The importer accepts exactly those values and preserves them. It rejects
unexpected values, including `255`; it does not reinterpret them as PV or as a
new ignore class. Shape and dtype mismatches also fail the import.

Every canonical record has:

- `annotation_status = "upstream"`: source annotations, without a claim of local
  review against the [annotation guide](annotation-guide.md).
- `capture_date = null`: the acquisition date is unavailable.
- `georeference = null`: no original orthophoto affine transform is established.
- `building_ids = []`: verified building identifiers are unavailable.
- `split = "unassigned"`: evaluation splits must be established separately.

Filename coordinates are retained in the provenance audit for grouping. The
upstream converter derives an approximate transform from those coordinates and
a nominal crop size; it is not the original orthophoto transform. HDF5 files
also contain pickled metadata attributes. The importer does not deserialize
them or execute pickle data.

In addition to `dataset.toml`, `manifest.jsonl`, `images/`, and `masks/`, the
output includes:

| File | Purpose |
| --- | --- |
| `provenance/default_partition.json` | Exact publisher partition file |
| `provenance/source_splits.json` | All original sample identifiers and publisher split assignments, including for limited imports |
| `provenance/samples.jsonl` | Source identifiers and approximate EPSG:4326 centres for imported samples |
| `provenance/upstream-LICENSE`, `upstream-README`, `SOURCES.md` | Preserved publisher notices and source/license explanation |
| `provenance/original-MIT-LICENSE` | Complete original MIT terms and copyright notice, bundled with the importer |
| `import-report.json` | Import configuration, source checksums, counts, and limitations |
| `validation-report.json` | Validator results and dataset fingerprint |

Unknown dates, missing georeferences, and upstream annotation status produce
expected validator warnings. A valid result confirms format and provenance
checks; it does not establish annotation accuracy or model performance.

## Geographic groups and source splits

The original partition contains 3,221 `train`, 403 `valid`, and 403 `test`
samples. These assignments remain available in the provenance sidecar, but are
not adopted as PVExtent's training and evaluation splits. Nearby crops can occur
on opposite sides of those source boundaries, including potentially overlapping
32 m crops.

The importer groups connected sample centres within 100 m using the coordinates
in the source identifiers. Groups are computed from all 4,027 identifiers before
applying `--limit`, so limiting the import does not alter group membership. This
is a proximity heuristic: a connected group can extend beyond 100 m through
intermediate samples, and its members are not verified building identities.

The next preparation step is to establish geographic holdouts, review labels,
and add representative hard negatives. These proximity groups alone do not
prove independent buildings, cities, acquisition campaigns, or independence
from a pretrained model's training data.

## Verified import

The [2026-10-20 import audit](research/pv4ger-import-audit.md) records a complete
import: 4,027 samples, no validation errors, and 12,081 expected metadata/review
warnings. There are 4,019 nonzero masks and eight all-zero masks. These are label
counts, not a verified count of images with or without PV. The source is heavily
biased toward PV-containing crops; all-zero masks still require review before
being treated as hard negatives.

## Source versions and licenses

Downloads are pinned to these exact files from Zenodo record 8277054:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `data.zip` | 316,013,928 | `70438a20797bba6eecd905358bf26ace7738709a38848f8a58145f6b0448f82d` |
| `default_partition.json` | 129,229 | `b0068e4a7516f29b47155dfb674d2003603af687779d91df29a2c17d334890f6` |
| `LICENSE` | 105 | `bdebe1f65b5d5a86435e11edd275dbcb5b61d4fcad90672ff1728cd6a2839e84` |
| `README` | 678 | `79bfe9f379d2e3b0ce7c72aca98d24ddc1548c2f5510b1c56d6af21acec7df31` |

Image and annotation terms are recorded separately:

- **Imagery:** `dl-de-zero-2.0`. The
  [original authors](https://github.com/kdmayer/3D-PV-Locator) identify openNRW
  as the image source. The
  [GeoBasis NRW license terms, dated March 2020](https://www.bezreg-koeln.nrw.de/system/files/media/document/file/lizenzbedingungen_geobasis_nrw.pdf)
  identify Germany's Data Licence Zero 2.0 for those geobasis data.
- **Annotations:** `MIT`. The bundled source `LICENSE` points to the
  [original MIT license](https://github.com/kdmayer/3D-PV-Locator/blob/master/LICENSE),
  copyright 2020 Kevin. Keep that source notice and the original license with
  any redistribution, alongside GEO-Bench provenance.

Credit the original dataset as
[Mayer et al., “3D-PV-Locator” (2022)](https://doi.org/10.1016/j.apenergy.2021.118469)
and the [GEO-Bench adaptation](https://github.com/ServiceNow/geo-bench).
PVExtent's Apache-2.0 code license does not replace the dataset's terms.
