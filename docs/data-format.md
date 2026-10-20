# Dataset format, version 1.0

PVExtent uses RGB orthophoto crops paired with pixel-aligned semantic masks.
The initial target is the visible extent of photovoltaic modules. Annotation
rules are defined in the [annotation guide](annotation-guide.md).

Keep real images, masks, and generated dataset snapshots outside Git. A dataset
directory has this layout:

```text
dataset/
  dataset.toml
  manifest.jsonl
  images/
    example.png
  masks/
    example.png
```

Paths in the manifest are relative to the dataset directory, use forward
slashes, and must resolve inside it. Absolute paths, parent traversal, and
symlinks that escape the directory are not valid dataset paths.

## Dataset metadata

`dataset.toml` contains `schema_version = "1.0"`, a stable `dataset_id`, and a
`dataset_version`. It also declares the sources referenced by the manifest.
Version 1.0 uses the fixed filename `manifest.jsonl`.

Each `[sources.<source_id>]` table requires these non-empty string fields:

| Field | Meaning |
| --- | --- |
| `name` | Human-readable source name |
| `url` | HTTP or HTTPS provenance URL |
| `version` | Specific dataset release, revision, or snapshot identifier |
| `image_license` | License identifier or terms reference for the imagery |
| `annotation_license` | License identifier or terms reference for the masks |
| `attribution` | Required attribution for images and annotations |

Source identifiers contain ASCII letters, digits, periods, underscores, or
hyphens, start with a letter or digit, and are at most 128 characters long.
Use a separate source entry when a license, annotation origin, or source
version differs. Record image and annotation licenses separately: importing a
mask does not transfer the underlying image rights. Validation checks that
provenance is recorded; it does not certify permission to train or redistribute.

## Manifest records

`manifest.jsonl` contains one JSON object per line. Every record must contain
exactly the following fields; additional fields are rejected in version 1.0.

| Field | Type and meaning |
| --- | --- |
| `id` | Stable sample identifier, unique within the dataset |
| `image` | Relative JPEG or PNG image path |
| `mask` | Relative PNG mask path |
| `image_sha256` | SHA-256 of the exact image file bytes |
| `mask_sha256` | SHA-256 of the exact mask file bytes |
| `width`, `height` | Positive integer pixel dimensions shared by image and mask |
| `gsd_m` | Positive, finite ground sampling distance in metres per pixel |
| `source_id` | Identifier of a declared source |
| `capture_date` | Actual image acquisition date as `YYYY-MM-DD`, or `null` if unknown |
| `spatial_group` | Stable geographic group shared by related crops |
| `building_ids` | List of stable building identifiers; an empty list means unknown |
| `split` | `unassigned`, `train`, `val`, or `test` |
| `annotation_status` | `unreviewed`, `upstream`, or `reviewed` |
| `georeference` | CRS and affine transform object, or `null` if unknown |

Hashes use 64 lowercase hexadecimal characters. JSON booleans are not valid
integer dimensions. NaN and infinity are not valid numeric metadata.

Do not substitute a download date for an unknown acquisition date. Missing
capture dates and georeferences are accepted with warnings so legacy data can
retain honest provenance. They remain limitations of the resulting dataset.

### Annotation status

- `unreviewed`: provisional labels. These may be stored as `unassigned` but
  cannot be assigned to a training, validation, or test split.
- `upstream`: curated annotations imported from the declared, versioned source.
  These are accepted with a warning and do not claim a local human review.
- `reviewed`: annotations explicitly reviewed locally against the source image
  using the annotation guide.

Record the reviewer type and process separately in provenance. The status alone
does not establish human authorship. The experimental negative-curation workflow
can accept two documented AI reviews when explicitly permitted; its labels must
remain distinguishable from human-reviewed benchmark annotations.

The validator cannot establish whether a claimed review really took place.
Keep the evidence for annotation origin and review with the dataset's source
records and annotation process.

### Spatial grouping and splits

Choose geographic groups before making crops. Related crops, repeat captures,
and overlapping images must retain a common group. A new group identifier per
image defeats this boundary. Use globally meaningful building identifiers when
available, including a source namespace where necessary.

The validator rejects duplicate sample IDs and detects image hashes, geographic
groups, and building IDs shared across different assigned splits. Repeated image
hashes within a split produce a warning. Identical mask hashes are allowed:
many genuine negative samples have the same all-zero mask.

These checks use the supplied identifiers. They do not prove geographic
independence, detect every overlapping crop, or establish independence from a
pretrained model's training data.

### Georeferencing

When available, record an object with exactly `crs` and `transform`, for example:

```json
{"crs":"EPSG:25832","transform":[0.1,0,400000,0,-0.1,5700000]}
```

The six finite affine coefficients are ordered `[a, b, c, d, e, f]`:

```text
x = a * column + b * row + c
y = d * column + e * row + f
```

Coordinates refer to pixel corners: `(column=0, row=0)` is the upper-left
corner, and `(0.5, 0.5)` is the first pixel centre. The transform must be
non-singular. CRS units are those of the stated CRS; `gsd_m` always uses metres.
The format does not infer ground resolution from an arbitrary CRS.

## Image and mask encoding

Images must be 8-bit RGB JPEG or PNG files. Bake any orientation into the pixels
before import: EXIF orientation other than normal (`1`) is rejected. The mask
must share the image's encoded dimensions and pixel grid.

Masks are single-channel, 8-bit PNG files, Pillow mode `L`, with these values:

| Value | Meaning |
| ---: | --- |
| `0` | Supervised visible background, according to the annotation status |
| `1` | Visible PV module surface |
| `255` | Ignore: uncertain, occluded, or padded area |

Palette, RGB, binary `1`-bit, and 16-bit masks require explicit conversion by an
importer. There is no automatic interpretation of a conventional white mask:
`255` means **ignore**, not PV.

An all-zero mask is a usable negative only with `reviewed` or `upstream` labels.
An unreviewed all-zero sample may remain unassigned with a warning. An all-255
mask has no supervised pixels: it warns when unassigned and fails validation
when assigned to `train`, `val`, or `test`.

The first baseline is planned around 10 cm/pixel, 320-by-320 crops, giving 32 m
of context. These are experiment settings, not format restrictions. Preserve
source resolution and physical context when importing other datasets. If
resampling is required, interpolate imagery appropriately, use nearest-neighbour
resampling for class masks, and mark artificial padding as `255`.

## Validation

```sh
pvextent data validate /path/to/dataset
pvextent data validate /path/to/dataset --json
# Equivalent module invocation:
python -m pvextent data validate /path/to/dataset --json
```

Exit code `0` means valid, including possible warnings; `1` means invalid data;
`2` means incorrect command usage. The validator runs offline and checks file
hashes, dimensions, encodings, class values, metadata, and supplied split groups.
It does not repair or rewrite inputs.

Validation is bounded to 64 MiB per image or mask, 16,777,216 pixels per image or
mask, 1 MiB for `dataset.toml`, 64 MiB for the manifest, 64 KiB per manifest
line, and 100,000 records. Reporting stops after 200 errors. Large source
orthophotos should be prepared as bounded crops before validation.

The [`dataset.toml`](../examples/dataset.toml) and
[`manifest.jsonl`](../examples/manifest.jsonl) files are metadata-only templates.
Their placeholder hashes and image paths do **not** form a valid, complete dataset.
Replace them with real metadata and exact file hashes in a dataset directory
outside Git before running validation.

For a complete, synthetic format demonstration, run
`uv run --locked python examples/create_demo_dataset.py data/demo`, then validate
`data/demo`. The generated pixels and GSD are illustrative, with no actual
capture date or georeference. The demonstration is not training evidence.

The JSON report includes hashes of the metadata and manifest and a combined
dataset fingerprint. File hashes are checked against the actual encoded bytes.
Warning details are capped at 200; `warning_count` still records the total.
