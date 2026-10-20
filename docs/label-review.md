# Local label review

PVExtent can export a dataset into a self-contained review gallery. The gallery
shows copied source images, unchanged masks, coloured label views, and overlays.
It records human decisions as a downloadable JSON file without editing the
source dataset.

## Open a review bundle

From the repository directory, export a validated dataset:

```sh
uv run --locked pvextent data review SOURCE OUTPUT --json
```

`SOURCE` is a canonical dataset directory. `OUTPUT` must be a new directory
separate from the source; an existing output is never overwritten. The command
runs offline and requires the base dependencies only. Real imagery, masks, and
review evidence belong outside Git; the repository's `data/` directory is
ignored.

For the imported PV4GER-SEG dataset:

```sh
uv run --locked pvextent data review \
  data/pv4ger-seg-v0.9.1 data/review-pv4ger-seg-v0.9.1 --json
```

Open `data/review-pv4ger-seg-v0.9.1/index.html` in a browser by double-clicking
the file. No web server or account is required. If this bundle already exists,
open it directly instead of exporting it again, or choose a new output path.

The default selection includes every sample whose mask contains no PV-labelled
pixels, followed by up to 12 positive examples in sorted identifier order.
All-ignore masks come first. This selection is deterministic, and there is no
cap on the samples without PV labels. The existing complete PV4GER-SEG import
produces **20 review examples: 8 all-zero source masks and 12 positive examples**.
Those eight masks are review candidates, not verified negative images.

The same command works on an unreviewed NRW candidate snapshot. For example,
with a candidate dataset stored at `data/nrw-roof-candidates`:

```sh
uv run --locked pvextent data review \
  data/nrw-roof-candidates data/review-nrw-roof-candidates --json
```

Open the resulting `data/review-nrw-roof-candidates/index.html`. Every all-ignore
candidate is included, regardless of how many candidates the source snapshot
contains.

## Inspect and record decisions

1. Enter your reviewer name. Decisions initially show **pending** for every
   example.
2. Click the RGB image to inspect the complete crop at its native resolution.
   Check orientation, roof visibility, blur, shadows, and alignment with the
   source mask. The label view shows background `0` in black, PV `1` in green,
   and ignore `255` in orange; the overlay uses the same PV and ignore colours.
   The original single-channel mask is also linked unchanged.
3. Check the entire image for visible modules, including cropped arrays at its
   edges. Compare skylights, glass roofs, equipment, and dark roof surfaces
   against the [annotation rules](annotation-guide.md). Keep resolvable gaps
   between modules as background. Unresolved shadows, occlusions, and ambiguous
   pixels require ignore labels rather than an invented module outline.
4. Choose **Needs edit**, **Exclude**, **Unsure**, or **Accept source mask**.
   Record concrete corrections or unresolved points in notes; optional tags
   can describe issues such as `skylight`, `occlusion`, or `misalignment`.
5. For a candidate without PV labels, **No visible PV (requires full image
   review)** is available. It requires checking the box confirming that you
   inspected the entire image. Use it only after establishing the absence of
   visible PV from the image, never from an empty source mask alone.
6. Click **Download review decisions (JSON)** and retain the file beside the
   source snapshot or review bundle outside Git. A reviewer name is required.
   The download includes the source dataset fingerprint, review-manifest hash,
   reviewer, export time, decisions, inspection flags, notes, and tags.

Decisions remain in the page's memory until downloaded. Refreshing or closing
the page loses unsaved decisions. **Reset to pending** clears a decision when
an example needs another look.

The JSON is review evidence only. Applying approvals, importing corrected
masks, resuming a downloaded review, and automatically updating canonical
annotation status are not implemented. No gallery button changes masks,
`annotation_status`, or split assignments. Keep a source mask marked `upstream`
or `unreviewed` until a separate, explicit curation step records the completed
review and any required corrections.

## Candidate masks and source metadata

New NRW roof candidates start with masks entirely filled with `255` and with
`annotation_status = "unreviewed"` and `split = "unassigned"`. They contain
**no supervised background or PV pixels**. Accepting an all-ignore source mask
does not establish a negative example. Even a recorded “No visible PV” decision
does not convert its mask into reviewed background pixels.

Source acquisition dates are recorded as `capture_date` only when official
imagery metadata supports them; missing or ambiguous dates remain `null`.
Building object currency, retrieval time, and imagery acquisition time are
different fields. Preserve the original metadata, request records, image
sampling information, and source notices during curation instead of filling
unknown dates or native resolution from assumptions.

The bundle retains a copy of `dataset.toml`, selected source pixel files, and
available source license notices. `review-manifest.json` records the original
sample metadata, exact source dataset fingerprint, display paths, and initial
pending decisions. Dataset licenses and attribution remain applicable to the
copied material.

Geographic split snapshots provide an internal holdout for known source
locations. Reviewing their labels does not establish independence from a
published model's training data. A small NRW convenience sample likewise needs
completed annotation and an explicit evaluation protocol before it can serve
as an independent benchmark.
