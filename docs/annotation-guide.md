# PV annotation guide

The target is the visible surface of photovoltaic modules in an orthophoto.
The result is a semantic mask aligned exactly with the image pixels. It records
what can be supported by this image and acquisition date.

## Class values

| Value | Label | Use |
| ---: | --- | --- |
| `0` | Background | Visible, reviewed pixels that are not PV |
| `1` | PV | Recognisable, visible PV module surfaces |
| `255` | Ignore | Pixels whose PV status cannot be resolved, occlusion, or padding |

Draw the visible module footprint. Exclude gaps between modules when those gaps
are resolvable at the source resolution. Do not replace an array with its
bounding rectangle, fill an entire roof, or bridge visible background merely
to create one connected component. Grouping modules into arrays belongs to a
later evaluation or postprocessing step.

## Edge cases

- **Image boundary:** label the visible part of a cropped module as PV. Do not
  extend a label beyond the image or infer the hidden remainder.
- **Shadow:** retain PV where module surfaces remain recognisable. Use ignore
  where darkness prevents a reliable decision; a shadow alone is not PV.
- **Occlusion:** do not complete modules under trees, equipment, or other
  obstructions. Mark unresolved pixels as ignore.
- **Blur or small objects:** use ignore when the image cannot distinguish PV
  from another structure. Do not invent finer detail than the pixels show.
- **Skylights, roof windows, glass roofs, ventilation equipment, and dark roof
  surfaces:** mark background when their non-PV identity is clear. Keep these
  hard negatives in the dataset; use ignore for genuinely ambiguous areas.
- **Artificial padding:** always ignore, even if its fill colour resembles
  background. Padding must not become training evidence.

An all-background mask means the inspected image contains no visible PV in its
supervised area. A missing annotation file or an unreviewed model prediction is
not evidence for this negative label.

## Review workflow

1. Inspect the source image at its native resolution and confirm its orientation.
2. Mark recognisable PV surfaces and unresolved regions. A model proposal may
   assist drawing, but must be checked against the image.
3. Check that resolved gaps and non-PV roof structures remain background.
4. Overlay the mask on the exact image and inspect edges, cropped modules,
   shadows, and mask alignment.
5. Export canonical single-channel PNG values `0`, `1`, and `255`. Confirm that
   palette colours or a conventional `0/255` binary export have been converted
   explicitly, rather than silently interpreted as the canonical values.
6. Record annotation origin and review status using the
   [dataset format](data-format.md). Preserve the source's image and annotation
   licenses and attribution separately.

Use `reviewed` only after local review. Imported curated labels remain
`upstream` until reviewed locally; provisional labels remain `unreviewed` and
unassigned. Validation checks these declarations and encodings, not annotation
truth or model quality.
