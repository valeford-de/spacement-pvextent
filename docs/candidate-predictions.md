# Candidate predictions

Candidate inference applies a local baseline checkpoint to unreviewed imagery
and exports an offline inspection gallery. It makes predictions without
evaluating them against placeholder masks or changing any annotation.

## Run the export

Install the baseline dependencies, then use a prepared candidate snapshot and
a verified local `best.pt` checkpoint:

```sh
uv sync --locked --extra baseline
uv run --locked --extra baseline pvextent evaluate candidates \
  data/nrw-roof-candidates-dortmund-v1 \
  runs/baseline-resnet18-001/best.pt \
  artifacts/nrw-candidate-predictions --device auto --json
```

The checkpoint's `best.pt.sha256` companion must remain beside it. Loading uses
`weights_only=True`, the built-in U-Net/ResNet18 factory, and strict model-state
restoration. The encoder is constructed with no weight download: this command
runs offline and requires no original pretrained-weight cache.

Choose a new output directory separate from both the candidate source and the
checkpoint directory. Existing outputs are never replaced. The bundle is
staged and published only after every prediction and artifact has been written
successfully. Source imagery, masks, metadata and checkpoint bytes are not
modified. Keep generated imagery under the ignored `artifacts/` directory or
another location outside Git.

The Python API is
`pvextent.inference.candidates.export_candidate_predictions(source_dir,
checkpoint_path, output_dir, *, device="auto", progress=None)`.
It returns the prediction report or raises `CandidatePredictionError`.

## Accepted source snapshots

The exporter accepts a canonical snapshot containing 1–24 records, all with
`split = "unassigned"`, `annotation_status = "unreviewed"`, and masks filled
entirely with `255` ignore pixels. Assigned training, validation or test records
are rejected from their declarations before raster paths are opened. Reviewed
records and source masks containing supervised `0` or `1` pixels are also
rejected; this command is for unannotated candidates.

Canonical validation verifies actual image and mask encodings, dimensions,
hashes and provenance before the model starts. All examples must have common
native dimensions, each side between 32 and 1,024 pixels and divisible by 32,
and one common declared GSD. RGB values become float32 in `[0,1]` and receive
the checkpoint model's ImageNet channel normalization. No image or mask is
resized, cropped or relabelled.

This checkpoint schema does not declare the training GSD. The report therefore
records the source GSD and leaves `training_gsd_m` unknown. A common source GSD
does not establish that the new imagery matches the training distribution.
Source-domain transfer remains unmeasured.

## Inspect predictions after the RGB image

Open `artifacts/nrw-candidate-predictions/index.html` directly in a browser.
Original RGB crops are always visible; **Show model prediction** panels start
closed. Inspect the image at native resolution before opening a panel, so a
model proposal does not become the starting assumption for annotation.

An opened prediction panel shows a green overlay, the number and fraction of
pixels classified as PV, and links to the raw prediction PNG and unchanged
source mask. The threshold comes from the checkpoint configuration. Prediction
PNG values are `0` for model-predicted background and `1` for model-predicted PV;
the preserved source mask still contains only `255` ignore values.

Every human review decision remains **pending**, including when the model
predicts zero PV pixels. The gallery does not approve samples, mark negatives,
edit masks or change annotation status. Use the separate
[local label-review workflow](label-review.md) to record named human decisions
and any needed corrections.

## Output and provenance

| File or directory | Contents |
| --- | --- |
| `index.html` | Local RGB gallery with initially hidden prediction panels |
| `prediction-report.json` | Source/checkpoint fingerprints, device, threshold, GSD, dimensions, versions and predicted-pixel counts |
| `review-manifest.json` | Source record snapshots, copied file paths, prediction/overlay hashes and pending decisions |
| `dataset.toml` | Exact source dataset metadata |
| `images/`, `masks/` | Byte-identical source images and all-ignore masks |
| `predictions/`, `overlays/` | Native-grid model outputs and display overlays |
| `provenance/` | Available source notices and license information |

The report records the actual Torch, Torchvision, SMP, timm and safetensors
versions alongside those recorded in the checkpoint. A comparison flag exposes
different version contexts; it is unknown when the checkpoint lacks a version
record. Matching file hashes and state keys do not promise identical numerical
outputs across different hardware or library versions.

The bundle is an inspection artifact, not a newly annotated canonical dataset.
Its source records and copied notices preserve attribution and the source
snapshot fingerprint. Dataset and model licenses continue to apply separately.

All source pixels are ignored for supervision, so there are no ground-truth
targets from which to compute accuracy, IoU, Dice, precision, recall, false
positives or false negatives. Predicted-pixel counts describe the model's
output only. Neither a nonempty prediction nor an empty prediction establishes
whether visible PV is actually present. Candidate imagery remains outside the
training snapshot and is not an independent labelled benchmark.
