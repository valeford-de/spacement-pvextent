# Pretrained U-Net/ResNet18 baseline

The baseline experiment trains a U-Net with a ResNet18 encoder initialized from
ImageNet weights. It uses the frozen training partition for optimization and
the validation partition for checkpoint selection. Test pixels remain unopened.
The small randomly initialized [training smoke workflow](training-smoke.md)
remains a separate check of the training pipeline.

The [baseline comparison guide](baseline-comparison.md) covers a fresh run on
`data/pvextent-training-v2` with the same training protocol and unchanged
validation/test boundaries. Completed comparison measurements belong in the
dedicated [comparison audit](research/baseline-negative-comparison-audit.md).

## Run the configured experiment

Install the optional baseline dependencies and use a new run directory:

```sh
uv sync --locked --extra baseline
uv run --locked --extra baseline pvextent train baseline \
  data/pv4ger-seg-geographic-v1 runs/baseline-001 \
  --config configs/baseline-resnet18.toml \
  --weights-dir artifacts/weights/resnet18-a1-in1k \
  --download-weights --device auto --json
```

`--download-weights` explicitly allows fetching the pinned encoder artifact,
model card and source configuration. Without the flag, all three files must
already exist in the local cache and pass size and SHA-256 checks. An existing
mismatched file is never overwritten. Model construction itself stays offline.
Keep the encoder cache and run output in separate directories; neither may contain the other.

The encoder source is
[`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k/tree/491b427b45c94c7fb0e78b5474cc919aff584bbf),
pinned to revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`.
The safetensors file's SHA-256 is
`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.
The cache retains the publisher's model card and configuration with their own
hashes. Published encoder provenance and package versions accompany run outputs.

Device choices are `auto`, `cpu`, `mps` and `cuda`. Automatic selection prefers
CUDA, then MPS, then CPU. An explicitly unavailable device fails. Use a fixed
explicit device when reproducing a run or continuing a checkpoint.

The flat TOML schema is strict: missing or additional fields fail.
[`configs/baseline-resnet18.toml`](../configs/baseline-resnet18.toml) contains all
required fields:

| Field | Initial experiment value |
| --- | --- |
| `schema_version` | `"1.0"` |
| `model` | `"unet-resnet18-v1"` |
| `epochs` | `3` total target epochs |
| `batch_size` | `2` |
| `learning_rate` | `0.0001` |
| `weight_decay` | `0.0001` |
| `seed` | `20261020` |
| `threshold` | `0.5` |
| `horizontal_flip` | `0.5` |
| `vertical_flip` | `0.5` |
| `max_train_samples` | `0`: full declared training split |
| `max_val_samples` | `0`: full declared validation split |

Positive sample limits create bounded pipeline experiments and are recorded in
the contract. Their results do not describe the full-split experiment. Epochs
are bounded to 1–100 and batch size to 1–16. No automatic hyperparameter search
or early stopping is performed.

## Data and evaluation scope

Use a versioned [geographic split snapshot](geographic-splits.md). The current
PV4GER snapshot contains 3,282 training examples and 278 validation examples,
with 421 examples reserved for a later test evaluation. Its 46 unassigned
examples remain outside training and validation. See the
[split audit](research/geographic-split-audit.md) for the actual distances,
exclusions and dataset fingerprint.

The initial experiment preserves native RGB pixels, 320 × 320 crops, and
0.1 m/pixel. Other input snapshots must provide matching train/validation
sizes divisible by 32 and a common GSD. The workflow does not resize imagery
or masks. Values `0` and `1` supply background and PV supervision; `255` is excluded from the loss and metric counts. Annotation
status stays unchanged. Upstream labels do not become locally reviewed labels
through training.

For native 32 × 32 crops, every training batch must contain at least two samples,
including the final batch, because ResNet18 BatchNorm otherwise has only one value per channel.

ImageNet encoder initialization is transfer learning from a general image
classification task. It is not a pretrained PV segmentation model. The decoder
starts with random weights, and the complete segmentation model is trained
against the PV masks. RGB channels are converted to float32 in `[0,1]`, then
normalized using ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. This channel normalization does not change resolution
or the pixel grid.

## Training and validation protocol

The initial experiment fixes three epochs, batch size two, learning rate
`0.0001`, weight decay `0.0001`, seed `20261020`, and probability threshold
`0.5`. Horizontal and vertical flips each have probability `0.5`; imagery,
target masks and valid-pixel masks receive the same flips. No spatial resampling
occurs. Freeze the source snapshot and configuration before running; do not
select seeds or thresholds from test results. The optimizer is AdamW, with
gradient norm clipped at `1.0`.

Validation uses evaluation mode with gradients disabled. Its pixel confusion
counts are accumulated over all supervised validation pixels before computing
IoU, Dice, precision and recall. These are global pixel metrics, not averages
of crop scores or array detection metrics. An initial validation pass records
the starting model before optimization.

Reported training and validation **loss** is the valid-pixel-weighted mean of
the per-batch objective, masked BCE plus soft Dice loss:

```text
reported_loss = sum(batch_valid_pixels * batch_loss) / sum(batch_valid_pixels)
```

Each soft Dice loss is computed within its batch. This loss summary depends on
batch composition and is not a single globally computed Dice loss. The reported
IoU, Dice, precision and recall instead use the globally aggregated integer
pixel confusion counts described above; the loss and those metrics are
different quantities.

The best checkpoint is selected by validation IoU among completed training
epochs; ties retain the earlier best epoch. The initial validation pass is a
starting reference. A separate last checkpoint retains the final completed
epoch and optimizer state for continuation. Neither choice uses test data. Validation results guide this experiment and are not an
independent final performance estimate.

## Artifacts and epoch-boundary continuation

A successful run publishes a new directory after training and verification:

| File | Contents |
| --- | --- |
| `config.json` | Frozen preprocessing, hyperparameters, dataset/split fingerprints, encoder identity, device, package versions and target epochs |
| `history.jsonl` | Initial validation followed by one record per completed epoch, with training loss, gradient norms and validation metrics |
| `last.pt` | Current model, AdamW state, completed epoch/global step, RNG state and the earlier best model/metrics |
| `best.pt` | Best model weights, selected epoch, validation metrics and provenance; no optimizer state |
| `last.pt.sha256`, `best.pt.sha256` | Hash companions checked before checkpoint loading |
| `baseline-report.json` | Initial/final/best validation, completed epoch counts, sample counts, reload checks, hashes, environment and limitations |

Checkpoints are bounded local state dictionaries loaded with
`weights_only=True`. Reloading `best.pt` must reproduce its recorded validation
confusion counts before a successful run is published. Existing output
directories are never replaced. Dataset pixels, labels and split assignments
are unchanged.

To continue after epoch 3, copy the original TOML configuration, change only
`epochs = 6`, and choose a new output directory:

```sh
uv run --locked --extra baseline pvextent train baseline \
  data/pv4ger-seg-geographic-v1 runs/baseline-001-continued \
  --config configs/baseline-resnet18-extended.toml \
  --weights-dir artifacts/weights/resnet18-a1-in1k \
  --device auto --resume runs/baseline-001/last.pt --json
```

The epoch target includes previously completed epochs, so this performs epochs
4–6. Every other configuration field must match, along with the dataset and
selected split fingerprints, encoder identity, resolved device and package
versions. `best.pt` is an inference weights bundle and cannot be used for resume.
The self-contained `last.pt` carries the earlier best weights, so continuation
does not require the old `best.pt` or original encoder cache. Its `.sha256`
companion is still required. Parent run files are not rewritten.

If a handled failure or keyboard interruption occurs after a complete epoch,
the requested success directory remains unpublished. Recovery normally retains
checkpoint artifacts in a separate `<run-name>.incomplete-*` directory. If that
directory cannot be published, the original hidden `.<run-name>-baseline-*`
working directory is retained instead. The error identifies the actual retained
path, including a previous checkpoint generation if rollback was needed.

A checkpoint is advertised for continuation only after its payload matches its
SHA-256 companion. Continue the verified path into another new run directory
after resolving the failure. Unverified artifacts are retained without a
resumable-checkpoint claim; a written failure report then has `resume_from =
null`. If the failure report itself cannot be written, checkpoint files are
still preserved. Failed staging directories without checkpoint artifacts are
cleaned up. This recovery path does not guarantee survival of a power loss,
process kill or storage failure.

Validation is rerun at the start of a continuation to check the current model;
the report also preserves the original run's initial validation. CPU tests
compare continuous training with epoch-boundary continuation exactly. Results
across devices, hardware or library versions are not promised to be identical.

## Inspect validation predictions

After the full run has finished successfully, create a qualitative preview from
its verified best checkpoint:

```sh
uv run --locked --extra baseline python examples/preview_baseline.py \
  data/pv4ger-seg-geographic-v1 runs/baseline-resnet18-001/best.pt \
  artifacts/baseline-resnet18-validation.png --samples 6
```

This command creates a new PNG containing RGB images, source PV labels,
thresholded predictions and pixel error maps. Six validation examples are
selected deterministically using the run's seed. Green marks true-positive
pixels, red false positives, blue false negatives, gray true negatives and
orange ignored pixels in the error map. The checkpoint's hash companion and
validation snapshot fingerprints are checked before predictions are made.

The preview runs locally on CPU, uses the checkpoint's fixed threshold, and
opens no test imagery. Its examples come from the same validation partition
used for checkpoint selection, so it is an inspection artifact rather than
an independent performance estimate. An existing PNG is not overwritten; use
a fresh output path for another preview. Keep the generated imagery under the
ignored `artifacts/` directory or another location outside Git.

## Limitations

This is an internal source holdout with approximate centre coordinates and
upstream masks. It does not establish independence from all pretrained models,
cities or acquisition campaigns. Representative reviewed negatives, array-level
matching and an independent test protocol remain separate work. Pixel IoU alone
does not establish acceptable false-positive or false-negative rates in use.

Keep datasets, encoder weights, checkpoints and run artifacts outside Git.
Original project code, third-party model code, encoder weights and training
data retain their respective source notices and terms.
