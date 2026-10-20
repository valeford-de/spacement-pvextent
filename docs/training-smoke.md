# Training smoke test

The smoke workflow checks that real training images can pass through a model,
masked loss, optimizer, and resumable checkpoint. It uses a small, randomly
initialized model and a bounded subset of the training split. It does not
measure segmentation quality or produce a research baseline.

## Model and supervised pixels

`tiny-unet-v1` is a built-in two-level U-Net with 29,761 parameters. Its double
3 × 3 convolution blocks use ReLU and widths 8, 16, and 32. Max pooling reduces
resolution; skip connections and bilinear interpolation restore the exact input
dimensions, including odd dimensions. The single output channel contains raw
logits. There is no batch normalization, dropout, pretrained encoder, external
weight download, or third-party segmentation-model framework.

The model accepts floating-point RGB batches `[N, 3, H, W]`, with `N >= 1` and
both spatial dimensions at least 4. Its output has shape `[N, 1, H, W]`.

The training data view selects only `train` records with `upstream` or
`reviewed` annotations. It retains native dimensions and resolution, converts
RGB values to float32 in `[0, 1]`, and performs no resizing or augmentation.
Selection is deterministic for a fixed source snapshot and seed; the source
and selected sample fingerprints are recorded separately.

Mask values `0` and `1` provide background and PV supervision. Value `255` is
excluded from both BCE-with-logits and soft Dice loss. All-ignore masks provide
no supervision and are skipped. The training loss is BCE plus soft Dice loss;
neither component is a score measured on validation or test data.

## Run on a frozen split snapshot

Install the optional training dependencies:

```sh
uv sync --locked --extra train
```

Use the geographic split snapshot, choosing a fresh output directory:

```sh
uv run --locked --extra train pvextent train smoke \
  data/pv4ger-seg-geographic-v1 runs/smoke-001 \
  --device auto --steps 8 --samples 8 --batch-size 2 --seed 20261020 --lr 0.001 --json
```

`--device` accepts `auto`, `cpu`, `mps`, or `cuda`. Use an explicit device when
comparing or reproducing a run. Automatic selection prefers CUDA, then MPS,
then CPU. An explicitly requested unavailable accelerator fails instead of
silently switching devices. The small sample and update counts check the
pipeline; they are not a full training schedule.

The current bounded workflow accepts 1–100 total update steps, 1–64 selected
samples, and batch sizes 1–8, with the batch no larger than the selected sample
count. It uses Adam and clips the gradient norm at 1.0. Selected images must
have matching dimensions and ground resolution. The learning rate must be
finite, greater than zero, and at most 1.

Keep outputs under the ignored `runs/` directory or another location outside
Git. Source imagery, masks, and split assignments are not modified. Validation
and test images are not selected, opened, optimized against, or evaluated by
the training data view.

## Run artifacts and checkpoint continuation

Each successful run publishes a new output directory containing:

| File | Contents |
| --- | --- |
| `config.json` | Model, seed, optimizer settings, preprocessing, selected IDs, data fingerprints, target steps, and resume checkpoint hash |
| `metrics.jsonl` | One row per update in this run, with global step, sample IDs, valid pixel count, gradient norm, BCE, Dice loss, and combined loss |
| `checkpoint.pt` | Built-in model state, optimizer state, completed global step, configuration, and CPU/accelerator RNG states |
| `checkpoint.pt.sha256` | Exact checkpoint byte hash, checked before loading |
| `smoke-report.json` | Training-probe losses, finite gradient/update checks, verified checkpoint reload, environment, timing, fingerprints, and limitations |

Losses before and after training use the same fixed probe batch from the
selected training examples. Checkpoint reload must reproduce that batch's
logits within the recorded numerical tolerance. Outputs are staged and
published only after these checks pass; an existing run directory is never
overwritten.

To continue a run that reached step 8, choose a new output directory and request
step 12, retaining the original device and settings:

```sh
uv run --locked --extra train pvextent train smoke \
  data/pv4ger-seg-geographic-v1 runs/smoke-001-resumed \
  --device auto --steps 12 --samples 8 --batch-size 2 --seed 20261020 --lr 0.001 \
  --resume runs/smoke-001/checkpoint.pt --json
```

`--steps` is the **total target global step**, so this performs four additional
updates. The original run and its logs remain unchanged; the new JSONL log
contains steps 9–12. A target step at or below the checkpoint step is rejected.

Continuation requires the same dataset and selected sample fingerprints,
sample count, batch size, seed, learning rate, selected device, model, and
PyTorch version. Keep the `.sha256` companion beside the checkpoint. Loading
uses `weights_only=True` and the built-in model factory; the workflow accepts
bounded local state checkpoints, with no external model code or weight
download. Optimizer state and RNG state are restored along with the model.
Exact CPU continuation is tested; identical results across accelerator types,
hardware, or library versions are not promised.

Measured local runs and their artifact fingerprints are recorded separately in
the [training smoke audit](research/training-smoke-audit.md).

## Scope

The next model experiment is the separately planned pretrained
U-Net/ResNet18 baseline. This smoke architecture establishes the training and
checkpoint path before that experiment. A finite or decreasing smoke loss is
not evidence of useful PV detection, independent generalization, or acceptable
false-positive and false-negative rates. Those claims require the reviewed
evaluation data and a separate, fixed evaluation protocol.
