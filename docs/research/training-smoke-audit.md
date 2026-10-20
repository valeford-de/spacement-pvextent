# Training smoke audit, 2026-10-20

The [training smoke workflow](../training-smoke.md) was executed on an Apple M2
using MPS, Python 3.12.14 and PyTorch 2.14.1. The source is the frozen geographic
PV4GER split, fingerprint
`597b9facb7f115448e81af3b88de7e1081d510c38108c294195de05763d841c5`.
The [machine-readable audit](training-smoke-mps-v1.json) records aggregate
measurements, environment and artifact hashes. No pixels, masks, logs or model
checkpoint binaries are committed.

## Configuration

- Model: randomly initialized `tiny-unet-v1`, 29,761 trainable parameters.
- Inputs: eight deterministically selected **training** samples, native
  320 × 320 RGB pixels at 0.1 m/pixel; batch size two, seed `20261020`.
- Preprocessing: float32 RGB in `[0,1]`, with no resizing or augmentation.
- Loss: valid-pixel BCE-with-logits plus global soft Dice loss; `255` is ignored.
- Optimizer: Adam, learning rate `0.001`, gradient norm clipped at `1.0`.
- No pretrained weights, validation metrics, test metrics or model selection.

## Executed checks

| Run | Updates performed | Initial → final fixed training-probe loss | Checkpoint reload |
| --- | --- | --- | --- |
| Initial MPS run | 1–8 | 1.385390 → 1.349318 | Identical probe logits |
| Continue initial checkpoint | 9–12 | 1.349318 → 1.314520 | Identical probe logits |
| Fresh uninterrupted MPS run | 1–12 | 1.385390 → 1.314520 | Identical probe logits |

All updates had finite, nonzero gradients and changed model parameters.
Observed pre-clipping gradient norms across twelve updates were approximately
0.08635–0.59661. The uninterrupted parameter update L2 norm was 0.79809.
Each checkpoint is 388,725 bytes and has an integrity hash companion.

The continued and uninterrupted runs produced **identical entire checkpoint
state**, including model parameters, Adam state, completed step, configuration
and RNG states. Their maximum parameter difference is zero; their checkpoint
file SHA-256 is the same:
`233231f1d8212da1e74417c17c6b26f09373afac58bf973653faaadca3b433a9`.
This is a measured result on this machine and dependency snapshot, not a promise
of bitwise equivalence across accelerators, hardware or PyTorch versions.

The first eight-update loop took approximately 1.02 s; its complete smoke
operation took 5.66 s, including setup, input checks, initial probe and checkpoint
verification. The resumed four-update operation took 2.19 s overall. These are
small local execution observations, not an accelerator benchmark; process
startup, backend warmup and filesystem caches affect them.

CPU regression tests independently compare continuous versus resumed training
on generated images. They also verify held-out isolation using deliberately
unreadable validation/test image files, zero gradient at ignored pixels,
configuration/fingerprint mismatch rejection, finite updates, checkpoint
integrity checks, and restoration of runtime state after failures.

## Interpretation

The data, backward pass, optimizer and checkpoint continuation paths work on
the available hardware. The loss is measured on a fixed minibatch drawn from
the same tiny training subset. The BCE component decreases while Dice loss
increases in this run; the combined loss alone does not demonstrate useful
PV segmentation. Source labels retain their imported status, hard negatives
still require review, and no generalization or detection-rate claim follows.

The next experiment is the separately configured pretrained U-Net/ResNet18
baseline, using the established split and reviewed evaluation protocol. This
smoke checkpoint is a local development artifact, not a released model.
