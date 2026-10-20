# U-Net/ResNet18 baseline audit, 2026-10-20

The first complete pretrained baseline reached **0.915717 validation pixel IoU**
and **0.956005 Dice** after three epochs. These measure agreement with upstream
annotations on the internal geographic validation split. The validation split
also selected the checkpoint; this is not an independent test result.

The [machine-readable audit](baseline-resnet18-001.json) preserves curated
aggregate measurements, configuration provenance and artifact hashes from the
completed local run `baseline-resnet18-001`. Raw logs, runtime configuration
dumps, imagery, masks and checkpoint binaries remain outside Git.

## Data and protocol

The experiment used the frozen [geographic PV4GER split](geographic-split-audit.md),
dataset fingerprint
`597b9facb7f115448e81af3b88de7e1081d510c38108c294195de05763d841c5`.

| Partition | Samples | Use in this run |
| --- | ---: | --- |
| Train | 3,282 | Optimization, complete declared split |
| Validation | 278 | Initial evaluation and checkpoint selection |
| Test | 421 | Reserved; test pixels remained unopened |
| Unassigned | 46 | Excluded from training and validation |

Inputs retained native **320 × 320 RGB pixels at 0.1 m/pixel**. No resizing or
additional cropping occurred. Float32 RGB values in `[0,1]` were normalized
with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation
`[0.229, 0.224, 0.225]`. Horizontal and vertical flips each had probability
`0.5`; each decision was applied identically to the image, target and valid-pixel
mask. Validation received no augmentation.

The model was `unet-resnet18-v1`, with 14,328,209 parameters. Its ResNet18 encoder
was initialized from ImageNet-1k classification weights, while the decoder
started randomly; all parameters were fine-tuned. The encoder source was
[`timm/resnet18.a1_in1k`](https://huggingface.co/timm/resnet18.a1_in1k/tree/491b427b45c94c7fb0e78b5474cc919aff584bbf),
pinned to revision `491b427b45c94c7fb0e78b5474cc919aff584bbf`. The safetensors
artifact SHA-256 was
`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.
The publisher declares Apache-2.0 for these weights; model code, weights and
training imagery retain their separate provenance and licensing. See
[pretrained weight provenance](../pretrained-weights.md).

The [versioned configuration](../../configs/baseline-resnet18.toml) fixed three
epochs, batch size `2`, seed `20261020`, AdamW learning rate `0.0001`, weight
decay `0.0001` and gradient clipping at norm `1.0`. This produced **4,923 optimizer
updates**. Loss combined valid-pixel BCE-with-logits and soft Dice with smoothing
`1.0`. Label `255` supplied no supervision. Reported losses are the
valid-pixel-weighted mean of per-batch objectives; their Dice component is not
one globally computed dataset loss.

Validation ran in evaluation mode without gradients. Integer TP/FP/FN/TN
counts were accumulated over all valid validation pixels before computing
metrics, with the probability threshold fixed at `0.5`. The best checkpoint
was selected by global validation pixel IoU among completed epochs. The
initial pass was a starting reference. The [complete protocol and continuation
workflow](../baseline-training.md) describe the implementation and recovery
checks.

Execution used an **Apple M2 with 8 GB unified memory and MPS**, Python 3.12.14,
PyTorch 2.14.1 and segmentation-models-pytorch 0.5.0. Total wall time was
**1,349.01 seconds**, approximately 22 minutes 29 seconds, including setup and
verification. This is one local execution observation, not a hardware benchmark.

## Measured results

Metrics below are fractions in `[0,1]`, computed globally over the same
28,467,200 supervised validation pixels. Training loss uses the batch-weighted
definition above.

| Completed epoch | Validation IoU | Dice | Precision | Recall | Training loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial | 0.052527 | 0.099811 | 0.284243 | 0.060534 | — |
| 1 | 0.898400 | 0.946481 | 0.950709 | 0.942291 | 0.362680 |
| 2 | 0.909116 | 0.952395 | 0.947246 | 0.957600 | 0.183985 |
| **3, selected best** | **0.915717** | **0.956005** | **0.954771** | **0.957241** | **0.155011** |

The final and best checkpoints came from epoch 3. Best validation counts were
**5,838,793 TP, 276,590 FP, 260,814 FN and 22,091,003 TN**. Validation loss was
`0.132027`; global pixel accuracy was `0.981122`. These are pixel counts and
ratios, not counts of correctly detected roofs or PV arrays.

## Qualitative inspection

Six validation examples were chosen deterministically by sorting
`SHA256("20261020:<sample-id>")` and taking the first six. Selection did not
depend on prediction quality. The preview used CPU inference with the verified
best checkpoint and its fixed threshold, and remains in the ignored local artifact
`artifacts/baseline-resnet18-validation.png`.

In these six examples, the main predicted extents followed the source labels.
Visible residual differences were mainly contour shifts, rounded corners,
small gaps and omitted portions near crop boundaries. The first row showed
blue false-negative pixels at an image edge where the source array was already
cut by the crop. The fourth row contained small isolated red false-positive
patches in a row array. No obvious alignment or orientation error was visible
in this small preview.

These observations are qualitative hypotheses from six selected validation
examples, not a complete error analysis. The source polygons fill spaces
between modules: the preview therefore concerns **array extent as defined by
the upstream labels**. It does not establish the physical area of individual
modules or exact panel boundaries.

## Artifact and implementation verification

Local artifact hashes were independently checked against the run report.
Reloading `best.pt` reproduced its recorded validation confusion counts
exactly.

| Artifact | SHA-256 |
| --- | --- |
| `best.pt` | `252e5ec9f20557fdce9ecd8a3c97b3421be3f63fb73e6c05f4df4e24e95e3635` |
| `last.pt` | `bdff7bc905539f34656809e999bf2137a5ebdf3c791db49c0e7f291e92cb39f8` |
| Runtime `config.json` | `a17d6cbf6ca0a823aba0d086bc9859fc5d9c2b303395d451d2562c0e4f458ac5` |
| Versioned `baseline-resnet18.toml` | `1bde83e401ca13a08ec4e3527751de27f179839317bf86bce87c5558525575ec` |

The implementation test suite passed **183 tests**, and the final
`uv build --quiet` completed successfully. The wheel's four new modules were
checked against the current source bytes; wheel and source distribution
contained no imagery, weights or runtime logs. Full-split pixel counts were
also checked: 336,076,800 supervised training pixels per epoch and 28,467,200
validation pixels. Regression coverage includes held-out pixel isolation,
exact global metric counts, ignored-label handling, encoder integrity,
checkpoint reload, epoch-boundary continuation and failure recovery.

## Interpretation and limits

This establishes a working transfer-learning baseline and strong agreement
with the internal validation annotations. PV4GER-SEG is biased toward
PV-positive crops and lacks a representative, locally reviewed collection of
PV-free roofs. Consequently, pixel precision here does not establish roof-level
false-positive performance in deployment.

Upstream masks retain their imported status. Coarse polygon boundaries and
annotation errors can affect both training and reported agreement; complete
local label review remains outstanding. Geographic separation was established
using approximate filename-derived centers, not verified building identities
or acquisition footprints. It does not prove independence from every city,
imagery campaign or published model's training data.

No independent test evaluation or released model weights accompany this audit.
The 421 reserved test examples remain outside this training and selection
workflow. Checkpoints are local research artifacts; representative reviewed
negatives, array-level evaluation and an independent test protocol remain
necessary before making operational detection claims.
