# Pretrained encoder provenance

The first pretrained segmentation baseline uses SMP's U-Net with a ResNet18
encoder. Its encoder starts from the publisher's
[timm/resnet18.a1_in1k checkpoint](https://huggingface.co/timm/resnet18.a1_in1k/blob/491b427b45c94c7fb0e78b5474cc919aff584bbf/README.md),
trained on ImageNet-1k. The U-Net decoder and one-channel segmentation head are
new, randomly initialized layers. These encoder weights are not a pretrained
PV detector and do not establish PV segmentation accuracy.

## Exact weight source

| Field | Pinned value |
| --- | --- |
| Model publisher | PyTorch Image Models (`timm`) |
| Model repository | `timm/resnet18.a1_in1k` |
| Revision | `491b427b45c94c7fb0e78b5474cc919aff584bbf` |
| Artifact | `model.safetensors` |
| Size | 46,807,446 bytes |
| SHA-256 | `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612` |
| Weight license declaration | `apache-2.0` in the publisher's pinned model card |

The [revision-specific file metadata](https://huggingface.co/api/models/timm/resnet18.a1_in1k/revision/491b427b45c94c7fb0e78b5474cc919aff584bbf?blobs=true)
provides the artifact size and LFS SHA-256. The adapter checks both against the
actual downloaded bytes. It uses this explicit publisher declaration for the
**weights**, separately from the software licenses below.

## Prepare a local cache

Install the baseline dependencies, then explicitly permit the pinned download:

```sh
uv sync --locked --extra baseline
uv run --locked --extra baseline python - <<'PY'
import json
from pvextent.models.resnet_unet import prepare_encoder_weights

provenance = prepare_encoder_weights(
    "data/pretrained/resnet18.a1_in1k", download=True
)
print(json.dumps(provenance, indent=2))
PY
```

The cache contains three original publisher files:

| Local file | Purpose | SHA-256 |
| --- | --- | --- |
| `model.safetensors` | ImageNet-1k encoder/classifier tensors | `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612` |
| `model-card.md` | Pinned publisher description, license declaration, and citations | `e96d7547dfaff6f0a0d74209819c461ec2d66556a9d6b059f1c2a50d413da5bb` |
| `config.json` | Pinned architecture and preprocessing metadata | `a15887d36094642dbffa237f2c7b66101d7f8ab704c45981b9fafb107cd99d28` |

All three files are size- and hash-checked. Downloads stream to temporary files
and are published only after verification, with a 64 MiB per-file upper bound.
Existing files with mismatched bytes fail instead of being replaced. Keeping
the cache under the ignored `data/` directory keeps weights outside Git.

Subsequent calls use `download=False`, the default, to verify the cache offline:

```python
from pvextent.models.resnet_unet import (
    build_resnet_unet, normalize_rgb, prepare_encoder_weights,
)

provenance = prepare_encoder_weights("data/pretrained/resnet18.a1_in1k")
model = build_resnet_unet(provenance["path"])
# images: float32 RGB tensors [N,3,H,W], already scaled to [0,1].
logits = model(normalize_rgb(images))
```

Model construction never downloads weights. `build_resnet_unet(None)` explicitly
constructs a random encoder for isolated tests or restoration of a complete
local segmentation checkpoint; it is not the pretrained initialization path.

## Architecture and preprocessing

The built-in model identifier is `unet-resnet18-v1`. Construction passes
`encoder_name="resnet18"` and `encoder_weights=None` to
[SMP U-Net](https://smp.readthedocs.io/en/latest/models.html#unet), preventing
implicit downloads. The model uses encoder depth 5, decoder widths
`256, 128, 64, 32, 16`, batch normalization, nearest-neighbour decoder
interpolation, three input channels, and one output channel with no activation.
The complete model has 14,328,209 trainable parameters with SMP 0.5.0.

The adapter loads the verified safetensors bytes directly, checks finite
floating-point tensors, and removes exactly `fc.weight` and `fc.bias`, the
ImageNet classification head. All 120 remaining encoder entries must match
the SMP ResNet18 encoder's keys, shapes, and dtypes. They are loaded strictly;
missing or additional encoder tensors fail instead of being silently skipped.
The decoder and segmentation head are not loaded from the classifier.

[The pinned publisher configuration](https://huggingface.co/timm/resnet18.a1_in1k/blob/491b427b45c94c7fb0e78b5474cc919aff584bbf/config.json)
specifies RGB means `[0.485, 0.456, 0.406]` and standard deviations
`[0.229, 0.224, 0.225]`. `normalize_rgb` applies those channel statistics to
finite float32 RGB values in `[0,1]`, preserving the grid and gradients. It
does not resize, crop, change channel order, or modify the input tensor.

PVExtent retains the canonical 320 × 320 pixel, 0.1 m/pixel context for this
experiment. This differs from the classifier's original image-size/crop
settings: applying classification centre crops would remove segmentation
context and require changing the masks. Model tests use dimensions divisible
by 32; training-mode batch normalization also requires sufficient samples at
the deepest resolution. The baseline's native 320-pixel crops and batch size
2 satisfy those constraints.

## Software licenses and retained attribution

The dependency versions used for initial compatibility checks are recorded in
`uv.lock`; exact versions also belong in experiment reports. The primary code
license sources checked for this adapter are:

| Dependency | Checked version | Code license source |
| --- | --- | --- |
| Segmentation Models PyTorch | 0.5.0 | [MIT, copyright 2019 Pavel Iakubovskii](https://github.com/qubvel-org/segmentation_models.pytorch/blob/v0.5.0/LICENSE) |
| Torchvision | 0.29.1 | [BSD-3-Clause, copyright Soumith Chintala 2016](https://github.com/pytorch/vision/blob/v0.29.1/LICENSE) |
| timm | 1.0.30 | [Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/v1.0.30/LICENSE) |
| safetensors | 0.8.0 | [Apache-2.0](https://github.com/huggingface/safetensors/blob/main/LICENSE) |

Dependency code licenses do not substitute for a checkpoint's own license
declaration. Preserve the pinned publisher model card, its citations, applicable
Apache-2.0 license text and notices, source weight hash, and the training-data
provenance with any derived weight release. Preserve dependency copyright and
license notices when redistributing their code. PVExtent's original adapter
code remains copyright Valeford under Apache-2.0.

The checkpoint's ImageNet-1k training origin remains part of its provenance;
this weight license declaration does not relicense the underlying image set.
No ImageNet images or upstream weights are committed to this repository.
