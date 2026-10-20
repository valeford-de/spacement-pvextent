![Spacement PVExtent](docs/assets/pvextent-banner.png)

![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)
![uv](https://img.shields.io/badge/uv-%23DE5FE9.svg?style=for-the-badge&logo=uv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=pytorch&logoColor=white)

Photovoltaic array segmentation from orthophotos.

Developed and maintained by **Valeford** under the **Spacement** brand.

PVExtent is a standalone research project for training and evaluating models
that identify the visible extent of photovoltaic arrays in aerial imagery.
Its modules separate data preparation, model construction, training,
evaluation, and inference.

## Status

The dataset format, annotation rules, and an offline dataset validator are
implemented. Training and inference are not implemented yet, and no trained
model or performance result has been released. See the
[research roadmap](docs/roadmap.md) for the next steps.

## Documentation

The [project wiki](https://github.com/valeford-de/spacement-pvextent/wiki)
provides an introduction and getting-started documentation. The versioned
[research roadmap](docs/roadmap.md) remains in this repository.

Start with the [dataset format](docs/data-format.md) and
[annotation guide](docs/annotation-guide.md) when preparing imagery and masks.

For private vulnerability reports, see the [security policy](.github/SECURITY.md).

## Development setup

Requires Python 3.11 or newer; development uses Python 3.12. Install
[uv](https://docs.astral.sh/uv/getting-started/installation/), then run:

```sh
uv sync --locked
uv run --locked pvextent --help
uv run --locked python -m unittest discover -s tests -v
```

## Validate a dataset

```sh
uv run --locked pvextent data validate /path/to/dataset --json
```

The validator checks metadata, file hashes, image/mask alignment, mask classes,
and declared split boundaries. A passing report does not certify annotation
truth, usage rights, or geographic independence.

To try the format using generated synthetic pixels, without downloading imagery:

```sh
uv run --locked python examples/create_demo_dataset.py data/demo
uv run --locked pvextent data validate data/demo --json
```

The demo is a structural example with unreviewed labels and unknown capture
dates/georeferences, so its valid report includes warnings. It is not a training
dataset.

## Package layout

| Module | Responsibility |
| --- | --- |
| `pvextent.data` | Dataset loading, labels, provenance, and preprocessing |
| `pvextent.models` | Model construction and architecture adapters |
| `pvextent.training` | Training and checkpoint creation |
| `pvextent.evaluation` | Segmentation and array detection evaluation |
| `pvextent.inference` | Predictions from images and trained checkpoints |

The data module provides validation; the remaining modules define project
boundaries for the planned training pipeline.

## License and attribution

Original project code is copyright Valeford and licensed under the
[Apache License 2.0](LICENSE). Preserve the applicable copyright, license,
and [NOTICE](NOTICE) information when redistributing it, as required by the
license.

Datasets and model checkpoints have their own provenance and licensing
requirements. No third-party imagery or model weights are included in this
initial scaffold. Future releases will document their sources and terms
separately.

## Maintained by

<a href="https://valeford.de">
  <img src="docs/assets/valeford-logo.png" alt="Valeford" width="220">
</a>
