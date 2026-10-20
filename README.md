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

This repository contains an early Python package scaffold. Training and
inference are not implemented yet, and no trained model or performance result
has been released. See the [research roadmap](docs/roadmap.md) for the next steps.

## Documentation

The [project wiki](https://github.com/valeford-de/spacement-pvextent/wiki)
provides an introduction and getting-started documentation. The versioned
[research roadmap](docs/roadmap.md) remains in this repository.

For private vulnerability reports, see the [security policy](.github/SECURITY.md).

## Development setup

Requires Python 3.11 or newer.

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import pvextent; print(pvextent.__version__)"
```

## Package layout

| Module | Responsibility |
| --- | --- |
| `pvextent.data` | Dataset loading, labels, provenance, and preprocessing |
| `pvextent.models` | Model construction and architecture adapters |
| `pvextent.training` | Training and checkpoint creation |
| `pvextent.evaluation` | Segmentation and array detection evaluation |
| `pvextent.inference` | Predictions from images and trained checkpoints |

The initial modules define project boundaries; they do not yet provide a
working training pipeline.

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
