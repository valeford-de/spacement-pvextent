# Research roadmap

PVExtent starts as a standalone experiment. The first goal is a reproducible
baseline for photovoltaic array segmentation from orthophotos.

## 1. Establish the dataset

Step 1's [data contract and validator](data-format.md),
[annotation rules](annotation-guide.md), and synthetic format demonstration are
implemented. Real dataset import, label review, and an independent test set are
the next milestones.

- Record imagery source, capture date, ground sampling distance, geographic
  reference, usage rights, and content hash.
- Define the target consistently: visible PV array regions, with documented
  rules for occlusion, shadows, small arrays, and image boundaries.
- Create reviewed segmentation masks and include hard negatives such as
  skylights, glass roofs, ventilation equipment, and dark roof surfaces.
- Keep uncertain labels explicit. Model predictions can assist annotation but
  are not ground truth.

## 2. Separate training and evaluation

- Group images by building and geographic area before creating splits.
- Keep repeat captures of a building together; reserve additional acquisition
  dates or sources to measure transfer where data permits.
- Freeze an independent test set before choosing models or thresholds. Use a
  separate validation set for experiment selection.
- Record possible overlap with any pretrained model's training data.

## 3. Train a baseline

- Select a pretrained segmentation backbone after checking code, weight, and
  dataset licenses.
- Add explicit experiment configuration for preprocessing, spatial resolution,
  augmentation, optimization, random seed, and data split identifiers.
- Fine-tune one baseline and save the complete configuration alongside each
  checkpoint. Add architectural alternatives only when comparison is useful.

## 4. Evaluate the resulting model

- Report pixel-level overlap metrics, including IoU and Dice/F1.
- Measure array detection precision and recall using a documented spatial
  matching rule, including false positives on negative images.
- Inspect errors by image source, resolution, array size, and acquisition date.
- Record runtime and hardware separately from prediction quality. Treat
  uncertainty and abstention explicitly if introduced.

## 5. Publish a reproducible research release

- Include a model card describing intended use, data provenance, limitations,
  evaluation protocol, and results on the frozen test set.
- Publish training and inference instructions, dependency versions, experiment
  configurations, and checkpoint hashes.
- State the license of each released artifact separately and retain applicable
  third-party attribution.

Unless explicitly marked implemented above, these are planned milestones, not
implemented capabilities or measured results.
