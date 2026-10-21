# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for DeepLabV3+ (ResNet50 ASPP) model construction and weight verification."""

from __future__ import annotations

import unittest
import torch

from pvextent.models.deeplabv3plus import (
    MODEL_NAME,
    WEIGHT_SPEC,
    build_deeplabv3plus_resnet50,
    prepare_deeplabv3plus_encoder_weights,
    EncoderWeightsError,
)
from pvextent.training.baseline import load_config, BaselineTrainingError


class DeepLabV3PlusTests(unittest.TestCase):
    def test_model_name_and_spec(self) -> None:
        self.assertEqual(MODEL_NAME, "deeplabv3plus-resnet50-v1")
        self.assertEqual(WEIGHT_SPEC["model_id"], "smp-hub/resnet50.imagenet")
        self.assertEqual(WEIGHT_SPEC["size"], 102464800)
        self.assertEqual(WEIGHT_SPEC["sha256"], "df1aad85e18536504a4c8597118364e291ff3a9c4b56dd9b3a4900642e4c3a7c")

    def test_build_random_deeplabv3plus(self) -> None:
        model = build_deeplabv3plus_resnet50(None)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 25_000_000)

    def test_config_loader_accepts_deeplabv3plus_schema_1_1(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """schema_version = "1.1"
model = "deeplabv3plus-resnet50-v1"
epochs = 3
batch_size = 2
learning_rate = 0.0001
weight_decay = 0.0001
seed = 20261020
threshold = 0.5
horizontal_flip = 0.5
vertical_flip = 0.5
max_train_samples = 0
max_val_samples = 0
"""
            )
            config = load_config(config_path)
            self.assertEqual(config["model"], "deeplabv3plus-resnet50-v1")
            self.assertEqual(config["schema_version"], "1.1")


if __name__ == "__main__":
    unittest.main()
