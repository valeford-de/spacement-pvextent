# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SegFormer (MiT-B2) model construction and weight verification."""

from __future__ import annotations

import unittest
from unittest.mock import patch
import torch

from pvextent.models.segformer import (
    MODEL_NAME,
    WEIGHT_SPEC,
    build_segformer_mit_b2,
    prepare_segformer_encoder_weights,
    EncoderWeightsError,
)
from pvextent.training.baseline import load_config, BaselineTrainingError


class SegformerTests(unittest.TestCase):
    def test_model_name_and_spec(self) -> None:
        self.assertEqual(MODEL_NAME, "segformer-mit-b2-v1")
        self.assertEqual(WEIGHT_SPEC["model_id"], "smp-hub/mit_b2.imagenet")
        self.assertEqual(WEIGHT_SPEC["size"], 98867640)
        self.assertEqual(WEIGHT_SPEC["sha256"], "ab18b54f81748427bae1183160822903470bcca0e3121944c0d185b4b1816a95")

    def test_build_random_segformer(self) -> None:
        model = build_segformer_mit_b2(None)
        x = torch.randn(2, 3, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (2, 1, 64, 64))
        params = sum(p.numel() for p in model.parameters())
        self.assertGreater(params, 20_000_000)

    def test_config_loader_accepts_segformer_schema_1_1(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.toml"
            config_path.write_text(
                """schema_version = "1.1"
model = "segformer-mit-b2-v1"
epochs = 3
batch_size = 2
learning_rate = 0.00006
weight_decay = 0.01
seed = 20261020
threshold = 0.5
horizontal_flip = 0.5
vertical_flip = 0.5
max_train_samples = 0
max_val_samples = 0
"""
            )
            config = load_config(config_path)
            self.assertEqual(config["model"], "segformer-mit-b2-v1")
            self.assertEqual(config["schema_version"], "1.1")


if __name__ == "__main__":
    unittest.main()
