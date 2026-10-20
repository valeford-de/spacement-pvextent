# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""A small, randomly initialized U-Net for training-pipeline smoke tests.

This architecture is not the planned pretrained research baseline. It requires
no external weights, model registry, segmentation framework, or network access.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

MODEL_NAME = "tiny-unet-v1"


class _DoubleConv(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )


class TinyUNet(nn.Module):
    """Two pooling levels and an unactivated binary-segmentation logits head."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = _DoubleConv(3, 8)
        self.encoder2 = _DoubleConv(8, 16)
        self.bottleneck = _DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.decoder2 = _DoubleConv(32 + 16, 16)
        self.decoder1 = _DoubleConv(16 + 8, 8)
        self.head = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        if not isinstance(inputs, Tensor) or inputs.ndim != 4:
            raise ValueError("TinyUNet expects a tensor with shape [N, 3, H, W].")
        if inputs.shape[0] < 1 or inputs.shape[1] != 3 or min(inputs.shape[-2:]) < 4:
            raise ValueError("TinyUNet requires N >= 1, exactly 3 RGB channels, and H/W >= 4.")
        if not inputs.is_floating_point():
            raise ValueError("TinyUNet expects floating-point RGB inputs, not integer or boolean pixels.")
        if inputs.dtype != self.head.weight.dtype:
            raise ValueError("TinyUNet input dtype must match the model's floating-point parameter dtype.")
        first = self.encoder1(inputs)
        second = self.encoder2(self.pool(first))
        features = self.bottleneck(self.pool(second))
        features = F.interpolate(features, size=second.shape[-2:], mode="bilinear", align_corners=False)
        features = self.decoder2(torch.cat((features, second), dim=1))
        features = F.interpolate(features, size=first.shape[-2:], mode="bilinear", align_corners=False)
        features = self.decoder1(torch.cat((features, first), dim=1))
        return self.head(features)


def build_model(name: str = MODEL_NAME) -> nn.Module:
    """Construct only the built-in smoke architecture with ordinary Torch defaults."""
    if not isinstance(name, str) or name != MODEL_NAME:
        raise ValueError(f"Unknown model {name!r}; supported model: {MODEL_NAME}.")
    return TinyUNet()
