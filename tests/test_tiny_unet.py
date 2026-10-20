# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Shape, logits and optimizer checks for the built-in smoke architecture."""

from __future__ import annotations

import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the train extra for model smoke tests")
class TinyUNetTests(unittest.TestCase):
    def build(self):
        from pvextent.models.tiny_unet import build_model
        return build_model()

    def test_rgb_batches_preserve_even_odd_and_small_spatial_dimensions(self) -> None:
        model = self.build()
        with torch.no_grad():
            for shape in ((2, 3, 16, 16), (1, 3, 13, 17), (1, 3, 4, 4), (1, 3, 4, 7)):
                with self.subTest(shape=shape):
                    output = model(torch.zeros(shape))
                    self.assertEqual(tuple(output.shape), (shape[0], 1, shape[2], shape[3]))
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_logits_are_not_probabilities_or_thresholded_predictions(self) -> None:
        model = self.build()
        with torch.no_grad():
            model.head.weight.zero_()
            for bias in (-2.0, 3.0):
                model.head.bias.fill_(bias)
                logits = model(torch.zeros(1, 3, 4, 5))
                self.assertTrue(torch.equal(logits, torch.full_like(logits, bias)))

    def test_loss_backpropagates_and_optimizer_updates_parameters(self) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            model = self.build()
            image = torch.rand(2, 3, 12, 13)
            target = torch.zeros(2, 1, 12, 13)
            target[:, :, 2:5, 3:6] = 1
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(image), target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss).item())
            gradients = [parameter.grad for parameter in model.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all().item() for gradient in gradients))
            self.assertTrue(any(torch.count_nonzero(gradient).item() for gradient in gradients))
            optimizer.step()
            self.assertTrue(any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters()))

    def test_invalid_rgb_shapes_and_pixel_dtypes_have_clear_errors(self) -> None:
        model = self.build()
        cases = (
            [], torch.zeros(3, 8, 8), torch.zeros(1, 1, 8, 8), torch.zeros(0, 3, 8, 8),
            torch.zeros(1, 3, 3, 8), torch.zeros(1, 3, 8, 3),
            torch.zeros(1, 3, 8, 8, dtype=torch.long),
            torch.zeros(1, 3, 8, 8, dtype=torch.bool),
            torch.zeros(1, 3, 8, 8, dtype=torch.float64),
        )
        for inputs in cases:
            with self.subTest(type=type(inputs), shape=getattr(inputs, "shape", None)):
                with self.assertRaisesRegex(ValueError, "TinyUNet"):
                    model(inputs)

    def test_explicit_float_model_conversion_is_supported(self) -> None:
        model = self.build().double()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, 5, 7, dtype=torch.float64))
        self.assertEqual(output.dtype, torch.float64)
        self.assertEqual(tuple(output.shape), (1, 1, 5, 7))

    def test_factory_rejects_unknown_architectures_without_side_effects(self) -> None:
        from pvextent.models.tiny_unet import build_model
        settings = (torch.get_num_threads(), torch.get_default_dtype(), torch.are_deterministic_algorithms_enabled())
        model = build_model()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 29761)
        self.assertFalse(any(isinstance(module, (torch.nn.BatchNorm2d, torch.nn.Dropout)) for module in model.modules()))
        self.assertEqual(settings, (torch.get_num_threads(), torch.get_default_dtype(), torch.are_deterministic_algorithms_enabled()))
        for name in ("resnet18", "tiny-unet-v2", None):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown model"):
                build_model(name)


if __name__ == "__main__":
    unittest.main()
