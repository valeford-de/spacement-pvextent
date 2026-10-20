# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Pinned encoder caching, preprocessing, and actual U-Net compatibility checks."""

from __future__ import annotations

import hashlib
from io import BytesIO
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pvextent.models import resnet_unet as module

BASELINE_AVAILABLE = all(importlib.util.find_spec(name) is not None for name in (
    "torch", "segmentation_models_pytorch", "safetensors"))


class EncoderCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name) / "encoder"
        self.payloads = {}
        for attribute, payload in (("WEIGHT_SPEC", b"synthetic weight fixture"),
                                   ("MODEL_CARD_SPEC", b"license: apache-2.0\n"),
                                   ("CONFIG_SPEC", b'{"synthetic":true}\n')):
            spec = {**getattr(module, attribute), "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            self.payloads[spec["url"]] = payload
            context = patch.object(module, attribute, spec)
            context.start()
            self.addCleanup(context.stop)

    def test_missing_offline_cache_never_downloads_or_creates_files(self) -> None:
        with patch.object(module, "urlopen") as network:
            with self.assertRaisesRegex(module.EncoderWeightsError, "download=True"):
                module.prepare_encoder_weights(self.cache)
            network.assert_not_called()
        self.assertFalse(self.cache.exists())

    def test_explicit_download_verifies_and_preserves_all_pinned_source_files(self) -> None:
        def response(request, timeout):
            return BytesIO(self.payloads[request.full_url])
        with patch.object(module, "urlopen", side_effect=response) as network:
            first = module.prepare_encoder_weights(self.cache, download=True)
            self.assertEqual(network.call_count, 3)
            times = {path.name: path.stat().st_mtime_ns for path in self.cache.iterdir()}
            second = module.prepare_encoder_weights(self.cache)
            self.assertEqual(network.call_count, 3)
        self.assertEqual(first, second)
        self.assertTrue(first["verified"])
        self.assertEqual(first["license"], "Apache-2.0")
        self.assertEqual(Path(first["path"]).read_bytes(), self.payloads[module.WEIGHT_SPEC["url"]])
        self.assertEqual(times, {path.name: path.stat().st_mtime_ns for path in self.cache.iterdir()})

    def test_existing_corrupt_file_is_rejected_without_replacement(self) -> None:
        self.cache.mkdir()
        file = self.cache / module.WEIGHT_SPEC["filename"]
        corrupted = b"x" * module.WEIGHT_SPEC["size"]
        file.write_bytes(corrupted)
        with patch.object(module, "urlopen") as network:
            with self.assertRaisesRegex(module.EncoderWeightsError, "checksum"):
                module.prepare_encoder_weights(self.cache, download=True)
            network.assert_not_called()
        self.assertEqual(file.read_bytes(), corrupted)

    def test_oversized_or_wrong_hash_downloads_are_not_published(self) -> None:
        expected = self.payloads[module.WEIGHT_SPEC["url"]]
        for payload in (expected + b"extra", b"x" * len(expected)):
            with self.subTest(size=len(payload)):
                with patch.object(module, "urlopen", return_value=BytesIO(payload)):
                    with self.assertRaises(module.EncoderWeightsError):
                        module.prepare_encoder_weights(self.cache, download=True)
                self.assertEqual(list(self.cache.iterdir()), [])


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for U-Net/ResNet18 tests")
class ResNetUnetTests(unittest.TestCase):
    def test_explicit_random_construction_is_offline_and_preserves_native_shapes(self) -> None:
        import torch
        with patch.object(module, "urlopen", side_effect=AssertionError("unexpected network")), \
             patch("torch.hub.load_state_dict_from_url", side_effect=AssertionError("unexpected weights download")), \
             patch("segmentation_models_pytorch.encoders.hf_hub_download", side_effect=AssertionError("unexpected HF download")), \
             patch("segmentation_models_pytorch.encoders.load_url", side_effect=AssertionError("unexpected URL download")):
            model = module.build_resnet_unet(None).eval()
            with torch.no_grad():
                for shape in ((2, 3, 32, 64), (2, 3, 64, 32)):
                    output = model(module.normalize_rgb(torch.zeros(shape)))
                    self.assertEqual(tuple(output.shape), (shape[0], 1, shape[2], shape[3]))
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_normalization_preserves_rgb_order_grid_input_and_gradient(self) -> None:
        import torch
        inputs = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32).reshape(1, 3, 1, 1).expand(2, 3, 3, 5).clone().requires_grad_()
        before = inputs.detach().clone()
        output = module.normalize_rgb(inputs)
        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.equal(inputs.detach(), before))
        self.assertTrue(torch.allclose(output[0, :, 0, 0], torch.tensor([-2.1179039, 0.1964286, 2.64]), atol=1e-6))
        output.sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all().item())
        self.assertTrue((inputs.grad > 0).all().item())

    def test_normalization_rejects_invalid_dtype_shape_range_and_nonfinite_pixels(self) -> None:
        import torch
        for inputs in ([], torch.zeros(3, 4, 4), torch.zeros(1, 1, 4, 4), torch.zeros(0, 3, 4, 4),
                       torch.zeros(1, 3, 4, 4, dtype=torch.uint8), torch.zeros(1, 3, 4, 4, dtype=torch.float64),
                       torch.full((1, 3, 4, 4), -0.1), torch.full((1, 3, 4, 4), 1.1),
                       torch.full((1, 3, 4, 4), float("nan")), torch.full((1, 3, 4, 4), float("inf"))):
            with self.subTest(shape=getattr(inputs, "shape", None)), self.assertRaises(ValueError):
                module.normalize_rgb(inputs)

    def test_unactivated_head_and_backpropagation_reach_encoder_and_decoder(self) -> None:
        import torch
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(51)
            model = module.build_resnet_unet(None)
            image = module.normalize_rgb(torch.rand(2, 3, 32, 32))
            target = torch.zeros(2, 1, 32, 32)
            target[:, :, 8:16, 8:16] = 1
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
            before = model.encoder.conv1.weight.detach().clone()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(image), target)
            loss.backward()
            for parameter in (model.encoder.conv1.weight, next(model.decoder.parameters())):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(torch.count_nonzero(parameter.grad).item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, model.encoder.conv1.weight))
            with torch.no_grad():
                model.segmentation_head[0].weight.zero_()
                model.segmentation_head[0].bias.fill_(-3)
                logits = model(image)
                self.assertTrue(torch.equal(logits, torch.full_like(logits, -3)))

    def test_verified_safetensors_load_every_encoder_key_and_discard_only_classifier(self) -> None:
        import torch
        from safetensors.torch import save
        model = module.build_resnet_unet(None)
        state = {key: value.detach().clone() for key, value in model.encoder.state_dict().items()}
        state["conv1.weight"].fill_(0.125)
        state.update({"fc.weight": torch.zeros(1000, 512), "fc.bias": torch.zeros(1000)})
        payload = save(state)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.safetensors"
            path.write_bytes(payload)
            spec = {**module.WEIGHT_SPEC, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            with patch.object(module, "WEIGHT_SPEC", spec):
                loaded = module.build_resnet_unet(path)
        self.assertFalse(hasattr(loaded.encoder, "fc"))
        self.assertEqual(set(loaded.encoder.state_dict()), set(state) - {"fc.weight", "fc.bias"})
        for key, value in loaded.encoder.state_dict().items():
            self.assertTrue(torch.equal(value, state[key]), key)

    def test_unpinned_or_symlinked_weights_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.safetensors"
            path.write_bytes(b"wrong bytes")
            with self.assertRaises(module.EncoderWeightsError):
                module.build_resnet_unet(path)
            link = Path(folder) / "linked.safetensors"
            link.symlink_to(path)
            with self.assertRaises(module.EncoderWeightsError):
                module.build_resnet_unet(link)


if __name__ == "__main__":
    unittest.main()
