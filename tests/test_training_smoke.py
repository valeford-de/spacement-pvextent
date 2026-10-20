# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Real CPU optimization and checkpoint-resume checks on synthetic train pixels."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Install the train extra for optimization smoke tests")
class TrainingSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.training import smoke

        self.module = smoke
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.dataset = self.root / "dataset"
        self.output = self.root / "run"
        (self.dataset / "images").mkdir(parents=True)
        (self.dataset / "masks").mkdir()
        (self.dataset / "dataset.toml").write_text(
            'schema_version = "1.0"\ndataset_id = "synthetic-training-fixture"\n'
            'dataset_version = "1"\n\n[sources.demo]\n'
            'name = "Synthetic training fixture"\nurl = "https://example.invalid/fixture"\n'
            'version = "1"\nimage_license = "CC0-1.0"\n'
            'annotation_license = "CC0-1.0"\nattribution = "Synthetic fixture"\n',
            encoding="utf-8",
        )
        self.records = []
        for index, split in enumerate(("train", "train", "train", "train", "val", "test")):
            identifier = f"synthetic-{index}"
            image_path = self.dataset / "images" / f"{identifier}.png"
            mask_path = self.dataset / "masks" / f"{identifier}.png"
            if split == "train":
                image = Image.new("RGB", (32, 32), (20 + index, 60, 140))
                label = Image.new("L", (32, 32), 0)
                for y in range(10, 22):
                    for x in range(8 + index, 20 + index):
                        image.putpixel((x, y), (5, 25, 80 + index))
                        label.putpixel((x, y), 1)
                label.putpixel((0, 0), 255)
                image.save(image_path)
                label.save(mask_path)
            else:
                # A trainer must not decode validation/test imagery at all.
                image_path.write_bytes(f"{split} image must never be opened".encode())
                mask_path.write_bytes(f"{split} mask must never be opened".encode())
            self.records.append({
                "id": identifier,
                "image": image_path.relative_to(self.dataset).as_posix(),
                "mask": mask_path.relative_to(self.dataset).as_posix(),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                "width": 32, "height": 32, "gsd_m": 0.1, "source_id": "demo",
                "capture_date": "2025-06-20", "georeference": None,
                "spatial_group": f"synthetic-area-{index}", "building_ids": [],
                "split": split, "annotation_status": "upstream",
            })
        self.write_manifest()

    def write_manifest(self) -> None:
        (self.dataset / "manifest.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in self.records),
            encoding="utf-8",
        )

    @staticmethod
    def snapshot(directory: Path) -> dict[str, bytes]:
        return {
            path.relative_to(directory).as_posix(): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()
        }

    def run_smoke(self, **changes: object) -> dict[str, object]:
        parameters = {"steps": 2, "max_samples": 4, "batch_size": 2, "device": "cpu"}
        parameters.update(changes)
        return self.module.run_smoke(self.dataset, self.output, **parameters)

    def checkpoint(self, directory: Path | None = None) -> dict[str, object]:
        return torch.load((directory or self.output) / "checkpoint.pt", map_location="cpu", weights_only=True)

    def assert_failure(self, **changes: object) -> None:
        before = self.snapshot(self.dataset)
        entries = {path.name for path in self.root.iterdir()}
        with self.assertRaises(self.module.TrainingSmokeError):
            self.run_smoke(**changes)
        self.assertFalse(self.output.exists())
        self.assertEqual(self.snapshot(self.dataset), before)
        self.assertEqual({path.name for path in self.root.iterdir()}, entries)

    def assert_nested_equal(self, left: object, right: object) -> None:
        if isinstance(left, torch.Tensor):
            self.assertIsInstance(right, torch.Tensor)
            self.assertTrue(torch.equal(left, right))
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(type(left), type(right))
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_nested_equal(first, second)
        else:
            self.assertEqual(left, right)

    def test_real_cpu_updates_are_finite_checkpointed_and_do_not_touch_held_out_pixels(self) -> None:
        before = self.snapshot(self.dataset)
        from pvextent.models.tiny_unet import build_model

        threads = torch.get_num_threads()
        rng = torch.get_rng_state().clone()
        deterministic = torch.are_deterministic_algorithms_enabled()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261020)
            initial_parameters = build_model().state_dict()
        with patch.object(torch, "load", wraps=torch.load) as loads:
            report = self.run_smoke()
        self.assertTrue(loads.call_args_list)
        for call in loads.call_args_list:
            self.assertIs(call.kwargs["weights_only"], True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["device"], "cpu")
        self.assertTrue(math.isfinite(report["initial_loss"]))
        self.assertTrue(math.isfinite(report["final_loss"]))
        self.assertGreater(report["parameter_update_l2"], 0)
        self.assertTrue(report["verified_reload"])
        self.assertEqual(report["steps_completed"], 2)
        self.assertEqual(report["samples"], 4)
        self.assertGreater(report["gradient_norm_min"], 0)
        self.assertTrue(math.isfinite(report["gradient_norm_max"]))
        self.assertFalse(report["validation_used"])
        self.assertFalse(report["test_used"])
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)
        checkpoint = self.checkpoint()
        self.assertTrue(any(
            not torch.equal(value, initial_parameters[name])
            for name, value in checkpoint["model_state_dict"].items()
        ))
        self.assertTrue(checkpoint["optimizer_state_dict"]["state"])
        self.assertTrue(all(
            int(state["step"].item()) == 2
            for state in checkpoint["optimizer_state_dict"]["state"].values()
        ))
        metrics = [json.loads(line) for line in (self.output / "metrics.jsonl").read_text().splitlines()]
        self.assertEqual([row["step"] for row in metrics], [1, 2])
        train_ids = {record["id"] for record in self.records if record["split"] == "train"}
        for row in metrics:
            self.assertTrue(set(row["batch_ids"]).issubset(train_ids))
            self.assertTrue(math.isfinite(row["loss"]))
            self.assertTrue(math.isfinite(row["gradient_norm"]))
            self.assertGreater(row["gradient_norm"], 0)
        checkpoint_path = self.output / "checkpoint.pt"
        self.assertEqual(report["checkpoint_sha256"], hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
        self.assertEqual(self.snapshot(self.dataset), before)
        json.dumps(report, allow_nan=False)

    def test_float64_caller_uses_float32_weights_and_keeps_its_default_on_success_and_failure(self) -> None:
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            report = self.run_smoke()
            self.assertTrue(report["ok"])
            self.assertEqual(torch.get_default_dtype(), torch.float64)
            self.assertTrue(all(
                tensor.dtype == torch.float32 and tensor.device.type == "cpu"
                for tensor in self.checkpoint()["model_state_dict"].values()
            ))
            self.output = self.root / "failed-run"
            with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("synthetic optimizer failure")):
                self.assert_failure()
            self.assertEqual(torch.get_default_dtype(), torch.float64)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_continuous_training_and_resume_have_identical_final_states(self) -> None:
        continuous = self.root / "continuous"
        prefix = self.root / "prefix"
        resumed = self.root / "resumed"
        parameters = {"max_samples": 4, "batch_size": 2, "device": "cpu"}
        self.module.run_smoke(self.dataset, continuous, steps=4, **parameters)
        self.module.run_smoke(self.dataset, prefix, steps=2, **parameters)
        prefix_checkpoint = prefix / "checkpoint.pt"
        before = prefix_checkpoint.read_bytes()
        report = self.module.run_smoke(
            self.dataset, resumed, steps=4, resume=prefix_checkpoint, **parameters
        )
        self.assertEqual(prefix_checkpoint.read_bytes(), before)
        self.assertEqual(report["start_step"], 2)
        self.assertEqual(report["total_steps"], 4)
        self.assert_nested_equal(self.checkpoint(continuous), self.checkpoint(resumed))

    def test_resume_rejects_changed_training_configuration(self) -> None:
        prefix = self.root / "prefix"
        self.module.run_smoke(self.dataset, prefix, steps=2, max_samples=4, batch_size=2, device="cpu")
        checkpoint = prefix / "checkpoint.pt"
        for changes in (
            {"seed": 7}, {"learning_rate": 0.002}, {"batch_size": 1}, {"max_samples": 3},
        ):
            with self.subTest(changes=changes):
                self.assert_failure(steps=4, resume=checkpoint, **changes)

    def test_resume_rejects_incompatible_checkpoint_metadata_and_completed_steps(self) -> None:
        prefix = self.root / "prefix"
        self.module.run_smoke(self.dataset, prefix, steps=2, max_samples=4, batch_size=2, device="cpu")
        path = prefix / "checkpoint.pt"
        original = self.checkpoint(prefix)
        for field, value in (
            ("device", "cuda"), ("torch_version", "0.0.incompatible"),
            ("checkpoint_schema_version", "unsupported"),
        ):
            with self.subTest(field=field):
                document = copy.deepcopy(original)
                if field == "device":
                    document["config"]["device"] = value
                else:
                    document[field] = value
                torch.save(document, path)
                path.with_suffix(".pt.sha256").write_text(
                    hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
                )
                self.assert_failure(steps=4, resume=path)
        torch.save(original, path)
        path.with_suffix(".pt.sha256").write_text(
            hashlib.sha256(path.read_bytes()).hexdigest() + "\n", encoding="ascii"
        )
        self.assert_failure(steps=2, resume=path)

    def test_optimizer_failure_removes_staging_and_restores_runtime_state(self) -> None:
        rng = torch.get_rng_state().clone()
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        with patch.object(torch.optim.Adam, "step", side_effect=RuntimeError("synthetic optimizer failure")):
            self.assert_failure()
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertEqual(torch.get_num_threads(), threads)
        self.assertEqual(torch.are_deterministic_algorithms_enabled(), deterministic)

    def test_resume_rejects_changed_dataset_fingerprint(self) -> None:
        prefix = self.root / "prefix"
        self.module.run_smoke(self.dataset, prefix, steps=2, max_samples=4, batch_size=2, device="cpu")
        self.records[0]["capture_date"] = "2025-06-21"
        self.write_manifest()
        self.assert_failure(steps=4, resume=prefix / "checkpoint.pt")

    def test_corrupt_resume_checksum_fails_before_any_torch_load(self) -> None:
        prefix = self.root / "prefix"
        self.module.run_smoke(self.dataset, prefix, steps=2, max_samples=4, batch_size=2, device="cpu")
        checkpoint = prefix / "checkpoint.pt"
        payload = bytearray(checkpoint.read_bytes())
        payload[-1] ^= 1
        checkpoint.write_bytes(payload)
        with patch.object(torch, "load", side_effect=AssertionError("Corrupt bytes must never reach torch.load")) as load:
            self.assert_failure(steps=4, resume=checkpoint)
        load.assert_not_called()

    def test_invalid_checkpoint_with_matching_hash_is_a_clean_weights_only_error(self) -> None:
        checkpoint = self.root / "invalid.pt"
        for payload in (b"", b"not a checkpoint"):
            with self.subTest(payload=payload):
                checkpoint.write_bytes(payload)
                checkpoint.with_suffix(".pt.sha256").write_text(
                    hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
                )
                with patch.object(torch, "load", wraps=torch.load) as loads:
                    self.assert_failure(steps=4, resume=checkpoint)
                loads.assert_called_once()
                self.assertIs(loads.call_args.kwargs["weights_only"], True)

    def test_existing_output_is_never_replaced(self) -> None:
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("existing work", encoding="utf-8")
        with self.assertRaises(self.module.TrainingSmokeError):
            self.run_smoke()
        self.assertEqual(marker.read_text(), "existing work")
        self.assertEqual({path.name for path in self.output.iterdir()}, {"keep.txt"})

    def test_invalid_training_arguments_fail_without_partial_output(self) -> None:
        for changes in (
            {"steps": 0}, {"steps": -1}, {"steps": True},
            {"max_samples": 0}, {"batch_size": 0}, {"batch_size": True},
            {"seed": -1}, {"seed": True},
            {"learning_rate": 0}, {"learning_rate": float("nan")},
            {"learning_rate": float("inf")}, {"device": "invalid"},
        ):
            with self.subTest(changes=changes):
                self.assert_failure(**changes)


if __name__ == "__main__":
    unittest.main()
