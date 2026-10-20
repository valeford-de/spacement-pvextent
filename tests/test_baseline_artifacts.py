# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint generation rollback and nondestructive failure recovery."""

from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    torch = None

BASELINE_AVAILABLE = torch is not None and all(importlib.util.find_spec(name) is not None for name in (
    "segmentation_models_pytorch", "torchvision", "timm", "safetensors"))


@unittest.skipIf(torch is None, "Install the train extra for checkpoint artifact tests")
class CheckpointPairTests(unittest.TestCase):
    def setUp(self) -> None:
        from pvextent.training import baseline
        self.module = baseline
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "last.pt"
        self.companion = self.path.with_suffix(".pt.sha256")
        self.module._write_checkpoint(self.path, self.document(1), torch)
        self.previous = self.path.read_bytes(), self.companion.read_bytes()

    @staticmethod
    def document(epoch: int) -> dict:
        return {"baseline_checkpoint_version": "1.0", "epoch": epoch, "tensor": torch.tensor([float(epoch)])}

    def assert_previous_pair(self) -> None:
        self.assertEqual(self.path.read_bytes(), self.previous[0])
        self.assertEqual(self.companion.read_bytes(), self.previous[1])
        saved, _ = self.module._read_checkpoint(self.path, torch)
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual({path.name for path in self.directory.iterdir()}, {"last.pt", "last.pt.sha256"})

    def test_hash_write_failure_leaves_previous_pair_untouched(self) -> None:
        original = Path.write_text

        def write(path, *args, **kwargs):
            if path.name == "last.pt.sha256.tmp":
                raise OSError("synthetic sidecar disk full")
            return original(path, *args, **kwargs)

        with patch.object(Path, "write_text", new=write):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.module._write_checkpoint(self.path, self.document(2), torch)
        self.assert_previous_pair()

    def test_sidecar_install_failure_or_interrupt_rolls_back_both_files(self) -> None:
        original = os.replace
        for error_type in (OSError, KeyboardInterrupt):
            with self.subTest(error=error_type.__name__):
                def replace(source, destination):
                    if Path(source).name == "last.pt.sha256.tmp":
                        raise error_type("synthetic sidecar install failure")
                    return original(source, destination)

                with patch.object(os, "replace", new=replace):
                    with self.assertRaises(error_type):
                        self.module._write_checkpoint(self.path, self.document(2), torch)
                self.assert_previous_pair()

    def test_failed_rollback_keeps_complete_previous_generation_for_recovery(self) -> None:
        original = os.replace

        def replace(source, destination):
            if Path(source).name in ("last.pt.sha256.tmp", ".restore-last.pt"):
                raise OSError("synthetic install or rollback failure")
            return original(source, destination)

        with patch.object(os, "replace", new=replace):
            with self.assertRaises(OSError):
                self.module._write_checkpoint(self.path, self.document(2), torch)
        backups = list(self.directory.glob(".last.pt-backup-*/last.pt"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), self.previous[0])
        self.assertEqual(backups[0].with_suffix(".pt.sha256").read_bytes(), self.previous[1])
        saved, _ = self.module._read_checkpoint(backups[0], torch)
        self.assertEqual(saved["epoch"], 1)


@unittest.skipUnless(BASELINE_AVAILABLE, "Install the baseline extra for failure recovery tests")
class BaselineRecoveryTests(unittest.TestCase):
    def fixture(self):
        # Reuse generated image/config fixtures without duplicating their test
        # class in this module's discovery or touching real datasets/runs.
        import test_baseline_training as fixtures
        fixture = fixtures.BaselineTrainingTests(methodName="test_failure_after_a_completed_epoch_retains_a_resumable_checkpoint")
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        return fixture

    def test_failure_report_write_failure_still_preserves_last_verified_checkpoint(self) -> None:
        fixture = self.fixture()
        original_step = torch.optim.AdamW.step
        original_write = fixture.module._write_json
        steps = 0

        def fail_later(optimizer, *args, **kwargs):
            nonlocal steps
            steps += 1
            if steps == 3:
                raise RuntimeError("synthetic epoch two failure")
            return original_step(optimizer, *args, **kwargs)

        def write(path, document):
            if path.name == "failure-report.json":
                raise OSError("synthetic recovery-report disk full")
            return original_write(path, document)

        with patch.object(torch.optim.AdamW, "step", new=fail_later), patch.object(fixture.module, "_write_json", new=write):
            with self.assertRaisesRegex(fixture.module.BaselineTrainingError, "retained"):
                fixture.train()
        self.assertFalse(fixture.output.exists())
        recovered = list(fixture.root.glob("run.incomplete-*"))
        self.assertEqual(len(recovered), 1)
        checkpoint, _ = fixture.module._read_checkpoint(recovered[0] / "last.pt", torch)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertEqual(checkpoint["global_step"], 2)

    def test_failed_incomplete_publication_retains_working_directory(self) -> None:
        fixture = self.fixture()
        with patch.object(fixture.module, "_publish", side_effect=OSError("synthetic publication failure")):
            with self.assertRaisesRegex(fixture.module.BaselineTrainingError, "retained"):
                fixture.train()
        self.assertFalse(fixture.output.exists())
        working = list(fixture.root.glob(".run-baseline-*"))
        self.assertEqual(len(working), 1)
        checkpoint, digest = fixture.module._read_checkpoint(working[0] / "last.pt", torch)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertEqual(digest, hashlib.sha256((working[0] / "last.pt").read_bytes()).hexdigest())

    def test_unverified_checkpoint_bytes_are_retained_without_resume_claim(self) -> None:
        import json
        fixture = self.fixture()
        original_step = torch.optim.AdamW.step
        steps = 0

        def corrupt_then_fail(optimizer, *args, **kwargs):
            nonlocal steps
            steps += 1
            if steps == 3:
                path = next(fixture.root.glob(".run-baseline-*/last.pt"))
                payload = bytearray(path.read_bytes())
                payload[-1] ^= 1
                path.write_bytes(payload)
                raise RuntimeError("synthetic checkpoint corruption before failure")
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, "step", new=corrupt_then_fail):
            with self.assertRaisesRegex(fixture.module.BaselineTrainingError, "no verified resumable pair"):
                fixture.train()
        recovered = list(fixture.root.glob("run.incomplete-*"))
        self.assertEqual(len(recovered), 1)
        self.assertTrue((recovered[0] / "last.pt").is_file())
        failure = json.loads((recovered[0] / "failure-report.json").read_text())
        self.assertIsNone(failure["resume_from"])
        with self.assertRaisesRegex(fixture.module.BaselineTrainingError, "SHA-256 mismatch"):
            fixture.module._read_checkpoint(recovered[0] / "last.pt", torch)


if __name__ == "__main__":
    unittest.main()
