# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Bounded training-pipeline checks with local, resumable state dictionaries.

Only selected training pixels are loaded. Probe losses describe those same
training examples; they are not validation metrics or a model-quality result.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import pickle
import platform
import re
import shutil
import tempfile
import time
from typing import Any, Callable

from pvextent.data.pv4ger import _publish, _sha256, _write_json

MAX_CHECKPOINT_BYTES = 8 * 1024 * 1024
MODEL_NAME = "tiny-unet-v1"
CHECKPOINT_SCHEMA = "1.0"


class TrainingSmokeError(ValueError):
    """The smoke run failed without publishing a partial output directory."""


def _device(torch: Any, requested: str) -> str:
    if requested not in ("auto", "cpu", "mps", "cuda"):
        raise TrainingSmokeError("Device must be auto, cpu, mps or cuda.")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise TrainingSmokeError("MPS was requested but is unavailable on this host.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise TrainingSmokeError("CUDA was requested but is unavailable on this host.")
    return requested


def _rng_state(torch: Any, device: str) -> dict:
    state = {"cpu": torch.get_rng_state().clone(), "device": None}
    if device == "mps":
        state["device"] = torch.mps.get_rng_state().clone()
    elif device == "cuda":
        state["device"] = torch.cuda.get_rng_state().clone()
    return state


def _restore_rng(torch: Any, device: str, state: dict) -> None:
    torch.set_rng_state(state["cpu"])
    if device == "mps":
        torch.mps.set_rng_state(state["device"])
    elif device == "cuda":
        torch.cuda.set_rng_state(state["device"])


@contextmanager
def _runtime(torch: Any, device: str, seed: int):
    rng = _rng_state(torch, device)
    threads = torch.get_num_threads()
    default_dtype = torch.get_default_dtype()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)
        torch.random.default_generator.manual_seed(seed)
        if device == "mps":
            torch.mps.manual_seed(seed)
        elif device == "cuda":
            torch.cuda.manual_seed(seed)
        if device == "cpu":
            torch.use_deterministic_algorithms(True)
        yield
    finally:
        _restore_rng(torch, device, rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.set_default_dtype(default_dtype)


def _cpu_tree(value: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu_tree(item, torch) for item in value)
    return value


def _finite_tree(value: Any, torch: Any, depth: int = 0) -> bool:
    if depth > 32:
        return False
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_tree(item, torch, depth + 1) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item, torch, depth + 1) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return value is None or isinstance(value, (int, str, bool))


def _read_checkpoint(path: Path, torch: Any) -> tuple[dict, str]:
    companion = path.with_suffix(path.suffix + ".sha256")
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES
            or companion.is_symlink() or not companion.is_file() or companion.stat().st_size > 128):
        raise TrainingSmokeError("Expected a bounded local checkpoint and its .sha256 companion.")
    expected = companion.read_text(encoding="ascii").strip()
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected:
        raise TrainingSmokeError("Checkpoint SHA-256 mismatch; checkpoint was not loaded.")
    try:
        document = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    except (OSError, ValueError, RuntimeError, pickle.UnpicklingError, EOFError) as error:
        raise TrainingSmokeError("Checkpoint is not a supported weights-only state dictionary.") from error
    keys = {"checkpoint_schema_version", "config", "global_step", "model_state_dict",
            "optimizer_state_dict", "rng_state", "original_initial_loss", "torch_version"}
    if (not isinstance(document, dict) or set(document) != keys
            or document["checkpoint_schema_version"] != CHECKPOINT_SCHEMA
            or type(document["global_step"]) is not int or not 1 <= document["global_step"] <= 100
            or not isinstance(document["config"], dict) or not _finite_tree(document, torch)):
        raise TrainingSmokeError("Unsupported or non-finite smoke checkpoint contents.")
    return document, actual


def _batch(dataset: Any, step: int, batch_size: int, torch: Any, device: str) -> dict:
    rows = [dataset[(step * batch_size + offset) % len(dataset)] for offset in range(batch_size)]
    return {**{key: torch.stack([row[key] for row in rows]).to(device)
               for key in ("image", "target", "valid")}, "ids": [row["id"] for row in rows]}


def _probe(model: Any, batch: dict, loss_function: Any, torch: Any) -> tuple[dict, Any]:
    model.eval()
    with torch.no_grad():
        logits = model(batch["image"])
        losses = loss_function(logits, batch["target"], batch["valid"])
    return {key: float(losses[key].item()) for key in ("loss", "bce", "dice_loss")}, logits.detach().clone()


def run_smoke(
    dataset_dir: Path | str, output_dir: Path | str, *, steps: int = 8,
    max_samples: int = 8, batch_size: int = 2, seed: int = 20261020,
    learning_rate: float = 0.001, device: str = "auto", resume: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run 1–100 total updates, then save/reload a checkpoint and publish atomically.

    ``steps`` is the total target global step, including updates in ``resume``.
    Resume requires the same data selection, optimizer settings, device and Torch
    version. Only an existing locally generated checkpoint is expected.
    """
    for name, value, lower, upper in (("steps", steps, 1, 100), ("max_samples", max_samples, 1, 64),
                                       ("batch_size", batch_size, 1, 8), ("seed", seed, 0, 2 ** 32 - 1)):
        if type(value) is not int or not lower <= value <= upper:
            raise TrainingSmokeError(f"{name} must be an integer between {lower} and {upper}.")
    try:
        valid_lr = (not isinstance(learning_rate, bool) and isinstance(learning_rate, (int, float))
                    and math.isfinite(learning_rate) and 0 < learning_rate <= 1)
    except OverflowError:
        valid_lr = False
    if not valid_lr:
        raise TrainingSmokeError("learning_rate must be finite and greater than 0, at most 1.")
    try:
        import torch
        from pvextent.data.training import TrainingDataError, TrainingDataset
        from pvextent.models.tiny_unet import build_model
        from pvextent.training.losses import masked_bce_dice
    except ImportError as error:
        raise TrainingSmokeError("Training dependencies missing. Run: uv sync --locked --extra train") from error
    source, output = Path(dataset_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise TrainingSmokeError(f"Output already exists: {output}")
    output = output.resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise TrainingSmokeError("Dataset and run output must be separate directories.")
    notify = progress or (lambda message: None)
    staging = None
    began = time.perf_counter()
    try:
        selected_device = _device(torch, device)
        dataset = TrainingDataset(source, max_samples=max_samples, seed=seed)
        if batch_size > len(dataset):
            raise TrainingSmokeError("batch_size exceeds the selected training sample count.")
        config = {
            "model": MODEL_NAME, "weights": "random-initialization", "seed": seed,
            "max_samples": max_samples, "batch_size": batch_size, "learning_rate": float(learning_rate),
            "optimizer": "Adam", "gradient_clip_norm": 1.0, "device": selected_device,
            "loss": "masked-bce-plus-soft-dice-v1", "preprocessing": "uint8-rgb-to-float32-0..1",
            "sampling": "seed-hash-selected-train-ids; global-step-cyclic-batches-v1",
            "dataset_fingerprint": dataset.dataset_fingerprint,
            "selection_fingerprint": dataset.selection_fingerprint,
            "sample_ids": [row["id"] for row in dataset.records],
        }
        resume_document, resume_hash = _read_checkpoint(Path(resume), torch) if resume is not None else (None, None)
        if resume_document is not None:
            if resume_document["config"] != config or resume_document["torch_version"] != str(torch.__version__):
                raise TrainingSmokeError("Resume configuration, data fingerprint/selection, device or PyTorch version differs.")
            if resume_document["global_step"] >= steps:
                raise TrainingSmokeError("--steps must exceed the checkpoint's completed global step.")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-train-", dir=output.parent))
        _write_json(staging / "config.json", {**config, "target_steps": steps, "resume_checkpoint_sha256": resume_hash})
        with _runtime(torch, selected_device, seed):
            with torch.device("cpu"):
                model = build_model().to(device=selected_device, dtype=torch.float32)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            start_step = 0
            if resume_document is not None:
                model.load_state_dict(resume_document["model_state_dict"], strict=True)
                optimizer.load_state_dict(resume_document["optimizer_state_dict"])
                _restore_rng(torch, selected_device, resume_document["rng_state"])
                start_step = resume_document["global_step"]
            probe_batch = _batch(dataset, 0, batch_size, torch, selected_device)
            initial, _ = _probe(model, probe_batch, masked_bce_dice, torch)
            original_initial_loss = initial["loss"] if resume_document is None else resume_document["original_initial_loss"]
            initial_parameters = _cpu_tree(model.state_dict(), torch)
            gradients = []
            notify(f"Training {MODEL_NAME} on {selected_device}: {len(dataset)} samples, steps {start_step + 1}..{steps}")
            trained_at = time.perf_counter()
            with (staging / "metrics.jsonl").open("w", encoding="utf-8") as log:
                for step in range(start_step, steps):
                    batch = _batch(dataset, step, batch_size, torch, selected_device)
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    losses = masked_bce_dice(model(batch["image"]), batch["target"], batch["valid"])
                    losses["loss"].backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                    gradient = float(norm.item())
                    if not math.isfinite(gradient) or gradient <= 0:
                        raise TrainingSmokeError("Expected finite, nonzero training gradients.")
                    optimizer.step()
                    if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                        raise TrainingSmokeError("Optimizer produced non-finite model parameters.")
                    gradients.append(gradient)
                    row = {"step": step + 1, "batch_ids": batch["ids"], "gradient_norm": gradient,
                           "valid_pixels": losses["valid_pixels"],
                           **{key: float(losses[key].item()) for key in ("loss", "bce", "dice_loss")}}
                    log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    notify(f"Step {step + 1}/{steps}: loss={row['loss']:.6f}, gradient_norm={gradient:.6f}")
            if selected_device == "mps":
                torch.mps.synchronize()
            elif selected_device == "cuda":
                torch.cuda.synchronize()
            training_seconds = time.perf_counter() - trained_at
            final, final_logits = _probe(model, probe_batch, masked_bce_dice, torch)
            state = _cpu_tree(model.state_dict(), torch)
            update = math.sqrt(sum(float((value.double() - initial_parameters[name].double()).square().sum())
                                   for name, value in state.items()))
            if not math.isfinite(update) or update <= 0:
                raise TrainingSmokeError("Smoke training did not change model parameters.")
            checkpoint = {
                "checkpoint_schema_version": CHECKPOINT_SCHEMA, "config": config, "global_step": steps,
                "model_state_dict": state, "optimizer_state_dict": _cpu_tree(optimizer.state_dict(), torch),
                "rng_state": _cpu_tree(_rng_state(torch, selected_device), torch),
                "original_initial_loss": original_initial_loss, "torch_version": str(torch.__version__),
            }
            checkpoint_path = staging / "checkpoint.pt"
            torch.save(checkpoint, checkpoint_path)
            checkpoint_hash = _sha256(checkpoint_path)
            checkpoint_path.with_suffix(".pt.sha256").write_text(checkpoint_hash + "\n", encoding="ascii")
            loaded, verified_hash = _read_checkpoint(checkpoint_path, torch)
            with torch.device("cpu"):
                reloaded_model = build_model().to(device=selected_device, dtype=torch.float32)
            reloaded_model.load_state_dict(loaded["model_state_dict"], strict=True)
            _, reloaded_logits = _probe(reloaded_model, probe_batch, masked_bce_dice, torch)
            max_difference = float((final_logits - reloaded_logits).abs().max().item())
            if not torch.allclose(final_logits, reloaded_logits, rtol=1e-5, atol=1e-6):
                raise TrainingSmokeError("Checkpoint reload changed the training-probe prediction.")
            report = {
                "ok": True, "purpose": "training-pipeline-smoke-only", "model": MODEL_NAME,
                "device": selected_device, "samples": len(dataset), "batch_size": batch_size,
                "image_dimensions": [dataset.records[0]["width"], dataset.records[0]["height"]],
                "gsd_m": dataset.records[0]["gsd_m"], "parameter_count": sum(p.numel() for p in model.parameters()),
                "start_step": start_step, "total_steps": steps, "steps_completed": steps - start_step,
                "initial_loss": initial["loss"], "final_loss": final["loss"],
                "original_initial_loss": original_initial_loss,
                "probe": {"initial": initial, "final": final, "split": "train", "sample_ids": probe_batch["ids"]},
                "gradient_norm_min": min(gradients), "gradient_norm_max": max(gradients),
                "parameter_update_l2": update, "checkpoint_file": "checkpoint.pt",
                "checkpoint_sha256": verified_hash, "verified_reload": True,
                "reload_max_abs_logit_difference": max_difference,
                "resume_checkpoint_sha256": resume_hash, "config_sha256": _sha256(staging / "config.json"),
                "dataset_fingerprint": dataset.dataset_fingerprint,
                "selection_fingerprint": dataset.selection_fingerprint,
                "input_validation_scope": "Full metadata/manifest and selected train pixels only; held-out pixels are not opened.",
                "validation_used": False, "test_used": False,
                "training_elapsed_seconds": training_seconds, "wall_time_seconds": time.perf_counter() - began,
                "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                                "platform": platform.system() + " " + platform.machine(),
                                "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")},
                "limitations": ["Randomly initialized tiny U-Net, not the planned pretrained baseline.",
                                "Losses use a fixed training probe; no accuracy or generalisation claim.",
                                "No validation or test metrics; source labels remain upstream/unreviewed as declared.",
                                "CPU continuation is regression-tested exactly; cross-device/version bitwise equality is not promised."],
            }
            _write_json(staging / "smoke-report.json", report)
        _publish(staging, output)
        staging = None
        return report
    except TrainingSmokeError:
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, OverflowError) as error:
        raise TrainingSmokeError(f"Training smoke failed: {error}") from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
