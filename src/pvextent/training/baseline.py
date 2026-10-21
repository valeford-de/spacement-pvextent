# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""A configured pretrained segmentation baseline using train and validation only."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
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
import tomllib
from typing import Any, Callable

from pvextent.data.pv4ger import _publish, _sha256, _write_json
from .smoke import _cpu_tree, _device, _finite_tree, _restore_rng, _rng_state, _runtime

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024
REQUIRED_CONFIG_FIELDS = frozenset(("schema_version", "model", "epochs", "batch_size", "learning_rate",
                                    "weight_decay", "seed", "threshold", "horizontal_flip", "vertical_flip",
                                    "max_train_samples", "max_val_samples"))
OPTIONAL_CONFIG_FIELDS = frozenset(("brightness_jitter", "contrast_jitter", "domain_balance", "loss_type"))
ALL_CONFIG_FIELDS = REQUIRED_CONFIG_FIELDS | OPTIONAL_CONFIG_FIELDS
CONFIG_FIELDS = REQUIRED_CONFIG_FIELDS
SUPPORTED_MODELS = frozenset(("unet-resnet18-v1", "segformer-mit-b2-v1", "deeplabv3plus-resnet50-v1"))
SUPPORTED_LOSS_TYPES = frozenset(("bce_dice", "focal_lovasz"))


class BaselineTrainingError(ValueError):
    """A baseline run failed; completed epoch checkpoints may be recoverable."""


def _telemetry(directory: Path, event: str, **values) -> None:
    """Append flushed scalar events for a read-only live monitor."""
    row = {"event": event, "pid": os.getpid(), "unix_time": time.time(), **values}
    with (directory / "telemetry.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()


def _number(value: Any, low: float, high: float, *, inclusive_low: bool = True) -> bool:
    try:
        return (not isinstance(value, bool) and isinstance(value, (float, int)) and math.isfinite(value)
                and (value >= low if inclusive_low else value > low) and value <= high)
    except OverflowError:
        return False


def load_config(path: Path | str) -> dict:
    path = Path(path)
    try:
        if path.stat().st_size > 64 * 1024:
            raise BaselineTrainingError("Baseline configuration exceeds 64 KiB.")
        config = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError) as error:
        raise BaselineTrainingError(f"Cannot read baseline TOML configuration: {error}") from error
    schema = config.get("schema_version")
    if schema == "1.0":
        if set(config) != REQUIRED_CONFIG_FIELDS or config["model"] != "unet-resnet18-v1":
            raise BaselineTrainingError("Expected the complete baseline schema 1.0 configuration and unet-resnet18-v1 model.")
        config["brightness_jitter"] = 0.0
        config["contrast_jitter"] = 0.0
        config["domain_balance"] = 0.0
        config["loss_type"] = "bce_dice"
    elif schema == "1.1":
        if not REQUIRED_CONFIG_FIELDS.issubset(set(config)) or not set(config).issubset(ALL_CONFIG_FIELDS) or config["model"] not in SUPPORTED_MODELS:
            raise BaselineTrainingError(f"Expected baseline schema 1.1 with valid fields and model in {sorted(SUPPORTED_MODELS)}.")
        for opt in ("brightness_jitter", "contrast_jitter", "domain_balance"):
            config.setdefault(opt, 0.0)
        config.setdefault("loss_type", "bce_dice")
        if config["loss_type"] not in SUPPORTED_LOSS_TYPES:
            raise BaselineTrainingError(f"loss_type must be one of {sorted(SUPPORTED_LOSS_TYPES)}.")
    else:
        raise BaselineTrainingError("Expected baseline schema_version 1.0 or 1.1.")
    for key, lower, upper in (("epochs", 1, 100), ("batch_size", 1, 16), ("seed", 0, 2 ** 32 - 1),
                              ("max_train_samples", 0, 100000), ("max_val_samples", 0, 100000)):
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise BaselineTrainingError(f"{key} must be an integer between {lower} and {upper}.")
    for key in ("learning_rate", "weight_decay", "threshold", "horizontal_flip", "vertical_flip",
                "brightness_jitter", "contrast_jitter", "domain_balance"):
        if not _number(config[key], 0, 1, inclusive_low=key not in ("learning_rate", "threshold")):
            raise BaselineTrainingError(f"Invalid finite {key} configuration in the range 0..1.")
        config[key] = float(config[key])
    if config["threshold"] >= 1:
        raise BaselineTrainingError("threshold must lie strictly between 0 and 1.")
    return config


def _write_checkpoint(path: Path, document: dict, torch: Any) -> str:
    """Install a checkpoint/hash pair with rollback on handled failures.

    Both new files are complete before installation. A hard-linked previous
    generation survives until both replacements succeed. This is not a claim
    of atomic two-file publication under SIGKILL or power loss.
    """
    temporary = path.with_suffix(path.suffix + ".tmp")
    companion = path.with_suffix(path.suffix + ".sha256")
    companion_tmp = companion.with_suffix(companion.suffix + ".tmp")
    backup = None
    keep_backup = False
    try:
        torch.save(document, temporary)
        if temporary.stat().st_size > MAX_CHECKPOINT_BYTES:
            raise BaselineTrainingError(f"Baseline checkpoint exceeds the {MAX_CHECKPOINT_BYTES // (1024 * 1024)} MiB bound.")
        digest = _sha256(temporary)
        companion_tmp.write_text(digest + "\n", encoding="ascii")
        old_payload, old_companion = os.path.lexists(path), os.path.lexists(companion)
        if old_payload or old_companion:
            if (not old_payload or not old_companion or path.is_symlink() or companion.is_symlink()
                    or not path.is_file() or not companion.is_file()
                    or path.stat().st_size > MAX_CHECKPOINT_BYTES or companion.stat().st_size > 128):
                raise BaselineTrainingError("Refusing to replace an incomplete checkpoint/hash pair.")
            previous_hash = companion.read_text(encoding="ascii").strip()
            if not re.fullmatch(r"[a-f0-9]{64}", previous_hash) or _sha256(path) != previous_hash:
                raise BaselineTrainingError("Refusing to replace an invalid previous checkpoint/hash pair.")
            backup = Path(tempfile.mkdtemp(prefix=f".{path.name}-backup-", dir=path.parent))
            os.link(path, backup / path.name)
            os.link(companion, backup / companion.name)
        try:
            os.replace(temporary, path)
            os.replace(companion_tmp, companion)
        except BaseException as error:
            rollback_errors = []
            for target, existed in ((path, old_payload), (companion, old_companion)):
                try:
                    if existed:
                        restore = backup / f".restore-{target.name}"
                        os.link(backup / target.name, restore)
                        os.replace(restore, target)
                    else:
                        target.unlink(missing_ok=True)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                keep_backup = True
                error.add_note(f"Checkpoint rollback failed; previous generation retained in {backup}.")
            raise
        return digest
    finally:
        for unfinished in (temporary, companion_tmp):
            try:
                unfinished.unlink(missing_ok=True)
            except OSError:
                pass
        if backup is not None and not keep_backup:
            try:
                shutil.rmtree(backup)
            except OSError:
                pass


def _read_checkpoint(path: Path, torch: Any, *, resume: bool = True) -> tuple[dict, str]:
    companion = path.with_suffix(path.suffix + ".sha256")
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CHECKPOINT_BYTES
            or companion.is_symlink() or not companion.is_file() or companion.stat().st_size > 128):
        raise BaselineTrainingError("Expected a bounded local checkpoint and .sha256 companion.")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    expected = companion.read_text(encoding="ascii").strip()
    if not re.fullmatch(r"[a-f0-9]{64}", expected) or digest != expected:
        raise BaselineTrainingError("Checkpoint SHA-256 mismatch; checkpoint was not loaded.")
    try:
        result = torch.load(BytesIO(payload), map_location="cpu", weights_only=True)
    except (OSError, ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        raise BaselineTrainingError("Invalid weights-only baseline checkpoint.") from error
    field = "baseline_checkpoint_version" if resume else "baseline_weights_version"
    if not isinstance(result, dict) or result.get(field) != "1.0" or not _finite_tree(result, torch):
        raise BaselineTrainingError("Unsupported baseline checkpoint; resume requires last.pt with optimizer/RNG state.")
    return result, digest


def _batch(dataset: Any, indices: list[int], torch: Any, device: str) -> dict:
    rows = [dataset[index] for index in indices]
    return {**{key: torch.stack([row[key] for row in rows]).to(device)
               for key in ("image", "target", "valid")}, "ids": [row["id"] for row in rows]}


def _augment(batch: dict, generator: Any, config: dict, torch: Any) -> None:
    decisions = torch.rand((len(batch["ids"]), 2), generator=generator)
    for index, (horizontal, vertical) in enumerate(decisions.tolist()):
        axes = ([2] if horizontal < config["horizontal_flip"] else []) + ([1] if vertical < config["vertical_flip"] else [])
        if axes:
            for key in ("image", "target", "valid"):
                batch[key][index] = batch[key][index].flip(axes)
    b_jitter = config.get("brightness_jitter", 0.0)
    c_jitter = config.get("contrast_jitter", 0.0)
    if b_jitter > 0.0 or c_jitter > 0.0:
        b_size = len(batch["ids"])
        images = batch["image"]
        if b_jitter > 0.0:
            b_factor = 1.0 + (torch.rand((b_size, 1, 1, 1), generator=generator, device="cpu").to(images.device) * 2 - 1) * b_jitter
            images = (images * b_factor).clamp(0.0, 1.0)
        if c_jitter > 0.0:
            c_factor = 1.0 + (torch.rand((b_size, 1, 1, 1), generator=generator, device="cpu").to(images.device) * 2 - 1) * c_jitter
            images = ((images - 0.5) * c_factor + 0.5).clamp(0.0, 1.0)
        batch["image"] = images


def _evaluate(model: Any, dataset: Any, config: dict, torch: Any, device: str, normalize: Any, loss: Any, metric_type: Any) -> dict:
    model.eval()
    metric = metric_type()
    weighted_loss, pixels = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(dataset), config["batch_size"]):
            batch = _batch(dataset, list(range(start, min(start + config["batch_size"], len(dataset)))), torch, device)
            logits = model(normalize(batch["image"]))
            losses = loss(logits, batch["target"], batch["valid"])
            metric.update(logits, batch["target"], batch["valid"], threshold=config["threshold"])
            weighted_loss += float(losses["loss"].item()) * losses["valid_pixels"]
            pixels += losses["valid_pixels"]
    result = {**metric.to_dict(), "loss": weighted_loss / pixels, "samples": len(dataset),
              "threshold": config["threshold"], "split": "val"}
    if result["tp"] + result["fn"] == 0:
        raise BaselineTrainingError("Validation requires supervised PV-positive pixels for baseline IoU selection.")
    return result


def run_baseline(
    dataset_dir: Path | str, output_dir: Path | str, *, config_path: Path | str,
    weights_dir: Path | str, device: str = "auto", download_weights: bool = False,
    resume: Path | str | None = None, progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fine-tune all model parameters; select only by validation pixel IoU.

    Epoch count may be extended on resume; every other config/data/weight/library
    contract is fixed. Unsuccessful runs retain completed epoch checkpoints under
    a distinctly named incomplete directory, never the requested success path.
    """
    config = load_config(config_path)
    try:
        import torch
        from pvextent.data.segmentation import SegmentationDataset
        from pvextent.evaluation.pixels import PixelConfusion
        from pvextent.models.resnet_unet import normalize_rgb
        if config["model"] == "unet-resnet18-v1":
            from pvextent.models.resnet_unet import (
                WEIGHT_SPEC,
                build_resnet_unet as build_model,
                prepare_encoder_weights as prepare_weights,
            )
        elif config["model"] == "segformer-mit-b2-v1":
            from pvextent.models.segformer import (
                WEIGHT_SPEC,
                build_segformer_mit_b2 as build_model,
                prepare_segformer_encoder_weights as prepare_weights,
            )
        elif config["model"] == "deeplabv3plus-resnet50-v1":
            from pvextent.models.deeplabv3plus import (
                WEIGHT_SPEC,
                build_deeplabv3plus_resnet50 as build_model,
                prepare_deeplabv3plus_encoder_weights as prepare_weights,
            )
        else:
            raise BaselineTrainingError(f"Unsupported model: {config['model']}")
        if config.get("loss_type") == "focal_lovasz":
            from pvextent.training.losses import masked_focal_lovasz as loss_fn
        else:
            from pvextent.training.losses import masked_bce_dice as loss_fn
    except ImportError as error:
        raise BaselineTrainingError("Baseline dependencies missing. Run: uv sync --locked --extra baseline") from error
    source, output = Path(dataset_dir).resolve(), Path(output_dir).absolute()
    if os.path.lexists(output):
        raise BaselineTrainingError(f"Output already exists: {output}")
    output = output.resolve()
    if source.is_relative_to(output) or output.is_relative_to(source):
        raise BaselineTrainingError("Dataset and run output must be separate directories.")
    encoder_cache = Path(weights_dir).resolve()
    if encoder_cache.is_relative_to(output) or output.is_relative_to(encoder_cache):
        raise BaselineTrainingError("Encoder cache and run output must be separate directories without overlap.")
    notify = progress or (lambda message: None)
    staging = None
    started = time.perf_counter()
    try:
        selected_device = _device(torch, device)
        train = SegmentationDataset(source, split="train", max_samples=config["max_train_samples"] or None, seed=config["seed"])
        val = SegmentationDataset(source, split="val", max_samples=config["max_val_samples"] or None, seed=config["seed"])
        if train.dataset_fingerprint != val.dataset_fingerprint:
            raise BaselineTrainingError("Dataset declarations changed while constructing split views.")
        if (train.dimensions != val.dimensions or train.gsd_m != val.gsd_m
                or any(side % 32 for side in train.dimensions)):
            raise BaselineTrainingError("Baseline needs matching native train/val dimensions divisible by 32 and one GSD.")
        if train.dimensions == (32, 32) and (
                config["batch_size"] == 1 or len(train) % config["batch_size"] == 1):
            raise BaselineTrainingError(
                "32x32 crops produce 1x1 ResNet18 features; BatchNorm cannot train a singleton batch. "
                "Use batches without a singleton or larger native crops."
            )
        libraries = {key: version(key) for key in ("torch", "torchvision", "segmentation-models-pytorch", "timm", "safetensors")}
        contract = {**{key: value for key, value in config.items() if key != "epochs"},
                    "device": selected_device, "dataset_fingerprint": train.dataset_fingerprint,
                    "train_selection_fingerprint": train.selection_fingerprint,
                    "val_selection_fingerprint": val.selection_fingerprint,
                    "encoder": WEIGHT_SPEC, "optimizer": "AdamW", "gradient_clip_norm": 1.0,
                    "preprocessing": "native-float32-rgb-imagenet-mean-std-v1", "package_versions": libraries}
        document, parent_hash = _read_checkpoint(Path(resume), torch) if resume is not None else (None, None)
        if document is not None:
            if document.get("config") != contract:
                raise BaselineTrainingError("Resume configuration, split/encoder fingerprint, device or library versions differ.")
            if type(document.get("epoch")) is not int or not 1 <= document["epoch"] < config["epochs"]:
                raise BaselineTrainingError("Target epochs must exceed the checkpoint's completed epoch.")
            weights = None  # The full checkpoint contains the fine-tuned encoder.
        else:
            weights = prepare_weights(weights_dir, download=download_weights)
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-baseline-", dir=output.parent))
        _write_json(staging / "config.json", {**contract, "epochs": config["epochs"], "resume_checkpoint_sha256": parent_hash})
        with _runtime(torch, selected_device, config["seed"]):
            with torch.device("cpu"):
                model = build_model(None if weights is None else weights["path"]).to(device=selected_device, dtype=torch.float32)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
            start_epoch, global_step = 0, 0
            best_iou, best_epoch, best_val, best_state = -1.0, 0, None, None
            if document is not None:
                model.load_state_dict(document["model_state_dict"], strict=True)
                optimizer.load_state_dict(document["optimizer_state_dict"])
                _restore_rng(torch, selected_device, document["rng_state"])
                start_epoch, global_step = document["epoch"], document["global_step"]
                best_iou, best_epoch, best_val = document["best_iou"], document["best_epoch"], document["best_val"]
                best_state = document["best_model_state_dict"]
            parameter_names = set(dict(model.named_parameters()))
            before = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
            notify(f"Baseline on {selected_device}: {len(train)} train, {len(val)} val; epochs {start_epoch + 1}..{config['epochs']}; test pixels excluded")
            _telemetry(staging, "run_started", device=selected_device, epoch=start_epoch,
                       total_epochs=config["epochs"], global_step=global_step,
                       train_samples=len(train), val_samples=len(val),
                       batches_per_epoch=math.ceil(len(train) / config["batch_size"]))
            evaluated_start = _evaluate(model, val, config, torch, selected_device, normalize_rgb, loss_fn, PixelConfusion)
            initial_val = evaluated_start if document is None else document["initial_val"]
            history = []
            with (staging / "history.jsonl").open("w", encoding="utf-8") as log:
                log.write(json.dumps({"event": "initial_validation", "completed_epoch": start_epoch, "validation": evaluated_start}, sort_keys=True) + "\n")
                log.flush()
                for epoch in range(start_epoch + 1, config["epochs"] + 1):
                    epoch_start = time.perf_counter()
                    generator = torch.Generator(device="cpu").manual_seed(config["seed"] + epoch)
                    domain_balance = config.get("domain_balance", 0.0)
                    if domain_balance > 0.0:
                        target_indices = [i for i, r in enumerate(train.records) if r["source_id"].startswith("ai-bulk-")]
                        base_indices = [i for i, r in enumerate(train.records) if not r["source_id"].startswith("ai-bulk-")]
                        if target_indices and base_indices:
                            n_total = len(train)
                            n_target = round(n_total * domain_balance)
                            n_base = n_total - n_target
                            perm_target = torch.randperm(len(target_indices), generator=generator).tolist()
                            while len(perm_target) < n_target:
                                perm_target.extend(torch.randperm(len(target_indices), generator=generator).tolist())
                            chosen_target = [target_indices[i] for i in perm_target[:n_target]]
                            perm_base = torch.randperm(len(base_indices), generator=generator).tolist()
                            while len(perm_base) < n_base:
                                perm_base.extend(torch.randperm(len(base_indices), generator=generator).tolist())
                            chosen_base = [base_indices[i] for i in perm_base[:n_base]]
                            combined = chosen_target + chosen_base
                            shuffle_perm = torch.randperm(len(combined), generator=generator).tolist()
                            order = [combined[i] for i in shuffle_perm]
                        else:
                            order = torch.randperm(len(train), generator=generator).tolist()
                    else:
                        order = torch.randperm(len(train), generator=generator).tolist()
                    model.train()
                    total_loss, valid_pixels, gradient_min, gradient_max = 0.0, 0, math.inf, 0.0
                    for batch_index, start in enumerate(range(0, len(order), config["batch_size"]), 1):
                        batch = _batch(train, order[start:start + config["batch_size"]], torch, selected_device)
                        _augment(batch, generator, config, torch)
                        optimizer.zero_grad(set_to_none=True)
                        losses = loss_fn(model(normalize_rgb(batch["image"])), batch["target"], batch["valid"])
                        losses["loss"].backward()
                        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True).item())
                        if not math.isfinite(norm):
                            raise BaselineTrainingError("Non-finite training gradients.")
                        optimizer.step()
                        global_step += 1
                        batch_loss = float(losses["loss"].item())
                        total_loss += batch_loss * losses["valid_pixels"]
                        valid_pixels += losses["valid_pixels"]
                        gradient_min, gradient_max = min(gradient_min, norm), max(gradient_max, norm)
                        _telemetry(staging, "train_batch", epoch=epoch, global_step=global_step,
                                   batch=batch_index, batches_per_epoch=math.ceil(len(train) / config["batch_size"]),
                                   loss=batch_loss, learning_rate=config["learning_rate"])
                        if batch_index == 1 or batch_index % 100 == 0:
                            notify(f"Epoch {epoch}/{config['epochs']}, batch {batch_index}/{math.ceil(len(train) / config['batch_size'])}: loss={float(losses['loss'].item()):.5f}")
                    measured_val = _evaluate(model, val, config, torch, selected_device, normalize_rgb, loss_fn, PixelConfusion)
                    state = _cpu_tree(model.state_dict(), torch)
                    if not _finite_tree(state, torch):
                        raise BaselineTrainingError("Non-finite model parameters or buffers.")
                    if measured_val["iou"] > best_iou:
                        best_iou, best_epoch, best_val, best_state = measured_val["iou"], epoch, measured_val, state
                    checkpoint = {"baseline_checkpoint_version": "1.0", "config": contract,
                                  "epoch": epoch, "global_step": global_step, "model_state_dict": state,
                                  "optimizer_state_dict": _cpu_tree(optimizer.state_dict(), torch),
                                  "rng_state": _cpu_tree(_rng_state(torch, selected_device), torch),
                                  "best_epoch": best_epoch, "best_iou": best_iou, "best_val": best_val,
                                  "best_model_state_dict": best_state, "initial_val": initial_val}
                    last_hash = _write_checkpoint(staging / "last.pt", checkpoint, torch)
                    best_hash = _write_checkpoint(staging / "best.pt", {
                        "baseline_weights_version": "1.0", "config": contract, "epoch": best_epoch,
                        "model_state_dict": best_state, "validation": best_val}, torch)
                    row = {"event": "epoch", "epoch": epoch, "global_step": global_step,
                           "train_loss": total_loss / valid_pixels, "train_valid_pixels": valid_pixels,
                           "gradient_norm_min": gradient_min, "gradient_norm_max": gradient_max,
                           "validation": measured_val, "best_epoch": best_epoch,
                           "elapsed_seconds": time.perf_counter() - epoch_start}
                    history.append(row)
                    log.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                    log.flush()
                    _telemetry(staging, "epoch_completed", epoch=epoch, global_step=global_step,
                               train_loss=row["train_loss"], validation=measured_val)
                    notify(f"Epoch {epoch} complete: val pixel IoU={measured_val['iou']:.5f}, Dice={measured_val['dice']:.5f}; best={best_epoch}")
            best_document, _ = _read_checkpoint(staging / "best.pt", torch, resume=False)
            with torch.device("cpu"):
                restored = build_model(None).to(device=selected_device, dtype=torch.float32)
            restored.load_state_dict(best_document["model_state_dict"], strict=True)
            reloaded_val = _evaluate(restored, val, config, torch, selected_device, normalize_rgb, loss_fn, PixelConfusion)
            if any(reloaded_val[key] != best_val[key] for key in ("tp", "fp", "fn", "tn")):
                raise BaselineTrainingError("Best checkpoint reload changed validation pixel counts.")
            update = math.sqrt(sum(float((value.double() - before[name].double()).square().sum())
                                   for name, value in state.items() if name in parameter_names))
            if not math.isfinite(update) or update <= 0:
                raise BaselineTrainingError("Training did not change model parameters.")
            report = {"ok": True, "model": config["model"], "device": selected_device,
                      "train_samples": len(train), "val_samples": len(val), "test_used": False,
                      "image_dimensions": list(train.dimensions), "gsd_m": train.gsd_m,
                      "train_selection_fingerprint": train.selection_fingerprint,
                      "val_selection_fingerprint": val.selection_fingerprint,
                      "dataset_fingerprint": train.dataset_fingerprint,
                      "epochs_completed": config["epochs"] - start_epoch, "start_epoch": start_epoch,
                      "total_epochs": config["epochs"], "global_step": global_step,
                      "initial_val": initial_val, "start_val": evaluated_start,
                      "final_val": measured_val, "best_val": best_val, "best_epoch": best_epoch,
                      "best_checkpoint_sha256": best_hash, "last_checkpoint_sha256": last_hash,
                      "best_reload_verified": True, "parameter_update_l2": update,
                      "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                      "config_sha256": _sha256(staging / "config.json"), "encoder": WEIGHT_SPEC,
                      "resume_checkpoint_sha256": parent_hash, "history": history,
                      "environment": {"python": platform.python_version(), "platform": platform.system() + " " + platform.machine(),
                                      "package_versions": libraries, "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset")},
                      "wall_time_seconds": time.perf_counter() - started,
                      "limitations": ["Pixel metrics on the internal upstream-annotated validation split; no independent test result.",
                                      "The threshold is fixed; checkpoints are selected by validation pixel IoU only.",
                                      "This source lacks representative reviewed PV-free roofs; pixel precision is not roof-level false-positive performance.",
                                      "Pretraining data and source annotations have separate provenance and licensing."]}
            _write_json(staging / "baseline-report.json", report)
            _telemetry(staging, "run_completed", epoch=config["epochs"], global_step=global_step,
                       device=selected_device, best_epoch=best_epoch, best_validation=best_val)
        _publish(staging, output)
        staging = None
        return report
    except BaseException as error:
        if staging is not None and staging.is_dir():
            try:
                _telemetry(staging, "run_failed", error_type=type(error).__name__)
            except OSError:
                pass
        recovered = None
        resume_from = None
        recovery_errors = []
        if staging is not None:
            candidates = [staging / "last.pt", *sorted(staging.glob(".last.pt-backup-*/last.pt"))]
            for candidate in candidates:
                companion = candidate.with_suffix(candidate.suffix + ".sha256")
                try:
                    if (candidate.is_symlink() or companion.is_symlink()
                            or not candidate.is_file() or not companion.is_file()
                            or candidate.stat().st_size > MAX_CHECKPOINT_BYTES or companion.stat().st_size > 128):
                        continue
                    expected = companion.read_text(encoding="ascii").strip()
                    if re.fullmatch(r"[a-f0-9]{64}", expected) and _sha256(candidate) == expected:
                        resume_from = candidate.relative_to(staging).as_posix()
                        break
                except (OSError, ValueError, RuntimeError):
                    continue
            # Keep completed checkpoint artifacts even if their hash pair cannot
            # be verified. Such artifacts are not advertised as resumable.
            artifacts = any(os.path.lexists(candidate) for candidate in candidates)
            if resume_from is not None or artifacts:
                destination = output.with_name(output.name + ".incomplete-" + staging.name.rsplit("-", 1)[-1])
                try:
                    _write_json(staging / "failure-report.json", {"ok": False, "error": str(error),
                                                                  "error_type": type(error).__name__, "resume_from": resume_from})
                except BaseException as recovery_error:
                    recovery_errors.append(f"failure-report write failed: {recovery_error}")
                try:
                    _publish(staging, destination)
                except BaseException as recovery_error:
                    recovery_errors.append(f"incomplete-directory publication failed: {recovery_error}")
                    recovered = staging
                else:
                    recovered = destination
                # A failed recovery write/publication must never allow finally
                # to delete the working directory containing completed epochs.
                staging = None
        message = f"Baseline training failed: {error}"
        if recovered is not None:
            if resume_from is not None:
                message += f" Completed epoch checkpoint retained at {recovered / resume_from}."
            else:
                message += f" Unverified checkpoint artifacts retained at {recovered}; no verified resumable pair was found."
        if recovery_errors:
            message += " Recovery details: " + "; ".join(recovery_errors) + "."
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            error.add_note(message)
            raise
        raise BaselineTrainingError(message) from error
    finally:
        if staging is not None:
            shutil.rmtree(staging)
