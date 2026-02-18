"""
loop_utils.py — Runtime utilities for MMT train loop.

This module groups runtime helpers used by the finetuning and pretraining loops:

    • Moving collated batches to device (CPU/GPU/MPS)
    • Logging train setup
    • Extracting LR from param groups
    • Running a full train or validation epoch
    • AMP-safe backward + grad accumulation

It contains NO global configuration logic (handled in config validation),
and NO optimizer construction logic (handled in scheduler.py).

The goal is to keep `loop.py` minimal, readable, and focused on the
high-level orchestration of stages, epochs, checkpoints, and metrics.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Mapping, Tuple, Hashable, Optional

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler

from mmt.utils.amp_utils import amp_ctx_for_model
from .losses import compute_loss_pred_space

logger = logging.getLogger("mmt.Train")

# Max gradient norm for clipping (matches original loop.py behaviour)
_MAX_GRAD_NORM = 1.0

# How many batch-level timing lines to show at INFO during the *first* global epoch.
# If DEBUG logging is enabled, we will log timing for every batch at DEBUG.
_LOG_FIRST_EPOCH_BATCHES_INFO = 5


def _maybe_log_batch_timing(
    *,
    batch_idx: int,
    epoch_global: Optional[int],
    train: bool,
    dt_dataloader: float,
    dt_move: float,
    dt_forward: float,
    dt_backward: Optional[float],
    dt_opt: Optional[float],
) -> None:
    """Log per-batch timing without spamming INFO logs."""
    if logger.isEnabledFor(logging.DEBUG):
        level = logging.DEBUG
    else:
        eg = 1 if epoch_global is None else int(epoch_global)
        if eg <= 1 and batch_idx < _LOG_FIRST_EPOCH_BATCHES_INFO:
            level = logging.INFO
        else:
            return

    phase = "TRAIN" if train else "VAL"
    parts: list[str] = [
        f"time dataloader={dt_dataloader:.4f}s",
        f"move={dt_move:.4f}s",
        f"forward={dt_forward:.4f}s",
    ]
    if train:
        if dt_backward is not None:
            parts.append(f"backward={dt_backward:.4f}s")
        if dt_opt is not None:
            parts.append(f"opt={dt_opt:.4f}s")

    logger.log(level, "[TIMING %s] batch %d: %s", phase, batch_idx, "  ".join(parts))


# ======================================================================
# Batch → device helpers
# ======================================================================


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensor components of the collated batch to the given device.

    Notes
    -----
    • On MPS we force synchronous copies (non_blocking=False). This avoids rare
      metadata corruption observed with num_workers=0.
    • On CUDA we use non_blocking=True only when the source tensor is CPU pinned
      memory (otherwise it provides no benefit).
    """
    if device.type == "cpu":
        return batch

    def _to(tens: Tensor) -> Tensor:
        if tens.device == device:
            return tens

        if device.type == "mps":
            return tens.to(device, non_blocking=False)

        if device.type == "cuda":
            # non_blocking only helps for CPU->CUDA when source is pinned
            nb = (tens.device.type == "cpu") and tens.is_pinned()
            return tens.to(device, non_blocking=nb)

        # fallback (e.g. xpu / other)
        return tens.to(device, non_blocking=False)

    # Core token tensors
    for key in (
        "pos",
        "id",
        "mod",
        "role",
        "padding_mask",
        "input_mask",
        "actuator_mask",
    ):
        val = batch.get(key, None)
        if isinstance(val, Tensor):
            batch[key] = _to(val)

    # Packed embeddings (by signal_id): Dict[int, Tensor] + Dict[int, LongTensor]
    emb = batch.get("emb", None)
    if isinstance(emb, dict):
        batch["emb"] = {k: _to(v) for k, v in emb.items() if isinstance(v, Tensor)}
    elif emb is not None:
        raise TypeError(
            f"batch['emb'] must be a dict[int, Tensor] (packed), got {type(emb)}"
        )

    emb_index = batch.get("emb_index", None)
    if isinstance(emb_index, dict):
        batch["emb_index"] = {
            k: _to(v) for k, v in emb_index.items() if isinstance(v, Tensor)
        }
    elif emb_index is not None:
        raise TypeError(
            f"batch['emb_index'] must be a dict[int, Tensor], got {type(emb_index)}"
        )

    # Outputs: coeff-space embeddings
    output_emb = batch.get("output_emb", None)
    if isinstance(output_emb, dict):
        new_oe: Dict[Hashable, Tensor] = {}
        for k, v in output_emb.items():
            if isinstance(v, Tensor):
                new_oe[k] = _to(v)
            elif isinstance(v, list):
                if not v:
                    continue
                stacked = torch.stack(
                    [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in v],
                    dim=0,
                )
                new_oe[k] = _to(stacked)
            else:
                raise TypeError(
                    f"output_emb[{k!r}] must be Tensor or list[Tensor], got {type(v)}"
                )
        batch["output_emb"] = new_oe

    # Output masks
    output_mask = batch.get("output_mask", None)
    if isinstance(output_mask, dict):
        batch["output_mask"] = {k: _to(v) for k, v in output_mask.items()}

    # Optional: native output
    output_native = batch.get("output_native", None)
    if isinstance(output_native, dict):
        batch["output_native"] = {k: _to(v) for k, v in output_native.items()}

    return batch


# ======================================================================
# LR helpers
# ======================================================================


def backbone_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """Return the current learning rate of the backbone param group (for logging)."""
    for g in optimizer.param_groups:
        if g.get("group_type") == "backbone":
            return float(g.get("lr", 0.0))
    return None


# ======================================================================
# Logging helpers
# ======================================================================


def log_train_setup(
    model,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    train_loader_len: int,
    stages: list[Dict[str, Any]],
    train_cfg: Dict[str, Any],
) -> None:
    """Compact logging of device, AMP, parameters, loss weights, and stage definitions."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("======== MMT Train setup ========")
    logger.info("Device       : %s", device)
    logger.info("AMP enabled  : %s (%s)", amp_enabled, amp_dtype)
    logger.info("Params       : total=%d, trainable=%d", n_params, n_trainable)
    logger.info("Train loader : %d batches/epoch", train_loader_len)

    # Loss weights (do not change across stages)
    loss_cfg = train_cfg.get("loss", {})
    output_weights = loss_cfg.get("output_weights")
    if isinstance(output_weights, dict) and output_weights:
        logger.info("Loss weights : %r", output_weights)
    else:
        logger.info("Loss weights : (uniform across outputs)")

    logger.info("Stages:")
    for s in stages:
        logger.info("  - %s", s["name"])
        logger.info("      epochs      : %s", s["epochs"])
        logger.info("      freeze      : %r", s["freeze"])
        logger.info("      lr          : %r", s["optimizer"]["lr"])
        logger.info("      wd          : %r", s["optimizer"]["wd"])
        logger.info("      grad_acc    : %d", s["scheduler"]["grad_accum_steps"])

    pat = train_cfg["early_stop"]["patience"]
    dlt = train_cfg["early_stop"]["delta"]
    logger.info("Early stopping:")
    logger.info("      patience    : %d", pat)
    logger.info("      delta       : %.4f", dlt)
    logger.info("====================================")


# ======================================================================
# Epoch runner
# ======================================================================


def run_one_epoch(
    model,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
    scaler: Optional[GradScaler],
    *,
    device: torch.device,
    amp_enabled: bool,
    output_weights: Mapping[Hashable, float],
    grad_accum_steps: int,
    train: bool,
    global_step: int,
    max_batches: Optional[int] = None,
    epoch_global: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Run one epoch over a DataLoader, in either train or eval mode.

    Notes
    -----
    • In streaming mode, pass `max_batches` to define the epoch length.
    • We do **not** do per-batch LR toggling. Missing outputs are already masked
      inside the loss via `output_mask`.
    """
    if train and optimizer is None:
        raise ValueError("optimizer must be provided when train=True.")

    model.train(train)

    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    n_batches = 0

    t_before_next = time.perf_counter()

    # Ensure gradient enablement is always restored, even on exceptions.
    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            t_after_next = time.perf_counter()

            # Early stop for streaming mode
            if max_batches is not None and batch_idx >= max_batches:
                break

            t0 = time.perf_counter()
            batch = move_batch_to_device(batch, device)
            t1 = time.perf_counter()

            output_mask = batch["output_mask"]

            # ----------------------- FORWARD -----------------------
            with amp_ctx_for_model(model, enable=amp_enabled):
                out = model(batch)
                preds = out.get("pred", {})

            # Compute loss outside autocast. The loss function itself forces
            # float32 computation for AMP stability.
            loss_t, loss_logs = compute_loss_pred_space(
                preds=preds,
                y_true=batch["output_emb"],
                output_mask=output_mask,
                output_weights=output_weights,
            )

            # Optional: per-output loss logging (enable with logger level DEBUG).
            if loss_logs and logger.isEnabledFor(logging.DEBUG):
                per_out_str = ", ".join(
                    f"{k}={v:.3e}"
                    for k, v in sorted(loss_logs.items(), key=lambda kv: str(kv[0]))
                )
                logger.debug(
                    "[LOSS] batch %d (%s) total=%.3e per-output: %s",
                    batch_idx,
                    "train" if train else "val",
                    float(loss_t.detach().cpu().item()),
                    per_out_str,
                )

            # Fail fast on non-finite loss to surface the offending output key.
            if (not torch.isfinite(loss_t).item()) or any(
                not math.isfinite(v) for v in loss_logs.values()
            ):
                bad_keys = [k for k, v in loss_logs.items() if not math.isfinite(v)]
                first_bad = bad_keys[0] if bad_keys else None
                per_out_str = ", ".join(
                    f"{k}={v:.3e}"
                    for k, v in sorted(loss_logs.items(), key=lambda kv: str(kv[0]))
                )
                logger.error(
                    "[LOSS] Non-finite loss detected at batch %d (%s). loss=%s first_bad=%r",
                    batch_idx,
                    "train" if train else "val",
                    float(loss_t.detach().cpu().item()),
                    first_bad,
                )
                if per_out_str:
                    logger.error("[LOSS] Per-output losses: %s", per_out_str)
                raise RuntimeError(
                    f"Non-finite loss detected (batch={batch_idx}, train={train}, "
                    f"first_bad={first_bad!r})."
                )

            if train and grad_accum_steps > 1:
                loss_for_backprop = loss_t / float(grad_accum_steps)
            else:
                loss_for_backprop = loss_t
            t2 = time.perf_counter()

            # ----------------------- BACKWARD ----------------------
            t3, t4 = 0.0, 0.0
            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss_for_backprop).backward()
                else:
                    loss_for_backprop.backward()
                t3 = time.perf_counter()

                # Gradient accumulation → optimizer step
                if (batch_idx + 1) % grad_accum_steps == 0:
                    did_step = True

                    if (
                        scaler is not None
                        and scaler.is_enabled()
                        and optimizer is not None
                    ):
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), _MAX_GRAD_NORM
                        )

                        prev_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()

                        # If scale decreased, step was skipped due to inf/nan grads
                        did_step = scaler.get_scale() >= prev_scale
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), _MAX_GRAD_NORM
                        )
                        if optimizer is not None:
                            optimizer.step()

                    if optimizer is not None:
                        optimizer.zero_grad(set_to_none=True)

                    if did_step:
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                    else:
                        # Optional: log once in a while
                        logger.warning(
                            "AMP overflow detected: skipped optimizer/scheduler step."
                        )

                t4 = time.perf_counter()

            running_loss += float(loss_t.detach().cpu())
            n_batches += 1

            _maybe_log_batch_timing(
                batch_idx=batch_idx,
                epoch_global=epoch_global,
                train=train,
                dt_dataloader=t_after_next - t_before_next,
                dt_move=t1 - t0,
                dt_forward=t2 - t1,
                dt_backward=(t3 - t2) if train else None,
                dt_opt=(t4 - t3) if train else None,
            )

            # update t before next loading
            t_before_next = time.perf_counter()

    avg_loss = running_loss / max(1, n_batches)
    return avg_loss, global_step
