"""
loop_utils.py — Runtime utilities for MMT training loop.

This module groups all runtime helpers used by the finetuning and
pretraining loops:

    • Moving collated batches to device (CPU/GPU/MPS)
    • Determining which outputs are active in a batch
    • Logging training setup
    • Extracting LR from param groups
    • Running a full training or validation epoch
    • AMP-safe backward + grad accumulation

It contains NO global configuration logic (handled in config_validation.py),
and NO optimizer construction logic (handled in scheduler.py).

The goal is to keep `loop.py` minimal, readable, and focused on the
high-level orchestration of stages, epochs, checkpoints, and metrics.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Mapping, Tuple, Hashable, Optional

import torch
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler

from .amp_utils import amp_ctx_for_model
from .losses import compute_loss_pred_space
from .scheduler import toggle_param_groups

import time

logger = logging.getLogger("mmt.Training")

# Max gradient norm for clipping (matches original loop.py behaviour)
_MAX_GRAD_NORM = 1.0


# ======================================================================
# Batch → device helpers
# ======================================================================


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """
    Move all tensor components of the collated batch to the given device.

    Expected structure (from MMTCollate) after this function:
      - "emb"            : List[List[Tensor]]
      - "pos", "id", "mod", "role" : LongTensor (B, L)
      - "padding_mask"   : BoolTensor (B, L)
      - "input_mask",
        "actuator_mask"  : BoolTensor (B, L)
      - "outputs_emb"    : Dict[int, Tensor]        # (B, K) per output
      - "outputs_mask"   : Dict[int, BoolTensor(B,)]

      - optional:
        "output_native"  : Dict[int, Tensor]

    For backward-compatibility it also accepts:
      - "outputs_emb"    : Dict[int, List[Tensor]]
                           and converts each list → stacked Tensor(B, K).

    Non-tensor entries (e.g. "signal_name") are left untouched.
    """
    if device.type == "cpu":
        # DataLoader already yields CPU tensors; nothing to do
        return batch

    def _to(tens: Tensor) -> Tensor:
        return tens.to(device, non_blocking=True)

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

    # Ragged embeddings: List[List[Tensor]]
    emb = batch.get("emb", None)
    if isinstance(emb, list):
        for i, row in enumerate(emb):
            if isinstance(row, list):
                for j, t in enumerate(row):
                    if isinstance(t, Tensor):
                        row[j] = _to(t)
        batch["emb"] = emb

    # Outputs: coeff-space embeddings
    outputs_emb = batch.get("outputs_emb", None)
    if isinstance(outputs_emb, dict):
        new_oe: Dict[Hashable, Tensor] = {}
        for k, v in outputs_emb.items():
            if isinstance(v, Tensor):
                # Already dense (B, K)
                new_oe[k] = _to(v)
            elif isinstance(v, list):
                # Legacy: list[Tensor] length B, each (K,)
                if not v:
                    continue
                # Ensure everything is Tensor and stack along batch dim
                stacked = torch.stack(
                    [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in v],
                    dim=0,
                )  # (B, K)
                new_oe[k] = _to(stacked)
            else:
                raise TypeError(
                    f"outputs_emb[{k!r}] must be Tensor or list[Tensor], got {type(v)}"
                )
        batch["outputs_emb"] = new_oe

    # Outputs: masks
    outputs_mask = batch.get("outputs_mask", None)
    if isinstance(outputs_mask, dict):
        batch["outputs_mask"] = {k: _to(v) for k, v in outputs_mask.items()}

    # Optional: native outputs
    output_native = batch.get("output_native", None)
    if isinstance(output_native, dict):
        batch["output_native"] = {k: _to(v) for k, v in output_native.items()}

    return batch


# ======================================================================
# Active outputs / LR helpers
# ======================================================================


def active_outputs_from_mask(outputs_mask: Mapping[Hashable, Tensor]) -> set[Hashable]:
    """
    Determine which output keys have at least one supervised sample in
    this batch. These keys drive the automatic per-output LR toggling.

    Parameters
    ----------
    outputs_mask : dict
        Mapping from output_key -> BoolTensor(B,).

    Returns
    -------
    set
        A set of output keys with outputs_mask[k].any() == True.
    """
    active = set()
    for key, mask in outputs_mask.items():
        if mask.dtype != torch.bool:
            raise RuntimeError(f"outputs_mask[{key!r}] must be a bool tensor.")
        if bool(mask.any()):
            active.add(key)
    return active


def backbone_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """
    Return the current learning rate of the backbone param group.
    Used for logging.
    """
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
    training_cfg: Dict[str, Any],
) -> None:
    """
    Compact logging of device, AMP, parameters, and stage definitions.
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("======== MMT Training setup ========")
    logger.info("Device       : %s", device)
    logger.info("AMP enabled  : %s (%s)", amp_enabled, amp_dtype)
    logger.info("Params       : total=%d, trainable=%d", n_params, n_trainable)
    logger.info("Train loader : %d batches/epoch", train_loader_len)
    logger.info("Stages:")
    for s in stages:
        logger.info(
            "  - %(name)s: epochs=%(epochs)s | freeze=%(freeze)r | lr=%(lr)r | wd=%(wd)r | grad_acc=%(acc)d",
            {
                "name": s["name"],
                "epochs": s["epochs"],
                "freeze": s["freeze"],
                "lr": s["optimizer"]["lr"],
                "wd": s["optimizer"]["wd"],
                "acc": s["scheduler"]["grad_accum_steps"],
            },
        )

    pat = training_cfg["early_stop"]["patience"]
    dlt = training_cfg["early_stop"]["delta"]
    logger.info("Early stopping: patience=%d, delta=%.4f", pat, dlt)
    logger.info("====================================")


# ======================================================================
# Epoch runner
# ======================================================================


def run_one_epoch(
    model,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[LRScheduler],
    scaler: Optional[torch.cuda.amp.GradScaler],
    *,
    device: torch.device,
    amp_enabled: bool,
    output_weights: Mapping[Hashable, float],
    output_to_modality: Mapping[Hashable, str],
    grad_accum_steps: int,
    train: bool,
    global_step: int,
) -> Tuple[float, int]:
    """
    Run one epoch in train or eval mode.

    Responsibilities:
      • Move batch to device
      • Automatic LR toggling (train only)
      • Forward + loss under AMP
      • Backward under AMP (train only)
      • Grad accumulation
      • Scheduler stepping
      • Loss aggregation

    Returns
    -------
    avg_loss : float
        Mean loss over all batches.
    global_step : int
        Updated optimizer-step counter (unchanged in eval mode).
    """
    if train and optimizer is None:
        raise ValueError("optimizer must be provided when train=True.")

    model.train(train)
    if not train:
        torch.set_grad_enabled(False)

    if train:
        optimizer.zero_grad(set_to_none=True)

    running_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        t0 = time.time()
        batch = move_batch_to_device(batch, device)
        t1 = time.time()
        outputs_mask = batch["outputs_mask"]

        # Per-batch automatic LR enabling/disabling
        if train:
            act = active_outputs_from_mask(outputs_mask)
            toggle_param_groups(
                optimizer,
                active_outputs=act,
                output_to_modality=output_to_modality,
            )

        # Forward pass
        with amp_ctx_for_model(model, enable=amp_enabled):
            out = model(batch)
            preds = out.get("pred", {})

            loss_t, _ = compute_loss_pred_space(
                preds=preds,
                y_true=batch["outputs_emb"],
                outputs_mask=outputs_mask,
                output_weights=output_weights,
            )

            if train and grad_accum_steps > 1:
                loss_for_backprop = loss_t / float(grad_accum_steps)
            else:
                loss_for_backprop = loss_t
        t2 = time.time()
        # Backward pass
        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_for_backprop).backward()
            else:
                loss_for_backprop.backward()

            t3 = time.time()

            # Accumulation step triggers optimizer step
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), _MAX_GRAD_NORM)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

            t4 = time.time()

        running_loss += float(loss_t.detach().cpu())
        n_batches += 1

        logger.info(
            f"[TIMING] batch {batch_idx}: "
            f"move={t1 - t0:.4f}s  "
            f"forward={t2 - t1:.4f}s  "
            f"backward={t3 - t2:.4f}s  "
            f"opt={t4 - t3:.4f}s"
        )

    avg_loss = running_loss / max(1, n_batches)

    if not train:
        torch.set_grad_enabled(True)

    return avg_loss, global_step
