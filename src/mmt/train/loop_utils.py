"""
loop_utils.py — Runtime utilities for MMT train loop.

This module groups all runtime helpers used by the finetuning and
pretraining loops:

    • Moving collated batches to device (CPU/GPU/MPS)
    • Determining which outputs are active in a batch
    • Logging train setup
    • Extracting LR from param groups
    • Running a full train or validation epoch
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

logger = logging.getLogger("mmt.Train")

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
    train_cfg: Dict[str, Any],
) -> None:
    """
    Compact logging of device, AMP, parameters, and stage definitions.
    """
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("======== MMT Train setup ========")
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

    pat = train_cfg["early_stop"]["patience"]
    dlt = train_cfg["early_stop"]["delta"]
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
    max_batches: Optional[int] = None,
) -> Tuple[float, int]:
    """
    Run one epoch over a DataLoader, in either train or eval mode.

    This is the low-level driver used by the high-level loop in loop.py.
    It is fully agnostic to cached vs streaming datasets; the only extra
    control is the optional `max_batches` argument.

    Parameters
    ----------
    model :
        The MMT model (nn.Module).
    loader :
        A PyTorch DataLoader yielding collated window batches.
    optimizer : torch.optim.Optimizer or None
        Required when train=True, ignored when train=False.
    scheduler : LRScheduler or None
        Optional LR scheduler, stepped after each optimizer step.
    scaler : torch.cuda.amp.GradScaler or None
        Optional AMP scaler (enabled only on CUDA devices).
    device : torch.device
        Target device for the batch tensors.
    amp_enabled : bool
        Whether to use autocast for forward pass.
    output_weights : Mapping[Hashable, float]
        Per-output weights for the prediction-space loss.
    output_to_modality : Mapping[Hashable, str]
        Mapping from output-id -> modality string, used by LR toggling.
    grad_accum_steps : int
        Number of micro-batches to accumulate before one optimizer step.
    train : bool
        If True, run in train mode with optimizer and scheduler steps.
        If False, run in eval mode (no gradients, no optimizer, no scheduler).
    global_step : int
        Current global optimizer-step counter (for logging / schedulers).
    max_batches : int or None, optional
        If not None, process at most this many batches from the loader.
        This is used in STREAMING mode to define an "epoch" by a fixed
        number of batches rather than a full pass over an IterableDataset.
        In cached mode, this should be left as None to process the full
        dataloader.

    Returns
    -------
    avg_loss : float
        Mean loss over all processed batches.
    global_step : int
        Updated global optimizer-step counter (unchanged in eval mode).

    Notes
    -----
    • When `max_batches` is not None and the loader yields fewer batches
      than requested (e.g. streaming dataset exhausts early), the loop
      simply runs over the available batches and stops when the iterator
      ends.
    • Timing information is logged per batch when train=True.
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
        # Early stop for streaming mode
        if max_batches is not None and batch_idx >= max_batches:
            break

        t0 = time.perf_counter()
        batch = move_batch_to_device(batch, device)
        t1 = time.perf_counter()

        outputs_mask = batch["outputs_mask"]

        # Per-batch automatic LR enabling/disabling (train only)
        if train:
            active = active_outputs_from_mask(outputs_mask)
            toggle_param_groups(
                optimizer,
                active_outputs=active,
                output_to_modality=output_to_modality,
            )

        # ----------------------- FORWARD -----------------------
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
        t2 = time.perf_counter()

        # ----------------------- BACKWARD ----------------------
        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_for_backprop).backward()
            else:
                loss_for_backprop.backward()
            t3 = time.perf_counter()

            # Gradient accumulation → optimizer step
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

            t4 = time.perf_counter()

            logger.info(
                f"[TIMING] batch {batch_idx}: "
                f"move={t1 - t0:.4f}s  "
                f"forward={t2 - t1:.4f}s  "
                f"backward={t3 - t2:.4f}s  "
                f"opt={t4 - t3:.4f}s"
            )

        running_loss += float(loss_t.detach().cpu())
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)

    if not train:
        torch.set_grad_enabled(True)

    return avg_loss, global_step
