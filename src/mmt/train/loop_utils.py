"""
loop_utils.py — Runtime utilities for MMT train loop.

This module groups runtime helpers used by the finetuning and pretraining loops:

    • Moving collated batches to device (CPU/GPU/MPS)
    • Logging train setup
    • Extracting LR from param groups
    • Running a full train or validation epoch
    • AMP-safe backward + grad accumulation

It contains NO global configuration logic (handled in config validation), and NO optimizer construction logic (handled
in scheduler.py).

The goal is to keep `loop.py` minimal, readable, and focused on the high-level orchestration of stages, epochs,
checkpoints, and metrics.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, MutableMapping
from typing import TYPE_CHECKING, Any, Hashable

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch import dtype as torch_dtype
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler

from mmt.utils.amp_utils import amp_ctx_for_model
from mmt.train.losses.base import LossComputeContext
from .losses import LossAggregator, build_loss_aggregator, resolve_loss_output_filters


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Train")

# Max gradient norm for clipping (matches original loop.py behavior)
_MAX_GRAD_NORM = 1.0

# How many batch-level timing lines to show at INFO during the *first* global epoch.
# If DEBUG logging is enabled, we will log timing for every batch at DEBUG.
_LOG_FIRST_EPOCH_BATCHES_INFO = 5


# ----------------------------------------------------------------------------------------------------------------------
def effective_stage_loss_cfg(train_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Return the loss config for one stage.

    Stage-level ``loss`` blocks shallow-override the global ``train.loss`` block. Stages without ``loss`` use the
    global block unchanged.

    Parameters
    ----------
    train_cfg : Mapping[str, Any]
        Train config containing the global ``loss`` block.
    stage_cfg : Mapping[str, Any]
        Stage config, optionally containing a ``loss`` block.

    Returns
    -------
    Mapping[str, Any]
        Effective loss config for the stage.

    """

    global_loss = train_cfg["loss"]
    stage_loss = stage_cfg.get("loss")
    if not isinstance(stage_loss, Mapping):
        return global_loss
    return {**global_loss, **stage_loss}


# ----------------------------------------------------------------------------------------------------------------------
def canonical_loss_cfg(loss_cfg: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    """
    Return a stable representation of a loss config for equality checks.

    Dictionary keys are sorted recursively so equivalent mappings compare equal even if written in a different order.
    List order is preserved because loss terms are applied in the configured order.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        Effective loss config to normalize.

    Returns
    -------
    tuple[tuple[str, Any], ...]
        Canonical, comparable loss config representation.

    """

    def canonical_value(value: Any) -> Any:
        if isinstance(value, Mapping):
            return tuple(
                (str(key), canonical_value(inner_value))
                for key, inner_value in sorted(value.items(), key=lambda item: str(item[0]))
            )
        if isinstance(value, list | tuple):
            return tuple(canonical_value(item) for item in value)
        return value

    return canonical_value(loss_cfg)


# ----------------------------------------------------------------------------------------------------------------------
def _resolve_output_weights_by_id(
    *,
    loss_cfg: Mapping[str, Any],
    output_name_to_id: Mapping[str, int],
    path: str,
) -> dict[int, float]:
    """
    Resolve loss ``output_weights`` from output names to signal IDs.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        Effective loss config for one stage.
    output_name_to_id : Mapping[str, int]
        Mapping from output signal name to runtime signal ID.
    path : str
        Human-readable config path used in error messages.

    Returns
    -------
    dict[int, float]
        Output weights keyed by signal ID.

    Raises
    ------
    KeyError
        If a configured output weight references an unknown output.

    """

    output_weights_cfg = loss_cfg.get("output_weights") or {}
    if not isinstance(output_weights_cfg, Mapping) or not output_weights_cfg:
        return {}

    unknown = [key for key in output_weights_cfg.keys() if str(key) not in output_name_to_id]
    if unknown:
        raise KeyError(
            f"Unknown {path}.output_weights keys: {unknown}. "
            f"Expected output signal names among: {sorted(output_name_to_id.keys())}."
        )

    return {int(output_name_to_id[str(name)]): float(weight) for name, weight in output_weights_cfg.items()}


# ----------------------------------------------------------------------------------------------------------------------
def build_loss_aggregator_for_stage(
    *,
    loss_cfg: Mapping[str, Any],
    output_specs: list[Any],
    output_name_to_id: Mapping[str, int],
    output_decoders: dict | None,
    signal_stats: Mapping[str, Mapping[str, Any]] | None,
    path: str,
    require_all_outputs: bool,
) -> LossAggregator:
    """
    Build the loss aggregator for one effective stage loss config.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        Effective loss config for one stage.
    output_specs : list[Any]
        Model output specs.
    output_name_to_id : Mapping[str, int]
        Mapping from output signal name to runtime signal ID.
    output_decoders : dict | None
        Per-output decoders.
    signal_stats : Mapping[str, Mapping[str, Any]] | None
        Per-signal metadata used by native destandardized losses.
    path : str
        Human-readable config path used in error messages.
    require_all_outputs : bool
        Whether all model outputs must be supervised by this loss config.

    Returns
    -------
    LossAggregator
        Built loss aggregator for the stage.

    """

    output_weights_by_id = _resolve_output_weights_by_id(
        loss_cfg=loss_cfg,
        output_name_to_id=output_name_to_id,
        path=path,
    )
    term_output_filters = resolve_loss_output_filters(
        loss_cfg=loss_cfg,
        output_specs=output_specs,
        decoders=output_decoders,
        path=path,
        require_all_outputs=require_all_outputs,
    )

    return build_loss_aggregator(
        loss_cfg=loss_cfg,
        output_weights_by_id=output_weights_by_id if output_weights_by_id else None,
        decoders=output_decoders,
        term_output_filters=term_output_filters,
        output_name_to_id=output_name_to_id,
        signal_stats=signal_stats,
    )


# ----------------------------------------------------------------------------------------------------------------------
def _maybe_log_batch_timing(
    *,
    batch_idx: int,
    epoch_global: int | None,
    train: bool,
    dt_dataloader: float,
    dt_move: float,
    dt_forward: float,
    dt_backward: float | None,
    dt_opt: float | None,
) -> None:
    """Log per-batch timing without spamming INFO logs."""

    if logger.isEnabledFor(logging.DEBUG):
        level = logging.DEBUG
    else:
        eg = 1 if (epoch_global is None) else int(epoch_global)
        if (eg <= 1) and (batch_idx < _LOG_FIRST_EPOCH_BATCHES_INFO):
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


# ======================================================================================================================
# Batch → device helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def move_batch_to_device(  # NOSONAR - Ignore cognitive complexity
    batch: MutableMapping[str, Any], device: torch.device
) -> MutableMapping[str, Any]:
    """
    Move all tensor components of the collated batch to the given device.

    Notes
    -----
    • On MPS we force synchronous copies (non_blocking=False). This avoids rare metadata corruption observed with
    num_workers=0.
    • On CUDA we use non_blocking=True only when the source tensor is CPU pinned memory (otherwise it provides no
    benefit).

    Returns
    -------
    MutableMapping[str, Any]
        Specified `batch` moved to specified `device`.

    Raises
    ------
    TypeError
        If `batch["emb"]` is not a valid mapping (dict[int, Tensor]).
        If `batch["emb_index"]` is not a valid mapping (dict[int, Tensor]).
        If `batch["output_emb"]` is not a Tensor or list[Tensor].

    """

    # ..................................................................................................................
    def _to(tens: Tensor) -> Tensor:
        """Tensor dtype and/or device conversion for input tensor."""

        if tens.device == device:
            return tens

        if device.type == "mps":
            return tens.to(device, non_blocking=False)

        if device.type == "cuda":
            # non_blocking only helps for CPU->CUDA when source is pinned
            nb = (tens.device.type == "cpu") and tens.is_pinned()
            return tens.to(device, non_blocking=nb)

        # Fallback (e.g., xpu / other)
        return tens.to(device, non_blocking=False)

    # ..................................................................................................................

    if device.type == "cpu":
        return batch

    # Core token tensors
    for key in [
        "pos",
        "id",
        "mod",
        "role",
        "padding_mask",
        "input_mask",
        "actuator_mask",
        "space_grid",
        "token_time",
        "t_cut",
    ]:
        val = batch.get(key, None)
        if isinstance(val, Tensor):
            batch[key] = _to(tens=val)

    # Packed embeddings (by signal_id): dict[int, Tensor] + dict[int, LongTensor]
    emb = batch.get("emb", None)
    if isinstance(emb, dict):
        batch["emb"] = {k: _to(tens=v) for k, v in emb.items() if isinstance(v, Tensor)}
    elif emb is not None:
        raise TypeError(f"`batch['emb']` must be a dict[int, Tensor] (packed), got {type(emb)}.")

    emb_index = batch.get("emb_index", None)
    if isinstance(emb_index, dict):
        batch["emb_index"] = {k: _to(tens=v) for k, v in emb_index.items() if isinstance(v, Tensor)}
    elif emb_index is not None:
        raise TypeError(f"`batch['emb_index']` must be a dict[int, Tensor], got {type(emb_index)}.")

    # Outputs: coeff-space embeddings
    output_emb = batch.get("output_emb", None)
    if isinstance(output_emb, dict):
        new_oe: dict[Hashable, Tensor] = {}
        for k, v in output_emb.items():
            if isinstance(v, Tensor):
                new_oe[k] = _to(tens=v)
            elif isinstance(v, list):
                if not v:
                    continue
                stacked = torch.stack(
                    [t if isinstance(t, Tensor) else torch.as_tensor(t) for t in v],
                    dim=0,
                )
                new_oe[k] = _to(tens=stacked)
            else:
                raise TypeError(f"`batch['output_emb'][{k!r}]` must be Tensor or list[Tensor], got {type(v)}.")
        batch["output_emb"] = new_oe

    # Output masks
    output_mask = batch.get("output_mask", None)
    if isinstance(output_mask, dict):
        batch["output_mask"] = {k: _to(tens=v) for k, v in output_mask.items()}

    # Output query timestamps used by coordinate-aware models.
    output_time = batch.get("output_time", None)
    if isinstance(output_time, dict):
        batch["output_time"] = {k: _to(tens=v) for k, v in output_time.items()}

    # Optional: native output
    output_native = batch.get("output_native", None)
    if isinstance(output_native, dict):
        batch["output_native"] = {k: _to(tens=v) for k, v in output_native.items()}

    return batch


# ======================================================================================================================
# LR helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def backbone_lr(optimizer: torch.optim.Optimizer) -> float | None:
    """Return the current learning rate of the primary backbone param group (for logging)."""
    for preferred_block in ("backbone", "backbone_encoder"):
        for g in optimizer.param_groups:
            if g.get("group_type") == preferred_block:
                return float(g.get("lr", 0.0))
    return None


# ======================================================================================================================
# Logging helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def log_train_setup(
    model,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch_dtype | None,
    train_loader_len: int,
    stages: list[Mapping[str, Any]],
    train_cfg: Mapping[str, Any],
) -> None:
    """Compact logging of device, AMP, parameters, loss weights, and stage definitions."""

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info("======== MMT Train setup ========")
    logger.info("Device       : %s", device)
    logger.info("AMP enabled  : %s (%s)", amp_enabled, amp_dtype)
    logger.info("Params       : total=%d, trainable=%d", n_params, n_trainable)
    logger.info("Train loader : %d batches/epoch", train_loader_len)

    # Global loss config. Individual stages may override it with their own loss block.
    loss_cfg = train_cfg.get("loss", {})
    terms = loss_cfg.get("terms", [])
    if terms:
        for t in terms:
            logger.info("Default loss : type=%s weight=%s", t.get("type"), t.get("weight", 1.0))
    else:
        output_weights = loss_cfg.get("output_weights")
        if isinstance(output_weights, dict) and output_weights:
            logger.info("Loss weights : %r", output_weights)
        else:
            logger.info("Default loss : (uniform across outputs)")

    logger.info("Stages:")
    for s in stages:
        logger.info("  - %s", s["name"])
        logger.info("      epochs      : %s", s["epochs"])
        logger.info("      freeze      : %r", s["freeze"])
        logger.info("      lr          : %r", s["optimizer"]["lr"])
        logger.info("      wd          : %r", s["optimizer"]["wd"])
        logger.info("      grad_acc    : %d", s["scheduler"]["grad_accum_steps"])
        if isinstance(s.get("loss"), Mapping):
            stage_loss_cfg = {**loss_cfg, **s["loss"]}
            stage_terms = stage_loss_cfg.get("terms", [])
            for term in stage_terms:
                logger.info("      loss        : type=%s weight=%s", term.get("type"), term.get("weight", 1.0))

    pat = train_cfg["early_stop"]["patience"]
    dlt = train_cfg["early_stop"]["delta"]
    logger.info("Early stopping:")
    logger.info("      patience    : %d", pat)
    logger.info("      delta       : %.4f", dlt)
    logger.info("====================================")


# ======================================================================================================================
# Epoch runner
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def run_one_epoch(  # NOSONAR - Ignore cognitive complexity
    model,
    loader,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    *,
    device: torch.device,
    amp_enabled: bool,
    loss_aggregator: LossAggregator,
    grad_accum_steps: int,
    train: bool,
    global_step: int,
    max_batches: int | None = None,
    epoch_global: int | None = None,
    epoch_in_stage: int | None = None,
    stage_index: int | None = None,
    stage_name: str | None = None,
    run_dir: str | None = None,
) -> tuple[float, dict[str, float], int]:
    """
    Run one epoch over a DataLoader, in either train or eval mode.

    Notes
    -----
    • In streaming mode, pass `max_batches` to define the epoch length.
    • We do **not** do per-batch LR toggling. Missing outputs are already masked inside the loss via `output_mask`.

    Returns
    -------
    tuple[float, dict[str, float], int]
        Tuple (avg_loss, avg_term_logs, global_step).
        ``avg_term_logs`` contains epoch-averaged values for each ``<term>/total`` key produced by ``LossAggregator``.
        Useful for per-term breakdown when multiple loss terms are active.

    Raises
    ------
    ValueError
        If `optimizer` is not provided when `train`is True.
    RuntimeError
        If non-finite loss detected.

    """

    if train:
        if optimizer is None:
            raise ValueError("optimizer must be provided when train=True.")
        # Type narrowing: optimizer is guaranteed non-None in train branch

        optimizer.zero_grad(set_to_none=True)

    model.train(train)

    running_loss = 0.0
    running_term_logs: dict[str, float] = {}
    n_batches = 0

    t_before_next = time.perf_counter()

    # Ensure gradient enablement is always restored, even on exceptions.
    with torch.set_grad_enabled(train):
        for batch_idx, batch in enumerate(loader):
            t_after_next = time.perf_counter()

            # Early stop for streaming mode
            if (max_batches is not None) and (batch_idx >= max_batches):
                break

            t0 = time.perf_counter()
            batch = move_batch_to_device(batch=batch, device=device)
            t1 = time.perf_counter()

            # ----------------------- FORWARD -----------------------
            with amp_ctx_for_model(model=model, enable=amp_enabled):
                out = model(batch)
                preds = out.get("pred", {})

            # Compute loss outside autocast. The loss functions force float32 for AMP stability.
            # pred_dist is present only for gaussian output heads and is forwarded to probabilistic terms (e.g. CRPS).
            loss_context = LossComputeContext(
                stage_index=stage_index,
                stage_name=stage_name,
                epoch_global=epoch_global,
                epoch_in_stage=epoch_in_stage,
                global_step=global_step,
                batch_index=batch_idx,
                train=train,
                run_dir=run_dir,
            )
            loss_t, loss_logs = loss_aggregator.compute(
                preds=preds,
                batch=batch,
                pred_dist=out.get("pred_dist"),
                context=loss_context,
            )

            # Optional: per-output loss logging (enable with logger level DEBUG).
            if loss_logs and logger.isEnabledFor(logging.DEBUG):
                per_out_str = ", ".join(f"{k}={v:.3e}" for k, v in sorted(loss_logs.items(), key=lambda kv: str(kv[0])))
                logger.debug(
                    "[LOSS] batch %d (%s) total=%.3e per-output: %s",
                    batch_idx,
                    "train" if train else "val",
                    float(loss_t.detach().cpu().item()),
                    per_out_str,
                )

            # Fail fast on non-finite loss to surface the offending output key.
            if (not torch.isfinite(loss_t).item()) or any(not math.isfinite(v) for v in loss_logs.values()):
                bad_keys = [k for k, v in loss_logs.items() if not math.isfinite(v)]
                first_bad = bad_keys[0] if bad_keys else None
                per_out_str = ", ".join(f"{k}={v:.3e}" for k, v in sorted(loss_logs.items(), key=lambda kv: str(kv[0])))
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
                    f"Non-finite loss detected (batch={batch_idx}, train={train}, first_bad={first_bad!r})."
                )

            if train and grad_accum_steps > 1:
                loss_for_backprop = loss_t / float(grad_accum_steps)
            else:
                loss_for_backprop = loss_t
            t2 = time.perf_counter()

            # ----------------------- BACKWARD ----------------------
            t3, t4 = 0.0, 0.0
            if train:
                if (scaler is not None) and scaler.is_enabled():
                    scaler.scale(loss_for_backprop).backward()
                else:
                    loss_for_backprop.backward()
                t3 = time.perf_counter()

                # Gradient accumulation → optimizer step
                if optimizer is None:
                    raise ValueError("optimizer must be provided when train=True.")
                if (batch_idx + 1) % grad_accum_steps == 0:
                    did_step = True

                    if (scaler is not None) and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=_MAX_GRAD_NORM)

                        prev_scale = scaler.get_scale()
                        scaler.step(optimizer)
                        scaler.update()

                        # If scale decreased, step was skipped due to inf/nan grads
                        did_step = scaler.get_scale() >= prev_scale

                    else:
                        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=_MAX_GRAD_NORM)
                        optimizer.step()

                    optimizer.zero_grad(set_to_none=True)

                    if did_step:
                        if scheduler is not None:
                            scheduler.step()
                        global_step += 1
                    else:
                        # Optional: log once in a while
                        logger.warning("AMP overflow detected: skipped optimizer/scheduler step.")

                t4 = time.perf_counter()

            running_loss += float(loss_t.detach().cpu())
            for k, v in loss_logs.items():
                if k.endswith("/weighted") and math.isfinite(v):
                    running_term_logs[k] = running_term_logs.get(k, 0.0) + v
            n_batches += 1

            _maybe_log_batch_timing(
                batch_idx=batch_idx,
                epoch_global=epoch_global,
                train=train,
                dt_dataloader=(t_after_next - t_before_next),
                dt_move=(t1 - t0),
                dt_forward=(t2 - t1),
                dt_backward=(t3 - t2) if train else None,
                dt_opt=(t4 - t3) if train else None,
            )

            # Update t before next loading
            t_before_next = time.perf_counter()

    avg_loss = running_loss / max(1, n_batches)
    avg_term_logs = {k: v / max(1, n_batches) for k, v in running_term_logs.items()}

    return avg_loss, avg_term_logs, global_step
