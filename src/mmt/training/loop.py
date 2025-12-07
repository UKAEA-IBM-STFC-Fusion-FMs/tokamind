from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Hashable

import torch
from torch import Tensor

from mmt.models.mmt import MultiModalTransformer

from .amp_utils import get_amp_config, amp_ctx_for_model
from .scheduler import (
    build_optimizer_and_scheduler,
    toggle_param_groups,
    apply_stage_freeze_policy,
)
from .checkpoint_io import save_best, save_latest, resume_from_latest
from .losses import compute_loss_pred_space

logger = logging.getLogger("mmt.Training")

# Max gradient norm for clipping
_MAX_GRAD_NORM = 1.0


# ======================================================================
# Small helpers
# ======================================================================


def _move_batch_to_device(
    batch: Dict[str, Any], device: torch.device
) -> Dict[str, Any]:
    """
    Move all tensor components of the collated batch to the given device.

    Expected structure (from MMTCollate):
      - "emb"            : List[List[Tensor]]
      - "pos", "id", "mod", "role" : LongTensor (B, L)
      - "padding_mask"   : BoolTensor (B, L)
      - "input_mask",
        "actuator_mask"  : BoolTensor (B, L)
      - "outputs_emb"    : Dict[int, List[Tensor]]  (per-output coeffs)
      - "outputs_mask"   : Dict[int, BoolTensor(B,)]

      - optional:
        "output_native"  : Dict[int, Tensor]

    Non-tensor entries (e.g. "signal_name") are left untouched.
    """
    if device.type == "cpu":
        # DataLoader already yields CPU tensors; nothing to do
        return batch

    def _to(t: Tensor) -> Tensor:
        return t.to(device, non_blocking=True)

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

    # Outputs: coeff space embeddings
    outputs_emb = batch.get("outputs_emb", None)
    if isinstance(outputs_emb, dict):
        new_oe: Dict[Hashable, list[Tensor]] = {}
        for k, v_list in outputs_emb.items():
            if isinstance(v_list, list):
                new_oe[k] = [_to(t) for t in v_list]
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


def _active_outputs_from_mask(
    outputs_mask: Mapping[Hashable, Tensor],
) -> set[Hashable]:
    """
    Determine which outputs are actually supervised in this batch.

    An output is considered "active" if its BoolTensor mask has at least
    one True entry (i.e., at least one sample carries a label).
    """
    active: set[Hashable] = set()
    for out_key, m in outputs_mask.items():
        if m.dtype != torch.bool:
            raise RuntimeError(
                f"outputs_mask[{out_key!r}] must be bool tensor, got {m.dtype}."
            )
        if bool(m.any()):
            active.add(out_key)
    return active


def _get_backbone_lr(optimizer: torch.optim.Optimizer) -> Optional[float]:
    """
    Convenience helper for logging: read current LR of the backbone group.
    """
    for g in optimizer.param_groups:
        if g.get("group_type") == "backbone":
            return float(g.get("lr", 0.0))
    return None


def _log_train_setup(
    model: MultiModalTransformer,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: Optional[torch.dtype],
    train_loader_len: int,
    stages: list[Dict[str, Any]],
    training_cfg: Dict[str, Any],
) -> None:
    """
    Compact, logger-based summary of the training setup.
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
            "  - %(name)s: epochs=%(epochs)s, freeze_backbone=%(freeze_backbone)s, "
            "freeze_modality_heads=%(freeze_modality_heads)s, "
            "freeze_output_adapters=%(freeze_output_adapters)s, "
            "lr_backbone=%(lr_backbone)s, lr_modality_heads=%(lr_modality_heads)s, "
            "lr_output_adapters=%(lr_output_adapters)s, grad_accum_steps=%(grad_accum_steps)s",
            s,
        )

    patience = int(training_cfg.get("early_stop_patience", 0))
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))
    logger.info("Early stopping: patience=%d, min_delta=%.3g", patience, min_delta)
    logger.info("====================================")


# ======================================================================
# Epoch runner
# ======================================================================


def _run_one_epoch(
    model: MultiModalTransformer,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
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
    Run a single epoch over `loader` in either train or eval mode.

    Returns
    -------
    avg_loss : float
        Average loss over batches (in coeff space, as returned by
        `compute_loss_pred_space`).

    global_step : int
        Updated global optimizer-step counter (unchanged in eval mode).
    """
    if train and optimizer is None:
        raise ValueError("optimizer must be provided when train=True.")

    model.train(mode=train)
    if not train:
        torch.set_grad_enabled(False)

    running_loss = 0.0
    n_batches = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        batch = _move_batch_to_device(batch, device)

        outputs_mask: Mapping[Hashable, Tensor] = batch["outputs_mask"]

        if train:
            # Automatic per-output + per-modality toggling
            active_outputs = _active_outputs_from_mask(outputs_mask)
            toggle_param_groups(
                optimizer,
                active_outputs=active_outputs,
                output_to_modality=output_to_modality,
            )

        # Forward + loss under AMP context
        with amp_ctx_for_model(model, enable=amp_enabled):
            out = model(batch)
            preds: Mapping[Hashable, Tensor] = out.get("pred", {})

            loss_t, _logs = compute_loss_pred_space(
                preds=preds,
                y_true=batch["outputs_emb"],
                outputs_mask=outputs_mask,
                output_weights=output_weights,
            )

            if train and grad_accum_steps > 1:
                loss_for_backprop = loss_t / float(grad_accum_steps)
            else:
                loss_for_backprop = loss_t

        if train:
            # Backward with optional GradScaler
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss_for_backprop).backward()
            else:
                loss_for_backprop.backward()

            # Optimiser step every `grad_accum_steps` batches
            if (batch_idx + 1) % grad_accum_steps == 0:
                if scaler is not None and scaler.is_enabled():
                    # Unscale, then clip, then step
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=_MAX_GRAD_NORM
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=_MAX_GRAD_NORM
                    )
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

        # Accumulate loss *as returned* by compute_loss_pred_space
        running_loss += float(loss_t.detach().cpu())
        n_batches += 1

    avg_loss = running_loss / max(1, n_batches)

    if not train:
        torch.set_grad_enabled(True)

    return avg_loss, global_step


# ======================================================================
# Public training entry point (finetune)
# ======================================================================


def train_finetune(
    model: MultiModalTransformer,
    train_loader,
    val_loader,
    *,
    run_dir: str,
    training_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Finetuning loop for the Multi-Modal Transformer.

    This function is designed to work with the YAML config structure of
    `finetune_default.yaml`, in particular:

        training:
          resume: false
          epochs: 50                # total epochs (informational)
          grad_accum_steps: 2       # default, can be overridden per-stage
          early_stop_patience: 5
          early_stop_min_delta: 0.0
          loss:
            output_weights: {}      # per-output weights (optional)
          optimizer:
            lr:
              backbone: 5e-4
              modality_heads: 2e-3
              output_adapters: 2e-3
            wd:
              backbone: 0.05
              modality_heads: 0.05
              output_adapters: 0.05
            use_adamw: true
          scheduler:
            warmup_steps_fraction: 0.02
          stages:
            - name: ft_warm
              epochs: 5
              freeze_backbone: true
              freeze_modality_heads: false
              freeze_output_adapters: false
              lr_backbone: 5e-4
              lr_modality_heads: 2e-3
              lr_output_adapters: 2e-3
              wd_backbone: 0.05
              wd_modality_heads: 0.05
              wd_output_adapters: 0.05
              grad_accum_steps: 2
            - name: ft_main
              ...

    Behaviour
    ---------
    - Uses stage-level configs for coarse freezing and LR/WD choices.
    - Within each stage:
        * applies `apply_stage_freeze_policy` once,
        * builds a fresh optimizer + scheduler,
        * runs for `stage["epochs"]` epochs with optional grad accumulation.
    - Each epoch:
        * trains on `train_loader`,
        * evaluates on `val_loader`,
        * updates checkpoints:
              - `save_best` on validation improvement,
              - `save_latest` every epoch.
    - Early stopping:
        * controlled by `early_stop_patience` + `early_stop_min_delta`.
    - Resume:
        * if `training.resume` is True, we try `resume_from_latest` and
          continue from there (model + RNG); optimizer/scheduler are
          rebuilt per-stage, so we ignore their saved state.

    Returns
    -------
    history : dict
        Dictionary with keys like:
          - "best_val"
          - "epochs_run"
          - "global_step"
    """
    os.makedirs(run_dir, exist_ok=True)

    stages = list(training_cfg.get("stages", []))
    if not stages:
        raise ValueError("training_cfg['stages'] must contain at least one stage.")

    # Fill in per-stage defaults from the global optimizer section if missing
    opt_cfg = training_cfg.get("optimizer", {})
    lr_default = opt_cfg.get("lr", {})
    wd_default = opt_cfg.get("wd", {})
    grad_accum_default = int(training_cfg.get("grad_accum_steps", 1))

    for s in stages:
        s.setdefault("name", "stage")
        s.setdefault("grad_accum_steps", grad_accum_default)
        s.setdefault("lr_backbone", lr_default.get("backbone", 1e-3))
        s.setdefault("lr_modality_heads", lr_default.get("modality_heads", 2e-3))
        s.setdefault("lr_output_adapters", lr_default.get("output_adapters", 2e-3))
        s.setdefault("wd_backbone", wd_default.get("backbone", 1e-4))
        s.setdefault("wd_modality_heads", wd_default.get("modality_heads", 0.0))
        s.setdefault("wd_output_adapters", wd_default.get("output_adapters", 0.0))

        for flag in (
            "freeze_backbone",
            "freeze_modality_heads",
            "freeze_output_adapters",
        ):
            if flag not in s:
                s[flag] = False

    # AMP config + device
    device, amp_enabled, amp_dtype = get_amp_config(model, enable=True)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and amp_enabled))

    # Logging initial setup
    _log_train_setup(
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        train_loader_len=len(train_loader),
        stages=stages,
        training_cfg=training_cfg,
    )

    # Early stopping
    patience = int(training_cfg.get("early_stop_patience", 0))
    min_delta = float(training_cfg.get("early_stop_min_delta", 0.0))

    # Loss weights
    loss_cfg = training_cfg.get("loss", {})
    output_weights: Dict[Hashable, float] = loss_cfg.get("output_weights", {}) or {}

    # Scheduler warmup
    sched_cfg = training_cfg.get("scheduler", {})
    warmup_frac = float(sched_cfg.get("warmup_steps_fraction", 0.0))
    warmup_frac = max(0.0, min(1.0, warmup_frac))

    # Resume (model + RNG only; optimizer/scheduler rebuilt per-stage)
    resume_flag = bool(training_cfg.get("resume", False))
    global_step = 0
    best_val = math.inf
    bad_epochs = 0
    start_stage_idx = 0
    start_epoch_in_stage = 1

    if resume_flag:
        try:
            start_epoch_global, best_val_so_far, meta = resume_from_latest(
                run_dir,
                model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location=device,
            )
            best_val = float(best_val_so_far)
            global_step = int(meta.get("global_step", 0))
            bad_epochs = int(meta.get("bad_epochs", 0))
            start_stage_idx = int(meta.get("stage_index", 0))
            start_epoch_in_stage = int(meta.get("epoch_in_stage", 1))

            logger.info(
                "[resume] Resumed from latest: epoch_global=%d, "
                "stage_index=%d, epoch_in_stage=%d, best_val=%.6f",
                start_epoch_global,
                start_stage_idx,
                start_epoch_in_stage,
                best_val,
            )
        except FileNotFoundError:
            logger.warning(
                "[resume] No latest checkpoint found under %s; starting from scratch.",
                run_dir,
            )
        except Exception as e:
            logger.warning(
                "[resume] Failed to resume from latest checkpoint: %s; "
                "starting from scratch.",
                str(e),
            )

    # Main training loop over stages
    total_epochs_run = 0
    n_batches_per_epoch = len(train_loader)
    history: Dict[str, Any] = {}

    for stage_idx, stage in enumerate(stages):
        if stage_idx < start_stage_idx:
            # This stage was fully completed already
            total_epochs_run += int(stage.get("epochs", 0))
            continue

        stage_name = stage.get("name", f"stage_{stage_idx}")
        stage_epochs = int(stage.get("epochs", 0))
        if stage_epochs <= 0:
            logger.info("Skipping stage %s (epochs <= 0).", stage_name)
            continue

        grad_accum_steps = int(stage.get("grad_accum_steps", grad_accum_default))
        # approximate optimizer steps per epoch (for scheduler)
        steps_per_epoch = math.ceil(n_batches_per_epoch / max(1, grad_accum_steps))
        total_steps = steps_per_epoch * stage_epochs
        warmup_steps = int(round(warmup_frac * total_steps))

        # Stage-level freezing
        apply_stage_freeze_policy(
            model,
            freeze_backbone=bool(stage["freeze_backbone"]),
            freeze_modality_heads=bool(stage["freeze_modality_heads"]),
            freeze_output_adapters=bool(stage["freeze_output_adapters"]),
        )

        # Build optimizer + scheduler for this stage
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr_backbone=float(stage["lr_backbone"]),
            lr_modality_heads=float(stage["lr_modality_heads"]),
            lr_output_adapters=float(stage["lr_output_adapters"]),
            wd_backbone=float(stage["wd_backbone"]),
            wd_modality_heads=float(stage["wd_modality_heads"]),
            wd_output_adapters=float(stage["wd_output_adapters"]),
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            use_adamw=bool(opt_cfg.get("use_adamw", True)),
        )

        logger.info(
            "----- Stage %s (index %d): epochs=%d, grad_accum_steps=%d, "
            "total_steps=%d, warmup_steps=%d -----",
            stage_name,
            stage_idx,
            stage_epochs,
            grad_accum_steps,
            total_steps,
            warmup_steps,
        )

        # Epoch loop within this stage
        for epoch_in_stage in range(
            start_epoch_in_stage if stage_idx == start_stage_idx else 1,
            stage_epochs + 1,
        ):
            epoch_global = total_epochs_run + epoch_in_stage

            # --- TRAIN ---
            train_loss, global_step = _run_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device=device,
                amp_enabled=amp_enabled,
                output_weights=output_weights,
                output_to_modality=model.output2modality,
                grad_accum_steps=grad_accum_steps,
                train=True,
                global_step=global_step,
            )

            # --- VALIDATION ---
            val_loss, _ = _run_one_epoch(
                model,
                val_loader,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
                amp_enabled=amp_enabled,
                output_weights=output_weights,
                output_to_modality=model.output2modality,
                grad_accum_steps=1,
                train=False,
                global_step=global_step,
            )

            bb_lr = _get_backbone_lr(optimizer)
            logger.info(
                "Stage %s | Epoch %d/%d (global=%d) | "
                "train_loss=%.6f, val_loss=%.6f, best_val=%.6f, lr_backbone=%s",
                stage_name,
                epoch_in_stage,
                stage_epochs,
                epoch_global,
                train_loss,
                val_loss,
                best_val,
                f"{bb_lr:.3e}" if bb_lr is not None else "n/a",
            )

            # --- Checkpoint: best ---
            improved = (val_loss + min_delta) < best_val
            if improved:
                best_val = val_loss
                bad_epochs = 0
                save_best(
                    run_dir,
                    model,
                    epoch=epoch_global,
                    best_val=best_val,
                    extra_meta={
                        "stage_index": stage_idx,
                        "stage_name": stage_name,
                        "epoch_in_stage": epoch_in_stage,
                    },
                )
                logger.info(
                    "[checkpoint] New best at epoch_global=%d (stage=%s, epoch_in_stage=%d): val_loss=%.6f",
                    epoch_global,
                    stage_name,
                    epoch_in_stage,
                    best_val,
                )
            else:
                bad_epochs += 1

            # --- Checkpoint: latest ---
            save_latest(
                run_dir,
                model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_global,
                global_step=global_step,
                best_val_so_far=best_val,
                bad_epochs=bad_epochs,
                extra_meta={
                    "stage_index": stage_idx,
                    "stage_name": stage_name,
                    "epoch_in_stage": epoch_in_stage,
                },
            )

            # Early stopping check (across all stages)
            if patience > 0 and bad_epochs >= patience:
                logger.info(
                    "[early_stop] Patience exhausted after %d bad epochs. "
                    "Stopping training.",
                    bad_epochs,
                )
                total_epochs_run += epoch_in_stage
                history.update(
                    {
                        "best_val": best_val,
                        "epochs_run": total_epochs_run,
                        "global_step": global_step,
                    }
                )
                return history

        total_epochs_run += stage_epochs
        # After finishing a stage, next one starts from epoch_in_stage = 1
        start_epoch_in_stage = 1

    history.update(
        {
            "best_val": best_val,
            "epochs_run": total_epochs_run,
            "global_step": global_step,
        }
    )
    logger.info(
        "Training finished: epochs_run=%d, best_val=%.6f, global_step=%d",
        total_epochs_run,
        best_val,
        global_step,
    )
    return history
