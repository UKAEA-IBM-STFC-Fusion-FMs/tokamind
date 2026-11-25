"""
loop.py — High-level train loop for the Multi-Modal Transformer (MMT)

This module orchestrates the complete finetuning flow of the MMT model, including:

    • Multi-stage train (warm, main, transfer, etc.)
    • Freezing/unfreezing backbone and modality-specific components
    • Per-stage optimizers and LR schedulers (cosine + warmup)
    • AMP train, gradient accumulation
    • Best and latest checkpointing
    • Strict resume of interrupted runs
    • Early stopping
    • Full evaluation pass per epoch

IMPORTANT:
    This module assumes configuration validity has already been checked by:

        from mmt.utils.config import validate_config
        validate_config(cfg.raw)

    Only runtime-context checks are performed here, such as resolving loss output names against model output specs.

-------------------------------------------------------------------------------
DATASET: CACHED AND STREAMED BEHAVIOR
-------------------------------------------------------------------------------

MMT supports two dataset regimes:

1) **Cached mode** (WindowCachedDataset)
   - Map-style dataset
   - len(dataloader) == true number of batches
   - Epoch = full pass over all windows

2) **Streaming mode** (WindowStreamedDataset)
   - IterableDataset yielding windows sequentially
   - __len__ returns number of shots, NOT windows
   - True number of windows is unknown without a full pre-scan
   - Therefore len(dataloader) CANNOT be used as epoch length
   - Epoch length must be defined via:

         loader.streaming.batches_per_epoch

   - Train stops after this many batches
   - Validation ALWAYS exhausts the dataloader

------------------------------------------------------------------------------------------------------------------------
The `history` object tracks structured per-epoch train statistics and is returned to the caller for logging,
visualization, or experiment tracking.
------------------------------------------------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader

from mmt.train.loop_utils import (
    backbone_lr,
    build_loss_aggregator_for_stage,
    canonical_loss_cfg,
    effective_stage_loss_cfg,
    log_train_setup,
    run_one_epoch,
)
from mmt.train.scheduler import build_optimizer_and_scheduler, apply_stage_freeze_policy
from mmt.checkpoints import save_best, save_latest, resume_from_latest
from mmt.utils.amp_utils import get_amp_config


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Train")


# ======================================================================================================================
# Entry point: train_finetune()
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def train_finetune(  # NOSONAR - Ignore cognitive complexity
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    run_dir: str,
    train_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    output_decoders: dict | None = None,
    signal_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Finetune the MMT model using the (already-validated) train configuration.

    Parameters
    ----------
    model : torch.nn.Module
        MMT-compatible model to be fine-tuned.
    train_loader : DataLoader
        Training DataLoader returned by initialize_mmt_dataloader().
    val_loader : DataLoader
        Validation DataLoader returned by initialize_mmt_dataloader().
    run_dir : str
        Directory used for checkpoints and logs.
    train_cfg : Mapping[str, Any]
        The validated train configuration.
    loader_cfg : Mapping[str, Any]
        The validated loader configuration.
    output_decoders : dict[int, TorchDecoder] | None
        Pre-built TorchDecoder instances keyed by signal_id, required when any loss term in
        ``train_cfg["loss"]["terms"]`` has ``requires_decode=True`` (e.g., ``native_sparse_mse``). Pass the result of
        ``build_decoders()`` from ``codec_utils``.
         Optional. Default: None.
    signal_stats : Mapping[str, Mapping[str, Any]] | None
        Per-signal mean/std metadata, required by losses that operate in destandardized native units.
        Optional. Default: None.

    Returns
    -------
    dict[str, Any]
        {
            "history": <structured history dictionary>,
            "best_val": float,
            "epochs_run": int,
            "global_step": int
        }

    Raises
    ------
    KeyError
        Unknown output_weights keys in `model["output_specs"]`.
    ValueError
        Streaming dataset detected, but `loader_cfg.batches_per_epoch` is not set.

    """

    os.makedirs(run_dir, exist_ok=True)

    # ..................................................................................................................
    # Extract fields from train_cfg
    # ..................................................................................................................

    stages = train_cfg["stages"]
    resume_flag = train_cfg["resume"]
    early_patience = int(train_cfg["early_stop"]["patience"])
    early_delta = float(train_cfg["early_stop"]["delta"])

    output_specs = list(getattr(model, "output_specs", []))
    output_name_to_id = {str(spec.name): int(spec.signal_id) for spec in output_specs}

    stage_loss_cfgs = [effective_stage_loss_cfg(train_cfg=train_cfg, stage_cfg=stage) for stage in stages]
    stage_loss_keys = [canonical_loss_cfg(loss_cfg) for loss_cfg in stage_loss_cfgs]
    loss_aggregators = [
        build_loss_aggregator_for_stage(
            loss_cfg=stage_loss_cfgs[stage_index],
            output_specs=output_specs,
            output_name_to_id=output_name_to_id,
            output_decoders=output_decoders,
            signal_stats=signal_stats,
            path=f"train.stages[{stage_index}].loss" if "loss" in stage else "train.loss",
            require_all_outputs="loss" not in stage,
        )
        for stage_index, stage in enumerate(stages)
    ]

    use_adamw = train_cfg["optimizer"]["use_adamw"]

    amp_enabled = train_cfg.get("amp", {}).get("enable", True)

    # Determine batches per epoch (streaming vs cached).
    bpe = loader_cfg.get("batches_per_epoch", None)
    if bpe is not None:
        train_batches_per_epoch = int(cast(Any, bpe))
    else:
        # For cached datasets, infer from dataloader length.
        try:
            train_batches_per_epoch = len(train_loader)
        except TypeError:
            # Streaming dataset without batches_per_epoch specified.
            raise ValueError(
                "Streaming dataset detected (no len(train_loader)), but loader.batches_per_epoch is not set. Please "
                "specify loader.batches_per_epoch in your config for streaming datasets."
            )

    # ..................................................................................................................
    # Device, AMP, scaler
    # ..................................................................................................................

    device, amp_enabled, amp_dtype = get_amp_config(model=model, enable=amp_enabled)
    use_scaler = (device.type == "cuda") and amp_enabled and (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_scaler)

    if output_decoders:
        for _d in output_decoders.values():
            _d.to(device).eval()

    logger.info("AMP enabled=%s dtype=%s scaler=%s", amp_enabled, amp_dtype, use_scaler)

    # ..................................................................................................................
    # Initial reporting
    # ..................................................................................................................

    log_train_setup(
        model=model,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        train_loader_len=train_batches_per_epoch,
        stages=stages,
        train_cfg=train_cfg,
    )

    # ..................................................................................................................
    # Resume metadata (if resuming)
    # ..................................................................................................................

    global_step = 0
    best_val = float("inf")
    bad_epochs = 0
    start_stage_idx = 0
    start_epoch_in_stage = 1

    if resume_flag:
        try:
            # Load model weights and resume metadata (optimizer/scheduler/scaler restored later per stage)
            start_epoch_global, best_so_far, meta = resume_from_latest(  # NOSONAR - Unused variable
                run_dir=run_dir,
                model=model,
                optimizer=None,
                scheduler=None,
                scaler=None,
                map_location=str(device),
            )
            best_val = float(best_so_far)
            global_step = int(meta.get("global_step", 0))
            bad_epochs = int(meta.get("bad_epochs", 0))
            start_stage_idx = int(meta.get("stage_index", 0))
            last_epoch_in_stage = int(meta.get("epoch_in_stage", 0))
            start_epoch_in_stage = last_epoch_in_stage + 1
            if start_epoch_in_stage < 1:
                start_epoch_in_stage = 1

            logger.info(
                f"[resume] Loaded model weights and metadata: stage_idx={start_stage_idx}, "
                f"last_epoch_in_stage={last_epoch_in_stage}, "
                f"next_epoch_in_stage={start_epoch_in_stage}, "
                f"best_val={best_val:.6f}, global_step={global_step}"
            )
        except Exception as e:
            logger.warning(f"[resume] Failed to resume from latest checkpoint: {e!s}. Starting from scratch.")
            resume_flag = False

    # ..................................................................................................................
    # History structure
    # ..................................................................................................................

    history: dict[str, Any] = {"stages": {}}

    # ..................................................................................................................
    # Stage loop
    # ..................................................................................................................

    total_epochs_run = 0

    for stage_idx, stage in enumerate(stages):
        if stage_idx < start_stage_idx:
            total_epochs_run += stage["epochs"]
            continue

        name = stage["name"]
        epochs = int(stage["epochs"])

        # ---- Freeze policy + optim hyperparameters ----
        freeze_cfg = stage["freeze"]
        lr_cfg = stage["optimizer"]["lr"]
        wd_cfg = stage["optimizer"]["wd"]
        loss_aggregator = loss_aggregators[stage_idx]

        grad_accum_steps = int(stage["scheduler"]["grad_accum_steps"])
        warmup_frac = float(stage["scheduler"].get("warmup_steps_fraction", 0.0))
        warmup_frac = float(max(0.0, min(1.0, warmup_frac)))

        lr_by_block = {str(k): float(v) for k, v in lr_cfg.items()}
        wd_by_block = {str(k): float(v) for k, v in wd_cfg.items()}
        freeze_by_block = {str(k): bool(v) for k, v in freeze_cfg.items()}

        # ---- Stage steps / scheduler steps ----
        steps_per_epoch = math.ceil(train_batches_per_epoch / max(1, grad_accum_steps))
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(round(warmup_frac * total_steps))

        # ---- Stage freezing ----
        apply_stage_freeze_policy(
            model=model,
            freeze_by_block=freeze_by_block,
        )

        # ---- New optimizer + scheduler per stage ----
        optimizer, scheduler = build_optimizer_and_scheduler(
            model=model,
            lr_by_block=lr_by_block,
            wd_by_block=wd_by_block,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            use_adamw=use_adamw,
        )

        # ---- Resume optimizer/scheduler/scaler state if resuming in this stage ----
        if resume_flag and stage_idx == start_stage_idx:
            try:
                # Restore optimizer/scheduler/scaler state (model already loaded above).
                _, _, _ = resume_from_latest(
                    run_dir=run_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    map_location=str(device),
                    load_model=False,  # Skip model loading (already done).
                )
                logger.info(f"[resume] Restored optimizer, scheduler, and scaler state for stage '{name}'")
            except Exception as e:
                logger.warning(
                    f"[resume] Failed to restore optimizer/scheduler/scaler state: {e!s}. "
                    f"Continuing with fresh optimizer/scheduler."
                )

        logger.info(f"----- Stage '{name}' (index {stage_idx}) -----")
        logger.info(f"  epochs={epochs}, grad_accum={grad_accum_steps}")
        logger.info(f"  total_steps={total_steps}, warmup_steps={warmup_steps}")

        # A new stage can optimize a different loss scale, so compare validation only within matching loss configs.
        first_epoch_in_stage = start_epoch_in_stage if stage_idx == start_stage_idx else 1
        if stage_idx > 0 and first_epoch_in_stage <= 1 and stage_loss_keys[stage_idx] != stage_loss_keys[stage_idx - 1]:
            logger.info(
                "  loss config changed from previous stage; resetting best validation and early-stop counter "
                "for this stage."
            )
            # After a reset, best.pt tracks the best checkpoint within the current loss regime.
            best_val = float("inf")
            bad_epochs = 0

        # Create history list for this stage
        history["stages"][name] = []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Epoch loop
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for epoch_in_stage in range(
            first_epoch_in_stage,
            epochs + 1,
        ):
            epoch_global = total_epochs_run + epoch_in_stage

            # ---------------------------- TRAIN ----------------------------
            train_loss, train_term_logs, global_step = run_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                amp_enabled=amp_enabled,
                loss_aggregator=loss_aggregator,
                grad_accum_steps=grad_accum_steps,
                train=True,
                global_step=global_step,
                max_batches=train_batches_per_epoch,
                epoch_global=epoch_global,
                epoch_in_stage=epoch_in_stage,
                stage_index=stage_idx,
                stage_name=name,
                run_dir=run_dir,
            )

            # ---------------------------- VALIDATION -----------------------
            val_loss, val_term_logs, _ = run_one_epoch(
                model=model,
                loader=val_loader,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
                amp_enabled=amp_enabled,
                loss_aggregator=loss_aggregator,
                grad_accum_steps=1,
                train=False,
                global_step=global_step,
                max_batches=None,  # Always full validation
                epoch_global=epoch_global,
                epoch_in_stage=epoch_in_stage,
                stage_index=stage_idx,
                stage_name=name,
                run_dir=run_dir,
            )

            # ---------------------------- BEST CHECKPOINT ------------------
            improved = (val_loss + early_delta) < best_val
            if improved:
                best_val = val_loss
                bad_epochs = 0

                save_best(
                    run_dir=run_dir,
                    model=model,
                    epoch=epoch_global,
                    best_val=best_val,
                    extra_meta={
                        "stage_index": stage_idx,
                        "stage_name": name,
                        "epoch_in_stage": epoch_in_stage,
                    },
                )

            else:
                bad_epochs += 1

            # ---------------------------- EPOCH LOG ------------------------
            if early_patience > 0:
                no_improve_str = f"{bad_epochs}/{early_patience}"
            else:
                no_improve_str = f"{bad_epochs}"

            logger.info(
                f"Stage {name} | Epoch {epoch_in_stage}/{epochs} "
                f"(global={epoch_global}) | step={global_step} | "
                f"train={train_loss:.6f}, val={val_loss:.6f}, best={best_val:.6f} | "
                f"no_improve={no_improve_str}"
            )

            # Per-term gradient share (only when multiple terms are active).
            # Each percentage = w_i * L_i / Σ(w_j * L_j): the actual gradient contribution
            # after applying term weights. 50%/50% means equal gradient pull.
            if len(train_term_logs) > 1:

                def _fmt_term_pcts(d: dict) -> str:
                    w_sum = sum(d.values())
                    parts = []
                    for k_, v_ in sorted(d.items()):
                        name_ = re.sub(r"_\d+/weighted$", "", k_)
                        pct = 100.0 * v_ / w_sum if w_sum > 0.0 else 0.0
                        parts.append(f"{name_}={pct:.0f}%")
                    return "  ".join(parts)

                logger.info("  terms train: %s", _fmt_term_pcts(train_term_logs))
                logger.info("  terms val:   %s", _fmt_term_pcts(val_term_logs))

            bb_lr = backbone_lr(optimizer=optimizer)

            # ---------------------------- HISTORY UPDATE -------------------
            epoch_record: dict = {
                "epoch_global": epoch_global,
                "epoch_in_stage": epoch_in_stage,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr_backbone": bb_lr,
                "best_val": best_val,
                "global_step": global_step,
                "bad_epochs": bad_epochs,
                "improved": improved,
            }
            for k, v in train_term_logs.items():
                epoch_record[f"train_{k}"] = v
            for k, v in val_term_logs.items():
                epoch_record[f"val_{k}"] = v
            history["stages"][name].append(epoch_record)

            # ---------------------------- LATEST CHECKPOINT ---------------
            save_latest(
                run_dir=run_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch_global,
                global_step=global_step,
                best_val_so_far=best_val,
                bad_epochs=bad_epochs,
                extra_meta={
                    "stage_index": stage_idx,
                    "stage_name": name,
                    "epoch_in_stage": epoch_in_stage,
                },
            )

            # ---------------------------- EARLY STOP -----------------------
            if 0 < early_patience <= bad_epochs:
                logger.info(f"[early_stop] Patience exhausted after {bad_epochs} epochs.")
                total_epochs_run += epoch_in_stage

                history["best_val"] = best_val
                history["epochs_run"] = total_epochs_run
                history["global_step"] = global_step

                return history

        total_epochs_run += epochs
        start_epoch_in_stage = 1

    # ..................................................................................................................
    # All stages done
    # ..................................................................................................................

    history["best_val"] = best_val
    history["epochs_run"] = total_epochs_run
    history["global_step"] = global_step
    logger.info(f"Train finished: epochs_run={total_epochs_run}, best_val={best_val:.6f}")

    return history

    # ..................................................................................................................
