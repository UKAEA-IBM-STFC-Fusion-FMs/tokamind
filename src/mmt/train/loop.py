"""
loop.py — High-level train loop for the Multi-Modal Transformer (MMT)

This module orchestrates the complete finetuning flow of the MMT model,
including:

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

    No config validation is performed here.

-------------------------------------------------------------------------------
DATASET: CACHED AND STREAMED BEHAVIOUR
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

-------------------------------------------------------------------------------
The `history` object tracks structured per-epoch train statistics and is
returned to the caller for logging, visualization, or experiment tracking.
-------------------------------------------------------------------------------
"""

from __future__ import annotations
import logging
import math
import os
from typing import Dict, Any, cast

import torch

from mmt.models.mmt import MultiModalTransformer
from mmt.train.loop_utils import (
    backbone_lr,
    log_train_setup,
    run_one_epoch,
)
from mmt.train.scheduler import (
    build_optimizer_and_scheduler,
    apply_stage_freeze_policy,
)
from mmt.checkpoints import (
    save_best,
    save_latest,
    resume_from_latest,
)
from mmt.utils.amp_utils import get_amp_config


logger = logging.getLogger("mmt.Train")


# ------------------------------------------------------------------
# Entry point: train_finetune()
# ------------------------------------------------------------------


def train_finetune(
    model: MultiModalTransformer,
    train_loader,
    val_loader,
    *,
    run_dir: str,
    train_cfg: Dict[str, Any],
    loader_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Finetune the MMT model using the (already-validated) train configuration.

    Parameters
    ----------
    model : MultiModalTransformer
    train_loader, val_loader : DataLoader
        DataLoaders returned by initialize_mmt_dataloader()
    run_dir : str
        Directory used for checkpoints and logs.
    train_cfg : dict
        The validated train configuration.
    loader_cfg : dict
        The validated loader configuration.

    Returns
    -------
    Dict[str, Any]
        {
            "history": <structured history dictionary>,
            "best_val": float,
            "epochs_run": int,
            "global_step": int
        }
    """
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Extract fields from train_cfg
    # ------------------------------------------------------------------
    stages = train_cfg["stages"]
    resume_flag = train_cfg["resume"]
    early_patience = int(train_cfg["early_stop"]["patience"])
    early_delta = float(train_cfg["early_stop"]["delta"])

    output_weights = train_cfg["loss"]["output_weights"] or {}
    # NOTE: config defines loss weights by *signal name*; model preds are keyed by signal_id (int).
    # Resolve name -> signal_id once here (same convention used for output adapters).
    _ow_cfg = output_weights
    output_weights = {}
    if isinstance(_ow_cfg, dict) and _ow_cfg:
        name_to_sid = {
            spec.name: spec.signal_id for spec in getattr(model, "output_specs", [])
        }
        unknown = [k for k in _ow_cfg.keys() if str(k) not in name_to_sid]
        if unknown:
            raise KeyError(
                f"Unknown output_weights keys: {unknown}. "
                f"Expected output signal names among: {sorted(name_to_sid.keys())}"
            )
        for name, w in _ow_cfg.items():
            output_weights[int(name_to_sid[str(name)])] = float(w)
        # Overwrite train_cfg for consistent logging downstream
        train_cfg["loss"]["output_weights"] = output_weights

    use_adamw = train_cfg["optimizer"]["use_adamw"]

    amp_enabled = train_cfg.get("amp", {}).get("enable", True)

    # Determine batches per epoch (streaming vs cached)
    bpe = loader_cfg.get("batches_per_epoch", None)
    if bpe is not None:
        train_batches_per_epoch = int(cast(Any, bpe))
    else:
        # For cached datasets, infer from dataloader length
        try:
            train_batches_per_epoch = len(train_loader)
        except TypeError:
            # Streaming dataset without batches_per_epoch specified
            raise ValueError(
                "Streaming dataset detected (no len(train_loader)), but "
                "loader.batches_per_epoch is not set. Please specify "
                "loader.batches_per_epoch in your config for streaming datasets."
            )

    # ------------------------------------------------------------------
    # Device, AMP, scaler
    # ------------------------------------------------------------------
    device, amp_enabled, amp_dtype = get_amp_config(model, enable=amp_enabled)
    use_scaler = device.type == "cuda" and amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    logger.info("AMP enabled=%s dtype=%s scaler=%s", amp_enabled, amp_dtype, use_scaler)

    # ------------------------------------------------------------------
    # Initial reporting
    # ------------------------------------------------------------------
    log_train_setup(
        model,
        device,
        amp_enabled,
        amp_dtype,
        train_batches_per_epoch,
        stages,
        train_cfg,
    )

    # ------------------------------------------------------------------
    # Resume metadata (if resuming)
    # ------------------------------------------------------------------
    global_step = 0
    best_val = float("inf")
    bad_epochs = 0
    start_stage_idx = 0
    start_epoch_in_stage = 1

    if resume_flag:
        try:
            # Load model weights and resume metadata (optimizer/scheduler/scaler restored later per stage)
            start_epoch_global, best_so_far, meta = resume_from_latest(
                run_dir,
                model,
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
            logger.warning(
                f"[resume] Failed to resume from latest checkpoint: {e!s}. "
                f"Starting from scratch."
            )
            resume_flag = False

    # ------------------------------------------------------------------
    # History structure
    # ------------------------------------------------------------------
    history: Dict[str, Any] = {"stages": {}}

    # ------------------------------------------------------------------
    # Stage loop
    # ------------------------------------------------------------------
    total_epochs_run = 0

    for stage_idx, stage in enumerate(stages):
        if stage_idx < start_stage_idx:
            total_epochs_run += stage["epochs"]
            continue

        name = stage["name"]
        epochs = int(stage["epochs"])

        # Freeze policy + optim hyperparameters
        freeze_cfg = stage["freeze"]
        lr_cfg = stage["optimizer"]["lr"]
        wd_cfg = stage["optimizer"]["wd"]

        grad_accum_steps = int(stage["scheduler"]["grad_accum_steps"])
        warmup_frac = float(stage["scheduler"].get("warmup_steps_fraction", 0.0))
        warmup_frac = float(max(0.0, min(1.0, warmup_frac)))

        lr_token_encoder = float(lr_cfg["token_encoder"])
        lr_backbone = float(lr_cfg["backbone"])
        lr_modality_heads = float(lr_cfg["modality_heads"])
        lr_output_adapters = float(lr_cfg["output_adapters"])

        wd_token_encoder = float(wd_cfg["token_encoder"])
        wd_backbone = float(wd_cfg["backbone"])
        wd_modality_heads = float(wd_cfg["modality_heads"])
        wd_output_adapters = float(wd_cfg["output_adapters"])

        # ---- Stage steps / scheduler steps ----
        steps_per_epoch = math.ceil(train_batches_per_epoch / max(1, grad_accum_steps))
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(round(warmup_frac * total_steps))

        # ---- Stage freezing ----
        apply_stage_freeze_policy(
            model,
            freeze_token_encoder=freeze_cfg["token_encoder"],  # NEW
            freeze_backbone=freeze_cfg["backbone"],
            freeze_modality_heads=freeze_cfg["modality_heads"],
            freeze_output_adapters=freeze_cfg["output_adapters"],
        )

        # ---- New optimizer + scheduler per stage ----
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr_token_encoder=lr_token_encoder,
            lr_backbone=lr_backbone,
            lr_modality_heads=lr_modality_heads,
            lr_output_adapters=lr_output_adapters,
            wd_token_encoder=wd_token_encoder,
            wd_backbone=wd_backbone,
            wd_modality_heads=wd_modality_heads,
            wd_output_adapters=wd_output_adapters,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            use_adamw=use_adamw,
        )

        # ---- Resume optimizer/scheduler/scaler state if resuming in this stage ----
        if resume_flag and stage_idx == start_stage_idx:
            try:
                # Restore optimizer/scheduler/scaler state (model already loaded above)
                _, _, _ = resume_from_latest(
                    run_dir,
                    model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    map_location=str(device),
                    load_model=False,  # Skip model loading (already done)
                )
                logger.info(
                    f"[resume] Restored optimizer, scheduler, and scaler state for stage '{name}'"
                )
            except Exception as e:
                logger.warning(
                    f"[resume] Failed to restore optimizer/scheduler/scaler state: {e!s}. "
                    f"Continuing with fresh optimizer/scheduler."
                )

        logger.info(f"----- Stage '{name}' (index {stage_idx}) -----")
        logger.info(f"  epochs={epochs}, grad_accum={grad_accum_steps}")
        logger.info(f"  total_steps={total_steps}, warmup_steps={warmup_steps}")

        # Create history list for this stage
        history["stages"][name] = []

        # ------------------------------------------------------------------
        # Epoch loop
        # ------------------------------------------------------------------
        for epoch_in_stage in range(
            start_epoch_in_stage if stage_idx == start_stage_idx else 1,
            epochs + 1,
        ):
            epoch_global = total_epochs_run + epoch_in_stage
            # ---------------------------- TRAIN ----------------------------
            train_loss, global_step = run_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device=device,
                amp_enabled=amp_enabled,
                output_weights=output_weights,
                grad_accum_steps=grad_accum_steps,
                train=True,
                global_step=global_step,
                max_batches=train_batches_per_epoch,
                epoch_global=epoch_global,
            )

            # ---------------------------- VALIDATION -----------------------
            val_loss, _ = run_one_epoch(
                model,
                val_loader,
                optimizer=None,
                scheduler=None,
                scaler=None,
                device=device,
                amp_enabled=amp_enabled,
                output_weights=output_weights,
                grad_accum_steps=1,
                train=False,
                global_step=global_step,
                max_batches=None,  # always full validation
                epoch_global=epoch_global,
            )

            # ---------------------------- BEST CHECKPOINT ------------------
            improved = (val_loss + early_delta) < best_val
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

            bb_lr = backbone_lr(optimizer)

            # ---------------------------- HISTORY UPDATE -------------------
            history["stages"][name].append(
                {
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
            )

            # ---------------------------- LATEST CHECKPOINT ---------------
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
                    "stage_name": name,
                    "epoch_in_stage": epoch_in_stage,
                },
            )

            # ---------------------------- EARLY STOP -----------------------
            if 0 < early_patience <= bad_epochs:
                logger.info(
                    f"[early_stop] Patience exhausted after {bad_epochs} epochs."
                )
                total_epochs_run += epoch_in_stage

                history["best_val"] = best_val
                history["epochs_run"] = total_epochs_run
                history["global_step"] = global_step
                return history

        total_epochs_run += epochs
        start_epoch_in_stage = 1

    # ------------------------------------------------------------------
    # All stages done
    # ------------------------------------------------------------------
    history["best_val"] = best_val
    history["epochs_run"] = total_epochs_run
    history["global_step"] = global_step
    logger.info(
        f"Train finished: epochs_run={total_epochs_run}, best_val={best_val:.6f}"
    )

    return history
