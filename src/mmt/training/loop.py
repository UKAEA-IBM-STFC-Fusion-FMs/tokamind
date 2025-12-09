"""
loop.py — High-level training loop for the Multi-Modal Transformer (MMT)

This module coordinates the entire finetuning workflow:

    • Strict validation of the training configuration.
    • Stage-by-stage orchestration (warm, main, transfer, etc.).
    • Stage-level coarse freezing (backbone / mod-heads / adapters).
    • Per-batch automatic freezing via param-group toggling.
    • Warmup+cosine LR scheduling.
    • AMP-safe forward/backward passes.
    • Grad accumulation.
    • Best & latest checkpointing.
    • Strict resume of interrupted runs.
    • Early stopping.

All low-level helpers are implemented in:
    - config_validation.py
    - loop_utils.py
    - scheduler.py
    - checkpoint_io.py
    - losses.py
    - amp_utils.py

The goal of this module is to remain clean, readable, and focused on
the high-level logic of the training pipeline.
"""

from __future__ import annotations
import logging
import math
import os
from typing import Dict, Any

import torch

from mmt.models.mmt import MultiModalTransformer

# Training utilities
from .config_validation import validate_training_config, validate_stage_config
from .loop_utils import (
    backbone_lr,
    log_train_setup,
    run_one_epoch,
)

from .scheduler import (
    build_optimizer_and_scheduler,
    apply_stage_freeze_policy,
)

from .checkpoint_io import (
    save_best,
    save_latest,
    resume_from_latest,
)

from .amp_utils import get_amp_config

logger = logging.getLogger("mmt.Training")


# ======================================================================
# Public API: train_finetune()
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
    Finetune the Multi-Modal Transformer using the NEW strict configuration.

    This entrypoint performs the following:

        1. Validate the training config.
        2. Validate each stage definition.
        3. Optionally resume training (strict).
        4. For each stage:
              - Apply coarse freeze policy.
              - Build optimizer + scheduler.
              - Run epochs with AMP, accumulation, LN scheduling.
              - Save best & latest checkpoints.
              - Early stopping across stages.
        5. Return training history.

    All low-level details (toggling, forward passes, scaler) are handled
    by the imported utilities and not reimplemented here.
    """
    os.makedirs(run_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Strict validation of global training config
    # ------------------------------------------------------------------
    validate_training_config(training_cfg)

    stages = training_cfg["stages"]
    if len(stages) == 0:
        raise ValueError("training.stages must contain at least one stage")

    for stage in stages:
        validate_stage_config(stage)

    # Extract global config
    resume_flag = training_cfg["resume"]
    early_patience = int(training_cfg["early_stop"]["patience"])
    early_delta = float(training_cfg["early_stop"]["delta"])
    output_weights = training_cfg["loss"]["output_weights"] or {}
    use_adamw = training_cfg["optimizer"]["use_adamw"]

    warmup_frac = float(training_cfg["scheduler"]["warmup_steps_fraction"])
    warmup_frac = float(max(0.0, min(1.0, warmup_frac)))

    # ------------------------------------------------------------------
    # 2. AMP + device setup
    # ------------------------------------------------------------------
    device, amp_enabled, amp_dtype = get_amp_config(model, enable=True)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and amp_enabled)
    )

    # ------------------------------------------------------------------
    # 3. Logging initial setup
    # ------------------------------------------------------------------
    log_train_setup(
        model,
        device,
        amp_enabled,
        amp_dtype,
        len(train_loader),
        stages,
        training_cfg,
    )

    # ------------------------------------------------------------------
    # 4. Optional strict resume
    # ------------------------------------------------------------------
    global_step = 0
    best_val = float("inf")
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
                f"[resume] Resumed training: "
                f"stage_idx={start_stage_idx}, "
                f"epoch_in_stage={start_epoch_in_stage}, "
                f"best_val={best_val:.6f}"
            )
        except Exception as e:
            logger.warning(
                f"[resume] Failed to resume from latest: {e!s} — starting from scratch."
            )

    # ==================================================================
    # 5. Main training loop over stages
    # ==================================================================
    total_epochs_run = 0
    history: Dict[str, Any] = {}
    n_batches = len(train_loader)

    for stage_idx, stage in enumerate(stages):
        # --------------------------------------------------------------
        # Skip completed stages if resuming
        # --------------------------------------------------------------
        if stage_idx < start_stage_idx:
            total_epochs_run += stage["epochs"]
            continue

        name = stage["name"]
        epochs = int(stage["epochs"])

        freeze_cfg = stage["freeze"]
        lr_cfg = stage["optimizer"]["lr"]
        wd_cfg = stage["optimizer"]["wd"]

        # Numeric fields: convert at point of use
        grad_accum_steps = int(stage["scheduler"]["grad_accum_steps"])

        lr_backbone = float(lr_cfg["backbone"])
        lr_modality_heads = float(lr_cfg["modality_heads"])
        lr_output_adapters = float(lr_cfg["output_adapters"])

        wd_backbone = float(wd_cfg["backbone"])
        wd_modality_heads = float(wd_cfg["modality_heads"])
        wd_output_adapters = float(wd_cfg["output_adapters"])

        steps_per_epoch = math.ceil(n_batches / max(1, grad_accum_steps))
        total_steps = steps_per_epoch * epochs
        warmup_steps = int(round(warmup_frac * total_steps))

        # --------------------------------------------------------------
        # 5a. Apply coarse freeze policy for this stage
        # --------------------------------------------------------------
        apply_stage_freeze_policy(
            model,
            freeze_backbone=freeze_cfg["backbone"],
            freeze_modality_heads=freeze_cfg["modality_heads"],
            freeze_output_adapters=freeze_cfg["output_adapters"],
        )

        # --------------------------------------------------------------
        # 5b. Build optimizer + scheduler (fresh per stage)
        # --------------------------------------------------------------
        optimizer, scheduler = build_optimizer_and_scheduler(
            model,
            lr_backbone=lr_backbone,
            lr_modality_heads=lr_modality_heads,
            lr_output_adapters=lr_output_adapters,
            wd_backbone=wd_backbone,
            wd_modality_heads=wd_modality_heads,
            wd_output_adapters=wd_output_adapters,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            use_adamw=use_adamw,
        )

        logger.info(
            f"----- Stage '{name}' (index {stage_idx}) "
            f"epochs={epochs}, grad_accum={grad_accum_steps}, "
            f"total_steps={total_steps}, warmup={warmup_steps} -----"
        )

        # --------------------------------------------------------------
        # 5c. Epoch loop
        # --------------------------------------------------------------
        for epoch_in_stage in range(
            start_epoch_in_stage if stage_idx == start_stage_idx else 1,
            epochs + 1,
        ):
            epoch_global = total_epochs_run + epoch_in_stage

            # ------------------------------ TRAIN ------------------------------
            train_loss, global_step = run_one_epoch(
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

            # ------------------------------ VALIDATE ---------------------------
            val_loss, _ = run_one_epoch(
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

            bb_lr = backbone_lr(optimizer)
            bb_lr_str = f"{bb_lr:.3e}" if bb_lr is not None else "n/a"
            logger.info(
                f"Stage {name} | Epoch {epoch_in_stage}/{epochs} "
                f"(global={epoch_global}) "
                f"| train={train_loss:.6f}, val={val_loss:.6f}, "
                f"best={best_val:.6f}, lr_backbone={bb_lr_str}"
            )

            # ------------------------ BEST CHECKPOINT --------------------------
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

                logger.info(
                    f"[checkpoint] New best @ global_epoch {epoch_global}: val={best_val:.6f}"
                )
            else:
                bad_epochs += 1

            # ------------------------ LATEST CHECKPOINT ------------------------
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

            # --------------------------- EARLY STOP ----------------------------
            if 0 < early_patience <= bad_epochs:
                logger.info(
                    f"[early_stop] Patience exhausted after {bad_epochs} bad epochs."
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

        total_epochs_run += epochs
        start_epoch_in_stage = 1  # reset for next stage

    # ==================================================================
    # 6. Final history & return
    # ==================================================================
    history.update(
        {
            "best_val": best_val,
            "epochs_run": total_epochs_run,
            "global_step": global_step,
        }
    )
    logger.info(
        f"Training finished: epochs_run={total_epochs_run}, best_val={best_val:.6f}"
    )
    return history
