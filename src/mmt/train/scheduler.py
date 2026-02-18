"""
mmt.train.scheduler

Optimizer + LR schedule helpers.

This module is intentionally *simple* and stage-driven.

Expected (validated) stage config fields
---------------------------------------
optimizer:
  lr:
    token_encoder: float
    backbone: float
    modality_heads: float
    output_adapters: float
  wd:
    token_encoder: float
    backbone: float
    modality_heads: float
    output_adapters: float

The validator is expected to already:
  • apply lr/wd inheritance (e.g. token_encoder inherits from backbone when null)
  • apply freeze policies by setting lr=0 and wd=0 for frozen blocks (optional),
    *and/or* the caller will apply requires_grad=False via apply_stage_freeze_policy.

Design choices
--------------
- **No per-batch LR toggling**. The loss is already masked by `output_mask`, so
  missing outputs do not contribute gradients.
- Param groups are 4 coarse blocks: token_encoder, backbone, modality_heads,
  output_adapters. This keeps the optimizer state compact and avoids brittle
  per-output bookkeeping.

"""

from __future__ import annotations

from typing import Dict, Optional, Iterable

import math

import torch
import torch.nn as nn


def _set_trainable(module: nn.Module, flag: bool) -> None:
    """Set requires_grad=flag for all parameters of a module."""
    for p in module.parameters():
        p.requires_grad = flag


def _flatten_params(modules: Iterable[nn.Module]) -> list[nn.Parameter]:
    """Return a flat list of parameters from a list/iterable of modules."""
    params: list[nn.Parameter] = []
    for m in modules:
        params.extend(list(m.parameters()))
    return params


def build_param_groups(
    model: nn.Module,
    *,
    lr_token_encoder: float,
    wd_token_encoder: float,
    lr_backbone: float,
    wd_backbone: float,
    lr_modality_heads: float,
    wd_modality_heads: float,
    lr_output_adapters: float,
    wd_output_adapters: float,
) -> list[Dict]:
    """
    Build optimizer parameter groups for four coarse blocks:

      • token_encoder   (model.tokens)
      • backbone        (model.backbone)
      • modality_heads  (all heads in model.modality_heads)
      • output_adapters (all adapters in model.output_adapters)

    Notes
    -----
    - Frozen params (requires_grad=False) are allowed in groups; PyTorch
      optimizers skip params with grad=None during step().
    - group_type is used by logging utilities (e.g. backbone_lr()).
    """
    groups: list[Dict] = []

    # ---- Token encoder ----
    if not hasattr(model, "tokens"):
        raise AttributeError("Model must expose .tokens (TokenEncoder).")
    tok_params = list(model.tokens.parameters())  # type: ignore[attr-defined]
    if tok_params:
        groups.append(
            {
                "params": tok_params,
                "lr": float(lr_token_encoder),
                "weight_decay": float(wd_token_encoder),
                "group_type": "token_encoder",
            }
        )

    # ---- Backbone ----
    if not hasattr(model, "backbone"):
        raise AttributeError("Model must expose .backbone.")
    back_params = list(model.backbone.parameters())  # type: ignore[attr-defined]
    if not back_params:
        raise RuntimeError("model.backbone has no parameters.")
    groups.append(
        {
            "params": back_params,
            "lr": float(lr_backbone),
            "weight_decay": float(wd_backbone),
            "group_type": "backbone",
        }
    )

    # ---- Modality heads (single group) ----
    if not hasattr(model, "modality_heads"):
        raise AttributeError("Model must expose .modality_heads (ModuleDict).")
    mh_params = _flatten_params(model.modality_heads.values())  # type: ignore[attr-defined]
    if mh_params:
        groups.append(
            {
                "params": mh_params,
                "lr": float(lr_modality_heads),
                "weight_decay": float(wd_modality_heads),
                "group_type": "modality_heads",
            }
        )

    # ---- Output adapters (single group) ----
    if not hasattr(model, "output_adapters"):
        raise AttributeError("Model must expose .output_adapters (ModuleDict).")
    oa_params = _flatten_params(model.output_adapters.values())  # type: ignore[attr-defined]
    if oa_params:
        groups.append(
            {
                "params": oa_params,
                "lr": float(lr_output_adapters),
                "weight_decay": float(wd_output_adapters),
                "group_type": "output_adapters",
            }
        )

    return groups


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_token_encoder: float,
    wd_token_encoder: float,
    lr_backbone: float,
    wd_backbone: float,
    lr_modality_heads: float,
    wd_modality_heads: float,
    lr_output_adapters: float,
    wd_output_adapters: float,
    total_epochs: int,
    use_adamw: bool,
    warmup_epochs: int = 1,
    min_lr_ratio: float = 0.10,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LRScheduler]]:
    """
    Build optimizer and epoch-based warmup + cosine annealing scheduler.

    Scheduler definition
    --------------------
    - Linear warmup from start_factor (0.01) to 1.0 over warmup_epochs
    - Cosine decay from 1.0 to min_lr_ratio over remaining epochs
    - Scheduler is stepped once per epoch (after training pass)

    This approach is universal and works for both cached and streaming datasets,
    as it doesn't require knowing the number of batches per epoch.

    Parameters
    ----------
    warmup_epochs : int, default=1
        Number of epochs for linear warmup. Automatically clamped to not exceed
        total_epochs - 1 to ensure at least one decay epoch.
    min_lr_ratio : float, default=0.10
        Minimum LR as fraction of initial LR. Prevents LR from collapsing to zero,
        allowing continued learning throughout training.

    Notes
    -----
    - For short stages (5 epochs), warmup_epochs=1 is recommended
    - For longer stages (15+ epochs), warmup_epochs=2-3 may be beneficial
    - min_lr_ratio=0.10 keeps 10% of initial LR as a floor, preventing dead epochs
    - Warmup starts at 1% of initial LR (start_factor=0.01) for stability
    """
    param_groups = build_param_groups(
        model,
        lr_token_encoder=lr_token_encoder,
        wd_token_encoder=wd_token_encoder,
        lr_backbone=lr_backbone,
        wd_backbone=wd_backbone,
        lr_modality_heads=lr_modality_heads,
        wd_modality_heads=wd_modality_heads,
        lr_output_adapters=lr_output_adapters,
        wd_output_adapters=wd_output_adapters,
    )

    OptimClass = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptimClass(param_groups, betas=(0.9, 0.999), eps=1e-8)

    total_epochs = int(total_epochs)
    if total_epochs <= 0:
        return optimizer, None

    warmup_epochs = max(0, int(warmup_epochs))
    # Keep at least 1 decay epoch whenever possible
    if total_epochs > 1:
        warmup_epochs = min(warmup_epochs, total_epochs - 1)
    else:
        warmup_epochs = 0

    start_factor = 0.01
    min_lr_ratio = float(min_lr_ratio)

    def lr_lambda(epoch: int) -> float:
        """
        LR multiplier as a function of epoch number.

        Note: PyTorch's LambdaLR calls this with epoch=0 at initialization,
        then epoch increments after each scheduler.step() call.
        Since we call scheduler.step() AFTER each epoch completes,
        epoch=0 corresponds to the LR used during epoch 1 training.
        """
        # Linear warmup
        if warmup_epochs > 0 and epoch < warmup_epochs:
            # epoch=0 -> start_factor (0.01), epoch=warmup_epochs-1 -> 1.0
            progress = epoch / float(warmup_epochs - 1) if warmup_epochs > 1 else 0.0
            return start_factor + (1.0 - start_factor) * progress

        # Cosine decay with floor
        decay_epochs = max(1, total_epochs - warmup_epochs)
        t = epoch - warmup_epochs
        progress = min(max(t / float(decay_epochs), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


def apply_stage_freeze_policy(
    model: nn.Module,
    *,
    freeze_token_encoder: bool,
    freeze_backbone: bool,
    freeze_modality_heads: bool,
    freeze_output_adapters: bool,
) -> None:
    """Freeze/unfreeze whole blocks at the beginning of a stage."""
    if hasattr(model, "tokens"):
        _set_trainable(model.tokens, not freeze_token_encoder)  # type: ignore[arg-type]

    if hasattr(model, "backbone"):
        _set_trainable(model.backbone, not freeze_backbone)  # type: ignore[arg-type]

    if hasattr(model, "modality_heads"):
        for head in model.modality_heads.values():  # type: ignore[attr-defined]
            _set_trainable(head, not freeze_modality_heads)  # type: ignore[arg-type]

    if hasattr(model, "output_adapters"):
        for adp in model.output_adapters.values():  # type: ignore[attr-defined]
            _set_trainable(adp, not freeze_output_adapters)  # type: ignore[arg-type]
