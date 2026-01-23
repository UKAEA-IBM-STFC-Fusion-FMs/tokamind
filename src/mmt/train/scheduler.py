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

We keep `toggle_param_groups(...)` as a backward-compatible **no-op** so older
loops that still call it won't break.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Iterable

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
    tok_params = list(model.tokens.parameters())
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
    back_params = list(model.backbone.parameters())
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
    mh_params = _flatten_params(model.modality_heads.values())
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
    oa_params = _flatten_params(model.output_adapters.values())
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
    total_steps: int,
    warmup_steps: int,
    use_adamw: bool,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    """
    Build optimizer and warmup+cosine scheduler.

    Scheduler definition
    --------------------
    - linear warmup from ~0 to 1 over warmup_steps
    - cosine decay from 1 to 0 over remaining steps
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

    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)

    if total_steps <= 0:
        return optimizer, None

    def lr_lambda(step: int) -> float:
        step = max(0, int(step))

        # Warmup
        if warmup_steps > 0 and step < warmup_steps:
            # Avoid exact 0 multiplier (can break some schedulers / logs)
            return max(1e-8, step / float(warmup_steps))

        # Cosine decay
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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
        _set_trainable(model.tokens, not freeze_token_encoder)

    if hasattr(model, "backbone"):
        _set_trainable(model.backbone, not freeze_backbone)

    if hasattr(model, "modality_heads"):
        for head in model.modality_heads.values():
            _set_trainable(head, not freeze_modality_heads)

    if hasattr(model, "output_adapters"):
        for adp in model.output_adapters.values():
            _set_trainable(adp, not freeze_output_adapters)
