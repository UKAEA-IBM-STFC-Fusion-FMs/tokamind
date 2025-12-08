# ======================================================================
# scheduler.py — Optimizer / LR schedule / per-output freezing utilities
#
# Fully updated for the NEW config specification:
#
# Stage config must provide (strict):
#
#   optimizer:
#     lr:
#       backbone: float
#       modality_heads: float
#       output_adapters: float
#     wd:
#       backbone: float
#       modality_heads: float
#       output_adapters: float
#
# No legacy behaviour:
# - No default LR/WD
# - No fallback based on model structure
# - No global lr.* fields
#
# Responsibilities of this module:
#   1. Build param groups (backbone, modality_heads, output_adapters)
#   2. Build optimizer (Adam or AdamW)
#   3. Build warmup+cosine LR scheduler
#   4. Apply coarse stage-level freeze
#   5. Apply per-batch LR toggling (automatic freezing of unused outputs)
#
# ======================================================================

from __future__ import annotations
import math
from typing import Dict, Optional, Set, Hashable, Mapping
import torch
import torch.nn as nn

# ======================================================================
# Helpers
# ======================================================================


def _trainable(params):
    """Return only parameters with requires_grad=True."""
    return [p for p in params if p.requires_grad]


def _set_trainable(module: nn.Module, flag: bool) -> None:
    """Set requires_grad=flag for all parameters of a module."""
    for p in module.parameters():
        p.requires_grad = flag


# ======================================================================
# Param groups
# ======================================================================


def build_param_groups(
    model: nn.Module,
    *,
    lr_backbone: float,
    wd_backbone: float,
    lr_modality_heads: float,
    wd_modality_heads: float,
    lr_output_adapters: float,
    wd_output_adapters: float,
) -> list[Dict]:
    """
    Build optimizer parameter groups for backbone, modality_heads,
    and output_adapters.

    All LR/WD values are explicit—no defaults, no inference.

    Param group structure:
      - backbone:
            group_type="backbone"
            base_lr=lr_backbone
            lr=lr_backbone
            weight_decay=wd_backbone

      - modality_head <modality>:
            group_type="modality_head"
            group_name=<modality>
            lr_on=lr_modality_heads
            wd_on=wd_modality_heads
            lr=0.0 (toggled per batch)
            weight_decay=0.0

      - output_adapter <output_key>:
            group_type="output_adapter"
            head_name=<output_key>
            lr_on=lr_output_adapters
            wd_on=wd_output_adapters
            lr=0.0 (toggled per batch)
            weight_decay=0.0
    """
    # ---- Backbone ----
    if not hasattr(model, "backbone"):
        raise AttributeError("Model must expose .backbone")

    backbone_params = _trainable(model.backbone.parameters())
    if not backbone_params:
        raise RuntimeError("No trainable parameters found in model.backbone.")

    groups = [
        {
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": wd_backbone,
            "group_type": "backbone",
            "base_lr": lr_backbone,
        }
    ]

    # ---- Modality heads ----
    if not hasattr(model, "modality_heads"):
        raise AttributeError("Model must expose .modality_heads (nn.ModuleDict)")

    for modality, head in model.modality_heads.items():
        params = _trainable(head.parameters())
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "lr": 0.0,
                "weight_decay": 0.0,
                "group_type": "modality_head",
                "group_name": modality,
                "lr_on": lr_modality_heads,
                "wd_on": wd_modality_heads,
            }
        )

    # ---- Output adapters ----
    if not hasattr(model, "output_adapters"):
        raise AttributeError("Model must expose .output_adapters (nn.ModuleDict)")

    for out_key, adapter in model.output_adapters.items():
        params = _trainable(adapter.parameters())
        if not params:
            continue
        groups.append(
            {
                "params": params,
                "lr": 0.0,
                "weight_decay": 0.0,
                "group_type": "output_adapter",
                "head_name": out_key,
                "lr_on": lr_output_adapters,
                "wd_on": wd_output_adapters,
            }
        )

    return groups


# ======================================================================
# LR scaling & automatic per-output freezing
# ======================================================================


def _backbone_lr_scale(optimizer: torch.optim.Optimizer) -> float:
    """
    Compute multiplicative LR scale applied by the scheduler:

        scale = current_backbone_lr / base_backbone_lr

    This allows modality_heads and output_adapters to follow exactly the
    same LR schedule as the backbone.
    """
    for g in optimizer.param_groups:
        if g.get("group_type") == "backbone":
            curr = float(g["lr"])
            base = float(g.get("base_lr", curr))
            return curr / base if base > 0 else 1.0
    return 1.0


def toggle_param_groups(
    optimizer: torch.optim.Optimizer,
    active_outputs: Set[Hashable],
    output_to_modality: Mapping[Hashable, str],
) -> None:
    """
    Automatic per-output & per-modality freezing:
      - For each output_adapter group:
            enabled iff output_key in active_outputs
      - For each modality_head group:
            enabled iff its modality is used by any active output
      - Backbone is untouched here.
    """
    scale = _backbone_lr_scale(optimizer)
    active_mods = {
        output_to_modality[o] for o in active_outputs if o in output_to_modality
    }

    for g in optimizer.param_groups:
        gtype = g.get("group_type", "")
        params = g.get("params", [])
        trainable = any(p.requires_grad for p in params)

        if gtype == "output_adapter":
            name = g["head_name"]
            is_on = trainable and (name in active_outputs)
            g["lr"] = g["lr_on"] * scale if is_on else 0.0
            g["weight_decay"] = g["wd_on"] if is_on else 0.0

        elif gtype == "modality_head":
            mod = g["group_name"]
            is_on = trainable and (mod in active_mods)
            g["lr"] = g["lr_on"] * scale if is_on else 0.0
            g["weight_decay"] = g["wd_on"] if is_on else 0.0

        # Backbone untouched.


# ======================================================================
# Build optimizer + scheduler
# ======================================================================


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_backbone: float,
    lr_modality_heads: float,
    lr_output_adapters: float,
    wd_backbone: float,
    wd_modality_heads: float,
    wd_output_adapters: float,
    total_steps: int,
    warmup_steps: int,
    use_adamw: bool,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    """
    Build optimizer + warmup+cosine scheduler.

    All LR/WD values must be provided explicitly by the stage config.

    Scheduler behaviour:
      - Linear warmup for warmup_steps
      - Cosine decay to 0.0 over remaining steps

    The scheduler is stepped **once per optimizer step** (not per batch).
    """
    param_groups = build_param_groups(
        model,
        lr_backbone=lr_backbone,
        wd_backbone=wd_backbone,
        lr_modality_heads=lr_modality_heads,
        wd_modality_heads=wd_modality_heads,
        lr_output_adapters=lr_output_adapters,
        wd_output_adapters=wd_output_adapters,
    )

    OptimClass = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptimClass(param_groups, betas=(0.9, 0.999), eps=1e-8)

    if total_steps <= 0:
        return optimizer, None

    # ---- LR schedule function
    def lr_lambda(step: int) -> float:
        step = max(0, int(step))

        # Warmup
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)

        # Cosine decay
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ======================================================================
# Stage-level coarse freezing
# ======================================================================


def apply_stage_freeze_policy(
    model: nn.Module,
    *,
    freeze_backbone: bool,
    freeze_modality_heads: bool,
    freeze_output_adapters: bool,
) -> None:
    """
    Freeze/unfreeze whole blocks at the beginning of a stage.

    Coarse-grained: modifies requires_grad.
    Fine-grained automatic freezing per batch is handled by toggle_param_groups().
    """
    if hasattr(model, "backbone"):
        _set_trainable(model.backbone, not freeze_backbone)

    if hasattr(model, "modality_heads"):
        for _, head in model.modality_heads.items():
            _set_trainable(head, not freeze_modality_heads)

    if hasattr(model, "output_adapters"):
        for _, adp in model.output_adapters.items():
            _set_trainable(adp, not freeze_output_adapters)
