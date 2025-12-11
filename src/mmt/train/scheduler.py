"""
scheduler.py — Optimizer / LR schedule / per-output freezing utilities

Fully updated for the NEW config specification:

Stage config MUST provide (after validator normalization):

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

The validator guarantees:
  • lr/wd inheritance from backbone
  • freeze → lr=0 and wd=0

Responsibilities of this module:
  1. Build param groups (token_encoder, backbone, modality_heads, output_adapters)
  2. Build optimizer (Adam or AdamW)
  3. Build warmup+cosine LR scheduler
  4. Apply coarse stage-level freezing (requires_grad)
  5. Per-batch dynamic toggling for heads/adapters (not token_encoder)

"""

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
# Param groups (NOW FOUR BLOCKS)
# ======================================================================


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
    Build optimizer parameter groups for:
        • token_encoder
        • backbone
        • modality_heads
        • output_adapters

    All LR/WD values are explicit — the validator guarantees proper defaults.

    Param group design:
      - token_encoder:
            lr = lr_token_encoder
            wd = wd_token_encoder

      - backbone:
            lr = lr_backbone
            wd = wd_backbone
            base_lr = lr_backbone   (for LR scaling in scheduler)

      - modality_head groups:
            lr = 0.0 (toggled per batch)
            lr_on = lr_modality_heads

      - output_adapter groups:
            lr = 0.0 (toggled per batch)
            lr_on = lr_output_adapters
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
                "lr": lr_token_encoder,
                "weight_decay": wd_token_encoder,
                "group_type": "token_encoder",
                "base_lr": lr_token_encoder,
            }
        )

    # ---- Backbone ----
    if not hasattr(model, "backbone"):
        raise AttributeError("Model must expose .backbone")

    back_params = list(model.backbone.parameters())
    if not back_params:
        raise RuntimeError("model.backbone has no parameters.")

    groups.append(
        {
            "params": back_params,
            "lr": lr_backbone,
            "weight_decay": wd_backbone,
            "group_type": "backbone",
            "base_lr": lr_backbone,
        }
    )

    # ---- Modality heads ----
    if not hasattr(model, "modality_heads"):
        raise AttributeError("Model must expose .modality_heads (ModuleDict).")

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
        raise AttributeError("Model must expose .output_adapters (ModuleDict).")

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
# LR scaling and dynamic per-output toggling
# ======================================================================


def _backbone_lr_scale(optimizer: torch.optim.Optimizer) -> float:
    """
    Compute multiplicative LR scale applied by the scheduler:

        scale = current_backbone_lr / base_backbone_lr
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

      - Output adapters enabled only for active outputs.
      - Modality heads enabled only for modalities supporting active outputs.
      - TokenEncoder and backbone are NOT toggled here.

    Called once per batch.
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

        # token_encoder and backbone untouched.


# ======================================================================
# Optimizer + scheduler builder
# ======================================================================


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

    All LRs/Wds provided explicitly by config (validator handles defaults).
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

    if total_steps <= 0:
        return optimizer, None

    def lr_lambda(step: int) -> float:
        step = max(0, int(step))

        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, step / warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)

        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


# ======================================================================
# Stage-level coarse freezing (requires_grad)
# ======================================================================


def apply_stage_freeze_policy(
    model: nn.Module,
    *,
    freeze_token_encoder: bool,
    freeze_backbone: bool,
    freeze_modality_heads: bool,
    freeze_output_adapters: bool,
) -> None:
    """
    Freeze/unfreeze whole blocks at the beginning of a stage.
    """

    if hasattr(model, "tokens"):
        _set_trainable(model.tokens, not freeze_token_encoder)

    if hasattr(model, "backbone"):
        _set_trainable(model.backbone, not freeze_backbone)

    if hasattr(model, "modality_heads"):
        for _, head in model.modality_heads.items():
            _set_trainable(head, not freeze_modality_heads)

    if hasattr(model, "output_adapters"):
        for _, adp in model.output_adapters.items():
            _set_trainable(adp, not freeze_output_adapters)
