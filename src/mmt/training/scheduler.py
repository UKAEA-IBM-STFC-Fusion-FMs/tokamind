from __future__ import annotations

import math
from typing import Dict, Optional, Set, Hashable, Mapping

import torch
import torch.nn as nn


"""
Optimiser / scheduler utilities for the Multi-Modal Transformer.

This module provides:

  • Param-group construction for the standard triplet:
        - backbone
        - modality_heads (per modality)
        - output_adapters (per output)

  • A simple LR scheduler:
        - linear warmup for `warmup_steps`
        - cosine decay for the remaining steps

  • Automatic per-output / per-modality "freezing":
        - for each batch, only heads/adapters whose outputs are *active*
          (have at least one supervised label) get a non-zero LR / WD.
        - outputs that never appear in the dataset for this task will
          *never* receive updates (LR stays 0 for them), without any
          extra config.

  • Stage-level freezing helper:
        - apply_stage_freeze_policy(model, freeze_backbone, freeze_modality_heads, freeze_output_adapters)
          to implement coarse-grained staged FT (e.g. "freeze backbone in
          the warm-up stage, unfreeze later").

Naming is aligned with the new project:
  - we speak about *outputs* (not targets),
  - and about backbone / modality_heads / output_adapters.

How automatic freezing works
----------------------------
The training loop is expected to do, once per batch:

    active_outputs = {out_key for out_key, m in outputs_mask.items() if m.any()}
    toggle_param_groups(optimizer, active_outputs, model.output2modality)

where:
  • outputs_mask[out_key] is a BoolTensor of shape (B,) indicating which
    samples in the batch have labels for that output.
  • model.output2modality maps each output key to its modality string.

Then:

  - For each output_adapter group:
        if head_name ∈ active_outputs:
            lr = lr_on * scale
            weight_decay = wd_on
        else:
            lr = 0.0
            weight_decay = 0.0

  - For each modality_head group:
        if group_name is the modality of *any* active output:
            lr = lr_on * scale
            weight_decay = wd_on
        else:
            lr = 0.0
            weight_decay = 0.0

Thus, outputs that are not present in the current batch are "frozen" for
that step (no parameter update). And outputs that never appear in this
task's dataset are frozen for the entire run, with **zero config**.

The backbone group is controlled only by the scheduler and by the
stage-level freeze policy.
"""


# ======================================================================
# Helpers
# ======================================================================


def _trainable(params):
    """Filter out parameters that are actually trainable (requires_grad=True)."""
    return [p for p in params if p.requires_grad]


def _set_trainable(module: nn.Module, flag: bool) -> None:
    """Flip requires_grad for all parameters in a module."""
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
    Build optimizer param groups with explicit tags for backbone,
    modality_heads, and output_adapters.

    The model is expected to expose:

        model.backbone        : nn.Module
        model.modality_heads  : nn.ModuleDict
        model.output_adapters : nn.ModuleDict

    Param groups:

      - backbone:
          group_type = "backbone"
          base_lr   = lr_backbone

      - one group per modality head:
          group_type = "modality_head"
          group_name = <modality name>  (e.g. "ts", "profile", "scalar")
          lr_on      = lr_modality_heads
          wd_on      = wd_modality_heads
          lr         = 0.0 initially (toggled per batch)
          weight_decay = 0.0 initially

      - one group per output adapter:
          group_type = "output_adapter"
          head_name  = <output key>  (e.g. signal ID / output name)
          lr_on      = lr_output_adapters
          wd_on      = wd_output_adapters
          lr         = 0.0 initially (toggled per batch)
          weight_decay = 0.0 initially

    Combined with `toggle_param_groups`, this gives:
      - automatic per-output / per-modality freezing (LR=0 for unused),
      - while the backbone LR follows the scheduler.
    """
    if not hasattr(model, "backbone") or not isinstance(model.backbone, nn.Module):
        raise AttributeError("Model must expose `.backbone` nn.Module.")

    if not hasattr(model, "modality_heads") or not isinstance(
        model.modality_heads, nn.ModuleDict
    ):
        raise AttributeError(
            "Model must expose `.modality_heads` as a nn.ModuleDict (per modality)."
        )

    if not hasattr(model, "output_adapters") or not isinstance(
        model.output_adapters, nn.ModuleDict
    ):
        raise AttributeError(
            "Model must expose `.output_adapters` as a nn.ModuleDict (per output)."
        )

    groups: list[Dict] = []

    # ---- backbone ----
    backbone_params = _trainable(model.backbone.parameters())
    if not backbone_params:
        raise RuntimeError("No trainable parameters found in `model.backbone`.")
    groups.append(
        {
            "params": backbone_params,
            "lr": lr_backbone,
            "weight_decay": wd_backbone,
            "group_type": "backbone",
            "base_lr": lr_backbone,
        }
    )

    # ---- per-modality heads (start disabled; toggled by `toggle_param_groups`) ----
    for modality, head in model.modality_heads.items():
        head_params = _trainable(head.parameters())
        if not head_params:
            continue
        groups.append(
            {
                "params": head_params,
                "lr": 0.0,
                "weight_decay": 0.0,
                "group_type": "modality_head",
                "group_name": modality,
                "lr_on": lr_modality_heads,
                "wd_on": wd_modality_heads,
            }
        )

    # ---- per-output adapters (start disabled; toggled by `toggle_param_groups`) ----
    for out_key, adapter in model.output_adapters.items():
        adp_params = _trainable(adapter.parameters())
        if not adp_params:
            continue
        groups.append(
            {
                "params": adp_params,
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
# LR scaling + per-output toggling (automatic freezing)
# ======================================================================


def _backbone_lr_scale(optimizer: torch.optim.Optimizer) -> float:
    """
    Compute the multiplicative LR scale applied by the scheduler to the
    backbone group:

        scale = current_backbone_lr / base_backbone_lr

    This allows us to propagate the same schedule to modality_heads and
    output_adapters by multiplying their "on" LR (lr_on) by this scale.
    """
    for g in optimizer.param_groups:
        if g.get("group_type") == "backbone":
            base = float(g.get("base_lr", g["lr"]))
            curr = float(g["lr"])
            if base <= 0.0:
                return 1.0
            return curr / base
    return 1.0


def toggle_param_groups(
    optimizer: torch.optim.Optimizer,
    active_outputs: Set[Hashable],
    output_to_modality: Mapping[Hashable, str],
) -> None:
    """
    Automatically enable / disable LR + weight decay for modality_heads and
    output_adapters based on which outputs are supervised in the current
    batch.

    This function implements the "automatic freezing of unused outputs &
    modalities" with zero config. The training loop must pass:

        - active_outputs: set of output keys that have at least one label
          in this batch.

        - output_to_modality: mapping from output key -> modality string,
          typically `model.output2modality`.

    Behaviour
    ---------
    1. We compute a global LR scale from the backbone group:

           scale = backbone_lr / base_backbone_lr

       so that all groups share the same LR schedule shape.

    2. Given active_outputs, we derive *active_modalities*:

           active_modalities = { output_to_modality[o] for o in active_outputs }

    3. For each param group:

       - if group_type == "output_adapter":

             head_name = group["head_name"]

             • if head_name in active_outputs and group has any trainable params:
                    lr = lr_on * scale
                    weight_decay = wd_on
               else:
                    lr = 0.0
                    weight_decay = 0.0

         This means that adapters for outputs that are not present in the
         current batch (or in the entire dataset) will *never* get a non-zero
         LR and thus never update, i.e. they are effectively frozen.

       - if group_type == "modality_head":

             group_name = group["group_name"]

             • if group_name in active_modalities and group has any trainable params:
                    lr = lr_on * scale
                    weight_decay = wd_on
               else:
                    lr = 0.0
                    weight_decay = 0.0

         Thus, modality heads for modalities that do not correspond to any
         active output in this batch (or in this task) are effectively frozen.

       - backbone group is left untouched here; its LR is driven purely by
         the scheduler and the stage-level freeze policy.

    Notes
    -----
    - This function does **not** change requires_grad; it only manipulates
      LR and weight_decay. So gradients may still be computed, but parameters
      with lr=0 will not change.

    - In a finetune scenario where the original model was trained on outputs
      {A, B, C} but the new task only uses {C}, and you always pass
      active_outputs={"C"}:
          - adapters A and B will keep lr=0 for the entire run
          - modality heads whose modalities are only used by A/B will keep
            lr=0 as well.
    """
    scale = _backbone_lr_scale(optimizer)
    active_modalities = {
        output_to_modality[o] for o in active_outputs if o in output_to_modality
    }

    for g in optimizer.param_groups:
        gt = g.get("group_type", "")
        params = g.get("params", [])
        is_trainable = any(p.requires_grad for p in params)

        if gt == "output_adapter":
            head_name = g.get("head_name")
            is_on = is_trainable and (head_name in active_outputs)
            g["lr"] = (float(g.get("lr_on", 0.0)) * scale) if is_on else 0.0
            g["weight_decay"] = float(g.get("wd_on", 0.0)) if is_on else 0.0

        elif gt == "modality_head":
            group_name = g.get("group_name")
            is_on = is_trainable and (group_name in active_modalities)
            g["lr"] = (float(g.get("lr_on", 0.0)) * scale) if is_on else 0.0
            g["weight_decay"] = float(g.get("wd_on", 0.0)) if is_on else 0.0

        # backbone group: unchanged here


# ======================================================================
# Optimiser + scheduler
# ======================================================================


def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_backbone: float = 1e-3,
    lr_modality_heads: float = 2e-3,
    lr_output_adapters: Optional[float] = None,
    wd_backbone: float = 1e-4,
    wd_modality_heads: float = 0.0,
    wd_output_adapters: Optional[float] = None,
    total_steps: Optional[int] = None,
    warmup_steps: int = 0,
    use_adamw: bool = True,
) -> tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler.LambdaLR]]:
    """
    Build an Adam/AdamW optimizer with param groups + a LambdaLR scheduler.

    Scheduler behaviour
    -------------------
    If `total_steps` is provided and > 0, we create a LambdaLR with:

        - linear warmup for steps in [0, warmup_steps)
        - cosine decay from step = warmup_steps to step = total_steps

    The returned scheduler is meant to be stepped *once per optimizer step*:

        for step, batch in enumerate(train_loader):
            ...
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    If `total_steps is None` or `<= 0`, we return `scheduler = None`.

    Notes
    -----
    - If lr_output_adapters / wd_output_adapters are None, they default to
      lr_modality_heads / wd_modality_heads respectively.

    - Param groups are built using `build_param_groups`, which tags each
      group with its role and "on" LR/WD, ready for `toggle_param_groups`.
    """
    if lr_output_adapters is None:
        lr_output_adapters = lr_modality_heads
    if wd_output_adapters is None:
        wd_output_adapters = wd_modality_heads

    param_groups = build_param_groups(
        model,
        lr_backbone=lr_backbone,
        wd_backbone=wd_backbone,
        lr_modality_heads=lr_modality_heads,
        wd_modality_heads=wd_modality_heads,
        lr_output_adapters=lr_output_adapters,
        wd_output_adapters=wd_output_adapters,
    )

    OptimCls = torch.optim.AdamW if use_adamw else torch.optim.Adam
    optimizer = OptimCls(param_groups, betas=(0.9, 0.999), eps=1e-8)

    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None
    if total_steps is not None and total_steps > 0:

        def lr_lambda(step: int) -> float:
            step = int(step)
            if step < 0:
                step = 0

            # warmup phase
            if warmup_steps > 0 and step < warmup_steps:
                return max(1e-8, step / max(1, warmup_steps))

            # cosine phase
            denom = max(1, total_steps - warmup_steps)
            progress = (step - warmup_steps) / denom
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


# ======================================================================
# Stage-level freezing (coarse-grained)
# ======================================================================


def apply_stage_freeze_policy(
    model: nn.Module,
    *,
    freeze_backbone: bool,
    freeze_modality_heads: bool,
    freeze_output_adapters: bool,
) -> None:
    """
    Apply a coarse-grained freeze policy for a training **stage**.

    This is separate from the automatic per-output freezing implemented
    by `toggle_param_groups`. This function controls whether whole blocks
    (backbone / all heads / all adapters) should be trainable in this
    stage at all.

    Typical use from a stage config:

        freeze_backbone: true / false
        freeze_modality_heads: false
        freeze_output_adapters: false

    Semantics
    ---------
    - freeze_backbone:
          * True  → backbone.requires_grad = False
          * False → backbone.requires_grad = True

    - freeze_modality_heads:
          * True  → all modality_heads modules frozen
          * False → all modality_heads modules trainable (subject to
                    per-batch LR toggling in `toggle_param_groups`)

    - freeze_output_adapters:
          * True  → all output_adapters modules frozen
          * False → all output_adapters modules trainable (subject to
                    per-batch LR toggling)

    Suggested usage
    ---------------
    At the beginning of each stage (and after a resume):

        apply_stage_freeze_policy(
            model,
            freeze_backbone=stage_cfg["freeze_backbone"],
            freeze_modality_heads=stage_cfg["freeze_modality_heads"],
            freeze_output_adapters=stage_cfg["freeze_output_adapters"],
        )

    Then build the optimizer and scheduler, and inside the training loop
    call `toggle_param_groups(...)` once per batch to automatically
    "freeze" unused outputs and modalities via LR=0.
    """
    if hasattr(model, "backbone"):
        _set_trainable(model.backbone, not freeze_backbone)

    if hasattr(model, "modality_heads"):
        for _, head in model.modality_heads.items():
            _set_trainable(head, not freeze_modality_heads)

    if hasattr(model, "output_adapters"):
        for _, adp in model.output_adapters.items():
            _set_trainable(adp, not freeze_output_adapters)
