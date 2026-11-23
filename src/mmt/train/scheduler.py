"""
mmt.train.scheduler

Optimizer + LR schedule helpers.

This module is intentionally *simple* and stage-driven.

Expected (validated) stage config fields
----------------------------------------
optimizer.lr, optimizer.wd, and freeze are mappings keyed by the selected model's ``get_named_blocks()`` names.

The validator is expected to already:
  • apply lr/wd inheritance (e.g., token_encoder inherits from backbone when null)
  • apply freeze policies by setting lr=0 and wd=0 for frozen blocks (optional), *and/or* the caller will apply
    requires_grad=False via apply_stage_freeze_policy.

Design choices
--------------
- **No per-batch LR toggling**. The loss is already masked by `output_mask`, so missing outputs do not contribute
  gradients.
- Param groups are coarse model-declared blocks. This keeps the optimizer state compact while allowing different model
  architectures to expose different transfer units.

"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from mmt.constants import FLOAT_STABILITY_EPS
from mmt.models.blocks import get_named_model_blocks


# ----------------------------------------------------------------------------------------------------------------------
def _set_trainable(module: nn.Module | torch.Tensor, flag: bool) -> None:
    """Set requires_grad=flag for all parameters of a module."""
    if isinstance(module, torch.Tensor):
        return
    for p in module.parameters():
        p.requires_grad = flag


# ----------------------------------------------------------------------------------------------------------------------
def build_param_groups(
    model: nn.Module,
    *,
    lr_by_block: dict[str, float],
    wd_by_block: dict[str, float],
) -> list[dict]:
    """
    Build optimizer parameter groups for model-declared blocks.

    Notes
    -----
    - Frozen params (requires_grad=False) are allowed in groups; PyTorch optimizers skip params with grad=None during
      step().
    - group_type is used by logging utilities.

    Raises
    ------
    RuntimeError
        If a declared block has no parameters.

    """

    groups: list[dict] = []
    for block_name, block in get_named_model_blocks(model=model).items():
        params = list(block.parameters())
        if not params:
            raise RuntimeError(f"Model block {block_name!r} has no parameters.")
        groups.append(
            {
                "params": params,
                "lr": float(lr_by_block[block_name]),
                "weight_decay": float(wd_by_block[block_name]),
                "group_type": block_name,
            }
        )

    return groups


# ----------------------------------------------------------------------------------------------------------------------
def build_optimizer_and_scheduler(
    model: nn.Module,
    *,
    lr_by_block: dict[str, float],
    wd_by_block: dict[str, float],
    total_steps: int,
    warmup_steps: int,
    use_adamw: bool,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR | None]:
    """
    Build optimizer and warmup+cosine scheduler.

    Scheduler definition
    --------------------
    - linear warmup from ~0 to 1 over warmup_steps
    - cosine decay from 1 to 0 over remaining steps
    """

    # ..................................................................................................................
    def lr_lambda(step: int) -> float:
        """Return the LR multiplier for a given step: linear warmup followed by cosine decay."""

        step = max(0, int(step))

        # Warmup
        if (warmup_steps > 0) and (step < warmup_steps):
            # Avoid exact 0 multiplier (can break some schedulers / logs)
            return max(FLOAT_STABILITY_EPS, step / float(warmup_steps))

        # Constant after warmup
        # return 1.0

        # Cosine with floor (set min_lr_ratio to 0 to have no floor)
        min_lr_ratio = 0.0
        denom = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / float(denom)
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # in [0, 1]

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    # ..................................................................................................................

    param_groups = build_param_groups(
        model=model,
        lr_by_block=lr_by_block,
        wd_by_block=wd_by_block,
    )

    OptimClass = torch.optim.AdamW if use_adamw else torch.optim.Adam  # NOSONAR # noqa - Ignore lowercase warning
    optimizer = OptimClass(param_groups, betas=(0.9, 0.999), eps=1e-8)

    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)

    if total_steps <= 0:
        return optimizer, None

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


# ----------------------------------------------------------------------------------------------------------------------
def apply_stage_freeze_policy(
    model: nn.Module,
    *,
    freeze_by_block: dict[str, bool],
) -> None:
    """Freeze/unfreeze whole blocks at the beginning of a stage."""

    for block_name, block in get_named_model_blocks(model=model).items():
        _set_trainable(module=block, flag=(not freeze_by_block[block_name]))
