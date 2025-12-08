"""
config_validation.py — Minimal configuration validation for MMT training

This module centralises *presence* checks for the training configuration.
It enforces that the new strict config schema contains all required
entries, but it does not perform type checking or numeric casting.

Type correctness is enforced later at the point of use, where numeric
fields are converted via float()/int() inside the training loop.

Typical usage:

    from mmt.training.config_validation import (
        validate_training_config,
        validate_stage_config,
    )

    validate_training_config(cfg["training"])
    for stage in cfg["training"]["stages"]:
        validate_stage_config(stage)
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List

# ======================================================================
# Required configuration fields
# ======================================================================

#: Required global-level fields under the `training:` block.
#: The second element (type) is purely documentary; it is not enforced here.
REQUIRED_GLOBAL_FIELDS: List[Tuple[str, type]] = [
    ("resume", bool),
    ("early_stop.patience", int),
    ("early_stop.delta", float),
    ("loss.output_weights", dict),
    ("optimizer.use_adamw", bool),
    ("scheduler.warmup_steps_fraction", float),
    ("stages", list),
]

#: Required per-stage fields inside each entry of `training.stages`.
REQUIRED_STAGE_FIELDS: List[Tuple[str, type]] = [
    ("name", str),
    ("epochs", int),
    ("scheduler.grad_accum_steps", int),
    ("optimizer.lr.backbone", float),
    ("optimizer.lr.modality_heads", float),
    ("optimizer.lr.output_adapters", float),
    ("optimizer.wd.backbone", float),
    ("optimizer.wd.modality_heads", float),
    ("optimizer.wd.output_adapters", float),
    ("freeze.backbone", bool),
    ("freeze.modality_heads", bool),
    ("freeze.output_adapters", bool),
]


# ======================================================================
# Nested config helpers
# ======================================================================


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    """
    Retrieve cfg[path], where path is a dotted string like 'a.b.c'.

    Raises
    ------
    KeyError
        If any part of the path does not exist.
    """
    node = cfg
    for part in path.split("."):
        if part not in node:
            raise KeyError(
                f"Missing required config entry '{path}' (failed at '{part}')"
            )
        node = node[part]
    return node


# ======================================================================
# Public validation API
# ======================================================================


def validate_training_config(training_cfg: Dict[str, Any]) -> None:
    """
    Validate the top-level training configuration block.

    This only checks that all required fields are present. Type and
    numeric correctness are enforced later, at the point of use.
    """
    for path, _expected_type in REQUIRED_GLOBAL_FIELDS:
        _ = _get_nested(training_cfg, path)


def validate_stage_config(stage_cfg: Dict[str, Any]) -> None:
    """
    Validate the configuration of a single training stage.

    This only checks that all required fields are present. Type and
    numeric correctness are enforced later, at the point of use.
    """
    for path, _expected_type in REQUIRED_STAGE_FIELDS:
        _ = _get_nested(stage_cfg, path)
