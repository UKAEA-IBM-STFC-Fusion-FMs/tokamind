"""
validator.py — Centralised configuration validation for the MMT pipeline
===============================================================================

This module performs ALL cross-field validation of experiment configuration
after YAML loading and merging, but BEFORE dataset/model/train objects
are constructed.

Why a dedicated validator?
--------------------------
MMT supports two fundamentally different data-loading regimes:

    1) Cached mode   (cache_tokens = true)
       • WindowCachedDataset
       • len(dataset) == true number of windows
       • len(dataloader) is well-defined
       • Dataloader epoch-size = full pass over windows

    2) Streaming mode (cache_tokens = false)
       • WindowStreamedDataset (IterableDataset)
       • __len__ reports number of *shots*, not windows
       • Total window count is unknown without a full pre-scan
       • Hence, epoch boundaries must be specified explicitly

In addition, the train pipeline requires:
    • num_workers >= 1
    • well-formed train stages
    • LR/WD consistency with freeze flags
    • loader.streaming.batches_per_epoch for streaming train

This validator ensures that the entire configuration is internally consistent
BEFORE the train loop begins. All checks raise descriptive ValueError
exceptions to guide the user.

Typical usage:
--------------
    from mmt.config.config_validator import validate_config

    cfg = load_experiment_config(...)
    validate_config(cfg.raw)

This function should be called exactly once at the start of run_finetune.py.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, List


# -------------------------------------------------------------------------
# Nested access helper
# -------------------------------------------------------------------------


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    """
    Retrieve cfg[path], where path is a dotted string like 'a.b.c'.

    Raises
    ------
    KeyError
        If any part of the path is missing.

    Notes
    -----
    Type checking is intentionally NOT enforced here: the loader or the
    consumer (train loop, dataloader, etc.) will later cast to int/float.
    """
    node = cfg
    for part in path.split("."):
        if part not in node:
            raise KeyError(
                f"Missing required config entry '{path}' (failed at '{part}')"
            )
        node = node[part]
    return node


# -------------------------------------------------------------------------
# TRAIN MAIN-BLOCK REQUIREMENTS
# -------------------------------------------------------------------------

REQUIRED_TRAIN_FIELDS: List[Tuple[str, type]] = [
    ("train.resume", bool),
    ("train.early_stop.patience", int),
    ("train.early_stop.delta", float),
    ("train.loss.output_weights", dict),
    ("train.optimizer.use_adamw", bool),
    ("train.scheduler.warmup_steps_fraction", float),
    ("train.stages", list),
]


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


def _validate_stage_consistency(stage_cfg: Dict[str, Any]) -> None:
    """
    Cross-field checks inside a single train stage.
    Ensures freeze.* implies LR/WD = 0 for that block.
    """
    name = stage_cfg.get("name", "<unnamed-stage>")

    freeze = stage_cfg["freeze"]
    lr_cfg = stage_cfg["optimizer"]["lr"]
    wd_cfg = stage_cfg["optimizer"]["wd"]

    for block in ("backbone", "modality_heads", "output_adapters"):
        if bool(freeze.get(block, False)):
            lr = float(lr_cfg.get(block, 0.0))
            wd = float(wd_cfg.get(block, 0.0))
            if lr > 0.0 or wd > 0.0:
                raise ValueError(
                    f"Stage '{name}': freeze.{block}=True but lr={lr} wd={wd}. "
                    "Frozen blocks must have LR and WD explicitly set to 0."
                )


# -------------------------------------------------------------------------
# LOADER VALIDATION (STREAMING VS CACHED)
# -------------------------------------------------------------------------


def _validate_loader(cfg: Dict[str, Any]) -> None:
    """
    Validate loader-wide rules:
    - num_workers >= 1
    - batch_size present
    - streaming mode requires batches_per_epoch
    """
    loader = cfg.get("loader", {})
    batch_size = loader.get("batch_size", None)

    if batch_size is None:
        raise ValueError("loader.batch_size is required.")

    # --- num_workers rule (global) ---
    num_workers = loader.get("num_workers", None)

    # Must be explicitly provided, and must be an integer ≥ 1
    if num_workers is None:
        raise ValueError("loader.num_workers must be specified (>= 1).")

    if not isinstance(num_workers, int):
        raise TypeError(
            f"loader.num_workers must be an int, got {type(num_workers).__name__}"
        )

    if num_workers <= 0:
        raise ValueError(
            "MMT pipeline does not support num_workers=0 because window objects "
            "are mutated in-place by transforms. Please set num_workers >= 1."
        )

    # --- Streaming rules ---
    train_cfg = cfg.get("train", {})
    cache_tokens = bool(train_cfg.get("cache_tokens", False))

    streaming_cfg = loader.get("streaming", {})

    if not cache_tokens:
        # streaming mode MUST specify batches_per_epoch
        if "batches_per_epoch" not in streaming_cfg:
            raise ValueError(
                "loader.streaming.batches_per_epoch is required when "
                "train.cache_tokens = false (streaming mode)."
            )
        if streaming_cfg["batches_per_epoch"] <= 0:
            raise ValueError("loader.streaming.batches_per_epoch must be > 0.")


# -------------------------------------------------------------------------
# VALIDATION ENTRYPOINT
# -------------------------------------------------------------------------


def validate_config(cfg: Dict[str, Any]) -> None:
    """
    Validate the FULL experiment configuration.

    This includes:
      • presence & minimal consistency of the train block
      • presence & consistency of all train stages
      • loader rules (streaming vs cached)
      • num_workers >= 1

    Parameters
    ----------
    cfg : dict
        The FULL YAML-merged configuration from config_loader.

    Raises
    ------
    KeyError or ValueError
        If any required entry is missing or inconsistent.
    """
    # --- Global train fields ---
    for path, _expected_type in REQUIRED_TRAIN_FIELDS:
        _ = _get_nested(cfg, path)

    # --- Stage-level checks ---
    stages = cfg["train"]["stages"]
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError("train.stages must be a non-empty list.")

    for stage in stages:
        for path, _expected_type in REQUIRED_STAGE_FIELDS:
            _ = _get_nested(stage, path)
        _validate_stage_consistency(stage)

    # --- Loader-level checks ---
    _validate_loader(cfg)

    # All good
    return None
