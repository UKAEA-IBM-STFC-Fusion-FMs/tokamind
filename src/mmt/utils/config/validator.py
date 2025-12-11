"""
Validator for the MMT experiment configuration (MMT v0).

This module validates and normalizes the full YAML-merged experiment
configuration produced by the config loader. The validator enforces
the MMT v0 specification:

  • mandatory global training fields,
  • mandatory stage-level fields (epochs, lr/wd, freeze, scheduler),
  • automatic inheritance of lr/wd from backbone,
  • freeze → force lr=0 and wd=0 (with warnings),
  • model_init.load_parts normalized (defaults to True),
  • dataset loader rules (streamed vs cached),
  • basic sanity checks (num_workers >= 1).

Any missing or inconsistent entry raises a clear KeyError/ValueError
before training starts.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Utility: nested retrieval
# ---------------------------------------------------------------------------


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    """
    Retrieve a nested key from dict `cfg` using a dotted path,
    raising KeyError if any component is missing.
    """
    node = cfg
    parts = path.split(".")
    for p in parts:
        if p not in node:
            raise KeyError(f"Missing required config entry: {path}")
        node = node[p]
    return node


# ---------------------------------------------------------------------------
# REQUIRED TOP-LEVEL TRAIN FIELDS (minimal global spec)
# ---------------------------------------------------------------------------

REQUIRED_TRAIN_FIELDS: List[Tuple[str, type]] = [
    ("train.resume", bool),
    ("train.early_stop.patience", int),
    ("train.early_stop.delta", float),
    ("train.loss.output_weights", dict),
    ("train.optimizer.use_adamw", bool),
    ("train.scheduler.warmup_steps_fraction", float),
    ("train.stages", list),
]


# ---------------------------------------------------------------------------
# REQUIRED STAGE FIELDS
# These fields must exist in each stage block.
# Types are validated loosely; lr/wd may be None (inherit).
# ---------------------------------------------------------------------------

REQUIRED_STAGE_FIELDS: List[Tuple[str, type]] = [
    ("name", str),
    ("epochs", int),
    ("scheduler.grad_accum_steps", int),
    ("optimizer.lr.backbone", float),
    ("optimizer.lr.modality_heads", (float, type(None))),
    ("optimizer.lr.output_adapters", (float, type(None))),
    ("optimizer.lr.token_encoder", (float, type(None))),
    ("optimizer.wd.backbone", float),
    ("optimizer.wd.modality_heads", (float, type(None))),
    ("optimizer.wd.output_adapters", (float, type(None))),
    ("optimizer.wd.token_encoder", (float, type(None))),
    ("freeze.backbone", bool),
    ("freeze.modality_heads", bool),
    ("freeze.output_adapters", bool),
    ("freeze.token_encoder", bool),
]


# ---------------------------------------------------------------------------
# LR/WD INHERITANCE
# Missing or None → inherit from backbone.
# ---------------------------------------------------------------------------


def _apply_lr_wd_inheritance(stage_cfg: Dict[str, Any]) -> None:
    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]

    backbone_lr = float(lr["backbone"])
    backbone_wd = float(wd["backbone"])

    for block in ("token_encoder", "modality_heads", "output_adapters"):
        if lr.get(block) is None:
            lr[block] = backbone_lr
        if wd.get(block) is None:
            wd[block] = backbone_wd


# ---------------------------------------------------------------------------
# FREEZE RULES
# freeze.<block> = True → force lr=0, wd=0 (with warning)
# ---------------------------------------------------------------------------


def _apply_freeze_rules(stage_cfg: Dict[str, Any]) -> None:
    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]
    freeze = stage_cfg["freeze"]

    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if freeze.get(block, False):
            if lr.get(block, 0.0) != 0.0 or wd.get(block, 0.0) != 0.0:
                warnings.warn(
                    f"[MMT config] freeze.{block}=True → forcing lr=0 and wd=0 "
                    f"(was lr={lr.get(block)}, wd={wd.get(block)})",
                    stacklevel=2,
                )
            lr[block] = 0.0
            wd[block] = 0.0


# ---------------------------------------------------------------------------
# POST-FREEZE CONSISTENCY CHECK
# Ensures freeze.<block> actually results in lr=0, wd=0.
# ---------------------------------------------------------------------------


def _validate_stage_consistency(stage_cfg: Dict[str, Any]) -> None:
    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]
    freeze = stage_cfg["freeze"]

    # 1) If frozen → lr and wd must already be zero (enforced by _apply_freeze_rules)
    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if freeze.get(block, False):
            if lr[block] != 0.0 or wd[block] != 0.0:
                raise RuntimeError(
                    f"Inconsistent config: block '{block}' is frozen "
                    f"but lr={lr[block]} or wd={wd[block]} is nonzero."
                )

    # 2) If NOT frozen, lr=0 for backbone/token_encoder is almost certainly a mistake.
    #    We do NOT apply this check to modality_heads/output_adapters,
    #    because their lr is dynamically toggled per batch.
    for block in ("token_encoder", "backbone"):
        if not freeze.get(block, False) and lr[block] == 0.0:
            raise ValueError(
                f"Inconsistent config: freeze.{block}=False but optimizer.lr.{block}=0. "
                f"This implicitly disables training for '{block}'. "
                f"Either set freeze.{block}=True or specify a positive learning rate."
            )


# ---------------------------------------------------------------------------
# LOADER VALIDATION (streamed vs cached dataset rules)
# ---------------------------------------------------------------------------


def _validate_loader(cfg: Dict[str, Any]) -> None:
    loader_cfg = cfg.get("loader", {})

    # num_workers must be >=1
    nw = loader_cfg.get("num_workers", 1)
    if not isinstance(nw, int) or nw < 1:
        raise ValueError("loader.num_workers must be an integer >= 1.")

    # Check mutually exclusive loader modes
    streamed = loader_cfg.get("streaming", False)
    cached = loader_cfg.get("cached", False)

    if streamed and cached:
        raise ValueError("loader.streaming and loader.cached cannot both be true.")

    if not streamed and not cached:
        warnings.warn(
            "[MMT config] Neither streaming nor cached dataset mode selected; "
            "defaulting to cached=False, streaming=False may degrade performance.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# MODEL_INIT.load_parts validation
# Missing entries → default to True (most user-friendly behaviour)
# ---------------------------------------------------------------------------


def _normalize_load_parts(cfg: Dict[str, Any]) -> None:
    mi = cfg.get("model_init", {})
    lp = mi.get("load_parts")

    if lp is None:
        raise KeyError("model_init.load_parts must be defined in MMT v0.")

    # Fill missing entries with True
    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if lp.get(block) is None:
            lp[block] = True


# ---------------------------------------------------------------------------
# MAIN VALIDATOR for TRAIN and EVAL
# ---------------------------------------------------------------------------


def validate_train_config(cfg: Dict[str, Any]) -> None:
    """
    Validate the full experiment configuration.

    This function performs:
      • validation of the global training block,
      • validation and normalization of all training stages
        (lr/wd inheritance, freeze→force-zero rules),
      • validation of model_init.load_parts,
      • validation of dataset loader rules (streamed vs cached),
      • basic sanity checks (num_workers >= 1).

    Parameters
    ----------
    cfg : dict
        The fully merged configuration loaded by config_loader.

    Raises
    ------
    KeyError or ValueError
        If any required entry is missing or inconsistent.
    """

    # -----------------------------
    # Validate global train fields
    # -----------------------------
    for path, _t in REQUIRED_TRAIN_FIELDS:
        _get_nested(cfg, path)

    stages = cfg["train"]["stages"]

    # -----------------------------
    # Validate and normalize stages
    # -----------------------------
    for stage in stages:
        # Ensure required fields exist
        for path, _t in REQUIRED_STAGE_FIELDS:
            _get_nested(stage, path)

        # 1) Inherit lr/wd from backbone
        _apply_lr_wd_inheritance(stage)

        # 2) Freeze → force lr=0, wd=0
        _apply_freeze_rules(stage)

        # 3) Check consistency post-freeze
        _validate_stage_consistency(stage)

    # -----------------------------
    # Validate model_init.load_parts
    # -----------------------------
    _normalize_load_parts(cfg)

    # -----------------------------
    # Validate dataset loader rules
    # -----------------------------
    _validate_loader(cfg)

    return None


def validate_eval_config(cfg: Dict[str, Any]) -> None:
    """
    Validate the evaluation configuration.

    Evaluation configuration is intentionally MUCH simpler than training:
    - No train.stages, no lr/wd rules, no freeze rules
      (these apply only to training).
    - No model_init.load_parts
      (evaluation ALWAYS loads all four model blocks).
    - We still validate loader.num_workers.
    - We enforce data.keep_output_native = True for metrics + traces.
    - We DO NOT enforce any streaming/cached logic here.
      Dataset mode is controlled exclusively via data.cache.enable.
    """

    # ---------------------------------------------------------
    # 1. Basic loader validation (num_workers >= 1, etc.)
    # ---------------------------------------------------------
    _validate_loader(cfg)

    # ---------------------------------------------------------
    # 2. keep_output_native MUST be True in eval
    # ---------------------------------------------------------
    data_cfg = cfg.get("data", {})
    kon = data_cfg.get("keep_output_native", None)
    if not kon:
        raise ValueError(
            "For phase='eval', data.keep_output_native must be True "
            "(native outputs are required for metrics and trace saving)."
        )

    # ---------------------------------------------------------
    # 3. No training-related fields required
    #    We deliberately DO NOT require:
    #       - train.*
    #       - model_init.load_parts
    #       - training stages
    #       - optimizer/scheduler
    # ---------------------------------------------------------

    return None
