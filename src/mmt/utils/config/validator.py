"""
Validator for the MMT experiment configuration (new config layout).

This module validates and normalizes the fully-merged configuration produced by
the convention-based loader (common + task + optional overrides).

We deliberately keep validation focused and simple:
  • common required fields (phase/task/task_config),
  • training stages validation (lr/wd inheritance, freeze rules),
  • loader rules for streaming vs cached datasets,
  • eval-specific requirements (model_source.run_dir, keep_output_native),
  • tune_dct3d minimal requirements.

No backwards compatibility: the config is expected to be in the new format.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple, Union


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _as_dict(cfg: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
    """
    Accept either a raw dict or an ExperimentConfig-like object with `.raw`.
    """
    if isinstance(cfg, dict):
        return cfg
    raw = getattr(cfg, "raw", None)
    if isinstance(raw, dict):
        return raw
    raise TypeError("cfg must be a dict or an object with a `.raw` dict attribute.")


def _get_nested(cfg: Dict[str, Any], path: str) -> Any:
    """
    Retrieve a nested key from dict `cfg` using a dotted path,
    raising KeyError if any component is missing.
    """
    node: Any = cfg
    for p in path.split("."):
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"Missing required config entry: {path}")
        node = node[p]
    return node


def _ensure_dict(cfg: Dict[str, Any], path: str) -> Dict[str, Any]:
    """
    Ensure a nested value exists and is a dict.
    """
    val = _get_nested(cfg, path)
    if not isinstance(val, dict):
        raise TypeError(f"Expected dict at '{path}', got {type(val).__name__}.")
    return val


def _normalize_null_to_empty_dict(cfg: Dict[str, Any], path: str) -> None:
    """
    YAML like:
        output_weights:
    parses as None. Normalize it to {} for downstream code.
    """
    parts = path.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise TypeError(f"Expected dict while walking '{path}', got {type(node)}")

    leaf = parts[-1]
    if leaf not in node or node[leaf] is None:
        node[leaf] = {}
    elif not isinstance(node[leaf], dict):
        raise TypeError(f"Expected dict at '{path}', got {type(node[leaf]).__name__}.")


# ---------------------------------------------------------------------------
# Common required fields
# ---------------------------------------------------------------------------

ALLOWED_PHASES = {"pretrain", "finetune", "eval", "tune_dct3d"}

REQUIRED_COMMON_FIELDS: List[Tuple[str, type]] = [
    ("phase", str),
    ("task", str),
]


# ---------------------------------------------------------------------------
# Training validation (same spec as before)
# ---------------------------------------------------------------------------

REQUIRED_TRAIN_FIELDS: List[Tuple[str, type]] = [
    ("train.resume", bool),
    ("train.early_stop.patience", int),
    ("train.early_stop.delta", float),
    ("train.loss.output_weights", dict),  # normalized if YAML gives null
    ("train.optimizer.use_adamw", bool),
    ("train.scheduler.warmup_steps_fraction", float),
    ("train.stages", list),
]

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


def _validate_stage_consistency(stage_cfg: Dict[str, Any]) -> None:
    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]
    freeze = stage_cfg["freeze"]

    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if freeze.get(block, False):
            if lr[block] != 0.0 or wd[block] != 0.0:
                raise RuntimeError(
                    f"Inconsistent config: block '{block}' is frozen but lr={lr[block]} or wd={wd[block]} is nonzero."
                )

    for block in ("token_encoder", "backbone"):
        if not freeze.get(block, False) and lr[block] == 0.0:
            raise ValueError(
                f"Inconsistent config: freeze.{block}=False but optimizer.lr.{block}=0. "
                f"Either set freeze.{block}=True or specify a positive learning rate."
            )


# ---------------------------------------------------------------------------
# Loader rules: streamed vs cached
# ---------------------------------------------------------------------------


def _validate_loader(cfg: Dict[str, Any]) -> None:
    loader_cfg = cfg.get("loader", {}) or {}
    phase = cfg.get("phase", None)

    data_cfg = cfg.get("data", {}) or {}
    cache_cfg = data_cfg.get("cache") or {}
    cache_enable = bool(cache_cfg.get("enable", False))

    bpe = loader_cfg.get("batches_per_epoch", None)

    # If cache is disabled (streaming windows), an epoch length must be defined.
    if phase in ("pretrain", "finetune"):
        if not cache_enable and bpe is None:
            raise ValueError(
                "When data.cache.enable=false (streaming), "
                "loader.batches_per_epoch must be set (int >= 1)."
            )


# ---------------------------------------------------------------------------
# model_source.load_parts normalization
# ---------------------------------------------------------------------------


def _normalize_load_parts(cfg: Dict[str, Any]) -> None:
    ms = cfg.get("model_source", None)
    if ms is None:
        return

    if not isinstance(ms, dict):
        raise TypeError("model_source must be a dict or null.")

    lp = ms.get("load_parts", None)
    if lp is None:
        lp = {}
        ms["load_parts"] = lp
    elif not isinstance(lp, dict):
        raise TypeError("model_source.load_parts must be a dict.")

    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if lp.get(block) is None:
            lp[block] = True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_config(cfg: Union[Dict[str, Any], Any]) -> None:
    """
    Validate config based on cfg.phase.
    """
    cfgd = _as_dict(cfg)

    # Common fields
    for k, _t in REQUIRED_COMMON_FIELDS:
        if k not in cfgd:
            raise KeyError(f"Missing required config entry: {k}")

    phase = cfgd["phase"]
    if phase not in ALLOWED_PHASES:
        raise ValueError(
            f"Unsupported phase: {phase} (allowed: {sorted(ALLOWED_PHASES)})"
        )

    if phase in ("pretrain", "finetune"):
        validate_train_config(cfgd)
    elif phase == "eval":
        validate_eval_config(cfgd)
    elif phase == "tune_dct3d":
        validate_tune_dct3d_config(cfgd)
    else:
        raise ValueError(f"Unsupported phase: {phase}")


def validate_train_config(cfg: Dict[str, Any]) -> None:
    """
    Validate configuration for training phases (pretrain/finetune).
    """
    # Normalize YAML-null dicts that are commonly left empty by users
    _normalize_null_to_empty_dict(cfg, "train.loss.output_weights")

    # Validate required train fields exist
    for path, _t in REQUIRED_TRAIN_FIELDS:
        _get_nested(cfg, path)

    # Mutual exclusion: resume vs warm-start from other run
    ms = cfg.get("model_source")
    has_warmstart = isinstance(ms, dict) and bool(ms.get("run_dir"))
    if cfg["train"]["resume"] is True and has_warmstart:
        raise ValueError(
            "Inconsistent config: train.resume=true is incompatible with model_source. "
            "Use resume to continue the same run, or set resume=false and use "
            "model_source.run_dir to warm-start from a different run."
        )

    stages = cfg["train"]["stages"]
    if not isinstance(stages, list) or len(stages) == 0:
        raise ValueError("train.stages must be a non-empty list for training phases.")

    for i, stage in enumerate(stages):
        for path, _t in REQUIRED_STAGE_FIELDS:
            _get_nested(stage, path)

        gas = stage["scheduler"]["grad_accum_steps"]
        if not isinstance(gas, int) or gas < 1:
            raise ValueError(
                "scheduler.grad_accum_steps must be an integer >= 1 "
                f"(got {gas}) in train.stages[{i}]."
            )

        _apply_lr_wd_inheritance(stage)
        _apply_freeze_rules(stage)
        _validate_stage_consistency(stage)

    _normalize_load_parts(cfg)
    _validate_loader(cfg)


def validate_eval_config(cfg: Dict[str, Any]) -> None:
    """
    Validate evaluation configuration.
    """
    _validate_loader(cfg)

    # Eval requires native outputs for metrics/traces.
    data_cfg = cfg.get("data", {}) or {}
    if not bool(data_cfg.get("keep_output_native", False)):
        raise ValueError(
            "For phase='eval', data.keep_output_native must be True "
            "(native outputs are required for metrics and trace saving)."
        )

    # Eval requires a run_dir to evaluate.
    ms = cfg.get("model_source", None)
    if not isinstance(ms, dict) or not ms.get("run_dir"):
        raise ValueError(
            "For phase='eval', model_source.run_dir must be set (path to a training run)."
        )


def validate_tune_dct3d_config(cfg: Dict[str, Any]) -> None:
    """
    Minimal validation for tune_dct3d phase.
    """
    td = cfg.get("tune_dct3d", None)
    if not isinstance(td, dict):
        raise KeyError("For phase='tune_dct3d', missing required block: tune_dct3d")

    # Required: search_space.keep_h/keep_w/keep_t
    ss = td.get("search_space", None)
    if not isinstance(ss, dict):
        raise KeyError("Missing required block: tune_dct3d.search_space")

    for k in ("keep_h", "keep_w", "keep_t"):
        v = ss.get(k)
        if not isinstance(v, list) or not v:
            raise ValueError(f"tune_dct3d.search_space.{k} must be a non-empty list.")
