"""
Validator for the MMT experiment configuration (new config layout).

This module validates and normalizes the fully-merged configuration produced by
the convention-based loader (common + task + optional overrides).

We deliberately keep validation focused and simple:
  • common required fields (phase/task),
  • training stages validation (lr/wd inheritance, freeze rules),
  • loader rules for streaming vs cached datasets,
  • eval-specific requirements (model_source.run_dir, keep_output_native).

"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple, Union


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# Common required fields
# ------------------------------------------------------------------

ALLOWED_PHASES = {"pretrain", "finetune", "eval"}

REQUIRED_COMMON_FIELDS: List[Tuple[str, type]] = [
    ("phase", str),
    ("task", str),
]

# ------------------------------------------------------------------
# Required run-context fields (explicit in phase configs)
# ------------------------------------------------------------------

# These fields are required for *all* phases. They capture execution/runtime knobs
# that should be explicit in the selected phase config rather than implicitly
# provided by the loader.
REQUIRED_RUN_CONTEXT_FIELDS: List[Tuple[str, Union[type, Tuple[type, ...]]]] = [
    ("seed", int),
    ("runtime", dict),
    ("data.local", bool),
    ("data.subset_of_shots", (int, type(None))),
]


# ------------------------------------------------------------------
# Training validation (same spec as before)
# ------------------------------------------------------------------

REQUIRED_TRAIN_FIELDS: List[Tuple[str, type]] = [
    ("train.resume", bool),
    ("train.early_stop.patience", int),
    ("train.early_stop.delta", float),
    ("train.loss.output_weights", dict),  # normalized if YAML gives null
    ("train.optimizer.use_adamw", bool),
    ("train.stages", list),
]

REQUIRED_STAGE_FIELDS: List[Tuple[str, Union[type, Tuple[type, ...]]]] = [
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


# ------------------------------------------------------------------
# Loader rules: streamed vs cached
# ------------------------------------------------------------------


def _validate_loader(cfg: Dict[str, Any]) -> None:
    loader_cfg = cfg.get("loader", {}) or {}

    data_cfg = cfg.get("data", {}) or {}
    cache_cfg = data_cfg.get("cache") or {}
    cache_enable = bool(cache_cfg.get("enable", False))

    # Cached windows are already precomputed and collation is typically Python-heavy.
    # In this mode, multi-worker DataLoaders rarely help and can be slower
    # (and on some systems may increase file-descriptor pressure).
    if cache_enable:
        num_workers = int(loader_cfg.get("num_workers", 0) or 0)
        if num_workers > 0:
            warnings.warn(
                "[MMT config] data.cache.enable=true: prefer loader.num_workers=0 (or at most 1). "
                "Multi-workers rarely help when each item is already precomputed and collation is Python-heavy.",
                stacklevel=2,
            )

    # Validate batches_per_epoch (optional, used only for streaming datasets)
    bpe = loader_cfg.get("batches_per_epoch")
    if bpe is not None:
        if not isinstance(bpe, int) or bpe < 1:
            raise ValueError(
                f"loader.batches_per_epoch must be an integer >= 1 (got {bpe})."
            )


# ------------------------------------------------------------------
# model_source.load_parts normalization
# ------------------------------------------------------------------


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


# ------------------------------------------------------------------
# model.output_adapters.hidden_dim (very simple validation / normalization)
# ------------------------------------------------------------------


def _validate_output_adapters_hidden_dim(cfg: Dict[str, Any]) -> None:
    """Minimal validation for model.output_adapters.hidden_dim.

    Semantics:
      - fill defaults if missing
      - coerce ints / 'd_model'
      - manual always wins
    """
    model = cfg.get("model")
    if not isinstance(model, dict):
        return

    oa = model.setdefault("output_adapters", {})
    if oa is None:
        oa = model["output_adapters"] = {}
    if not isinstance(oa, dict):
        raise TypeError("model.output_adapters must be a dict.")

    hd = oa.get("hidden_dim")
    if hd is None:
        oa["hidden_dim"] = {
            "default": 0,
            "bucketed": {"enable": False, "rules": []},
            "manual": {},
        }
        return
    if not isinstance(hd, dict):
        raise TypeError("model.output_adapters.hidden_dim must be a dict.")

    def _hid(v):
        if v == "d_model":
            return "d_model"
        v = int(v)
        if v < 0:
            raise ValueError("hidden_dim values must be >= 0 or 'd_model'.")
        return v

    hd["default"] = _hid(hd.get("default", 0))

    bucketed = hd.get("bucketed") or {}
    if not isinstance(bucketed, dict):
        raise TypeError("hidden_dim.bucketed must be a dict.")
    bucketed["enable"] = bool(bucketed.get("enable", False))

    rules = bucketed.get("rules") or []
    if not isinstance(rules, list):
        raise TypeError("hidden_dim.bucketed.rules must be a list.")
    cleaned = []
    for r in rules:
        if not isinstance(r, dict) or "hidden" not in r:
            continue
        max_out = r.get("max_out_dim")
        max_out = None if max_out is None else int(max_out)
        cleaned.append({"max_out_dim": max_out, "hidden": _hid(r["hidden"])})
    bucketed["rules"] = cleaned
    hd["bucketed"] = bucketed

    manual = hd.get("manual") or {}
    if not isinstance(manual, dict):
        raise TypeError("hidden_dim.manual must be a dict.")
    hd["manual"] = {str(k): _hid(v) for k, v in manual.items()}


def _validate_required_run_context(cfg: Dict[str, Any]) -> None:
    """Validate presence and basic types for run-context keys."""
    # Presence checks
    for path, _t in REQUIRED_RUN_CONTEXT_FIELDS:
        val = _get_nested(cfg, path)
        # Basic type checks (match existing validator style: simple and explicit)
        if not isinstance(val, _t):
            raise TypeError(
                f"Expected type {getattr(_t, '__name__', str(_t))} at '{path}', "
                f"got {type(val).__name__}."
            )

    # runtime must be a dict (already checked), but allow empty dict; no further checks here.


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


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

    _validate_required_run_context(cfgd)
    # validate phase-specific config
    if phase in ("pretrain", "finetune"):
        validate_train_config(cfgd)
    elif phase == "eval":
        validate_eval_config(cfgd)
    else:
        raise ValueError(f"Unsupported phase: {phase}")

    # validate common fields
    data = cfgd.get("data") or {}
    cache = data.get("cache") or {}
    if bool(cache.get("enable", False)):
        dt = cache.get("dtype", None)
        if dt is None:
            cache["dtype"] = "float32"
        elif dt not in ("float16", "float32"):
            raise ValueError(
                "data.cache.dtype must be 'float16' or 'float32' (or null)."
            )

    # Model config validation (common to all phases)
    _validate_output_adapters_hidden_dim(cfgd)


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

        # Validate warmup_steps_fraction (optional, defaults to 0.1 in loop.py)
        warmup_frac = stage["scheduler"].get("warmup_steps_fraction")
        if warmup_frac is not None:
            if not isinstance(warmup_frac, (int, float)):
                raise ValueError(
                    "scheduler.warmup_steps_fraction must be a number "
                    f"(got {warmup_frac}) in train.stages[{i}]."
                )
            if not (0.0 <= warmup_frac < 1.0):
                raise ValueError(
                    "scheduler.warmup_steps_fraction must be in [0.0, 1.0) "
                    f"(got {warmup_frac}) in train.stages[{i}]."
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
