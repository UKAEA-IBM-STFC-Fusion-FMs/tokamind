"""
Validator for the MMT experiment configuration (new config layout).

This module validates and normalizes the fully-merged configuration produced by the convention-based loader (common +
task + optional overrides).

We deliberately keep validation focused and simple:
  • common required fields (phase/task),
  • training stages validation (lr/wd inheritance, freeze rules),
  • loader rules for streaming vs cached datasets,
  • eval-specific requirements (model_source.run_dir),
  • automatic derivation of data.keep_output_native from phase and loss terms.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, MutableMapping
from typing import Any, Union

from mmt.models.blocks import MODEL_BLOCKS
from mmt.train.losses.constants import (
    ALL_LOSS_TYPES,
    EMBED_MSE_LOSS_TYPE,
    NATIVE_SPACE_LOSS_TYPES,
    PRED_DIST_LOSS_TYPES,
    SAMPLE_LOSS_TYPES,
)
from mmt.train.losses.registry import get_loss_class


# ======================================================================================================================
# Common required fields
# ======================================================================================================================

ALLOWED_PHASES = {"pretrain", "finetune", "eval"}
ALLOWED_MODEL_NAMES = set(MODEL_BLOCKS)
ALLOWED_OUTPUT_ADAPTER_TYPES = {"deterministic", "gaussian"}

# Loss term types that require batch['output_native'] — add one entry here when introducing a new native-space loss.
_NATIVE_TARGET_TERMS: frozenset[str] = NATIVE_SPACE_LOSS_TYPES

REQUIRED_COMMON_FIELDS: list[tuple[str, type]] = [("phase", str), ("task", str)]


# ======================================================================================================================
# Helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _as_dict(cfg: Union[Mapping[str, Any], Any]) -> dict[str, Any]:
    """
    Accept either a raw dict or an ExperimentConfig-like object with `.raw`.

    Returns
    -------
    dict[str, Any]
        Either `cfg` if it is dict, or `cfg.raw` if it exists and is a dict.

    Raises
    ------
    TypeError
        If `cfg` is not a dict or an object with a `.raw` dict attribute.

    """

    if isinstance(cfg, dict):
        return cfg

    raw = getattr(cfg, "raw", None)
    if isinstance(raw, dict):
        return raw

    raise TypeError("`cfg` must be a dict or an object with a `.raw` dict attribute.")


# ----------------------------------------------------------------------------------------------------------------------
def _get_nested(cfg: Mapping[str, Any], path: str) -> Any:
    """Retrieve a nested key from dict `cfg` using a dotted path, raising KeyError if any component is missing."""

    node: Any = cfg
    for p in path.split("."):
        if (not isinstance(node, dict)) or (p not in node):
            raise KeyError(f"Missing required config entry: {path}")

        node = node[p]

    return node


# ----------------------------------------------------------------------------------------------------------------------
def _ensure_dict(cfg: Mapping[str, Any], path: str) -> dict[str, Any]:
    """Ensure a nested value exists and is a dict, raising TypeError if `path` does not lead to a dictionary."""

    val = _get_nested(cfg=cfg, path=path)
    if not isinstance(val, dict):
        raise TypeError(f"Expected dict at '{path}', got {type(val).__name__}.")

    return val


# ----------------------------------------------------------------------------------------------------------------------
def _normalize_null_to_empty_dict(cfg: Mapping[str, Any], path: str) -> None:
    """
    YAML like:
        output_weights:
    parses as None. Normalize it to {} for downstream code.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If a dict is not obtained while walking `path`.
        If a dict is not obtained at `path`.

    """

    parts = path.split(".")
    node = cfg
    for p in parts[:-1]:
        node = node.setdefault(p, {})
        if not isinstance(node, dict):
            raise TypeError(f"Expected dict while walking '{path}', got {type(node)}.")

    leaf = parts[-1]
    if (leaf not in node) or (node[leaf] is None):
        node[leaf] = {}
    elif not isinstance(node[leaf], dict):
        raise TypeError(f"Expected dict at '{path}', got {type(node[leaf]).__name__}.")


# ======================================================================================================================
# Required run-context fields (explicit in phase configs)
# ======================================================================================================================

# These fields are required for *all* phases. They capture execution/runtime knobs that should be explicit in the
# selected phase config rather than implicitly provided by the loader.
REQUIRED_RUN_CONTEXT_FIELDS: list[tuple[str, Union[type, tuple[type, ...]]]] = [
    ("seed", int),
    ("runtime", dict),
]

# ======================================================================================================================
# Training validation (same spec as before)
# ======================================================================================================================

REQUIRED_TRAIN_FIELDS: list[tuple[str, type]] = [
    ("train.resume", bool),
    ("train.early_stop.patience", int),
    ("train.early_stop.delta", float),
    ("train.loss.output_weights", dict),  # Normalized if YAML gives null
    ("train.optimizer.use_adamw", bool),
    ("train.stages", list),
]

REQUIRED_STAGE_FIELDS: list[tuple[str, Union[type, tuple[type, ...]]]] = [
    ("name", str),
    ("epochs", int),
    ("scheduler.grad_accum_steps", int),
    ("optimizer.lr", dict),
    ("optimizer.wd", dict),
    ("freeze", dict),
]


# ----------------------------------------------------------------------------------------------------------------------
def _model_blocks_for_config(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Return declared model block names for the materialized config."""

    model = cfg.get("model")
    if not isinstance(model, dict):
        return MODEL_BLOCKS["mmt"]

    model_name = str(model.get("name", "mmt"))
    if model_name not in MODEL_BLOCKS:
        raise ValueError(f"Unsupported model.name={model_name!r}. Allowed values are: {sorted(MODEL_BLOCKS)}.")

    return MODEL_BLOCKS[model_name]


# ----------------------------------------------------------------------------------------------------------------------
def _validate_block_mapping_keys(mapping: Mapping[str, Any], *, path: str, blocks: tuple[str, ...]) -> None:
    """
    Validate that a block-keyed mapping exactly matches the selected model blocks.

    Raises
    ------
    KeyError
        If required block keys are missing or unknown block keys are present.

    """

    keys = set(mapping)
    expected = set(blocks)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if unknown:
            parts.append(f"unknown={unknown}")
        raise KeyError(f"{path} must match model blocks {sorted(expected)} ({', '.join(parts)}).")


# ----------------------------------------------------------------------------------------------------------------------
def _apply_lr_wd_inheritance(stage_cfg: Mapping[str, Any], *, blocks: tuple[str, ...]) -> None:
    """Apply learning rate (lr) and weight decay (wd) inheritance from stage configuration mapping."""

    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]

    inherit_block = "backbone" if "backbone" in blocks else "backbone_encoder"
    inherit_lr = float(lr[inherit_block])
    inherit_wd = float(wd[inherit_block])

    for block in blocks:
        if lr.get(block) is None:
            lr[block] = inherit_lr
        if wd.get(block) is None:
            wd[block] = inherit_wd


# ----------------------------------------------------------------------------------------------------------------------
def _apply_freeze_rules(stage_cfg: Mapping[str, Any], *, blocks: tuple[str, ...]) -> None:
    """Apply freeze rules from stage configuration mapping."""

    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]
    freeze = stage_cfg["freeze"]

    for block in blocks:
        if freeze.get(block, False):
            if (lr.get(block, 0.0) != 0) or (wd.get(block, 0.0) != 0):
                warnings.warn(
                    f"[MMT config] freeze.{block}=True -> forcing lr=0 and wd=0 "
                    f"(was lr={lr.get(block)}, wd={wd.get(block)})",
                    stacklevel=2,
                )
            lr[block] = 0.0
            wd[block] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
def _validate_stage_consistency(stage_cfg: Mapping[str, Any], *, blocks: tuple[str, ...]) -> None:
    """
    Validate stage consistency based on stage configuration mapping.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If inconsistency found in `stage_cfg["freeze"]` for a frozen block with nonzero learning rate or weight decay.
        If inconsistency found in `stage_cfg["freeze"]` for an unfrozen block with learning rate equal to 0.

    """

    lr = stage_cfg["optimizer"]["lr"]
    wd = stage_cfg["optimizer"]["wd"]
    freeze = stage_cfg["freeze"]

    for block in blocks:
        if freeze.get(block, False) and ((lr[block] != 0) or (wd[block] != 0)):
            raise ValueError(
                f"Inconsistent config: block '{block}' is frozen but lr={lr[block]} or wd={wd[block]} is nonzero."
            )

    for block in blocks:
        if (not freeze.get(block, False)) and (lr[block] == 0):
            raise ValueError(
                f"Inconsistent config: freeze.{block}=False but optimizer.lr.{block}=0. "
                f"Either set freeze.{block}=True or specify a positive learning rate."
            )


# ======================================================================================================================
# Loss → data dependency resolution
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _effective_stage_loss_cfgs(cfg: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """
    Return loss configs actually used by training stages.

    Stage-level ``loss`` blocks shallow-override the global ``train.loss`` block. Stages without a ``loss`` block use
    the global loss unchanged.

    Parameters
    ----------
    cfg : Mapping[str, Any]
        Full experiment config.

    Returns
    -------
    list[Mapping[str, Any]]
        Effective loss configs. When stages are not available yet, this falls back to the global train loss.

    """

    train_cfg = cfg.get("train") or {}
    global_loss = train_cfg.get("loss") or {}
    stages = train_cfg.get("stages")
    if not isinstance(stages, list) or not stages:
        return [global_loss]

    effective: list[Mapping[str, Any]] = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            effective.append(global_loss)
            continue

        stage_loss = stage.get("loss")
        if isinstance(stage_loss, Mapping):
            effective.append({**global_loss, **stage_loss})
        else:
            effective.append(global_loss)

    return effective


# ----------------------------------------------------------------------------------------------------------------------
def _resolve_keep_output_native(cfg: MutableMapping[str, Any], phase: str) -> None:
    """
    Compute and write ``data.keep_output_native`` — it is always derived, never set manually.

    Rules:
      • eval phase  → always ``True`` (native outputs are required for metrics and trace saving).
      • train phase → ``True`` iff any effective global/stage loss term is in ``_NATIVE_TARGET_TERMS``.

    The computed value is written back into ``cfg["data"]`` so all downstream code (collate, dataset) reads it
    transparently without knowing how it was determined.
    """

    if phase == "eval":
        keep = True
    else:
        keep = False
        for loss_cfg in _effective_stage_loss_cfgs(cfg=cfg):
            terms = loss_cfg.get("terms") or []
            if any(isinstance(t, dict) and t.get("type", EMBED_MSE_LOSS_TYPE) in _NATIVE_TARGET_TERMS for t in terms):
                keep = True
                break

    cfg.setdefault("data", {})["keep_output_native"] = keep


# ----------------------------------------------------------------------------------------------------------------------
def _validate_loss_terms_block(loss_cfg: Mapping[str, Any], *, path: str, head_type: str) -> None:
    """
    Validate one configured loss block with early, user-facing errors.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        Loss config block.
    path : str
        Human-readable config path used in error messages.
    head_type : str
        Configured model output adapter type.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If ``terms`` or a term-specific field has the wrong type.
    ValueError
        If a loss term has an unsupported type or invalid Grad-Shafranov option.
    KeyError
        If a Grad-Shafranov loss term is missing a required field.

    """

    terms = loss_cfg.get("terms")
    if terms is None:
        return
    if not isinstance(terms, list):
        raise TypeError(f"{path}.terms must be a list when provided.")

    supported = sorted(ALL_LOSS_TYPES)
    for term_index, term_def in enumerate(terms):
        if not isinstance(term_def, dict):
            raise TypeError(f"{path}.terms[{term_index}] must be a mapping.")

        term_type = str(term_def.get("type", EMBED_MSE_LOSS_TYPE))

        # ..............................................................................................................
        if term_type not in ALL_LOSS_TYPES:
            raise ValueError(f"Unsupported {path}.terms[{term_index}].type={term_type!r}. Supported: {supported}.")

        # Head <-> loss check: terms that consume pred_dist need a probabilistic output adapter. Fail at config time
        # (naming both the term and the knob) instead of crashing at the first training step.
        if (term_type in PRED_DIST_LOSS_TYPES) and (head_type == "deterministic"):
            raise ValueError(
                f"{path}.terms[{term_index}].type={term_type!r} requires a probabilistic output adapter "
                "(currently: 'gaussian'), but model.output_adapters.type='deterministic'. "
                "Set model.output_adapters.type: gaussian to use this loss term."
            )

        # `n_samples` is required (int >= 2) for sample-based probabilistic terms and rejected for all others, so a
        # misplaced key fails loudly instead of being silently ignored.
        has_n_samples = "n_samples" in term_def
        if term_type in SAMPLE_LOSS_TYPES:
            n_samples = term_def.get("n_samples")
            if (n_samples is not None) and (not isinstance(n_samples, int) or isinstance(n_samples, bool)):
                raise TypeError(f"{path}.terms[{term_index}].n_samples must be an integer.")
            if (n_samples is not None) and (n_samples < 2):
                raise ValueError(f"{path}.terms[{term_index}].n_samples must be >= 2, got {n_samples}.")
        elif has_n_samples:
            raise ValueError(
                f"{path}.terms[{term_index}] (type={term_type!r}) does not accept 'n_samples'; "
                f"it is only valid for sampled terms {sorted(SAMPLE_LOSS_TYPES)}."
            )

        # Loss-specific fields are owned by the concrete loss class, so the generic validator can stay structural.
        get_loss_class(term_type).validate_term_cfg(term_def=term_def, path=f"{path}.terms[{term_index}]")


# ----------------------------------------------------------------------------------------------------------------------
def _validate_loss_terms(cfg: Mapping[str, Any]) -> None:
    """
    Validate global and stage-level train loss terms.

    Stage-level ``loss`` blocks shallow-override the global ``train.loss`` block before validation, so stages can
    replace ``terms`` while inheriting other global loss settings such as ``output_weights``.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If a loss block or term-specific field has the wrong type.
    ValueError
        If a loss term has an unsupported type or invalid Grad-Shafranov option.
    KeyError
        If a Grad-Shafranov loss term is missing a required field.

    """

    train_cfg = cfg.get("train") or {}
    global_loss = train_cfg.get("loss") or {}
    if not isinstance(global_loss, Mapping):
        raise TypeError("train.loss must be a mapping.")

    head_type = str((cfg.get("model") or {}).get("output_adapters", {}).get("type", "deterministic"))
    _validate_loss_terms_block(loss_cfg=global_loss, path="train.loss", head_type=head_type)

    stages = train_cfg.get("stages") or []
    if not isinstance(stages, list):
        return

    for stage_index, stage in enumerate(stages):
        if not isinstance(stage, Mapping) or "loss" not in stage:
            continue

        stage_loss = stage["loss"]
        if not isinstance(stage_loss, Mapping):
            raise TypeError(f"train.stages[{stage_index}].loss must be a mapping when provided.")

        if "output_weights" in stage_loss:
            output_weights = stage_loss["output_weights"]
            if output_weights is None:
                stage_loss["output_weights"] = {}
            elif not isinstance(output_weights, dict):
                raise TypeError(f"train.stages[{stage_index}].loss.output_weights must be a mapping when provided.")

        effective_loss = {**global_loss, **stage_loss}
        _validate_loss_terms_block(
            loss_cfg=effective_loss,
            path=f"train.stages[{stage_index}].loss",
            head_type=head_type,
        )


# ======================================================================================================================
# Loader rules: streamed vs cached
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _validate_loader(cfg: Mapping[str, Any]) -> None:
    """
    Loader validation using stage configuration mapping.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        cfg["loader"].batches_per_epoch is not an integer >= 1.

    """

    loader_cfg = cfg.get("loader", {}) or {}

    data_cfg = cfg.get("data", {}) or {}
    cache_cfg = data_cfg.get("cache") or {}
    cache_enable = bool(cache_cfg.get("enable", False))

    # Cached windows are already precomputed and collation is typically Python-heavy.
    # In this mode, multi-worker DataLoaders rarely help and can be slower (and on some systems may increase
    # file-descriptor pressure).
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
    if (bpe is not None) and ((not isinstance(bpe, int)) or (bpe < 1)):
        raise ValueError(f"loader.batches_per_epoch must be an integer >= 1 (got {bpe}).")


# ======================================================================================================================
# model_source.load_parts normalization
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _normalize_load_parts(cfg: Mapping[str, Any], *, blocks: tuple[str, ...]) -> None:
    """
    Normalization of load parts.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `cfg["model_source"]` is not a dict or null.
        If `cfg["model_source"].load_parts` is not a dict."

    """

    ms = cfg.get("model_source")
    if ms is None:
        return

    if not isinstance(ms, dict):
        raise TypeError("`cfg['model_source']` must be a dict or null.")

    lp = ms.get("load_parts")
    if lp is None:
        lp = {}
        ms["load_parts"] = lp
    elif not isinstance(lp, dict):
        raise TypeError("`cfg['model_source'].load_parts` must be a dict.")

    unknown = sorted(set(lp) - set(blocks))
    if unknown:
        raise KeyError(f"model_source.load_parts has unknown keys {unknown}. Expected: {sorted(blocks)}.")
    for block in blocks:
        if lp.get(block) is None:
            lp[block] = True


# ======================================================================================================================
# model.output_adapters.hidden_dim (very simple validation / normalization)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _validate_output_adapters_hidden_dim(  # NOSONAR - Ignore cognitive complexity
    cfg: Mapping[str, Any],
) -> None:
    """
    Minimal validation for model.output_adapters.hidden_dim.

    Semantics:
      - fill defaults if missing
      - coerce ints / 'd_model'
      - manual always wins

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `cfg["model"]["output_adapters"]` is not a dict.
        If `cfg["model"]["output_adapters"]["hidden_dim"]` is not a dict.
        If `cfg["model"]["output_adapters"]["hidden_dim"]["bucketed"]` is not a dict.
        If `cfg["model"]["output_adapters"]["hidden_dim"]["bucketed"]["rules"]` is not a list.
        If `cfg["model"]["output_adapters"]["hidden_dim"]["manual"]` is not a dict.
    ValueError
        If hidden values are neither >= 0 nor "d_model".

    """

    # ..................................................................................................................
    def _validate_hidden_dim(values):
        """Validate hidden dimension values, raising ValueError if `values` are neither >= 0 nor 'd_model'."""

        if values == "d_model":
            return "d_model"

        v = int(values)
        if v < 0:
            raise ValueError("`hidden_dim` values must be >= 0 or 'd_model'.")

        return v

    # ..................................................................................................................

    model = cfg.get("model")
    if not isinstance(model, dict):
        return

    oa = model.setdefault("output_adapters", {})
    if oa is None:
        oa = model["output_adapters"] = {}
    if not isinstance(oa, dict):
        raise TypeError("`cfg['model']['output_adapters']` must be a dict.")  # noqa - Ignore unreachable code warning

    adapter_type = str(oa.get("type", "deterministic"))
    if adapter_type not in ALLOWED_OUTPUT_ADAPTER_TYPES:
        raise ValueError(
            f"Unsupported `cfg['model']['output_adapters']['type']`={adapter_type!r}. "
            f"Allowed values are: {sorted(ALLOWED_OUTPUT_ADAPTER_TYPES)}."
        )
    oa["type"] = adapter_type

    # Individual gaussian keys are not whitelisted here: they are forwarded as kwargs to
    # ProbabilisticOutputAdapter, so unknown keys fail loudly at model construction.
    gaussian_cfg = oa.get("gaussian")
    if gaussian_cfg is None:
        gaussian_cfg = {}
    if not isinstance(gaussian_cfg, dict):
        raise TypeError("`cfg['model']['output_adapters']['gaussian']` must be a dict if provided.")
    oa["gaussian"] = dict(gaussian_cfg)

    hd = oa.get("hidden_dim")
    if hd is None:
        oa["hidden_dim"] = {
            "default": 0,
            "bucketed": {"enable": False, "rules": []},
            "manual": {},
        }
        return

    if not isinstance(hd, dict):
        raise TypeError("`cfg['model']['output_adapters']['hidden_dim']` must be a dict.")

    hd["default"] = _validate_hidden_dim(values=hd.get("default", 0))

    bucketed = hd.get("bucketed") or {}
    if not isinstance(bucketed, dict):
        raise TypeError("`cfg['model']['output_adapters']['hidden_dim']['bucketed']` must be a dict.")

    bucketed["enable"] = bool(bucketed.get("enable", False))

    rules = bucketed.get("rules") or []
    if not isinstance(rules, list):
        raise TypeError("`cfg['model']['output_adapters']['hidden_dim']['bucketed']['rules']` must be a list.")

    cleaned = []
    for r in rules:
        if (not isinstance(r, dict)) or ("hidden" not in r):
            continue
        max_out = r.get("max_out_dim")
        max_out = None if max_out is None else int(max_out)  # type: ignore[arg-type]
        cleaned.append({"max_out_dim": max_out, "hidden": _validate_hidden_dim(values=r["hidden"])})

    bucketed["rules"] = cleaned
    hd["bucketed"] = bucketed

    manual = hd.get("manual") or {}
    if not isinstance(manual, dict):
        raise TypeError("`cfg['model']['output_adapters']['hidden_dim']['manual']` must be a dict.")

    hd["manual"] = {str(k): _validate_hidden_dim(values=v) for k, v in manual.items()}


# ----------------------------------------------------------------------------------------------------------------------
def _validate_model_name(cfg: Mapping[str, Any]) -> None:
    """
    Validate the materialized model architecture name.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `cfg["model"]` is present but not a mapping.
    ValueError
        If `cfg["model"].name` is unsupported, or inconsistent with `cfg["model_profile"]`.

    """

    model = cfg.get("model")
    if model is None:
        return
    if not isinstance(model, dict):
        raise TypeError("`cfg['model']` must be a dict.")

    model_name = str(model.get("name", "mmt"))
    if model_name not in ALLOWED_MODEL_NAMES:
        raise ValueError(f"Unsupported model.name={model_name!r}. Allowed values are: {sorted(ALLOWED_MODEL_NAMES)}.")

    model_profile = cfg.get("model_profile")
    if model_profile is not None and str(model_profile) != model_name:
        raise ValueError(f"model_profile={model_profile!r} is inconsistent with model.name={model_name!r}.")

    chunks = (cfg.get("preprocess") or {}).get("chunks")
    if isinstance(chunks, Mapping) and model_name == "mmt":
        output_cfg = chunks.get("output") or {}
        if output_cfg.get("max_chunks", 1) != 1:
            raise ValueError(f"model.name={model_name!r} currently requires preprocess.chunks.output.max_chunks=1.")


# ----------------------------------------------------------------------------------------------------------------------
def _validate_output_adapter_residual(cfg: Mapping[str, Any]) -> None:
    """Validate the optional output-adapter residual switch."""

    model_cfg = cfg.get("model") or {}
    residual_cfg = (model_cfg.get("output_adapters") or {}).get("residual")
    if residual_cfg is None:
        return
    if not isinstance(residual_cfg, Mapping):
        raise TypeError("model.output_adapters.residual must be a mapping when provided.")

    unknown = sorted(str(key) for key in residual_cfg if key not in {"enable", "zero_init"})
    if unknown:
        raise KeyError(
            f"Unknown model.output_adapters.residual keys: {unknown}. "
            "Only 'enable' and 'zero_init' are currently configurable; source and mapping use fixed defaults."
        )

    enable = residual_cfg.get("enable", False)
    if not isinstance(enable, bool):
        raise TypeError(f"model.output_adapters.residual.enable must be bool, got {type(enable).__name__}.")

    zero_init = residual_cfg.get("zero_init", True)
    if not isinstance(zero_init, bool):
        raise TypeError(f"model.output_adapters.residual.zero_init must be bool, got {type(zero_init).__name__}.")


# ----------------------------------------------------------------------------------------------------------------------
def _validate_required_run_context(cfg: Mapping[str, Any]) -> None:
    """Validate presence and basic types for run-context keys, raising TypeError for type mismatches."""

    # Presence checks
    for path, _t in REQUIRED_RUN_CONTEXT_FIELDS:
        val = _get_nested(cfg=cfg, path=path)

        # Basic type checks (match existing validator style: simple and explicit)
        if not isinstance(val, _t):
            raise TypeError(f"Expected type {getattr(_t, '__name__', str(_t))} at '{path}', got {type(val).__name__}.")

    # runtime must be a dict (already checked), but allow empty dict; no further checks here.


# ======================================================================================================================
# Public API
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def validate_config(cfg: Union[Mapping[str, Any], Any]) -> None:
    """
    Validate config based on cfg.phase.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `cfg` misses required entry.
    ValueError
        If `cfg["phase"]` is not an allowed phase.
        If `cfg["data"]["cache"]["dtype"]` not in ["float16", "float32", None].

    """

    cfgd = _as_dict(cfg)

    # Common fields
    for k, _t in REQUIRED_COMMON_FIELDS:
        if k not in cfgd:
            raise KeyError(f"`cfg` missing required config entry: {k}.")

    phase = cfgd["phase"]
    if phase not in ALLOWED_PHASES:
        raise ValueError(f"Unsupported phase '{phase}' in `cfg['phase']` (allowed values: {sorted(ALLOWED_PHASES)}).")

    _validate_required_run_context(cfg=cfgd)
    # Derive keep_output_native from phase and loss terms — always computed, never set by the user.
    _resolve_keep_output_native(cfg=cfgd, phase=phase)

    # Validate phase-specific config
    if phase in ("pretrain", "finetune"):
        validate_train_config(cfg=cfgd)
    elif phase == "eval":
        validate_eval_config(cfg=cfgd)

    # Validate common fields
    data = cfgd.get("data") or {}
    cache = data.get("cache") or {}
    if bool(cache.get("enable", False)):
        dt = cache.get("dtype")
        if dt is None:
            cache["dtype"] = "float32"
        elif dt not in ("float16", "float32"):
            raise ValueError(  # noqa  - Ignore unreachable code warning
                "`cfg['data']['cache']['dtype']` must be 'float16', 'float32', or null (None)."
            )

    # Model config validation (common to all phases)
    _validate_model_name(cfg=cfgd)
    _validate_output_adapter_residual(cfg=cfgd)
    _validate_output_adapters_hidden_dim(cfg=cfgd)


# ----------------------------------------------------------------------------------------------------------------------
def validate_train_config(  # NOSONAR - Ignore cognitive complexity
    cfg: Mapping[str, Any],
) -> None:
    """
    Validate configuration for training phases (pretrain/finetune).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `cfg["train"]["resume"]` is True and `cfg["model_source"]["run_dir"]` is also True.
        If `cfg["train"]["stages"]` is not a non-empty list.
        If a stage in `cfg["train"]["stages"]` has an item `stage["scheduler"]["grad_accum_steps"]` that is not an
        integer >= 1.
        If a stage in `cfg["train"]["stages"]` has an item `stage["scheduler"]["warmup_steps_fraction"]` with a
        non-numerical value, or not in [0.0, 1.0).

    """

    # Normalize YAML-null dicts that are commonly left empty by users
    _normalize_null_to_empty_dict(cfg=cfg, path="train.loss.output_weights")
    _validate_loss_terms(cfg=cfg)
    blocks = _model_blocks_for_config(cfg=cfg)

    # Validate required train fields exist
    for path, _t in REQUIRED_TRAIN_FIELDS:
        _get_nested(cfg=cfg, path=path)

    # Mutual exclusion: resume vs warm-start from other run
    ms = cfg.get("model_source")
    has_warmstart = isinstance(ms, dict) and bool(ms.get("run_dir"))
    if (cfg["train"]["resume"] is True) and has_warmstart:
        raise ValueError(
            "Inconsistent config: train.resume=true is incompatible with model_source. "
            "Use resume to continue the same run, or set resume=false and use model_source.run_dir to warm-start from "
            "a different run."
        )

    stages = cfg["train"]["stages"]
    if (not isinstance(stages, list)) or (len(stages) == 0):
        raise ValueError("train.stages must be a non-empty list for training phases.")

    for i, stage in enumerate(stages):
        for path, _t in REQUIRED_STAGE_FIELDS:
            _get_nested(cfg=stage, path=path)

        for block_path in ("optimizer.lr", "optimizer.wd", "freeze"):
            block_mapping = _get_nested(cfg=stage, path=block_path)
            if not isinstance(block_mapping, dict):
                raise TypeError(f"Expected dict at train.stages[{i}].{block_path}.")
            _validate_block_mapping_keys(mapping=block_mapping, path=f"train.stages[{i}].{block_path}", blocks=blocks)

        for block in blocks:
            if not isinstance(stage["freeze"][block], bool):
                raise TypeError(f"Expected bool at train.stages[{i}].freeze.{block}.")
            for optim_key in ("lr", "wd"):
                value = stage["optimizer"][optim_key][block]
                if value is not None:
                    try:
                        stage["optimizer"][optim_key][block] = float(value)
                    except (TypeError, ValueError) as e:
                        raise TypeError(
                            f"Expected number or null at train.stages[{i}].optimizer.{optim_key}.{block}."
                        ) from e

        gas = stage["scheduler"]["grad_accum_steps"]
        if (not isinstance(gas, int)) or (gas < 1):
            raise ValueError(f"scheduler.grad_accum_steps must be an integer >= 1 (got {gas}) in train.stages[{i}].")

        # Validate warmup_steps_fraction (optional, defaults to 0.1 in loop.py)
        warmup_frac = stage["scheduler"].get("warmup_steps_fraction")
        if warmup_frac is not None:
            if not isinstance(warmup_frac, (int, float)):
                raise ValueError(
                    f"scheduler.warmup_steps_fraction must be a number (got {warmup_frac}) in train.stages[{i}]."
                )

            if not (0.0 <= warmup_frac < 1.0):
                raise ValueError(
                    f"scheduler.warmup_steps_fraction must be in [0.0, 1.0) (got {warmup_frac}) in train.stages[{i}]."
                )

        _apply_lr_wd_inheritance(stage_cfg=stage, blocks=blocks)
        _apply_freeze_rules(stage_cfg=stage, blocks=blocks)
        _validate_stage_consistency(stage_cfg=stage, blocks=blocks)

    _normalize_load_parts(cfg=cfg, blocks=blocks)
    _validate_loader(cfg=cfg)


# ----------------------------------------------------------------------------------------------------------------------
def validate_eval_config(cfg: Mapping[str, Any]) -> None:
    """
    Validate evaluation configuration.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `cfg["model_source"]["run_dir"]` is not defined.
        If `data.split` is set (split is a training-time setting).

    """

    _validate_loader(cfg=cfg)

    # Eval requires a run_dir to evaluate.
    ms = cfg.get("model_source", None)
    if (not isinstance(ms, dict)) or (not ms.get("run_dir")):
        raise ValueError("For phase='eval', model_source.run_dir must be set (path to a training run).")
