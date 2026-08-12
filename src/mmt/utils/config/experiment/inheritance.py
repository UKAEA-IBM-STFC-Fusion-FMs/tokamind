"""
Source-run inheritance and finetune model semantics.

This module handles config inheritance from source models for warmstart/eval:
- Resolves source model directories (run_id or path)
- Loads source run config snapshots
- Optionally inherits representation preprocessing settings (chunk, trim_chunks, embed_chunks)
- Applies finetune model semantics (scratch model vs warmstart overrides)
- Merges source model config with current overrides

Key concepts:
- Warmstart: load source model weights/config, keep finetune preprocess from current task config
- Scratch: use complete model_scratch architecture, no source inheritance
- Eval: always inherits from source model (weights + config + embeddings), including representation preprocess settings
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal, Union
from pathlib import Path

from mmt.utils.paths import REPO_ROOT

from .merge import deep_merge, load_yaml, resolve_from_repo_root


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.ConfigLoader")


# ----------------------------------------------------------------------------------------------------------------------
def resolve_run_id_to_run_dir(run_id: str) -> Path:
    """
    Resolve a training run ID to <repo_root>/runs/<run_id>.

    Parameters
    ----------
    run_id : str
        Input run ID.

    Returns
    -------
    Path
        Resolved run ID path.

    Raises
    ------
    ValueError
        If `model_source["run_id"]` is an empty string.
        If `model_source["run_id"]` is an invalid string (e.g., it contains path separators).

    """

    s = str(run_id).strip()
    if not s:
        raise ValueError(
            "Value for `model_source['run_id']` must be a non-empty string (folder name under <repo_root>/runs/)."
        )

    p = Path(s)
    if p.is_absolute() or (len(p.parts) != 1):
        raise ValueError(
            "Value for `model_source['run_id']` must be a valid run ID (folder name under <repo_root>/runs/), "
            "e.g., 'pretrain_base'. Do not include 'runs/' or any path separators."
        )

    return (REPO_ROOT / "runs" / p.parts[0]).resolve()


# ----------------------------------------------------------------------------------------------------------------------
def resolve_model_source_dir(
    model_source: Mapping[str, Any], *, phase: Literal["pretrain", "finetune", "eval"]
) -> tuple[Path, Union[str, None]]:
    """
    Resolve model_source to absolute source run directory.

    Parameters
    ----------
    model_source : Mapping[str, Any]
        Dictionary with model source configuration.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".

    Returns
    -------
    tuple[Path, Union[str, None]]
        Tuple (Path, None) if `model_source["model_path"]` is not None and no errors are raised, or (Path, str) if
        `model_source["run_id"]` is not None.

    Raises
    ------
    TypeError
        If `model_source` is not a mapping (dict).
    ValueError
        If `model_source["model_path"]`, if provided, is an empty path string.
    FileNotFoundError
        If `model_source["model_path"]`, if provided, points to a nonexisting file.
        If neither valid values for `model_source["run_id"]` nor `model_source["model_path"]` are provided.

    """

    if not isinstance(model_source, dict):
        raise TypeError("Parameter `model_source` must be a mapping (dict).")

    model_path = model_source.get("model_path")
    if model_path is not None:
        mp = str(model_path).strip()
        if not mp:
            raise ValueError("Value for `model_source['model_path']`, if provided, must be a non-empty path string.")

        path = resolve_from_repo_root(mp)
        if not path.is_dir():
            raise FileNotFoundError(
                f"`phase` {phase!r} requires `model_source['model_path']` to point to an existing directory.\n"
                f"Got: {path}"
            )

        return path, None

    run_id = model_source.get("run_id")
    if run_id is None:
        raise ValueError(
            f"Phase '{phase}' requires a valid `model_source['run_id']` (a training run ID under <repo_root>/runs/) "
            "or a valid `model_source['model_path']` (external run directory)."
        )

    return (resolve_run_id_to_run_dir(run_id=str(run_id)), str(run_id).strip())


# ----------------------------------------------------------------------------------------------------------------------
def load_source_run_config_yaml(model_run_dir: Path) -> dict[str, Any]:
    """
    Load saved merged source run config from <run_dir>/<run_id>.yaml.

    Parameters
    ----------
    model_run_dir : Path
        Path to model directory.

    Returns
    -------
    dict[str, Any]
        Dictionary with loaded configuration from resulting YAML file.

    Raises
    ------
    FileNotFoundError
        If `model_run_dir` does not lead to an existing configuration YAML file.

    """

    src_cfg_path = model_run_dir / f"{model_run_dir.name}.yaml"
    if not src_cfg_path.is_file():
        raise FileNotFoundError(
            "Required source run config YAML not found.\n"
            "Warm-start and evaluation require the saved merged config at:\n"
            f"  {src_cfg_path}\n"
        )

    return load_yaml(path=src_cfg_path)


# ----------------------------------------------------------------------------------------------------------------------
def inherit_preprocess_representation(  # NOSONAR - Ignore cognitive complexity
    merged: MutableMapping[str, Any],
    src_cfg: Mapping[str, Any],
    *,
    allow_override: bool,
) -> None:
    """
    Inherit representation-defining preprocess settings from source config.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary (modified in-place).
    src_cfg : Mapping[str, Any]
        Dictionary with source configuration.
    allow_override : bool
        If True, mapping override is allowed (and override wins at each level).

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `src_cfg` does not have a key 'preprocess'.
        If `src_cfg["preprocess"]` misses required representation preprocess keys.
    TypeError
        If `merged["preprocess"]` is not of type dict.

    """

    src_pre = src_cfg.get("preprocess")
    if not isinstance(src_pre, dict):
        raise KeyError(
            "Source run config `src_cfg` is missing required key 'preprocess' with mapping (dict) value.\n"
            "Expected preprocess.chunks plus preprocess.embed_chunks, or the legacy chunk/trim_chunks contract."
        )
    uses_role_specific_chunks = "chunks" in src_pre
    representation_keys = (
        ("chunks", "embed_chunks")
        if uses_role_specific_chunks
        else (
            "chunk",
            "trim_chunks",
            "embed_chunks",
        )
    )
    missing = [key for key in representation_keys if key not in src_pre]
    if missing:
        raise KeyError(
            "Source run config key `src_cfg['preprocess']` is missing required representation preprocess keys "
            f"{missing}.\nExpected: preprocess.chunks + preprocess.embed_chunks, or "
            "preprocess.chunk + preprocess.trim_chunks + preprocess.embed_chunks"
        )

    merged_pre = merged.get("preprocess")
    if merged_pre is None:
        merged_pre = {}
        merged["preprocess"] = merged_pre
    if not isinstance(merged_pre, dict):
        raise TypeError("Config key `merged['preprocess']` must be a mapping (dict).")

    overrides: dict[str, Any] = {}
    if allow_override:
        overrides = {key: copy.deepcopy(merged_pre.get(key)) for key in representation_keys}

    for key in representation_keys:
        merged_pre[key] = copy.deepcopy(src_pre[key])

    if allow_override:
        for key, override in overrides.items():
            if override is None:
                continue
            if isinstance(override, dict) and isinstance(merged_pre[key], dict):
                merged_pre[key] = deep_merge(base=merged_pre[key], override=override)
            else:
                merged_pre[key] = override


# ----------------------------------------------------------------------------------------------------------------------
def apply_finetune_model_semantics(
    merged: MutableMapping[str, Any], *, init_mode: Literal["warmstart", "scratch"]
) -> None:
    """
    Materialize canonical ``model`` for scratch and validate warmstart model overrides.

    Scratch configs define the full model under ``model_scratch``.
    Warmstart configs define optional ``model_overrides`` that are applied later on top of the source model.
    """

    if init_mode not in ["warmstart", "scratch"]:
        raise ValueError(f"Unsupported finetune init mode: {init_mode!r}")

    if "model" in merged:
        raise KeyError(
            "Finetune config now uses explicit init-specific model keys: "
            "'model_scratch' for scratch or 'model_overrides' for warmstart. "
            "Remove top-level 'model' from finetune configs."
        )

    if init_mode == "scratch":
        model_scratch = merged.get("model_scratch")
        if not isinstance(model_scratch, dict):
            raise TypeError(
                "Finetune `init_mode='scratch'` requires `merged['model_scratch']` to be defined as a mapping (dict)."
            )
        merged["model"] = copy.deepcopy(model_scratch)

    else:
        model_overrides = merged.get("model_overrides")
        if model_overrides is None:
            merged["model_overrides"] = {}
        elif not isinstance(model_overrides, dict):
            raise TypeError("`merged['model_overrides']` must be a mapping (dict) for warmstart finetune.")


# ----------------------------------------------------------------------------------------------------------------------
def _model_name_from_config(config: Mapping[str, Any]) -> str:
    """
    Return the model architecture name from a run config.

    Older configs did not persist ``model.name`` because MMT was the only architecture. Treat those snapshots as
    ``mmt``.
    """

    model_cfg = config.get("model")
    if not isinstance(model_cfg, Mapping):
        return "mmt"
    return str(model_cfg.get("name", "mmt"))


# ----------------------------------------------------------------------------------------------------------------------
def infer_model_profile_from_source(model_source: Mapping[str, Any], *, phase: Literal["finetune", "eval"]) -> str:
    """
    Infer the model profile/name from a source run's saved config.

    Parameters
    ----------
    model_source : Mapping[str, Any]
        Model source mapping with either ``run_id`` or ``model_path``.
    phase : Literal["finetune", "eval"]
        Phase requesting the source model. Used for source-path validation/error messages.

    Returns
    -------
    str
        Source model name, e.g. ``"mmt"``.

    """

    src_run_dir, _src_run_id = resolve_model_source_dir(model_source=model_source, phase=phase)
    src_cfg = load_source_run_config_yaml(model_run_dir=src_run_dir)

    return _model_name_from_config(config=src_cfg)


# ----------------------------------------------------------------------------------------------------------------------
def infer_embeddings_profile_from_source(model_source: Mapping[str, Any], *, phase: Literal["finetune", "eval"]) -> str:
    """Return the embedding profile recorded in a source run configuration.

    Parameters
    ----------
    model_source : Mapping[str, Any]
        Model source mapping with either ``run_id`` or ``model_path``.
    phase : Literal["finetune", "eval"]
        Phase requesting the source model. Used for source-path validation/error messages.

    Returns
    -------
    str
        Source embedding profile, defaulting to ``"dct3d"`` for legacy snapshots
        that predate the explicit profile field.

    Raises
    ------
    TypeError
        If the source embedding profile is present but is not a string.

    """

    src_run_dir, _src_run_id = resolve_model_source_dir(model_source=model_source, phase=phase)
    src_cfg = load_source_run_config_yaml(model_run_dir=src_run_dir)
    profile = src_cfg.get("embeddings_profile", "dct3d")
    if not isinstance(profile, str) or not profile.strip():
        raise TypeError(
            f"Source run '{src_run_dir}' has invalid embeddings_profile={profile!r}; expected a non-empty string."
        )
    return profile.strip()


# ----------------------------------------------------------------------------------------------------------------------
def inherit_from_source_model(  # NOSONAR - Ignore cognitive complexity
    merged: MutableMapping[str, Any], *, phase: Literal["finetune", "eval"]
) -> None:
    """
    Load source run config/checkpoints and inherit config for finetune/warmstart or eval.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary (modified in-place).
    phase : Literal["finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If `merged["model_source"]` is not set as a mapping (dict) for finetune/eval phases.
        If `merged["model_overrides"]` is not a mapping (dict) for finetune warmstart.
    KeyError
        If a loaded source config derived from `merged["model_source"]` does not have a key "model".
        If a loaded source config derived from `merged["model_source"]` does not have a key "embeddings" for eval phase.
    FileNotFoundError
        If no checkpoints are found.

    """

    if phase not in ["finetune", "eval"]:
        raise ValueError("Checkpoints are only loaded for finetune/eval phases, got `phase={phase}`.")

    model_source = merged.get("model_source", {})
    if not isinstance(model_source, dict):
        raise TypeError("`merged['model_source']` must be set as a mapping (dict) for finetune/eval phases.")

    src_run_dir, src_run_id_for_yaml = resolve_model_source_dir(model_source=model_source, phase=phase)

    src_cfg = load_source_run_config_yaml(model_run_dir=src_run_dir)
    if "model" not in src_cfg:
        raise KeyError(f"Loaded source config from '{src_run_dir}' does not have a key 'model'.")

    ckpt_root = src_run_dir / "checkpoints"
    best_dir = ckpt_root / "best"
    latest_dir = ckpt_root / "latest"
    if (not best_dir.is_dir()) and (not latest_dir.is_dir()):
        raise FileNotFoundError(
            f"No checkpoints found in {src_run_dir}/checkpoints/\nExpected: {best_dir} or {latest_dir}"
        )

    if phase == "finetune":
        model_overrides = merged.get("model_overrides", {})
        if not isinstance(model_overrides, dict):
            raise TypeError("`merged['model_overrides']` must be a mapping (dict) for warmstart finetune.")

        source_model_name = _model_name_from_config(config=src_cfg)
        requested_model_name = str(merged.get("model_profile", source_model_name))
        override_model_name = model_overrides.get("name")
        if override_model_name is not None and str(override_model_name) != source_model_name:
            raise ValueError(
                "Cross-model warmstart is disabled. "
                f"Source model is {source_model_name!r}, but model_overrides.name={override_model_name!r}."
            )
        if requested_model_name != source_model_name:
            raise ValueError(
                "Cross-model warmstart is disabled. "
                f"Selected model_profile={requested_model_name!r}, but source model is {source_model_name!r}."
            )

        merged["model"] = copy.deepcopy(src_cfg["model"])
        if model_overrides:
            merged["model"] = deep_merge(base=merged["model"], override=model_overrides)

        source_residual = bool((src_cfg["model"]["output_adapters"].get("residual") or {}).get("enable", False))
        requested_residual = bool((merged["model"]["output_adapters"].get("residual") or {}).get("enable", False))
        if requested_residual != source_residual:
            raise ValueError(
                "Warm-start cannot change model.output_adapters.residual.enable because loaded output-adapter "
                f"weights have different semantics (source={source_residual}, requested={requested_residual})."
            )

        # Finetune warmstart: keep preprocess settings from current merged config (common + task overrides), do not
        # force source representation-preprocess inheritance.

    else:  # -> I.e., phase is "eval"
        if "embeddings" not in src_cfg:
            raise KeyError(
                f"Loaded source config from '{src_run_dir}' does not have a key 'embeddings' for eval phase."
            )

        merged["model"] = copy.deepcopy(src_cfg["model"])
        merged["model_profile"] = _model_name_from_config(config=src_cfg)
        merged["embeddings"] = copy.deepcopy(src_cfg["embeddings"])
        merged["embeddings_profile"] = src_cfg.get("embeddings_profile", merged.get("embeddings_profile"))
        inherit_preprocess_representation(merged=merged, src_cfg=src_cfg, allow_override=False)

    merged["model_source"]["run_dir"] = str(src_run_dir)
    if src_run_id_for_yaml:
        merged["model_source"]["run_id"] = src_run_id_for_yaml
