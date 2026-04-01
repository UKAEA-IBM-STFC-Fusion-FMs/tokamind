"""
Source-run inheritance and finetune model semantics.

This module handles config inheritance from source models for warmstart/eval:
- Resolves source model directories (run_id or path)
- Loads source run config snapshots
- Optionally inherits preprocessing settings (chunk, trim_chunks)
- Applies finetune model semantics (scratch vs warmstart)
- Merges source model config with current overrides

Key concepts:
- Warmstart: load source model weights/config, keep finetune preprocess from current task config
- Scratch: use model_scratch architecture, no source inheritance
- Eval: always inherits from source model (weights + config + embeddings), including preprocess chunk/trim settings
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
def inherit_preprocess_chunk_trim(  # NOSONAR - Ignore cognitive complexity
    merged: MutableMapping[str, Any],
    src_cfg: Mapping[str, Any],
    *,
    allow_override: bool,
) -> None:
    """
    Inherit preprocess.chunk and preprocess.trim_chunks from source config.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary (modified in-place).
    src_cfg : Mapping[str, Any]
        Dictionary with source configuration.
    allow_override : bool
        If True, mapping override is allowed (and override wins at each levels).

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `src_cfg` does not have a key 'preprocess'.
        If `src_cfg["preprocess"]` does not have a required key 'chunk' or 'trim_chunks'.
    TypeError
        If `merged["preprocess"]` is not of type dict.

    """

    src_pre = src_cfg.get("preprocess")
    if not isinstance(src_pre, dict):
        raise KeyError(
            "Source run config `src_cfg` is missing required key 'preprocess' with mapping (dict) value.\n"
            "Expected 'preprocess.chunk' and 'preprocess.trim_chunks' to be present."
        )
    if ("chunk" not in src_pre) or ("trim_chunks" not in src_pre):
        raise KeyError(
            "Source run config key `src_cfg['preprocess']` is missing a required key 'chunk' or 'trim_chunks'.\n"
            "Expected: preprocess.chunk and preprocess.trim_chunks"
        )

    merged_pre = merged.get("preprocess")
    if merged_pre is None:
        merged_pre = {}
        merged["preprocess"] = merged_pre
    if not isinstance(merged_pre, dict):
        raise TypeError("Config key `merged['preprocess']` must be a mapping (dict).")

    override_chunk = None
    override_trim = None
    if allow_override:
        override_chunk = copy.deepcopy(merged_pre.get("chunk"))
        override_trim = copy.deepcopy(merged_pre.get("trim_chunks"))

    merged_pre["chunk"] = copy.deepcopy(src_pre["chunk"])
    merged_pre["trim_chunks"] = copy.deepcopy(src_pre["trim_chunks"])

    if allow_override:
        if override_chunk is not None:
            if isinstance(override_chunk, dict) and isinstance(merged_pre["chunk"], dict):
                merged_pre["chunk"] = deep_merge(base=merged_pre["chunk"], override=override_chunk)
            else:
                merged_pre["chunk"] = override_chunk

        if override_trim is not None:
            if isinstance(override_trim, dict) and isinstance(merged_pre["trim_chunks"], dict):
                merged_pre["trim_chunks"] = deep_merge(base=merged_pre["trim_chunks"], override=override_trim)
            else:
                merged_pre["trim_chunks"] = override_trim


# ----------------------------------------------------------------------------------------------------------------------
def apply_finetune_model_semantics(
    merged: MutableMapping[str, Any], *, init_mode: Literal["warmstart", "scratch"]
) -> None:
    """
    Materialize canonical `model` for scratch and validate warmstart blocks.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary (modified in-place).
    init_mode : Literal["warmstart", "scratch"]
        Finetune initialization mode, either "warmstart" or "scratch".

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If `merged` contains a key 'model'.
    TypeError
        If `merged['finetune_model_overrides']` is not a mapping.
        If `init_mode='scratch'` and `merged['model_scratch']` is not a mapping (dict).
        If `merged['warmstart']` is not a mapping (dict).
        If `merged['warmstart']['model_overrides'] is not a mapping (dict)."

    Notes
    =====

    New finetune semantics:
      - `model_scratch`: full architecture base used for scratch init
      - `finetune_model_overrides`: overrides applied in both scratch and warmstart
      - `warmstart.model_overrides`: additional overrides applied only in warmstart

    """

    if init_mode not in ["warmstart", "scratch"]:
        raise ValueError(f"Unsupported finetune init mode: {init_mode!r}")

    if "model" in merged:
        raise KeyError(
            "Finetune config now uses explicit keys: "
            "'model_scratch', 'finetune_model_overrides', and 'warmstart.model_overrides'. "
            "Remove top-level 'model' from finetune configs."
        )

    common_overrides = merged.get("finetune_model_overrides")
    if common_overrides is None:
        common_overrides = {}
        merged["finetune_model_overrides"] = common_overrides
    if not isinstance(common_overrides, dict):
        raise TypeError("`merged['finetune_model_overrides']` must be a mapping (dict).")

    if init_mode == "scratch":
        model_scratch = merged.get("model_scratch")
        if not isinstance(model_scratch, dict):
            raise TypeError(
                "Finetune `init_mode='scratch'` requires `merged['model_scratch']` to be defined as a mapping (dict)."
            )

        merged["model"] = deep_merge(base=copy.deepcopy(model_scratch), override=common_overrides)

    else:  # -> I.e., init_mode is "warmstart"
        warm_cfg = merged.get("warmstart")
        if warm_cfg is None:
            warm_cfg = {}
            merged["warmstart"] = warm_cfg
        if not isinstance(warm_cfg, dict):
            raise TypeError("`merged['warmstart']` must be a mapping (dict).")

        ws_model_overrides = warm_cfg.get("model_overrides")
        if ws_model_overrides is None:
            warm_cfg["model_overrides"] = {}
        elif not isinstance(ws_model_overrides, dict):
            raise TypeError("`merged['warmstart']['model_overrides'] must be a mapping (dict).")


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
        If `merged["finetune_model_overrides"]` is not a mapping (dict) for finetune.
        iF `merged["warmstart"]["model_overrides"] is not a mapping (dict) for finetune.
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
        common_overrides = merged.get("finetune_model_overrides", {})
        if not isinstance(common_overrides, dict):
            raise TypeError("`merged['finetune_model_overrides']` must be a mapping (dict) for finetune.")

        warm_cfg = merged.get("warmstart", {})
        if not isinstance(warm_cfg, dict):
            raise TypeError("`merged['warmstart']` must be a mapping (dict) for finetune.")

        ws_model_overrides = warm_cfg.get("model_overrides") or {}
        if ws_model_overrides is None:
            ws_model_overrides = {}
        if not isinstance(ws_model_overrides, dict):
            raise TypeError("`merged['warmstart']['model_overrides'] must be a mapping (dict) for finetune.")

        merged["model"] = copy.deepcopy(src_cfg["model"])
        if common_overrides:
            merged["model"] = deep_merge(base=merged["model"], override=common_overrides)
        if ws_model_overrides:
            merged["model"] = deep_merge(base=merged["model"], override=ws_model_overrides)
        # Finetune warmstart: keep preprocess settings from current merged config (common + task overrides), do not
        # force source chunk/trim inheritance.

    else:  # -> I.e., phase is "eval"
        if "embeddings" not in src_cfg:
            raise KeyError(
                f"Loaded source config from '{src_run_dir}' does not have a key 'embeddings' for eval phase."
            )

        merged["model"] = copy.deepcopy(src_cfg["model"])
        merged["embeddings"] = copy.deepcopy(src_cfg["embeddings"])
        merged["embeddings_profile"] = src_cfg.get("embeddings_profile", merged.get("embeddings_profile"))
        inherit_preprocess_chunk_trim(merged=merged, src_cfg=src_cfg, allow_override=False)

    merged["model_source"]["run_dir"] = str(src_run_dir)
    if src_run_id_for_yaml:
        merged["model_source"]["run_id"] = src_run_id_for_yaml
