"""
YAML loading and deep-merge utilities for experiment config assembly.

This module provides the core config merging logic used to assemble experiment configurations from multiple YAML files.
It handles:
- Path resolution (relative to repo root or absolute)
- YAML file loading with safe defaults
- Deep dictionary merging with special handling for train.stages lists
- Convention-based config file discovery and loading

The merge strategy preserves nested structure while allowing overrides at any level, with special logic for merging
training stage lists by stage name.
"""

from __future__ import annotations

import copy
import logging
import yaml
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal
from pathlib import Path

from mmt.data.embeddings.dct3d import DCT3D_BOOTSTRAP_DEFAULTS
from mmt.utils.paths import REPO_ROOT


logger = logging.getLogger("mmt.ConfigLoader")


# ----------------------------------------------------------------------------------------------------------------------
def resolve_from_repo_root(rel_or_abs: str) -> Path:
    """
    Resolve a path relative to repo root unless already absolute.

    Parameters
    ----------
    rel_or_abs : str
        Input path to be resolved.

    Returns
    -------
    Path
        Resolved path.

    """

    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# ----------------------------------------------------------------------------------------------------------------------
def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML file as dict. Empty files result in {}.

    Parameters
    ----------
    path : Path
        Path to target YAML file to be loaded as dictionary.

    Returns
    -------
    dict[str, Any]
        Dictionary with contents loaded from target YAML file.

    """

    with path.open(mode="r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ----------------------------------------------------------------------------------------------------------------------
def merge_stage_lists(base: list[Any], override: list[Any]) -> list[Any]:
    """
    Merge `train.stages` lists by stage name for partial overrides.

    Parameters
    ----------
    base : list[Any]
        Base list.
    override : list[Any]
        Override list.

    Returns
    -------
    list[Any]
        Resulting merged list.

    """

    if not (
        all(isinstance(x, dict) and "name" in x for x in base)
        and all(isinstance(x, dict) and "name" in x for x in override)
    ):
        return copy.deepcopy(override)

    override_map: dict[str, dict[str, Any]] = {}
    override_order: list[str] = []
    for stage in override:
        name = str(stage.get("name"))
        override_map[name] = stage
        override_order.append(name)

    base_names: set[str] = set()
    merged_list: list[Any] = []
    for stage in base:
        name = str(stage.get("name"))
        base_names.add(name)
        if name in override_map:
            merged_list.append(deep_merge(base=stage, override=override_map[name]))
        else:
            merged_list.append(copy.deepcopy(stage))

    for name in override_order:
        if name not in base_names:
            merged_list.append(copy.deepcopy(override_map[name]))

    return merged_list


# ----------------------------------------------------------------------------------------------------------------------
def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge nested mappings; override wins at each level.

    Parameters
    ----------
    base : Mapping[str, Any]
        Base mapping
    override : Mapping[str, Any]
        Override mapping.

    Returns
    -------
    dict[str, Any]
        Resulting deep-merged mapping.

    """

    out = dict(copy.deepcopy(base))  # -> The dict() wrapping is to set the return type as dict.
    for key, val in override.items():
        if (key in out and isinstance(out[key], dict)) and isinstance(val, dict):
            out[key] = deep_merge(base=out[key], override=val)
        elif key in out and key == "stages" and isinstance(out[key], list) and isinstance(val, list):
            out[key] = merge_stage_lists(base=out[key], override=val)
        else:
            out[key] = copy.deepcopy(val)

    return out


# ----------------------------------------------------------------------------------------------------------------------
def _role_expanded_dct3d_defaults() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Return DCT3D bootstrap defaults in the role-keyed shape consumed by ``build_signal_specs``.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Defaults keyed by role, then modality.

    """

    return {role: copy.deepcopy(DCT3D_BOOTSTRAP_DEFAULTS) for role in ("input", "actuator", "output")}


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def _role_expanded_identity_defaults() -> dict[str, dict[str, dict[str, Any]]]:
    """
    Return identity-encoder defaults for every role and modality.

    The identity profile applies no transform (signals are flattened at full/raw dimension) and does not tune. It is a
    dataset-agnostic baseline usable by any model profile in either integration layer.

    Returns
    -------
    dict[str, dict[str, dict[str, Any]]]
        Identity defaults keyed by role, then modality.

    """

    modality_defaults = {
        "timeseries": {"encoder_name": "identity", "encoder_kwargs": {}},
        "profile": {"encoder_name": "identity", "encoder_kwargs": {}},
        "video": {"encoder_name": "identity", "encoder_kwargs": {}},
    }
    return {role: copy.deepcopy(modality_defaults) for role in ("input", "actuator", "output")}


# ----------------------------------------------------------------------------------------------------------------------
def _load_task_block(path: Path, *, task: str, phase: str) -> dict[str, Any]:
    """
    Load one task entry from an aggregated phase override file.

    Aggregated files are shaped as ``tasks: {<task>: {...}}``. Missing files or missing task entries are treated as
    empty overrides.

    Parameters
    ----------
    path : Path
        Path to the aggregated phase override YAML file.
    task : str
        Task identifier whose override block should be loaded.
    phase : str
        Phase name, used only for error messages.

    Returns
    -------
    dict[str, Any]
        Task-specific override mapping, or an empty dict if the file/task entry is absent.

    Raises
    ------
    TypeError
        If the file-level ``tasks`` key or selected task entry is not a mapping.
    """

    if not path.is_file():
        return {}

    data = load_yaml(path=path)
    tasks = data.get("tasks") or {}
    if not isinstance(tasks, Mapping):
        raise TypeError(f"Config file {path} must define a mapping at key 'tasks' for {phase} overrides.")

    task_cfg = tasks.get(task) or {}
    if not isinstance(task_cfg, Mapping):
        raise TypeError(f"Config file {path} has non-mapping override for task={task!r}.")

    return dict(task_cfg)


# ----------------------------------------------------------------------------------------------------------------------
def _profile_default_path(configs_root_path: Path, model_profile: str, embeddings_profile: str) -> Path:
    """
    Return the default YAML path for an embedding profile.

    Parameters
    ----------
    configs_root_path : Path
        Root directory containing configuration files.
    model_profile : str
        Model profile name; embeddings live under that model's folder.
    embeddings_profile : str
        Embedding profile name.

    Returns
    -------
    Path
        Path to ``<model_profile>/embeddings/<profile>/_default.yaml``.

    """

    return configs_root_path / model_profile / "embeddings" / embeddings_profile / "_default.yaml"


# ----------------------------------------------------------------------------------------------------------------------
def _profile_task_path(configs_root_path: Path, model_profile: str, embeddings_profile: str, task: str) -> Path:
    """
    Return the task-specific YAML path for an embedding profile.

    Parameters
    ----------
    configs_root_path : Path
        Root directory containing configuration files.
    model_profile : str
        Model profile name; embeddings live under that model's folder.
    embeddings_profile : str
        Embedding profile name.
    task : str
        Task identifier.

    Returns
    -------
    Path
        Path to ``<model_profile>/embeddings/<profile>/<task>.yaml``.

    """

    return configs_root_path / model_profile / "embeddings" / embeddings_profile / f"{task}.yaml"


# ----------------------------------------------------------------------------------------------------------------------
def _materialize_embedding_runtime_blocks(
    merged: MutableMapping[str, Any], *, phase: str, embeddings_profile: str
) -> None:
    """
    Materialize runtime embedding defaults, signal overrides, and tuning from the selected rank-tuned profile block.

    Runtime code expects shared defaults at ``embeddings.defaults``, signal overrides at
    ``embeddings.per_signal_overrides``, and the existing tuner reads ``embeddings.tuning``. DCT3D role/modality
    defaults are intentionally internal: normal DCT3D runs tune rank-mode artifacts, while non-DCT3D profiles still
    need DCT3D fallback defaults for any signals not explicitly overridden.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary to update in place.
    phase : str
        Experiment phase used to select ``embeddings.<profile>.tuning.<phase>``.
    embeddings_profile : str
        Selected embedding profile name.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If the new DCT3D profile blocks are present but not mappings.
    """

    profile_name = str(embeddings_profile).lower()

    if profile_name == "identity":
        # Virtual profile: every signal uses the identity codec (no transform, no tuning). Works for any model
        # profile in either integration layer; per-signal overrides (if any) still apply on top of these defaults.
        embeddings_cfg = merged.get("embeddings")
        if not isinstance(embeddings_cfg, MutableMapping):
            embeddings_cfg = {}
            merged["embeddings"] = embeddings_cfg
        embeddings_cfg["defaults"] = _role_expanded_identity_defaults()
        return

    embeddings_cfg = merged.get("embeddings")
    if not isinstance(embeddings_cfg, MutableMapping):
        return

    is_dct3d_profile = profile_name.startswith("dct3d")
    dct3d_cfg = embeddings_cfg.get("dct3d")
    if not isinstance(dct3d_cfg, Mapping):
        return
    embeddings_cfg["defaults"] = _role_expanded_dct3d_defaults()
    tuning_block_name = "dct3d"

    is_rank_tuned_profile = is_dct3d_profile
    if not is_rank_tuned_profile:
        # Task-only profiles such as VAE still need baseline defaults for signals they do not override, but they do not
        # participate in rank-tuning/source policy. Preserve their top-level per-signal overrides.
        top_level_overrides = embeddings_cfg.get("per_signal_overrides")
        if (top_level_overrides is not None) and (not isinstance(top_level_overrides, Mapping)):
            raise TypeError("embeddings.per_signal_overrides must be a mapping when provided.")
        return

    tuning_profile_cfg = embeddings_cfg.get(tuning_block_name)
    if not isinstance(tuning_profile_cfg, Mapping):
        raise TypeError(f"embeddings.{tuning_block_name} must be a mapping for profile {embeddings_profile!r}.")

    profile_overrides = tuning_profile_cfg.get("per_signal_overrides")
    if isinstance(profile_overrides, Mapping):
        embeddings_cfg["per_signal_overrides"] = copy.deepcopy(profile_overrides)
    elif profile_overrides is not None:
        raise TypeError(f"embeddings.{tuning_block_name}.per_signal_overrides must be a mapping when provided.")

    if phase == "eval":
        embeddings_cfg.pop("tuning", None)
        return

    tuning_cfg = tuning_profile_cfg.get("tuning") or {}
    if not isinstance(tuning_cfg, Mapping):
        raise TypeError(f"embeddings.{tuning_block_name}.tuning must be a mapping when provided.")

    common_tuning = tuning_cfg.get("common") or {}
    if not isinstance(common_tuning, Mapping):
        raise TypeError(f"embeddings.{tuning_block_name}.tuning.common must be a mapping when provided.")

    phase_tuning = tuning_cfg.get(phase)
    if not isinstance(phase_tuning, Mapping):
        raise TypeError(f"embeddings.{tuning_block_name}.tuning.{phase} must be a mapping for phase={phase!r}.")

    effective_tuning = deep_merge(base=common_tuning, override=phase_tuning)
    embeddings_cfg["tuning"] = effective_tuning


# ----------------------------------------------------------------------------------------------------------------------
def load_and_merge_base_configs(
    *,
    task: str,
    phase: Literal["pretrain", "finetune", "eval"],
    model_profile: str,
    embeddings_profile: str,
    configs_root_path: Path,
    finetune_init: Literal["warmstart", "scratch"] | None = None,
) -> dict[str, Any]:
    """
    Load and merge common/task configs using the standard hierarchy.

    Parameters
    ----------
    task : str
        Task identifier.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".
    model_profile : str
        Model profile name; its folder ``configs/<model_profile>/`` holds ``phases/``, ``tasks/`` and ``embeddings/``.
    embeddings_profile : str
        Embedding profile name under ``configs/<model_profile>/embeddings/``.
    finetune_init : Literal["warmstart", "scratch"] | None
        Finetune initialization mode. Used only when `phase == "finetune"` to load the mode-specific common config.
    configs_root_path : Path
        Path to the root directory for configuration files.
    Returns
    -------
    dict[str, Any]
        Resulting merged mapping from loaded common/task configs mappings.

    Raises
    ------
    FileNotFoundError
        If required config file is not found.
        If required embedding profile defaults are missing.

    """

    model_profile = str(model_profile).strip()
    if not model_profile:
        raise ValueError("`model_profile` must be a non-empty string.")
    if any(part in model_profile for part in ["/", "\\", ".."]):
        raise ValueError(f"`model_profile` must be a simple folder name, got {model_profile!r}.")

    model_dir = configs_root_path / model_profile
    phases_dir = model_dir / "phases"
    if not phases_dir.is_dir():
        raise FileNotFoundError(
            f"Required model profile config directory not found for model_profile={model_profile!r}:\n  {phases_dir}"
        )

    if phase == "finetune":
        if finetune_init not in ["warmstart", "scratch"]:
            raise ValueError(
                "`finetune_init` must be provided when loading finetune configs and must be one of "
                f"['warmstart', 'scratch'], got {finetune_init!r}."
            )
        phase_common_path = phases_dir / f"finetune_{finetune_init}.yaml"
        if not phase_common_path.is_file():
            raise FileNotFoundError(f"Required init-specific finetune config not found at path {phase_common_path}.")
    else:
        phase_common_path = phases_dir / f"{phase}.yaml"
        if not phase_common_path.is_file():
            raise FileNotFoundError(f"Required config file not found at path {phase_common_path}.")

    # The virtual "identity" profile has no files on disk; its defaults are synthesized in
    # _materialize_embedding_runtime_blocks, so skip embedding-file resolution entirely.
    is_identity_profile = str(embeddings_profile).lower() == "identity"

    task_profile_path = _profile_task_path(
        configs_root_path=configs_root_path,
        model_profile=model_profile,
        embeddings_profile=embeddings_profile,
        task=task,
    )

    merged: dict[str, Any] = {}
    if not is_identity_profile:
        requested_profile_default_path = _profile_default_path(
            configs_root_path=configs_root_path,
            model_profile=model_profile,
            embeddings_profile=embeddings_profile,
        )
        profile_default_path = requested_profile_default_path

        if not profile_default_path.is_file():
            if not task_profile_path.is_file():
                embeddings_root = configs_root_path / model_profile / "embeddings"
                available = (
                    sorted(p.name for p in embeddings_root.iterdir() if p.is_dir()) if embeddings_root.is_dir() else []
                )
                raise FileNotFoundError(
                    f"Embedding profile={embeddings_profile!r} has no default file and no task file for task={task!r} "
                    f"under model_profile={model_profile!r}.\n"
                    f"Expected one of:\n"
                    f"  {requested_profile_default_path}\n"
                    f"  {task_profile_path}\n"
                    f"Model profile {model_profile!r} provides these embedding profiles: {available or '(none)'} "
                    f"(plus the virtual 'identity' profile). "
                    "Pass --emb_profile with one of the listed embedding profiles."
                )
            profile_default_path = _profile_default_path(
                configs_root_path=configs_root_path,
                model_profile=model_profile,
                embeddings_profile="dct3d",
            )

        if not profile_default_path.is_file():
            raise FileNotFoundError(
                f"Required embedding profile default not found for profile={embeddings_profile!r}.\n"
                f"Expected file:\n"
                f"  {profile_default_path}"
            )

        merged = deep_merge(base=merged, override=load_yaml(path=profile_default_path))

    merged = deep_merge(base=merged, override=load_yaml(path=phase_common_path))

    # Model-specific per-task overrides live under this model's own tasks/ folder.
    phase_tasks_path = model_dir / "tasks" / f"{phase}_tasks.yaml"
    phase_task_overrides = _load_task_block(path=phase_tasks_path, task=task, phase=phase)
    if phase_task_overrides:
        merged = deep_merge(base=merged, override=phase_task_overrides)

    if phase in ["pretrain", "finetune"]:
        if task_profile_path.is_file():
            merged = deep_merge(base=merged, override=load_yaml(path=task_profile_path))

    # Optional local overrides (gitignored, never committed).
    # Create scripts_mast/configs/local_overrides.yaml to override any config value
    # for machine-specific settings (e.g. local data paths on a laptop or HPC cluster).
    local_overrides_path = configs_root_path / "local_overrides.yaml"
    if local_overrides_path.is_file():
        logger.info(f"Applying local overrides from {local_overrides_path}")
        merged = deep_merge(base=merged, override=load_yaml(path=local_overrides_path))

    _materialize_embedding_runtime_blocks(merged=merged, phase=phase, embeddings_profile=embeddings_profile)

    return merged
