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
import yaml
from collections.abc import Mapping
from typing import Any, Literal
from pathlib import Path

from mmt.utils.paths import REPO_ROOT


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
def load_and_merge_base_configs(
    *,
    task: str,
    phase: Literal["pretrain", "finetune", "eval"],
    embeddings_profile: str,
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> dict[str, Any]:
    """
    Load and merge common/task configs using the standard hierarchy.

    Parameters
    ----------
    task : str
        Task identifier.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".
    embeddings_profile : str
        Path to YAML file with embeddings profile.
    configs_root_path : Path
        Path to the root directory for configuration files.
    tasks_overrides_dir : Path
        Path to the task overrides directory.

    Returns
    -------
    dict[str, Any]
        Resulting merged mapping from loaded common/task configs mappings.

    Raises
    ------
    FileNotFoundError
        If required config file is not found.
        If required task-level embedding overrides for profile is missing.

    """

    common_dir = configs_root_path / "common"

    embeddings_path = common_dir / "embeddings.yaml"
    phase_common_path = common_dir / f"{phase}.yaml"
    for path in [embeddings_path, phase_common_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required config file not found at path {path}.")

    merged: dict[str, Any] = {}
    merged = deep_merge(base=merged, override=load_yaml(path=embeddings_path))
    merged = deep_merge(base=merged, override=load_yaml(path=phase_common_path))

    phase_overrides_path = tasks_overrides_dir / f"{phase}_overrides.yaml"
    if phase_overrides_path.is_file():
        merged = deep_merge(base=merged, override=load_yaml(path=phase_overrides_path))

    if phase in ["pretrain", "finetune"]:
        embeddings_overrides_path = tasks_overrides_dir / "embeddings_overrides" / f"{embeddings_profile}.yaml"
        if not embeddings_overrides_path.is_file():
            raise FileNotFoundError(
                f"Missing required task-level embedding overrides for profile={embeddings_profile!r}, task={task!r}.\n"
                f"Expected file:\n"
                f"  {embeddings_overrides_path}"
            )
        merged = deep_merge(base=merged, override=load_yaml(path=embeddings_overrides_path))

    return merged
