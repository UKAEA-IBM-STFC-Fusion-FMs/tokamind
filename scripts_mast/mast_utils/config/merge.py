"""
YAML loading and deep-merge utilities for experiment config assembly.

This module provides the core config merging logic used to assemble experiment
configurations from multiple YAML files. It handles:
- Path resolution (relative to repo root or absolute)
- YAML file loading with safe defaults
- Deep dictionary merging with special handling for train.stages lists
- Convention-based config file discovery and loading

The merge strategy preserves nested structure while allowing overrides at any
level, with special logic for merging training stage lists by stage name.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root


def resolve_from_repo_root(rel_or_abs: str) -> Path:
    """Resolve a path relative to repo root unless already absolute."""
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file as dict; empty files become {}."""
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_stage_lists(base: list[Any], override: list[Any]) -> list[Any]:
    """Merge `train.stages` lists by stage name for partial overrides."""
    if not (
        all(isinstance(x, dict) and "name" in x for x in base)
        and all(isinstance(x, dict) and "name" in x for x in override)
    ):
        return copy.deepcopy(override)

    override_map: Dict[str, Dict[str, Any]] = {}
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
            merged_list.append(deep_merge(stage, override_map[name]))
        else:
            merged_list.append(copy.deepcopy(stage))

    for name in override_order:
        if name not in base_names:
            merged_list.append(copy.deepcopy(override_map[name]))

    return merged_list


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge nested mappings; override wins at each level."""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = deep_merge(out[key], val)
        elif (
            key in out
            and key == "stages"
            and isinstance(out[key], list)
            and isinstance(val, list)
        ):
            out[key] = merge_stage_lists(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_and_merge_base_configs(
    *,
    task: str,
    phase: str,
    embeddings_profile: str,
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> Dict[str, Any]:
    """Load and merge common/task configs using the standard hierarchy."""
    common_dir = configs_root_path / "common"

    embeddings_path = common_dir / "embeddings.yaml"
    phase_common_path = common_dir / f"{phase}.yaml"
    for path in (embeddings_path, phase_common_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required config not found: {path}")

    phase_overrides_path = tasks_overrides_dir / f"{phase}_overrides.yaml"
    embeddings_overrides_path = (
        tasks_overrides_dir / "embeddings_overrides" / f"{embeddings_profile}.yaml"
    )

    merged: Dict[str, Any] = {}
    merged = deep_merge(merged, load_yaml(embeddings_path))
    merged = deep_merge(merged, load_yaml(phase_common_path))

    if phase_overrides_path.is_file():
        merged = deep_merge(merged, load_yaml(phase_overrides_path))

    if phase in ("pretrain", "finetune"):
        if not embeddings_overrides_path.is_file():
            raise FileNotFoundError(
                "Missing required task-level embedding overrides for "
                f"profile={embeddings_profile!r}, task={task!r}.\n"
                "Expected file:\n"
                f"  {embeddings_overrides_path}"
            )
        merged = deep_merge(merged, load_yaml(embeddings_overrides_path))

    return merged
