"""
Task-config helpers (MAST integration layer).

This module loads a task configuration YAML and applies a small set of overrides
coming from our ExperimentConfig.

Task config resolution
----------------------
This integration layer supports two ways of specifying `task_config`:

1) Repo-root relative path (recommended):
      task_config: "scripts_mast/configs/task_2-1/config_task_2-1.yaml"

2) Baseline package path (legacy / convenience):
      task_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

The second form is resolved via importlib.resources by treating the first path
component as a Python package name (e.g. "scripts") and resolving the rest of
the path inside that package.

Overrides supported (when provided in `cfg.data`):
  - subset_of_shots
  - local
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _repo_root() -> Path:
    """
    Return repository root assuming this file lives at:
      <repo>/scripts_mast/mast_utils/task_config.py
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _resolve_task_config_path(task_config: Optional[str]) -> Path:
    """
    Resolve a task_config string to a concrete file path.

    Supported forms:
      - absolute path
      - repo-root relative path
      - baseline package path (first component is the package name)
    """
    if not task_config:
        raise ValueError("task_config must be a non-empty string.")

    p = Path(task_config)

    # 1) Absolute path
    if p.is_absolute():
        if p.is_file():
            return p
        raise FileNotFoundError(f"Task config not found: {p}")

    # 2) Repo-root relative
    candidate = (_repo_root() / p).resolve()
    if candidate.is_file():
        return candidate

    # 3) Baseline package path: <pkg>/<path/inside/pkg>
    parts = p.parts
    if len(parts) < 2:
        raise FileNotFoundError(
            f"Task config not found: {candidate} and cannot be resolved as a package path: {task_config!r}"
        )

    pkg_name = parts[0]
    rel = Path(*parts[1:])

    try:
        base = pkg_resources.files(pkg_name)
    except Exception as e:
        raise FileNotFoundError(
            f"Cannot resolve task_config as package path because package {pkg_name!r} "
            f"is not importable. task_config={task_config!r}"
        ) from e

    resolved_path = Path(str(base / rel)).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            "Task config not found. Tried:\n"
            f"  - repo-root: {candidate}\n"
            f"  - package:   {pkg_name}/{rel} -> {resolved_path}"
        )

    return resolved_path


def build_task_config(cfg) -> Dict[str, Any]:
    """
    Load task config and apply overrides from ExperimentConfig.

    This function resolves cfg.raw["task_config"] to a file path, loads the YAML,
    and then applies (when provided in cfg.raw["data"]):
      - subset_of_shots
      - local
    """
    task_cfg_str = None
    if hasattr(cfg, "raw") and isinstance(cfg.raw, dict):
        task_cfg_str = cfg.raw.get("task_config")
        data_cfg = cfg.raw.get("data") or {}
    else:
        # Fallback: allow passing a plain dict-like cfg in tests
        task_cfg_str = getattr(cfg, "task_config", None)
        data_cfg = getattr(cfg, "data", {}) or {}

    task_cfg_path = _resolve_task_config_path(task_cfg_str)
    config_task = _load_yaml(task_cfg_path)

    subset = data_cfg.get("subset_of_shots")
    if subset is not None:
        config_task["subset_of_shots"] = subset

    local_flag = data_cfg.get("local")
    if local_flag is not None:
        config_task["local"] = local_flag

    return config_task
