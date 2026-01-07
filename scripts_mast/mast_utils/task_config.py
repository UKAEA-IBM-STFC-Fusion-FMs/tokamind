"""
Task-config helpers (MAST integration layer).

This module loads a task configuration YAML and applies a small set of overrides
coming from the merged ExperimentConfig.

`task_config` resolution
------------------------
We support:

1) Absolute path:
      task_config: "/abs/path/to/config_task.yaml"

2) Repo-root relative path (recommended):
      task_config: "scripts_mast/configs/pretrain_global/pretrain_task_all_to_inputs_outputs.yaml"

3) Baseline package path (convenience):
      task_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

For (3), the first path component is treated as a Python package name and the
rest as a path inside that package, resolved via importlib.resources.

Overrides applied (if present in cfg.raw["data"]):
  - subset_of_shots
  - local
"""

from __future__ import annotations

import importlib.resources as pkg_resources
from pathlib import Path
from typing import Any, Dict

import yaml


def _repo_root() -> Path:
    """
    Repository root assuming this file lives at:
      <repo>/scripts_mast/mast_utils/task_config.py
    """
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_task_config_path(task_config: str) -> Path:
    """
    Resolve `task_config` to an existing file path.

    Supported:
      - absolute path
      - repo-root relative path
      - baseline package path (<pkg>/<path/inside/pkg>)
    """
    if not isinstance(task_config, str) or not task_config.strip():
        raise ValueError("task_config must be a non-empty string.")

    p = Path(task_config)

    # 1) Absolute path
    if p.is_absolute():
        if p.is_file():
            return p
        raise FileNotFoundError(f"Task config not found: {p}")

    # 2) Repo-root relative
    repo_candidate = (_repo_root() / p).resolve()
    if repo_candidate.is_file():
        return repo_candidate

    # 3) Baseline package path: <pkg>/<path/inside/pkg>
    parts = p.parts
    if len(parts) < 2:
        raise FileNotFoundError(
            f"Task config not found at repo-root path: {repo_candidate} "
            f"and cannot be resolved as a package path: {task_config!r}"
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

    resolved = Path(str(base / rel)).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(
            "Task config not found. Tried:\n"
            f"  - repo-root: {repo_candidate}\n"
            f"  - package:   {pkg_name}/{rel} -> {resolved}"
        )

    return resolved


def build_task_config(cfg) -> Dict[str, Any]:
    """
    Load the task config specified by cfg.raw["task_config"] and apply overrides
    from cfg.raw["data"].

    Expected cfg:
      - cfg.raw is a dict
      - cfg.raw["task_config"] exists (string)
      - cfg.raw.get("data", {}) may contain subset_of_shots/local
    """
    if not hasattr(cfg, "raw") or not isinstance(cfg.raw, dict):
        raise TypeError(
            "build_task_config expects an ExperimentConfig-like object with `.raw` dict."
        )

    task_cfg_str = cfg.raw.get("task_config")
    if task_cfg_str is None:
        raise KeyError("Missing required config entry: task_config")

    data_cfg = cfg.raw.get("data") or {}
    if not isinstance(data_cfg, dict):
        raise TypeError("cfg.raw['data'] must be a dict if provided.")

    task_cfg_path = _resolve_task_config_path(task_cfg_str)
    config_task = _load_yaml(task_cfg_path)

    # Apply supported overrides
    if "subset_of_shots" in data_cfg and data_cfg["subset_of_shots"] is not None:
        config_task["subset_of_shots"] = data_cfg["subset_of_shots"]

    if "local" in data_cfg and data_cfg["local"] is not None:
        config_task["local"] = data_cfg["local"]

    return config_task
