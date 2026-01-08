"""
Task-config helpers (MAST integration layer).

This module loads a task configuration YAML and applies a small set of overrides
coming from the merged ExperimentConfig.

`task_config` resolution
------------------------
We support:

1) Absolute path:
      task_config: "/abs/path/to/config_task.yaml"

2) Repo-root relative (multi-modal-transformer repo):
      task_config: "scripts_mast/configs/pretrain_global/pretrain_task_*.yaml"

3) Baseline-repo relative (fairmast-data-preprocessing repo):
      task_config: "configs_task/task_2_magnetics_dynamics/config_task_2-1.yaml"

4) Legacy baseline package path (convenience; only if it exists):
      task_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

Overrides applied (if present in cfg.raw["data"]):
  - subset_of_shots
  - local
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _mmt_repo_root() -> Path:
    """
    Repository root assuming this file lives at:
      <mmt_repo>/scripts_mast/mast_utils/task_config.py
    """
    return Path(__file__).resolve().parents[2]


def _baseline_repo_root() -> Path | None:
    """
    Try to locate the baseline (fairmast-data-preprocessing) repo root by
    using the installed baseline module path.

    Returns None if the baseline package is not importable.
    """
    try:
        import MAST_benchmark.tasks as tasks  # baseline module
    except Exception:
        return None

    # If the baseline exposes REPO_ROOT, prefer it.
    repo_root = getattr(tasks, "REPO_ROOT", None)
    if isinstance(repo_root, str) and repo_root:
        return Path(repo_root).resolve()

    # Otherwise infer from module location:
    # <baseline_root>/scripts/pipelines/utils/preprocessing_utils.py
    p = Path(tasks.__file__).resolve()
    # parents: utils(0) -> pipelines(1) -> scripts(2) -> baseline_root(3)
    if len(p.parents) >= 4:
        return p.parents[1]
    return None


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_task_config_path(task_config: str) -> Path:
    """
    Resolve `task_config` to an existing file path.

    Supported:
      - absolute path
      - mmt repo-root relative
      - baseline repo-root relative
      - legacy baseline relative path (if still present)
    """
    if not isinstance(task_config, str) or not task_config.strip():
        raise ValueError("task_config must be a non-empty string.")

    p = Path(task_config)

    # 1) Absolute path
    if p.is_absolute():
        if p.is_file():
            return p
        raise FileNotFoundError(f"Task config not found: {p}")

    tried: list[Path] = []

    # 2) MMT repo-root relative
    mmt_candidate = (_mmt_repo_root() / p).resolve()
    tried.append(mmt_candidate)
    if mmt_candidate.is_file():
        return mmt_candidate

    # 3) Baseline repo-root relative (new baseline layout)
    base_root = _baseline_repo_root()
    if base_root is not None:
        baseline_candidate = (base_root / p).resolve()
        tried.append(baseline_candidate)
        if baseline_candidate.is_file():
            return baseline_candidate

    # Nothing worked
    msg = "Task config not found. Tried:\n" + "\n".join(f"  - {x}" for x in tried)
    raise FileNotFoundError(msg)


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
