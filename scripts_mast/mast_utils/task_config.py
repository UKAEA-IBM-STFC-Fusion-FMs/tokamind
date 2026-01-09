"""
Task config resolver (MAST integration layer).

New rule (no backward compatibility)
------------------------------------
- We do NOT accept cfg.raw["task_config"] anymore.
- We infer the benchmark-style task config ONLY from cfg.raw["task"].

Resolution strategy
-------------------
1) Try benchmark by name via benchmark API:
       cfg_task = benchmark_get_task_config(task_name)

2) If benchmark doesn't know that task (KeyError), load locally via an explicit map:
       LOCAL_TASK_CONFIGS_MAP[task_name] -> YAML path under scripts_mast/configs/

Overrides applied (from cfg.raw["data"] if present)
---------------------------------------------------
- subset_of_shots
- local
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root
from .benchmark_imports import benchmark_get_task_config


# Root under which local benchmark-style task YAMLs live.
# (Keep this consistent with DEFAULT_CONFIGS_ROOT used by load_experiment_config)
DEFAULT_CONFIGS_ROOT = "scripts_mast/configs"


# Local task registry (explicit, symmetric to benchmark tasks_configs_map)
# Map: local_task_name -> path relative to scripts_mast/configs/
#
# Add a new local task by:
#  1) adding the YAML file under scripts_mast/configs/<...>
#  2) registering it here

LOCAL_TASK_CONFIGS_MAP: dict[str, str] = {
    # default task used by run_* scripts
    "_test": "local_tasks_def/_test.yaml",
    "pretrain_all_to_inputs_outputs": "local_tasks_def/pretrain_all_to_inputs_outputs.yaml",
    "pretrain_inputs_actuators_to_inputs_outputs": "local_tasks_def/pretrain_inputs_actuators_to_inputs_outputs.yaml",
    "pretrain_inputs_actuators_to_outputs": "local_tasks_def/pretrain_inputs_actuators_to_outputs.yaml",
}


def _configs_root() -> Path:
    return (get_repo_root() / DEFAULT_CONFIGS_ROOT).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Task YAML must be a mapping (dict). Got {type(data).__name__} at {path}"
        )
    return data


def _load_local_task_config(task_name: str) -> Dict[str, Any]:
    rel = LOCAL_TASK_CONFIGS_MAP.get(task_name)
    if rel is None:
        known = ", ".join(sorted(LOCAL_TASK_CONFIGS_MAP.keys())) or "<none>"
        raise KeyError(
            f"Task '{task_name}' is not a known benchmark task and is not registered as a local task.\n"
            f"Register it in LOCAL_TASK_CONFIGS_MAP. Known local tasks: {known}"
        )

    path = (_configs_root() / rel).resolve()
    if not path.is_file():
        raise FileNotFoundError(
            "Local task config file not found.\n"
            f"  task_name: {task_name}\n"
            f"  mapped_to: {rel}\n"
            f"  resolved:  {path}\n"
        )

    return _load_yaml(path)


def build_task_config(cfg: Any) -> Dict[str, Any]:
    """
    Build the benchmark-style task config dict inferred from cfg.task, then apply overrides.

    Expected cfg:
      - cfg.raw is a dict
      - cfg.raw["task"] exists (string)
      - cfg.raw["task_config"] MUST NOT exist
      - cfg.raw.get("data", {}) may contain subset_of_shots/local
    """
    if not hasattr(cfg, "raw") or not isinstance(cfg.raw, dict):
        raise TypeError(
            "build_task_config expects an ExperimentConfig-like object with `.raw` dict."
        )

    task_name = cfg.raw.get("task")
    if not isinstance(task_name, str) or not task_name.strip():
        raise KeyError("Missing required config entry: task (string)")

    data_cfg = cfg.raw.get("data") or {}
    if not isinstance(data_cfg, dict):
        raise TypeError("cfg.raw['data'] must be a dict if provided.")

    # 1) Benchmark by name (preferred)
    try:
        cfg_task = benchmark_get_task_config(task_name)
        if not isinstance(cfg_task, dict):
            raise TypeError(
                f"benchmark_get_task_config('{task_name}') returned {type(cfg_task).__name__}, expected dict."
            )
        config_task = copy.deepcopy(cfg_task)

    # 2) Local registry fallback
    except KeyError:
        config_task = _load_local_task_config(task_name)

    # Ensure task_name exists inside the task config
    config_task.setdefault("task_name", task_name)

    # Apply supported overrides from cfg.data
    if "subset_of_shots" in data_cfg and data_cfg["subset_of_shots"] is not None:
        config_task["subset_of_shots"] = data_cfg["subset_of_shots"]

    if "local" in data_cfg and data_cfg["local"] is not None:
        config_task["local"] = data_cfg["local"]

    return config_task
