"""
Benchmark/local task definition resolution (scripts_mast).

This module resolves the *benchmark-style task definition* (signals, segmenter, etc.)
by task name:

1) Prefer benchmark by name via the benchmark API:
      benchmark_get_task_config(task_key)

2) If the benchmark does not know that task (KeyError), load a local definition via
   an explicit registry map:
      LOCAL_TASK_DEFS_MAP[task_key] -> YAML path under scripts_mast/configs/

The returned dict must contain:
- task_name: str   (declared by the task definition, not inferred from folder name)

No backwards compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root
from .benchmark_imports import benchmark_get_task_config


# Map: local task key -> YAML path relative to configs_root
# Keep this explicit (symmetric to the benchmark's own tasks_configs_map).
LOCAL_TASK_DEFS_MAP: dict[str, str] = {
    "_test": "local_tasks_def/_test.yaml",
    "pretrain_inputs_actuators_to_inputs_outputs": "local_tasks_def/pretrain_inputs_actuators_to_inputs_outputs.yaml",
    "pretrain_all_to_inputs_outputs": "local_tasks_def/pretrain_all_to_inputs_outputs.yaml",
    "pretrain_inputs_actuators_to_outputs": "local_tasks_def/pretrain_inputs_actuators_to_outputs.yaml",
}


def _resolve_configs_root(configs_root: str | Path) -> Path:
    p = Path(configs_root)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"Task definition YAML must be a mapping (dict). Got {type(data).__name__} at {path}"
        )
    return data


def load_task_definition(
    task_key: str,
    *,
    configs_root: str | Path = "scripts_mast/configs",
    local_map: dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Load the benchmark-style task definition for a given task key."""
    if not isinstance(task_key, str) or not task_key.strip():
        raise ValueError("task_key must be a non-empty string")

    # 1) benchmark task definition
    try:
        cfg_task = benchmark_get_task_config(task_key)
        if not isinstance(cfg_task, dict):
            raise TypeError(
                f"benchmark_get_task_config('{task_key}') returned {type(cfg_task).__name__}, expected dict."
            )
        task_def = dict(cfg_task)
    except KeyError:
        # 2) local fallback
        mp = local_map if local_map is not None else LOCAL_TASK_DEFS_MAP
        rel = mp.get(task_key)
        if rel is None:
            known = ", ".join(sorted(mp.keys())) or "<none>"
            raise KeyError(
                f"Unknown task '{task_key}'. Not found in benchmark and not registered locally. "
                f"Known local tasks: {known}"
            )
        root = _resolve_configs_root(configs_root)
        path = (root / rel).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Local task definition file not found.\n"
                f"  task_key:  {task_key}\n"
                f"  mapped_to: {rel}\n"
                f"  resolved:  {path}\n"
            )
        task_def = _load_yaml(path)

    task_name = task_def.get("task_name")
    if not isinstance(task_name, str) or not task_name.strip():
        raise KeyError(
            f"Task definition loaded for '{task_key}' is missing required key 'task_name'. "
            "Add `task_name: <name>` to the benchmark/local task definition YAML."
        )

    return task_def
