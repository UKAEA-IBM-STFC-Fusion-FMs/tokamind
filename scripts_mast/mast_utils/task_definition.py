"""
Benchmark/local task definition resolution (scripts_mast).

This module resolves the *benchmark-style task definition* (signals, segmenter, etc.) by task name:

1) Prefer benchmark by name via the benchmark API:
      benchmark_get_task_config(task_key)

2) If the benchmark does not know that task (KeyError), load a local definition via an explicit registry map:
      LOCAL_TASK_DEFS_MAP[task_key] -> YAML path under scripts_mast/configs/

The returned dict must contain:
- task_name: str   (declared by the task definition, not inferred from folder name)

No backwards compatibility.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

from mmt.utils.paths import REPO_ROOT

from .benchmark_imports import benchmark_get_task_config


# ----------------------------------------------------------------------------------------------------------------------
# Map: local task key -> YAML path relative to configs_root
# Keep this explicit (symmetric to the benchmark's own tasks_configs_map).

LOCAL_TASK_DEFS_MAP: dict[str, str] = {
    "_test": "local_tasks_def/_test.yaml",
    "pretrain_inputs_actuators_to_inputs_outputs": "local_tasks_def/pretrain_inputs_actuators_to_inputs_outputs.yaml",
    "pretrain_all_to_inputs_outputs": "local_tasks_def/pretrain_all_to_inputs_outputs.yaml",
    "pretrain_inputs_actuators_to_outputs": "local_tasks_def/pretrain_inputs_actuators_to_outputs.yaml",
}


# ----------------------------------------------------------------------------------------------------------------------
def _resolve_configs_root(configs_root: str | Path) -> Path:
    """
    Resolve path to the root directory for configuration files.

    Parameters
    ----------
    configs_root : str | Path
        Path to the root directory for configuration files.

    Returns
    -------
    Path
        Resolved `configs_root`.

    """

    p = Path(configs_root)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


# ----------------------------------------------------------------------------------------------------------------------
def _load_yaml(path: Path) -> dict[str, Any]:
    """
    Load data in YAML file as dictionary.

    Parameters
    ----------
    path : Path
        Path to target YAML file.

    Returns
    -------
    dict[str, Any]
        Loaded data in target YAML file as dictionary.

    Raises
    ------
    TypeError
        If loading of target `path` does not result in a dictionary.

    """

    with path.open(mode="r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Task definition YAML must be a mapping (dict), got {type(data).__name__} at {path}.")
    return data


# ----------------------------------------------------------------------------------------------------------------------
def load_task_definition(
    task_key: str,
    *,
    configs_root: str | Path = "scripts_mast/configs",
    local_map: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Load the benchmark-style task definition for a given task key.

    Parameters
    ----------
    task_key : str
        Task key.
    configs_root : str | Path
        Path to the root directory for configuration files, only used within the local fallback pipeline (i.e., when
        configuration cannot be obtained from TokaMark). It is locally resolved in case it is not an absolute path.
        Optional. Default: "scripts_mast/configs".
    local_map : dict[str, str] | None
        Local mapping, used if configuration mapping cannot be obtained from TokaMark.
        Optional. Default: None.

    Returns
    -------
    dict[str, Any]
        Dictionary with task definition.

    Raises
    ------
    KeyError
        If `task_key` cannot be found in the loaded configuration mapping.
        If loaded task definition for `task_key` does not include the required key 'task_name'.
    FileNotFoundError
        If path for task definition configuration is not found within the local fallback pipeline (i.e., when
        configuration cannot be obtained from TokaMark).

    """

    if not isinstance(task_key, str) or not task_key.strip():
        raise ValueError("`task_key` must be a non-empty string")

    # 1) Benchmark task definition
    try:
        task_def = benchmark_get_task_config(task_name=task_key)
        if not isinstance(task_def, dict):  # Guard against unexpected benchmark API changes.
            raise TypeError(  # noqa - Ignore unreachable code warning
                f"`benchmark_get_task_config('{task_key}')` returned {type(task_def).__name__}, expected dict."
            )

    except KeyError:
        # 2) Local fallback
        mp = local_map if (local_map is not None) else LOCAL_TASK_DEFS_MAP
        rel = mp.get(task_key)
        if rel is None:
            known = ", ".join(sorted(mp.keys())) or "<none>"
            raise KeyError(
                f"Unknown task {task_key!r}. Not found in benchmark and not registered locally. "
                f"Known local tasks: {known}"
            )
        root = _resolve_configs_root(configs_root=configs_root)
        path = (root / rel).resolve()
        if not path.is_file():
            raise FileNotFoundError(
                "Local task definition file not found.\n"
                f"  task_key:  {task_key}\n"
                f"  mapped_to: {rel}\n"
                f"  resolved:  {path}\n"
            )
        task_def = _load_yaml(path=path)

    task_name = task_def.get("task_name")
    if (not isinstance(task_name, str)) or (not task_name.strip()):
        raise KeyError(
            f"Loaded task definition for {task_key!r} is missing required key 'task_name'. "
            "Add `task_name: <name>` to the benchmark/local task definition YAML file."
        )

    return task_def
