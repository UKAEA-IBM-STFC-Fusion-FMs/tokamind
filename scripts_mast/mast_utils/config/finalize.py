"""
Final path computation and config snapshot persistence.

This module handles the final stage of config loading:
1. Compute all output paths (run_dir, eval_dir, etc.) based on phase
2. Inject computed paths into config
3. Persist merged config snapshot to disk for reproducibility
"""

from __future__ import annotations

import yaml
import logging
from collections.abc import Mapping, MutableMapping
from typing import Any, Literal
from pathlib import Path

from mmt.utils.paths import REPO_ROOT


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.ConfigLoader")


# ----------------------------------------------------------------------------------------------------------------------
def compute_paths(
    merged: Mapping[str, Any],
    *,
    configs_root: Path,
    task_dir: Path,
) -> dict[str, str]:
    """
    Compute output paths for pretrain, finetune, and eval phases.

    Parameters
    ----------
    merged : Mapping[str, Any]
        Merged config dictionary.
    configs_root : str | Path
        Path to the root directory for configuration files.
    task_dir : Path
        Path to the task directory.

    Returns
    -------
    dict[str, str]
        Dictionary with phase-dependent computed paths.

    Raises
    ------
    ValueError
        Unsupported value for `merged['phase']`.
        Value for `merged['model_source']` must define a 'run_id' or 'model_path' key with valid value.
    TypeError
        Value for `merged['model_source'] must be a dictionary when `merged['phase'] is 'eval'.
    KeyError
        If "phase" key not in `merged`.
        If "run_id" key not in `merged` when `merged["phase"]` is either "pretrain" or "finetune".

    """

    global_runs_root = REPO_ROOT / "runs"

    if "phase" not in merged:
        raise KeyError("Missing required key 'phase' in `merged`.")

    phase = merged["phase"]
    if phase not in ["pretrain", "finetune", "eval"]:
        raise ValueError(f"Unsupported value '{phase}' for `merged['phase']`.")

    task = merged.get("task", "unknown_task")

    if phase in ["pretrain", "finetune"]:
        if "run_id" not in merged:
            raise KeyError(f"Missing key 'run_id' in `merged` when `merged['phase']` is '{phase}'.")

        run_id = merged["run_id"]
        run_dir = global_runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        return {
            "repo_root": str(REPO_ROOT),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": str(phase),
            "run_id": str(run_id),
            "run_dir": str(run_dir),
            "task_config_dir": str(task_dir),
        }

    else:  # -> I.e., phase is "eval"
        model_source = merged.get("model_source", {})
        if not isinstance(model_source, dict):
            raise TypeError("Value for `merged['model_source'] must be a dictionary when `merged['phase'] is 'eval'.")

        model_id: str | None = model_source.get("run_id")
        model_path: str | None = model_source.get("model_path")

        if model_path:
            model_dir = Path(model_path)
        elif model_id:
            model_dir = global_runs_root / model_id
        else:
            raise ValueError(
                "Value for `merged['model_source']` must define a 'run_id' or 'model_path' key with valid value."
            )

        eval_id = "eval"
        eval_dir = model_dir / eval_id

        return {
            "repo_root": str(REPO_ROOT),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": "eval",
            "eval_id": str(eval_id),
            "run_dir": str(eval_dir),
            "model_run_dir": str(model_dir),
            "task_config_dir": str(task_dir),
        }


# ----------------------------------------------------------------------------------------------------------------------
def finalize_and_save_config(
    merged: MutableMapping[str, Any],
    *,
    phase: Literal["pretrain", "finetune", "eval"],
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> None:
    """
    Finalize IDs/paths and write the merged config snapshot to disk.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged config dictionary (modified in-place).
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".
    configs_root_path : Path
        Path to the root directory for configuration files.
    tasks_overrides_dir : Path
        Path to the task overrides directory.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `phase` not in ["pretrain", "finetune", "eval"].

    """

    merged["paths"] = compute_paths(merged=merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir)

    if phase in ["pretrain", "finetune"]:
        merged["run_id"] = merged["paths"]["run_id"]
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['run_id']}.yaml"

    elif phase == "eval":
        merged["eval_id"] = merged["paths"]["eval_id"]
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['eval_id']}.yaml"

    else:
        raise ValueError(f"Value for `phase` must be in ['pretrain', 'finetune', 'eval'], got {phase!r}.")

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / config_name
    with config_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    logger.info("Saved config snapshot: %s", config_path)
