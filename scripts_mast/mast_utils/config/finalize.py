"""
Final path computation and config snapshot persistence.

This module handles the final stage of config loading:
1. Compute all output paths (run_dir, eval_dir, etc.) based on phase
2. Inject computed paths into config
3. Persist merged config snapshot to disk for reproducibility
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root

logger = logging.getLogger("mmt.ConfigLoader")


def compute_paths(
    merged: Dict[str, Any],
    *,
    configs_root: Path,
    task_dir: Path,
) -> Dict[str, str]:
    """Compute output paths for pretrain, finetune, and eval phases."""
    repo_root = get_repo_root()
    global_runs_root = repo_root / "runs"

    phase = merged.get("phase")
    if phase is None:
        raise ValueError("Missing required argument: phase.")

    task = merged.get("task", "unknown_task")

    if phase in ("pretrain", "finetune"):
        run_id = merged.get("run_id")
        if run_id is None:
            raise ValueError(
                f"run_id must be set for phase={phase} before calling compute_paths"
            )
        run_dir = global_runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "repo_root": str(repo_root),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": str(phase),
            "run_id": str(run_id),
            "run_dir": str(run_dir),
            "task_config_dir": str(task_dir),
        }

    if phase == "eval":
        model_source = merged.get("model_source", {})
        if not isinstance(model_source, dict):
            raise TypeError("model_source must be set for eval phase")

        model_id = model_source.get("run_id")
        model_path = model_source.get("model_path")
        if model_path:
            model_dir = Path(model_path)
            model_id = model_dir.name
        elif model_id:
            model_dir = global_runs_root / model_id
        else:
            raise ValueError("model_source must have either run_id or model_path set")

        eval_id = "eval"
        eval_dir = model_dir / eval_id
        return {
            "repo_root": str(repo_root),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": "eval",
            "eval_id": str(eval_id),
            "run_dir": str(eval_dir),
            "model_run_dir": str(model_dir),
            "task_config_dir": str(task_dir),
        }

    raise ValueError(f"Unsupported phase: {phase}")


def finalize_and_save_config(
    merged: Dict[str, Any],
    *,
    phase: str,
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> None:
    """Finalize ids/paths and write the merged config snapshot to disk."""
    merged["paths"] = compute_paths(
        merged,
        configs_root=configs_root_path,
        task_dir=tasks_overrides_dir,
    )

    if phase in ("pretrain", "finetune"):
        merged["run_id"] = merged["paths"]["run_id"]
    if phase == "eval":
        merged["eval_id"] = merged["paths"]["eval_id"]

    if phase == "eval":
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['eval_id']}.yaml"
    else:
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['run_id']}.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / config_name
    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    logger.info("Saved config snapshot: %s", config_path)
