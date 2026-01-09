"""
Project-level config loader for the MMT + MAST integration (scripts_mast).

Convention-based config layout (no pointers inside YAML):

scripts_mast/configs/
  common/
    core.yaml
    embeddings.yaml
    finetune.yaml
    pretrain.yaml
    eval.yaml
    tune_dct3d.yaml
  tasks_overrides/
    <task>/                         (optional for baseline tasks_overrides)
      core_overrides.yaml           (optional)
      finetune_overrides.yaml       (optional)
      pretrain_overrides.yaml       (optional)
      eval_overrides.yaml           (optional)
      tune_dct3d_overrides.yaml     (optional)
      embeddings_overrides.yaml     (optional; generally produced by tune_dct3d)

Merge order (later wins):
  1) common/core.yaml
  2) common/embeddings.yaml
  3) common/<phase>.yaml
  4) tasks_overrides/<task>/core_overrides.yaml                (optional)
  5) tasks_overrides/<task>/<phase>_overrides.yaml             (optional)
  6) tasks_overrides/<task>/embeddings_overrides.yaml          (optional; merged by default for
     pretrain/finetune/eval, NOT merged for tune_dct3d)

NOTE: Benchmark task *definitions* are resolved separately (scripts_mast layer),
not here.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root
from mmt.utils.config.schema import ExperimentConfig


def _resolve_from_repo_root(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _compute_paths(
    merged: Dict[str, Any], *, configs_root: Path, task_dir: Path
) -> Dict[str, str]:
    repo_root = get_repo_root()
    global_runs_root = repo_root / "runs"

    phase = merged.get("phase")
    if phase is None:
        raise ValueError("Missing required argument: phase.")

    task = merged.get("task", "unknown_task")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # tune_dct3d writes into the task folder
    if phase == "tune_dct3d":
        return {
            "repo_root": str(repo_root),
            "configs_root": str(configs_root),
            "run_dir": str(task_dir),
            "config_dir": str(task_dir),
        }

    # train phases write into runs/<run_id>
    if phase in ("pretrain", "finetune"):
        run_id = merged.get("run_id") or f"{task}__{phase}__{timestamp}"
        run_dir = global_runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return {
            "repo_root": str(repo_root),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": str(phase),
            "run_id": str(run_id),
            "run_dir": str(run_dir),
        }

    # eval writes into <model_run_dir>/eval/<eval_id>
    if phase == "eval":
        init_cfg = merged.get("model_source", {})
        model_dir = init_cfg.get("run_dir")
        if model_dir is None:
            raise ValueError(
                "Eval phase requires model_source.run_dir pointing to a training run."
            )
        model_dir = _resolve_from_repo_root(str(model_dir))

        eval_id = merged.get("eval_id") or f"{task}__eval__{timestamp}"
        eval_dir = model_dir / "eval" / eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)
        return {
            "repo_root": str(repo_root),
            "configs_root": str(configs_root),
            "task": str(task),
            "phase": "eval",
            "eval_id": str(eval_id),
            "run_dir": str(eval_dir),
            "model_run_dir": str(model_dir),
        }

    raise ValueError(f"Unsupported phase: {phase}")


def load_experiment_config(
    *,
    task: str,
    phase: str,
    configs_root: str | Path = "scripts_mast/configs",
    run_id: str | None = None,
    eval_id: str | None = None,
    use_embeddings_overrides: bool = True,
) -> ExperimentConfig:
    if phase not in ("pretrain", "finetune", "eval", "tune_dct3d"):
        raise ValueError(f"Unsupported phase: {phase}")

    configs_root_path = _resolve_from_repo_root(str(configs_root))
    common_dir = configs_root_path / "common"
    tasks_overrides_dir = configs_root_path / "tasks_overrides"
    tasks_overrides_dir = tasks_overrides_dir / task  # may not exist (OK)

    # Required common files
    core_path = common_dir / "core.yaml"
    embeddings_path = common_dir / "embeddings.yaml"
    phase_common_path = common_dir / f"{phase}.yaml"

    for p in (core_path, embeddings_path, phase_common_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required config not found: {p}")

    # Optional task overrides (all optional)
    core_overrides_path = tasks_overrides_dir / "core_overrides.yaml"
    phase_overrides_path = tasks_overrides_dir / f"{phase}_overrides.yaml"
    embeddings_overrides_path = tasks_overrides_dir / "embeddings_overrides.yaml"

    # For tune_dct3d we write outputs into tasks_overrides_dir, so ensure it exists
    if phase == "tune_dct3d":
        tasks_overrides_dir.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {}
    merged = _deep_merge(merged, _load_yaml(core_path))
    merged = _deep_merge(merged, _load_yaml(embeddings_path))
    merged = _deep_merge(merged, _load_yaml(phase_common_path))

    # Merge optional core overrides
    if core_overrides_path.is_file():
        merged = _deep_merge(merged, _load_yaml(core_overrides_path))

    # Merge optional phase overrides
    if phase_overrides_path.is_file():
        merged = _deep_merge(merged, _load_yaml(phase_overrides_path))

    # Merge optional embeddings overrides for run phases (default on), but NOT for tune_dct3d
    if phase != "tune_dct3d" and use_embeddings_overrides:
        if embeddings_overrides_path.is_file():
            merged = _deep_merge(merged, _load_yaml(embeddings_overrides_path))

    # Enforce task name from CLI/folder selection
    task_in_yaml = merged.get("task", None)
    if task_in_yaml is not None and str(task_in_yaml) != str(task):
        raise ValueError(
            f"Task mismatch: requested task='{task}' but an overrides file defines task='{task_in_yaml}'."
        )
    merged["task"] = task
    merged["phase"] = phase  # enforce phase from CLI

    # Inject run_id / eval_id if provided
    if run_id is not None:
        merged["run_id"] = run_id
    if eval_id is not None:
        merged["eval_id"] = eval_id

    merged["paths"] = _compute_paths(
        merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir
    )

    # Resolve model_source.run_dir to absolute path (if present)
    if isinstance(merged.get("model_source"), dict):
        model_dir = merged["model_source"].get("run_dir", None)
        if model_dir is not None:
            merged["model_source"]["run_dir"] = str(
                _resolve_from_repo_root(str(model_dir))
            )

    # Save merged config for run-based phases only
    if phase != "tune_dct3d":
        run_dir_path = Path(merged["paths"]["run_dir"])
        run_dir_path.mkdir(parents=True, exist_ok=True)
        with (run_dir_path / "config_merged.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, sort_keys=False)

    return ExperimentConfig(raw=merged)
