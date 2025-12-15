"""
Configuration loading utilities for the Multi-Modal Transformer (MMT) project.

This module loads and merges experiment configuration files in a flexible,
reproducible, and open-source-friendly way.

An experiment is defined by three YAML files:

  • experiment_base.yaml        – global experiment structure and defaults
  • embeddings_default.yaml     – embedding configuration (defaults + overrides)
  • <phase>_default.yaml        – phase-specific config (pretrain / finetune / eval / tune_dct3d)

Given a *phase config path*, the loader performs:

  1) Resolve config file paths relative to the repository root.
  2) Load phase config, base config, and embeddings config.
  3) Deep-merge dictionaries in the order:
         experiment_base ← embeddings ← phase_config
  4) Resolve FAIRMAST baseline_config path inside the installed baseline package.
  5) Resolve model_init.model_dir to an absolute path when present.
  6) Compute paths (run_dir, etc.) depending on the phase.

Phase-specific behavior
-----------------------
• pretrain / finetune:
    - Creates <repo_root>/runs/<run_id> as run_dir
    - Saves config_merged.yaml into run_dir

• eval:
    - Requires model_init.model_dir (a training run directory)
    - Creates <model_dir>/<eval_id> as run_dir
    - Saves config_merged.yaml into run_dir

• tune_dct3d:
    - Does NOT create a runs/ directory
    - Uses the directory containing the phase YAML as run_dir/config_dir
    - Does NOT save config_merged.yaml (the primary artifact is embeddings_tuned.yaml)

The returned ExperimentConfig object provides dynamic attribute access
(e.g., cfg.model, cfg.preprocess, cfg.loader, …) and keeps the merged raw
dictionary in cfg.raw.
"""

from __future__ import annotations

import copy
import datetime
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, cast

import yaml

from mmt.utils.paths import get_repo_root


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries.

    - Dict values are merged recursively
    - Non-dict values are replaced
    - Lists are replaced (not concatenated)
    """
    merged = copy.deepcopy(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = copy.deepcopy(v)
    return merged


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _resolve_from_repo_root(path_str: str) -> Path:
    """
    Resolve a path relative to the repository root.

    Rules:
      • absolute paths are returned unchanged
      • paths beginning with "mmt/" map to <repo_root>/src/mmt/...
      • all other relative paths are resolved under <repo_root>
    """
    p = Path(path_str)
    if p.is_absolute():
        return p

    repo_root = get_repo_root()

    # Path inside src/mmt
    if p.parts and p.parts[0] == "mmt":
        return (repo_root / "src" / p).resolve()

    return (repo_root / p).resolve()


def _resolve_baseline_config(baseline_config_str: str) -> Path:
    """
    Resolve a FAIRMAST baseline config path inside the installed baseline package.

    Convention example:
      baseline_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

    The first component is treated as the installed Python package name
    (e.g. "scripts"). The rest is resolved inside that package.
    """
    parts = Path(baseline_config_str).parts
    if not parts:
        raise ValueError("baseline_config is empty")

    pkg_name = parts[0]
    rel_inside_pkg = Path(*parts[1:])

    pkg_root = files(pkg_name)
    # Two-step cast due to imperfect stubs for importlib.resources
    pkg_root_path = cast(Path, cast(object, pkg_root))

    return (Path(pkg_root_path) / rel_inside_pkg).resolve()


# ---------------------------------------------------------------------------
# Paths computation
# ---------------------------------------------------------------------------


def _compute_paths(
    merged: Dict[str, Any], *, phase_config_path: Path
) -> Dict[str, str]:
    """
    Compute repo/run paths depending on phase.

    For tune_dct3d:
      - run_dir = directory containing the phase config
      - no runs/ directory is created
    """
    repo_root = get_repo_root()
    global_runs_root = repo_root / "runs"

    phase = merged.get("phase")
    if phase is None:
        raise ValueError(
            "Config must define phase (pretrain/finetune/eval/tune_dct3d)."
        )

    task = merged.get("task", "unknown_task")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------------------------------------
    # TUNING PHASE: tune_dct3d (no runs/, local outputs)
    # ----------------------------------------------------------
    if phase == "tune_dct3d":
        cfg_dir = phase_config_path.parent.resolve()
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return {
            "repo_root": str(repo_root),
            "run_dir": str(cfg_dir),  # for logging convenience
            "config_dir": str(cfg_dir),  # explicit alias
        }

    # ----------------------------------------------------------
    # TRAINING PHASE: pretrain or finetune
    # ----------------------------------------------------------
    if phase in ("pretrain", "finetune"):
        run_id = merged.get("run_id") or f"{task}__{phase}__{timestamp}"
        run_dir = global_runs_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        return {
            "repo_root": str(repo_root),
            "runs_root": str(global_runs_root),
            "run_dir": str(run_dir),
        }

    # ----------------------------------------------------------
    # EVALUATION PHASE
    # ----------------------------------------------------------
    if phase == "eval":
        init_cfg = merged.get("model_init", {})
        model_dir = init_cfg.get("model_dir")
        if model_dir is None:
            raise ValueError(
                "Eval phase requires model_init.model_dir pointing to a training run."
            )

        model_dir = _resolve_from_repo_root(str(model_dir))
        if not model_dir.exists():
            raise FileNotFoundError(f"Training run directory not found: {model_dir}")

        eval_id = merged.get("eval_id") or f"eval__{timestamp}"
        eval_dir = model_dir / eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        return {
            "repo_root": str(repo_root),
            "runs_root": str(model_dir),
            "run_dir": str(eval_dir),
            "eval_dir": str(eval_dir),
        }

    raise ValueError(f"Unsupported phase: {phase}")


# ---------------------------------------------------------------------------
# Dynamic ExperimentConfig
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Dynamic configuration object.

    Every top-level key in the merged YAML dictionary is available
    as an attribute: cfg.model, cfg.preprocess, cfg.collate, etc.

    The full merged dictionary is stored in cfg.raw.
    """

    raw: Dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        if key in self.raw:
            return self.raw[key]
        raise AttributeError(f"'ExperimentConfig' has no attribute '{key}'")

    def get(self, key: str, default=None):
        return self.raw.get(key, default)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


def load_experiment_config(phase_config_path: str | Path) -> ExperimentConfig:
    """
    Load and merge experiment configuration files.

    Parameters
    ----------
    phase_config_path:
        Path to the phase-specific config file, e.g.:
          mmt/configs/task_2-1/finetune_default.yaml
          mmt/configs/task_2-1/eval_default.yaml
          mmt/configs/task_2-1/tune_dct3d.yaml

    Returns
    -------
    ExperimentConfig
        Dynamic configuration object with merged fields and computed paths.
    """
    # --- Resolve & load phase config ---
    phase_config_path = _resolve_from_repo_root(str(phase_config_path))
    if not phase_config_path.is_file():
        raise FileNotFoundError(f"Phase config not found: {phase_config_path}")

    phase_cfg = _load_yaml(phase_config_path)

    # Required keys
    if "experiment_base" not in phase_cfg:
        raise KeyError("Phase config must contain 'experiment_base'.")
    if "embedding_config" not in phase_cfg:
        raise KeyError("Phase config must contain 'embedding_config'.")

    exp_base_path = _resolve_from_repo_root(str(phase_cfg["experiment_base"]))
    emb_cfg_path = _resolve_from_repo_root(str(phase_cfg["embedding_config"]))

    base_cfg = _load_yaml(exp_base_path)
    emb_cfg = _load_yaml(emb_cfg_path)

    # --- Merge dictionaries ---
    merged: Dict[str, Any] = {}
    merged = _deep_merge(merged, base_cfg)
    merged = _deep_merge(merged, emb_cfg)
    merged = _deep_merge(merged, phase_cfg)

    # --- Compute paths (phase-aware) ---
    merged["paths"] = _compute_paths(merged, phase_config_path=phase_config_path)

    # --- Resolve baseline config path (inside baseline package) ---
    if "baseline_config" in merged and merged["baseline_config"]:
        merged["baseline_config_path"] = str(
            _resolve_baseline_config(str(merged["baseline_config"]))
        )

    # --- Resolve model_init.model_dir to absolute path (if present) ---
    if isinstance(merged.get("model_init"), dict):
        model_dir = merged["model_init"].get("model_dir", None)
        if model_dir is not None:
            merged["model_init"]["model_dir"] = str(
                _resolve_from_repo_root(str(model_dir))
            )

    # --- Save merged config for run-based phases only ---
    phase = merged.get("phase")
    if phase != "tune_dct3d":
        run_dir = Path(merged["paths"]["run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        merged_yaml_out = run_dir / "config_merged.yaml"
        with merged_yaml_out.open("w") as f:
            yaml.safe_dump(merged, f, sort_keys=False)

    return ExperimentConfig(raw=merged)
