from __future__ import annotations

import copy
import yaml
import datetime

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, cast
from importlib.resources import files

from mmt.utils.paths import get_repo_root


"""
Configuration loading utilities for the Multi-Modal Transformer project.

This module loads and merges experiment configuration files in a flexible,
reproducible, and open-source-friendly way.

An experiment is defined by three YAML files located in the same directory:

    • experiment_base.yaml        – global experiment structure
    • embeddings_default.yaml     – embedding configuration
    • <phase>_default.yaml        – phase-specific config (pretrain / finetune / eval)

Given a *phase config path*, the loader performs:

  1. Resolve all config file paths relative to the repository root.
  2. Load the phase config (e.g., finetune_default.yaml).
  3. Load experiment_base.yaml and the embedding configuration referenced by the phase.
  4. Deep-merge dictionaries in the order:
         experiment_base ← embeddings ← phase_config
  5. Resolve the FAIRMAST baseline_config path inside the installed baseline package.
  6. Resolve model_init.model_dir to an absolute path when present.
  7. Create an output run directory (<repo_root>/runs/<run_id>) and save config_merged.yaml.
  8. Return an ExperimentConfig object providing dynamic attribute access
     (e.g., cfg.model, cfg.preprocess, cfg.loader, …).

The loader preserves full compatibility with the FAIRMAST baseline pipeline while
remaining simple, extensible, and suitable for open-source use.
"""


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries."""
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
      • all other relative paths are resolved under repo_root
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
    Resolve a FAIRMAST baseline config path.

    Convention:
      baseline_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

    The first component is the installed baseline package name (e.g. "scripts").
    The rest is a relative path inside that package.

    This mechanism keeps the baseline repository independent of the MMT one.
    """
    parts = Path(baseline_config_str).parts
    if not parts:
        raise ValueError("baseline_config is empty")

    pkg_name = parts[0]  # baseline package
    rel_inside_pkg = Path(*parts[1:])  # internal path in that package

    # Two-step cast to satisfy type checker due to bad stubs
    pkg_root = files(pkg_name)
    pkg_root_path = cast(Path, cast(object, pkg_root))

    return (Path(pkg_root_path) / rel_inside_pkg).resolve()


def _compute_paths(merged: Dict[str, Any]) -> Dict[str, str]:
    repo_root = get_repo_root()
    global_runs_root = repo_root / "runs"

    phase = merged.get("phase")
    if phase is None:
        raise ValueError("Config must define phase (pretrain/finetune/eval).")

    task = merged.get("task", "unknown_task")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

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

        # Resolve path (absolute)
        model_dir = Path(_resolve_from_repo_root(model_dir))
        if not model_dir.exists():
            raise FileNotFoundError(f"Training run directory not found: {model_dir}")

        eval_id = merged.get("eval_id") or f"eval__{timestamp}"

        eval_dir = model_dir / eval_id
        eval_dir.mkdir(parents=True, exist_ok=True)

        return {
            "repo_root": str(repo_root),
            "runs_root": str(model_dir),  # parent is training run
            "run_dir": str(eval_dir),  # eval run directory
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
    phase_config_path : str or Path
        Path to the phase-specific config file, e.g.:
        mmt/configs/task_2-1/finetune_default.yaml

    Returns
    -------
    ExperimentConfig
        Dynamic configuration object with merged fields and path metadata.
    """

    # --- Resolve & load phase config ---
    phase_config_path = _resolve_from_repo_root(str(phase_config_path))
    if not phase_config_path.is_file():
        raise FileNotFoundError(f"Phase config not found: {phase_config_path}")

    phase_cfg = _load_yaml(phase_config_path)

    # Required keys
    try:
        exp_base_rel = phase_cfg["experiment_base"]
        emb_cfg_rel = phase_cfg["embedding_config"]
    except KeyError:
        raise KeyError(
            "Phase config must contain 'experiment_base' and 'embedding_config'"
        )

    # --- Load base & embedding YAML ---
    exp_base_path = _resolve_from_repo_root(exp_base_rel)
    emb_cfg_path = _resolve_from_repo_root(emb_cfg_rel)

    base_cfg = _load_yaml(exp_base_path)
    emb_cfg = _load_yaml(emb_cfg_path)

    # --- Merge dictionaries ---
    merged = {}
    merged = _deep_merge(merged, base_cfg)
    merged = _deep_merge(merged, emb_cfg)
    merged = _deep_merge(merged, phase_cfg)

    merged["paths"] = _compute_paths(merged)

    # --- Resolve baseline config exactly as expected by baseline bridge ---
    if "baseline_config" in merged:
        merged["baseline_config_path"] = str(
            _resolve_baseline_config(merged["baseline_config"])
        )

    # --- Resolve init.model_dir  ---
    if "model_init" in merged and isinstance(merged["model_init"], dict):
        model_dir = merged["model_init"].get("model_dir", None)
        if model_dir is not None:
            merged["model_init"]["model_dir"] = str(_resolve_from_repo_root(model_dir))

    # --- Save merged config in run directory ---
    run_dir = Path(merged["paths"]["run_dir"])
    merged_yaml_out = run_dir / "config_merged.yaml"
    with merged_yaml_out.open("w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    # --- Return dynamic config ---
    return ExperimentConfig(raw=merged)
