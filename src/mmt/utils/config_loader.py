"""
Configuration loading and merging utilities for the multi-modal-transformer project.

This module is responsible for turning a *phase config* (e.g.
`mmt/configs/task1_1/finetune_default.yaml`) into a single structured
`ExperimentConfig` object.

The loader:
  1. Reads the phase config (finetune / eval / pretrain).
  2. Reads the corresponding `experiment_base.yaml`.
  3. Reads the embedding config (e.g. `embeddings_default.yaml`).
  4. Merges them into a single nested dictionary.
  5. Resolves the baseline task config path (from the FAIRMAST baseline repo).
  6. Creates a unique run directory under `runs/` with:
       - `output_root` and `cache_root` subfolders,
       - a `config_merged.yaml` file containing the full merged config.
  7. Returns an `ExperimentConfig` dataclass with convenient access to:
       - `phase`, `task`, `seed`
       - `preprocessing`, `model`, `data`, `embeddings`
       - all important paths (output, cache, baseline config, etc.).

Typical usage from scripts:

    from mmt.utils.config_loader import load_experiment_config

    cfg = load_experiment_config("mmt/configs/task1_1/finetune_default.yaml")
    # use cfg.model, cfg.preprocessing, cfg.paths["output_root"], ...

This keeps all YAML handling, path resolution and run bookkeeping in one place,
so the rest of the code can stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from importlib.resources import files

import datetime
import yaml

from .paths import get_repo_root


# ----------------------------------------------------------------------------------------------------------------------
# Small helpers

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base. Values in override win."""
    out = dict(base)
    for k, v in (override or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_from_repo_root(path_str: str) -> Path:
    """
    Resolve a path that is either absolute or relative to the repo root.

    Convention:
      - if `path_str` is absolute, return it as-is;
      - if it starts with "mmt/", treat it as inside `src/`:
            repo_root / "src" / "mmt/..."
      - otherwise, treat it as relative to repo_root directly.
    """
    p = Path(path_str)

    # Absolute path → trust the caller
    if p.is_absolute():
        return p

    # Paths like "mmt/configs/..." live under src/mmt in this repo layout
    if p.parts and p.parts[0] == "mmt":
        return (get_repo_root() / "src" / p).resolve()

    # Fallback: relative to repo root
    return (get_repo_root() / p).resolve()


def _resolve_baseline_config(baseline_config_str: str) -> Path:
    """
    Resolve the baseline config path.

    Convention for now:
      baseline_config: "scripts/pipelines/configs/.../config_task_2-1.yaml"

    i.e. first component is the installed package name of the baseline ("scripts"),
    the rest is a relative path inside that package.

    If you later change the format, adapt this function only.
    """
    parts = Path(baseline_config_str).parts
    if not parts:
        raise ValueError("baseline_config is empty")

    pkg_name = parts[0]         # "scripts"
    rel_inside_pkg = Path(*parts[1:])  # "pipelines/configs/.../config_task_2-1.yaml"

    pkg_root = files(pkg_name)
    return (pkg_root / rel_inside_pkg).resolve()


# ----------------------------------------------------------------------------------------------------------------------
# Main config object
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """
    Unified configuration for a single experiment run.

    This is what the rest of the code should use, instead of reading YAMLs
    directly and re-implementing path/merge logic everywhere.
    """

    phase: str                    # "finetune", "eval", "pretrain", ...
    task: str                     # e.g. "task_2-1"
    seed: int
    baseline_config_path: Path

    # Sub-configs (typed views over the raw dict)
    preprocessing: Dict[str, Any]
    model: Dict[str, Any]
    data: Dict[str, Any]
    embeddings: Dict[str, Any]

    # Paths that are important for the run
    paths: Dict[str, Path]

    # Full merged dict (useful for debugging / saving)
    raw: Dict[str, Any]


# ----------------------------------------------------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------------------------------------------------


def load_experiment_config(phase_config_path: str | Path) -> ExperimentConfig:
    """
    Load and merge:
      - phase config (e.g. finetune_default.yaml or eval_default.yaml)
      - experiment_base.yaml
      - embeddings_default.yaml (or tuned)
    Then:
      - resolve the baseline config path,
      - compute run_id, output_root, cache_root,
      - save the merged config under output_root/config_merged.yaml.

    Parameters
    ----------
    phase_config_path:
        Path to the phase config file, either absolute or relative to the
        repo root (e.g. "mmt/configs/task1_1/finetune_default.yaml").

    Returns
    -------
    ExperimentConfig
        Structured object with all the information needed by the rest
        of the pipeline.
    """
    # 1) Resolve and load phase config
    phase_config_path = _resolve_from_repo_root(str(phase_config_path))
    phase_cfg = _load_yaml(phase_config_path)

    # Required keys in the phase config
    exp_base_rel = phase_cfg["experiment_base"]
    emb_cfg_rel = phase_cfg["embedding_config"]
    phase = phase_cfg.get("phase", "unknown")

    # 2) Load experiment_base and embeddings configs
    exp_base_path = _resolve_from_repo_root(exp_base_rel)
    emb_cfg_path = _resolve_from_repo_root(emb_cfg_rel)

    exp_cfg = _load_yaml(exp_base_path)
    emb_cfg = _load_yaml(emb_cfg_path)

    # 3) Start building the raw merged dict
    raw: Dict[str, Any] = dict(exp_cfg)  # copy

    # Attach embeddings under a clear top-level key
    raw["embeddings"] = emb_cfg.get("embeddings", {})

    # Attach phase
    raw["phase"] = phase

    # Merge extra keys from phase config (training, evaluation, run, data overrides, etc.)
    for key, value in phase_cfg.items():
        if key in {"experiment_base", "embedding_config", "phase"}:
            continue
        if key in raw and isinstance(raw[key], dict) and isinstance(value, dict):
            raw[key] = _deep_merge(raw[key], value)
        else:
            raw[key] = value

    # 4) Resolve baseline config path
    baseline_config_str = raw["baseline_config"]
    baseline_cfg_path = _resolve_baseline_config(baseline_config_str)

    # 5) Compute run_id, output_root and cache_root
    task = raw["task"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{task}__{phase}__{timestamp}"

    repo_root = get_repo_root()
    runs_root = repo_root / "runs"
    output_root = runs_root / run_id
    cache_root = output_root / "cache"

    output_root.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, Path] = {
        "repo_root": repo_root,
        "runs_root": runs_root,
        "output_root": output_root,
        "cache_root": cache_root,
        "phase_config": phase_config_path,
        "experiment_base": exp_base_path,
        "embedding_config": emb_cfg_path,
        "baseline_config": baseline_cfg_path,
    }

    # Inject paths into raw (useful when saving / inspecting)
    raw.setdefault("paths", {})
    raw["paths"]["run_id"] = run_id
    raw["paths"]["output_root"] = str(output_root)
    raw["paths"]["phase_config"] = str(phase_config_path)
    raw["paths"]["experiment_base"] = str(exp_base_path)
    raw["paths"]["embedding_config"] = str(emb_cfg_path)
    raw["paths"]["baseline_config"] = str(baseline_cfg_path)

    # 6) Save merged config for reproducibility
    merged_cfg_path = output_root / "config_merged.yaml"
    with merged_cfg_path.open("w") as f:
        yaml.safe_dump(raw, f, sort_keys=False)
    paths["merged_config"] = merged_cfg_path

    # 7) Extract sub-configs and seed
    global_cfg = raw.get("global", {})
    preprocessing = raw.get("preprocessing", {})
    model = raw.get("model", {})
    data = raw.get("data", {})
    embeddings = raw.get("embeddings", {})

    seed = int(global_cfg.get("seed", 0))

    return ExperimentConfig(
        phase=phase,
        task=task,
        seed=seed,
        baseline_config_path=baseline_cfg_path,
        preprocessing=preprocessing,
        model=model,
        data=data,
        embeddings=embeddings,
        paths=paths,
        raw=raw,
    )
