"""
Project-level config loader for the MMT + MAST integration (scripts_mast).

Convention-based config layout (no pointers inside YAML):

scripts_mast/configs/
  common/
    embeddings.yaml
    pretrain.yaml
    finetune.yaml
    eval.yaml
    tune_dct3d.yaml

  tasks_overrides/
    <task>/                                   (one folder per task)
      pretrain_overrides.yaml                 (optional)
      finetune_overrides.yaml                 (optional)
      eval_overrides.yaml                     (optional)
      tune_dct3d_overrides.yaml               (optional)
      embeddings_overrides/
        <profile>.yaml                        (task-level embedding overrides; required for
                                              pretrain/finetune; may be empty)

Embedding profile selection
---------------------------
Embedding overrides are selected by an explicit *profile* (default: ``dct3d``).
The loader looks for:

    tasks_overrides/<task>/embeddings_overrides/<profile>.yaml

Merge order (later wins)
------------------------
  1) common/embeddings.yaml
  2) common/<phase>.yaml
  3) tasks_overrides/<task>/<phase>_overrides.yaml                  (optional)
  4) tasks_overrides/<task>/embeddings_overrides/<profile>.yaml     (merged for
     pretrain/finetune; NOT merged for eval/tune_dct3d)

Required keys (explicit per phase)
----------------------------------
To keep configs self-contained and avoid implicit inheritance, each phase config
must define:

- ``seed``
- ``runtime`` (mapping; individual keys are user-defined)
- ``data.local``
- ``data.subset_of_shots``

Model + preprocess inheritance from model_source
------------------------------------------------
To ensure consistent evaluation and avoid config drift:

- ``finetune`` requires ``model_source.run_dir`` (a run id under ``runs/``).
  The loader loads ``model`` and ``preprocess.chunk``/``preprocess.trim_chunks`` from the
  source run config YAML (``runs/<run_id>/<run_id>.yaml``), then applies any finetune-side
  overrides on top.

- ``eval`` requires ``model_source.run_dir`` and rebuilds the model spec from the
  source run config YAML, taking:
    - ``model``
    - ``embeddings``
    - ``preprocess.chunk`` / ``preprocess.trim_chunks``

  Eval-side YAMLs should not redefine these blocks.

Notes
-----
- Benchmark task *definitions* are resolved separately (scripts_mast layer),
  not here.
- ``tune_dct3d`` intentionally does not merge the task-level embedding overrides so
  the tuner always writes deltas relative to common/embeddings.yaml.
"""

from __future__ import annotations

import copy
import datetime
from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.paths import get_repo_root
from mmt.utils.config.schema import ExperimentConfig

import logging

logger = logging.getLogger("mmt.ConfigLoader")


def _resolve_from_repo_root(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (get_repo_root() / p).resolve()


def _resolve_run_id_to_run_dir(run_id: str) -> Path:
    """Resolve a training run id to an absolute run directory path.

    Convention: all training runs live under ``<repo_root>/runs/<run_id>/``.

    Notes
    -----
    - ``run_id`` must be a *single* path component (no slashes).
    - Do not pass ``runs/<run_id>``; pass only ``<run_id>``.
    """
    s = str(run_id).strip()
    if not s:
        raise ValueError(
            "model_source.run_dir must be a non-empty run id (folder name under <repo_root>/runs/)."
        )

    p = Path(s)

    # Enforce "run id only" (no relative/absolute paths).
    if p.is_absolute() or len(p.parts) != 1:
        raise ValueError(
            "model_source.run_dir must be a run id (folder name under <repo_root>/runs/), "
            "e.g. 'pretrain_base'. Do not include 'runs/' or any path separators."
        )

    return (get_repo_root() / "runs" / p.parts[0]).resolve()


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


def _load_source_run_config_yaml(model_run_dir: Path) -> Dict[str, Any]:
    """Load the saved merged config YAML for a previous training run."""
    src_cfg_path = model_run_dir / f"{model_run_dir.name}.yaml"
    if not src_cfg_path.is_file():
        raise FileNotFoundError(
            "Required source run config YAML not found.\n"
            "Warm-start and evaluation require the saved merged config at:\n"
            f"  {src_cfg_path}\n"
        )
    return _load_yaml(src_cfg_path)


def _inherit_preprocess_chunk_trim(
    merged: Dict[str, Any],
    src_cfg: Dict[str, Any],
    *,
    allow_override: bool,
) -> None:
    """Inherit preprocess.chunk and preprocess.trim_chunks from a source config.

    Keeps any other preprocess keys (e.g. preprocess.valid_windows) from the
    current merged config.
    """
    src_pre = src_cfg.get("preprocess")
    if not isinstance(src_pre, dict):
        raise KeyError(
            "Source run config is missing required mapping: 'preprocess'.\n"
            "Expected 'preprocess.chunk' and 'preprocess.trim_chunks' to be present."
        )
    if "chunk" not in src_pre or "trim_chunks" not in src_pre:
        raise KeyError(
            "Source run config is missing required keys under 'preprocess'.\n"
            "Expected: preprocess.chunk and preprocess.trim_chunks"
        )

    merged_pre = merged.get("preprocess")
    if merged_pre is None:
        merged_pre = {}
        merged["preprocess"] = merged_pre
    if not isinstance(merged_pre, dict):
        raise TypeError("Config key 'preprocess' must be a mapping (dict).")

    override_chunk = None
    override_trim = None
    if allow_override:
        override_chunk = copy.deepcopy(merged_pre.get("chunk"))
        override_trim = copy.deepcopy(merged_pre.get("trim_chunks"))

    merged_pre["chunk"] = copy.deepcopy(src_pre["chunk"])
    merged_pre["trim_chunks"] = copy.deepcopy(src_pre["trim_chunks"])

    if allow_override:
        if override_chunk is not None:
            if isinstance(override_chunk, dict) and isinstance(
                merged_pre["chunk"], dict
            ):
                merged_pre["chunk"] = _deep_merge(merged_pre["chunk"], override_chunk)
            else:
                merged_pre["chunk"] = override_chunk

        if override_trim is not None:
            if isinstance(override_trim, dict) and isinstance(
                merged_pre["trim_chunks"], dict
            ):
                merged_pre["trim_chunks"] = _deep_merge(
                    merged_pre["trim_chunks"], override_trim
                )
            else:
                merged_pre["trim_chunks"] = override_trim


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
            "tune_dir": str(task_dir / "embeddings_overrides"),
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

    # eval writes into <model_run_dir>/<eval_id>
    # NOTE: do *not* create directories here.
    # We only create the eval folder after confirming the source run config exists
    if phase == "eval":
        init_cfg = merged.get("model_source", {})
        model_dir = init_cfg.get("run_dir") if isinstance(init_cfg, dict) else None
        if model_dir is None:
            raise ValueError(
                "Eval phase requires model_source.run_dir set to a training run id (folder name under <repo_root>/runs/)."
            )
        model_dir = _resolve_run_id_to_run_dir(str(model_dir))

        eval_id = merged.get("eval_id") or f"{task}__eval__{timestamp}"
        eval_dir = model_dir / eval_id
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
    embeddings_profile: str = "dct3d",
) -> ExperimentConfig:
    """Load and merge YAML configuration for a given task + phase.

    Parameters
    ----------
    task:
        Task folder name under ``scripts_mast/configs/tasks_overrides/<task>/``.
    phase:
        One of: ``pretrain``, ``finetune``, ``eval``, ``tune_dct3d``.
    configs_root:
        Root folder containing the convention-based config tree.
    run_id:
        Optional explicit run identifier (training phases only).
    eval_id:
        Optional explicit eval identifier (eval phase only).
    embeddings_profile:
        Embedding profile key. The loader merges the corresponding task-level
        embedding overrides file:

            tasks_overrides/<task>/embeddings_overrides/<embeddings_profile>.yaml (pretrain/finetune only)

        Default: ``dct3d``.

    Returns
    -------
    ExperimentConfig
        A dynamic wrapper around the merged raw dict (``cfg.raw``).
    """
    if phase not in ("pretrain", "finetune", "eval", "tune_dct3d"):
        raise ValueError(f"Unsupported phase: {phase}")

    emb_profile = str(embeddings_profile).strip()
    if not emb_profile:
        raise ValueError("embeddings_profile must be a non-empty string")

    configs_root_path = _resolve_from_repo_root(str(configs_root))
    common_dir = configs_root_path / "common"
    tasks_overrides_dir = (
        configs_root_path / "tasks_overrides" / task
    )  # may not exist (OK)

    # Required common files
    embeddings_path = common_dir / "embeddings.yaml"
    phase_common_path = common_dir / f"{phase}.yaml"

    for p in (embeddings_path, phase_common_path):
        if not p.is_file():
            raise FileNotFoundError(f"Required config not found: {p}")

    # Optional task overrides
    phase_overrides_path = tasks_overrides_dir / f"{phase}_overrides.yaml"

    # Task-level embedding overrides are selected by *profile*
    embeddings_overrides_dir = tasks_overrides_dir / "embeddings_overrides"
    embeddings_overrides_path = embeddings_overrides_dir / f"{emb_profile}.yaml"

    # For tune_dct3d we write outputs into tasks_overrides/<task>/, so ensure it exists.
    if phase == "tune_dct3d":
        tasks_overrides_dir.mkdir(parents=True, exist_ok=True)
        embeddings_overrides_dir.mkdir(parents=True, exist_ok=True)

    merged: Dict[str, Any] = {}
    merged = _deep_merge(merged, _load_yaml(embeddings_path))
    merged = _deep_merge(merged, _load_yaml(phase_common_path))

    # Merge optional phase overrides
    if phase_overrides_path.is_file():
        merged = _deep_merge(merged, _load_yaml(phase_overrides_path))

    # Raise warning if the embeddings overrides are not provided.
    # We will use the common embeddings.
    if phase not in ("tune_dct3d", "eval"):
        if not embeddings_overrides_path.is_file():
            logger.warning(
                "[WARNING] Missing task-level embedding overrides for profile=%s (task=%s). "
                "Proceeding without per-signal overrides. Expected: %s",
                emb_profile,
                task,
                embeddings_overrides_path,
            )
        else:
            merged = _deep_merge(merged, _load_yaml(embeddings_overrides_path))

    # Enforce task + phase from CLI
    task_in_yaml = merged.get("task", None)
    if task_in_yaml is not None and str(task_in_yaml) != str(task):
        raise ValueError(
            f"Task mismatch: requested task={task!r} but an overrides file defines task={task_in_yaml!r}."
        )
    merged["task"] = task
    merged["phase"] = phase
    merged["embeddings_profile"] = emb_profile

    # Inject run_id / eval_id if provided
    if run_id is not None:
        merged["run_id"] = run_id
    if eval_id is not None:
        merged["eval_id"] = eval_id

    # ------------------------------------------------------------------
    # For finetune/eval: validate the source run (config + checkpoints)
    # *before* creating any output directories (e.g., finetune run_dir).
    # ------------------------------------------------------------------
    src_run_dir: Path | None = None
    src_cfg: Dict[str, Any] | None = None
    if phase in ("finetune", "eval"):
        init_cfg = merged.get("model_source", {})
        if not isinstance(init_cfg, dict):
            raise TypeError("Config key 'model_source' must be a mapping (dict).")

        src_run_id = init_cfg.get("run_dir", None)
        if src_run_id is None:
            raise ValueError(
                f"{phase} phase requires model_source.run_dir set to a training run id (folder name under <repo_root>/runs/)."
            )

        src_run_dir = _resolve_run_id_to_run_dir(str(src_run_id))

        # 1) Source run config must exist (and we load it once here).
        src_cfg = _load_source_run_config_yaml(src_run_dir)

        # 2) Source run checkpoints must exist (avoid silently evaluating an untrained model).
        ckpt_root = src_run_dir / "checkpoints"
        best_dir = ckpt_root / "best"
        latest_dir = ckpt_root / "latest"
        if not best_dir.is_dir() and not latest_dir.is_dir():
            raise FileNotFoundError(
                "Required source run checkpoints not found.\n"
                f"Phase '{phase}' requires an existing trained run with checkpoints under:\n"
                f"  {best_dir}\n"
                f"or\n"
                f"  {latest_dir}\n"
                "If you want to train from scratch, use the 'pretrain' phase instead.\n"
            )

    merged["paths"] = _compute_paths(
        merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir
    )

    # Ensure ids exist at top-level, even when auto-generated in _compute_paths().
    if phase in ("pretrain", "finetune") and merged.get("run_id") is None:
        merged["run_id"] = merged["paths"].get("run_id")
    if phase == "eval" and merged.get("eval_id") is None:
        merged["eval_id"] = merged["paths"].get("eval_id")

    # ------------------------------------------------------------------
    # Phase-specific inheritance from the *source run config*.
    #
    # - finetune requires model_source.run_dir and inherits model + preprocess chunk/trim,
    #   then applies finetune-side overrides on top.
    # - eval requires model_source.run_dir and rebuilds model + embeddings + preprocess chunk/trim.
    #
    # NOTE: src_run_dir/src_cfg are validated + loaded above (before _compute_paths).
    # ------------------------------------------------------------------
    if phase in ("finetune", "eval"):
        if src_run_dir is None or src_cfg is None:
            raise RuntimeError(
                "Internal error: source run should have been validated before _compute_paths()."
            )

        if "model" not in src_cfg:
            raise KeyError(
                "Source run config YAML is missing required key 'model'.\n"
                f"path={src_run_dir}/{src_run_dir.name}.yaml"
            )

        if phase == "finetune":
            # Model: inherit base model spec from source and apply finetune overrides.
            model_override = copy.deepcopy(merged.get("model", {}))
            merged["model"] = copy.deepcopy(src_cfg["model"])
            if model_override:
                if not isinstance(model_override, dict):
                    raise TypeError(
                        "Finetune 'model' overrides must be a mapping (dict)."
                    )
                merged["model"] = _deep_merge(merged["model"], model_override)

            # Preprocess: inherit chunk/trim from source (keep finetune valid_windows, etc.).
            _inherit_preprocess_chunk_trim(merged, src_cfg, allow_override=True)

        else:  # eval
            if "embeddings" not in src_cfg:
                raise KeyError(
                    "Source run config YAML is missing required key 'embeddings'.\n"
                    f"path={src_run_dir}/{src_run_dir.name}.yaml"
                )

            merged["model"] = copy.deepcopy(src_cfg["model"])
            merged["embeddings"] = copy.deepcopy(src_cfg["embeddings"])

            # For eval, the embedding profile is defined by the source run.
            # Any CLI/profile selection is ignored to avoid config drift.
            merged["embeddings_profile"] = src_cfg.get(
                "embeddings_profile", merged.get("embeddings_profile")
            )
            _inherit_preprocess_chunk_trim(merged, src_cfg, allow_override=False)

    # Resolve model_source.run_dir to absolute path (if present)
    if isinstance(merged.get("model_source"), dict):
        model_dir = merged["model_source"].get("run_dir", None)
        if model_dir is not None:
            merged["model_source"]["run_dir"] = str(
                _resolve_run_id_to_run_dir(str(model_dir))
            )

    # Save merged config
    if phase == "tune_dct3d":
        out_dir = Path(merged["paths"]["tune_dir"]) / "history"
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = f"tune_dct3d_{timestamp}.yaml"
    elif phase == "eval":
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['eval_id']}.yaml"
    else:
        out_dir = Path(merged["paths"]["run_dir"])
        config_name = f"{merged['run_id']}.yaml"

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / config_name).open("w", encoding="utf-8") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    return ExperimentConfig(raw=merged)
