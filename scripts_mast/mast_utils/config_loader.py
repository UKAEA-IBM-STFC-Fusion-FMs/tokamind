"""
Project-level config loader for the MMT + MAST integration (scripts_mast).

Convention-based config layout (no pointers inside YAML):

scripts_mast/configs/
  common/
    embeddings.yaml
    pretrain.yaml
    finetune.yaml
    eval.yaml

  tasks_overrides/
    <task>/                                   (one folder per task)
      pretrain_overrides.yaml                 (optional)
      finetune_overrides.yaml                 (optional)
      eval_overrides.yaml                     (optional)
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
     pretrain/finetune; NOT merged for eval)

Required keys (explicit per phase)
----------------------------------
To keep configs self-contained and avoid implicit inheritance, each phase config
must define:

- ``seed``
- ``runtime`` (mapping; individual keys are user-defined)
- ``data.local``
- ``data.subset_of_shots``

CLI-based model selection (NEW)
--------------------------------
Model sources are now specified via CLI arguments, not YAML configs:

- ``finetune``: Use ``--model <run_id_or_path>`` to specify the warm-start source.
  The loader automatically sets ``model_source.run_id`` and inherits model/preprocess
  from the source run config.

- ``eval``: Use ``--model <run_id_or_path>`` to specify which model to evaluate.
  The loader rebuilds the model spec from the source run config.

Optional ``--tag`` argument allows versioning multiple experiments with the same source.

All resolved model sources and CLI arguments are saved in the config snapshot for
full reproducibility.

Notes
-----
- Benchmark task *definitions* are resolved separately (scripts_mast layer),
  not here.
"""

from __future__ import annotations

import copy
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
            "model_source.run_id must be a non-empty run id (folder name under <repo_root>/runs/)."
        )

    p = Path(s)

    # Enforce "run id only" (no relative/absolute paths).
    if p.is_absolute() or len(p.parts) != 1:
        raise ValueError(
            "model_source.run_id must be a run id (folder name under <repo_root>/runs/), "
            "e.g. 'pretrain_base'. Do not include 'runs/' or any path separators."
        )

    return (get_repo_root() / "runs" / p.parts[0]).resolve()


def _resolve_model_source_dir(model_source: Dict[str, Any], *, phase: str) -> tuple[Path, str | None]:
    """Resolve model_source to an absolute directory.

    Recommended config:
      model_source:
        run_id: pretrain_12345
        model_path: null   # if set, overrides run_id

    Returns
    -------
    (src_run_dir, src_run_id_for_yaml)
        src_run_dir: absolute path to the directory containing checkpoints/
        src_run_id_for_yaml: run_id to locate <run_id>.yaml inside src_run_dir (may be None)
    """
    if not isinstance(model_source, dict):
        raise TypeError("Config key 'model_source' must be a mapping (dict).")

    model_path = model_source.get("model_path", None)
    run_id = model_source.get("run_id", None)

    # External directory override.
    if model_path is not None:
        mp = str(model_path).strip()
        if not mp:
            raise ValueError("model_source.model_path, if provided, must be a non-empty path string.")
        p = _resolve_from_repo_root(mp)
        if not p.is_dir():
            raise FileNotFoundError(
                f"{phase} phase requires model_source.model_path to point to an existing directory.\n"
                f"Got: {p}"
            )
        return p, None

    # Otherwise require a run id under runs/
    if run_id is None:
        raise ValueError(
            f"{phase} phase requires model_source.run_id (a training run id under <repo_root>/runs/) "
            "or model_source.model_path (external run directory)."
        )

    return _resolve_run_id_to_run_dir(str(run_id)), str(run_id).strip()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _merge_stage_lists(base: list[Any], override: list[Any]) -> list[Any]:
    """Merge train.stages lists by stage 'name' so overrides can be partial.

    - If both lists contain dict items with a 'name' key, we deep-merge matching
      stages (by name) and preserve the base order.
    - Stages present only in the override are appended (in override order).
    - If the lists are not in the expected format, we fall back to replacement.
    """
    if not (
        all(isinstance(x, dict) and "name" in x for x in base)
        and all(isinstance(x, dict) and "name" in x for x in override)
    ):
        return copy.deepcopy(override)

    override_map: Dict[str, Dict[str, Any]] = {}
    override_order: list[str] = []
    for st in override:
        name = str(st.get("name"))
        override_map[name] = st
        override_order.append(name)

    base_names: set[str] = set()
    merged_list: list[Any] = []
    for st in base:
        name = str(st.get("name"))
        base_names.add(name)
        if name in override_map:
            merged_list.append(_deep_merge(st, override_map[name]))
        else:
            merged_list.append(copy.deepcopy(st))

    # Append new stages defined only in overrides
    for name in override_order:
        if name not in base_names:
            merged_list.append(copy.deepcopy(override_map[name]))

    return merged_list


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        elif k in out and k == "stages" and isinstance(out[k], list) and isinstance(v, list):
            # Special-case: allow partial overrides of train.stages by merging on stage name.
            out[k] = _merge_stage_lists(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_source_run_config_yaml(model_run_dir: Path) -> Dict[str, Any]:
    """Load the saved merged config YAML for a previous training run.

    Convention: the merged config snapshot lives at:
        <run_dir>/<run_dir.name>.yaml
    """
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
            if isinstance(override_chunk, dict) and isinstance(merged_pre["chunk"], dict):
                merged_pre["chunk"] = _deep_merge(merged_pre["chunk"], override_chunk)
            else:
                merged_pre["chunk"] = override_chunk

        if override_trim is not None:
            if isinstance(override_trim, dict) and isinstance(merged_pre["trim_chunks"], dict):
                merged_pre["trim_chunks"] = _deep_merge(merged_pre["trim_chunks"], override_trim)
            else:
                merged_pre["trim_chunks"] = override_trim


def _extract_model_id(model: str) -> str:
    """Extract model ID from path or return as-is if already an ID.

    Examples:
        'tokamind_base_v1' -> 'tokamind_base_v1'
        '/path/to/runs/my_model' -> 'my_model'
        'runs/my_model' -> 'my_model'
    """
    if "/" in model or "\\" in model:
        return Path(model).name
    return model


def _generate_finetune_run_id(task: str, model: str, tag: str | None) -> str:
    """Generate run_id for finetune: ft-{task}-{tag}-{model_id}

    If tag is None, format is: ft-{task}-{model_id}
    """
    model_id = _extract_model_id(model)
    if tag:
        return f"ft-{task}-{tag}-{model_id}"
    else:
        return f"ft-{task}-{model_id}"


def _inject_cli_overrides_finetune(
    merged: Dict[str, Any],
    model: str | None,
    tag: str | None,
) -> None:
    """Inject CLI overrides for finetune phase."""
    if model is None:
        raise ValueError(
            "Finetune phase requires --model <run_id_or_path> to specify the warm-start source.\n"
            "Example: python run_finetune.py --task task_1-1 --model tokamind_base_v1"
        )

    # Determine if model is a path or run_id
    model_is_path = "/" in model or "\\" in model

    # Set model_source
    merged.setdefault("model_source", {})
    if model_is_path:
        merged["model_source"]["model_path"] = str(Path(model).resolve())
        merged["model_source"]["run_id"] = None
    else:
        merged["model_source"]["run_id"] = model
        merged["model_source"]["model_path"] = None

    # Generate run_id if not explicitly set in YAML
    if merged.get("run_id") is None:
        merged["run_id"] = _generate_finetune_run_id(
            task=merged["task"],
            model=model,
            tag=tag,
        )

    # Store CLI metadata for reproducibility
    merged["cli"] = {
        "model": model,
        "tag": tag,
        "phase": "finetune",
    }


def _inject_cli_overrides_eval(
    merged: Dict[str, Any],
    model: str | None,
) -> None:
    """Inject CLI overrides for eval phase."""
    if model is None:
        raise ValueError(
            "Eval phase requires --model <run_id_or_path> to specify which model to evaluate.\n"
            "Example: python run_eval.py --task task_1-1 --model ft_task_1-1_from_base_v1"
        )

    # Determine if model is a path or run_id
    model_is_path = "/" in model or "\\" in model

    # Set model_source
    merged.setdefault("model_source", {})
    if model_is_path:
        merged["model_source"]["model_path"] = str(Path(model).resolve())
        merged["model_source"]["run_id"] = None
    else:
        merged["model_source"]["run_id"] = model
        merged["model_source"]["model_path"] = None

    # Store CLI metadata for reproducibility
    merged["cli"] = {
        "model": model,
        "phase": "eval",
    }


def _generate_pretrain_run_id(task: str, run_id: str | None, tag: str | None) -> str:
    """Generate run_id for pretrain phase.

    Priority:
        1. If run_id provided → use as-is
        2. If tag provided → {task}_{tag}
        3. Otherwise → {task}

    Examples:
        _generate_pretrain_run_id("task_1-1", "my_model", None) → "my_model"
        _generate_pretrain_run_id("task_1-1", None, "v2") → "task_1-1_v2"
        _generate_pretrain_run_id("task_1-1", None, None) → "task_1-1"
        _generate_pretrain_run_id("pretrain_inputs_actuators_to_inputs_outputs", "tokamind_v1", None) → "tokamind_v1"
    """
    if run_id:
        return run_id
    elif tag:
        return f"{task}_{tag}"
    else:
        return task


def _inject_cli_overrides_pretrain(
    merged: Dict[str, Any],
    task: str,
    run_id: str | None,
    tag: str | None,
) -> None:
    """Inject CLI overrides for pretrain phase.

    Generates run_id based on CLI arguments:
        - If --run-id provided: use as-is (full control)
        - If --tag provided: generate {task}_{tag}
        - Otherwise: use task name as run_id

    Parameters
    ----------
    merged : Dict[str, Any]
        Configuration dictionary to modify in-place
    task : str
        Task name
    run_id : str | None
        Explicit run_id from CLI (takes precedence)
    tag : str | None
        Optional tag for versioning
    """
    generated_run_id = _generate_pretrain_run_id(task, run_id, tag)
    merged["run_id"] = generated_run_id

    # Store CLI metadata for reproducibility
    merged["cli"] = {
        "run_id": run_id,
        "tag": tag,
        "phase": "pretrain",
    }


def _compute_paths(
    merged: Dict[str, Any], *, configs_root: Path, task_dir: Path
) -> Dict[str, str]:
    repo_root = get_repo_root()
    global_runs_root = repo_root / "runs"

    phase = merged.get("phase")
    if phase is None:
        raise ValueError("Missing required argument: phase.")

    task = merged.get("task", "unknown_task")

    # pretrain and finetune write into runs/<run_id>
    if phase in ("pretrain", "finetune"):
        run_id = merged.get("run_id")
        if run_id is None:
            raise ValueError(f"run_id must be set for phase={phase} before calling _compute_paths")

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

    # eval writes into runs/<model_id>/eval/
    if phase == "eval":
        model_source = merged.get("model_source", {})
        if not isinstance(model_source, dict):
            raise TypeError("model_source must be set for eval phase")

        # Get model_id from CLI-injected model_source
        model_id = model_source.get("run_id")
        model_path = model_source.get("model_path")

        if model_path:
            # If path provided, use its basename as model_id
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


def _load_and_merge_base_configs(
    *,
    task: str,
    phase: str,
    embeddings_profile: str,
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> Dict[str, Any]:
    """Load and merge base configuration files following the convention-based hierarchy.

    Merge order (later wins):
        1. common/embeddings.yaml
        2. common/{phase}.yaml
        3. tasks_overrides/{task}/{phase}_overrides.yaml (optional)
        4. tasks_overrides/{task}/embeddings_overrides/{profile}.yaml (pretrain/finetune only)

    Parameters
    ----------
    task : str
        Task identifier (folder name under tasks_overrides/)
    phase : str
        Training phase: pretrain, finetune, or eval
    embeddings_profile : str
        Embedding profile name (e.g., 'dct3d', 'vae')
    configs_root_path : Path
        Absolute path to configs root directory
    tasks_overrides_dir : Path
        Absolute path to task-specific overrides directory

    Returns
    -------
    Dict[str, Any]
        Merged configuration dictionary

    Notes
    -----
    - Embedding overrides are NOT merged for eval
    - Embedding overrides are required for pretrain/finetune
    """
    common_dir = configs_root_path / "common"

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
    embeddings_overrides_path = embeddings_overrides_dir / f"{embeddings_profile}.yaml"

    # Start merging
    merged: Dict[str, Any] = {}
    merged = _deep_merge(merged, _load_yaml(embeddings_path))
    merged = _deep_merge(merged, _load_yaml(phase_common_path))

    # Merge optional phase overrides
    if phase_overrides_path.is_file():
        merged = _deep_merge(merged, _load_yaml(phase_overrides_path))

    # Merge embedding overrides (pretrain/finetune only)
    if phase in ("pretrain", "finetune"):
        if not embeddings_overrides_path.is_file():
            raise FileNotFoundError(
                "Missing required task-level embedding overrides for "
                f"profile={embeddings_profile!r}, task={task!r}.\n"
                "Expected file:\n"
                f"  {embeddings_overrides_path}"
            )
        merged = _deep_merge(merged, _load_yaml(embeddings_overrides_path))

    return merged


def _inject_cli_model_overrides(
    merged: Dict[str, Any],
    *,
    phase: str,
    task: str,
    model: str | None,
    run_id: str | None,
    tag: str | None,
) -> None:
    """Inject CLI-provided model selection and auto-generate run/eval IDs.

    For pretrain phase:
        - Auto-generates run_id based on --run-id, --tag, or task name

    For finetune phase:
        - Sets model_source.run_id from --model CLI argument
        - Auto-generates run_id as: ft-{task}-{tag}-{model_id}
        - If no tag provided, omits it: ft-{task}-{model_id}

    For eval phase:
        - Sets model_source.run_id from --model CLI argument
        - Auto-generates eval_id based on model being evaluated

    Parameters
    ----------
    merged : Dict[str, Any]
        Configuration dictionary to modify in-place
    phase : str
        Training phase (pretrain, finetune, eval)
    task : str
        Task name
    model : str | None
        Model source (run_id or path) from CLI (for finetune/eval)
    run_id : str | None
        Explicit run_id from CLI (for pretrain)
    tag : str | None
        Optional experiment tag for versioning

    Notes
    -----
    - Modifies merged dict in-place
    - Model argument is required for finetune and eval phases
    - For pretrain, uses run_id > tag > task name priority
    """
    if phase == "pretrain":
        _inject_cli_overrides_pretrain(merged, task=task, run_id=run_id, tag=tag)
    elif phase == "finetune":
        _inject_cli_overrides_finetune(merged, model=model, tag=tag)
    elif phase == "eval":
        _inject_cli_overrides_eval(merged, model=model)
    else:
        raise ValueError(f"Unsupported phase for CLI override injection: {phase}")


def _inherit_from_source_model(merged: Dict[str, Any], *, phase: str) -> None:
    """Load source model configuration and inherit settings for warm-start or evaluation.

    This function implements the model inheritance logic for finetune and eval phases:
    - Loads the saved config from the source model's run directory
    - Validates that checkpoints exist
    - Inherits model architecture, embeddings, and preprocessing settings

    Finetune phase:
        - Inherits model architecture (allows task-specific overrides)
        - Inherits preprocessing settings (allows overrides)
        - Does NOT inherit embeddings (uses task-specific embeddings)

    Eval phase:
        - Inherits model architecture (no overrides allowed)
        - Inherits embeddings (no overrides allowed)
        - Inherits preprocessing settings (no overrides allowed)

    Parameters
    ----------
    merged : Dict[str, Any]
        Configuration dictionary to modify in-place
    phase : str
        Training phase (must be 'finetune' or 'eval')

    Raises
    ------
    TypeError
        If model_source is not properly set in merged config
    FileNotFoundError
        If source run directory or checkpoints don't exist
    KeyError
        If source config is missing required keys (model, embeddings)

    Notes
    -----
    - Modifies merged dict in-place
    - Stores resolved source paths in merged['model_source']
    - Requires model_source.run_id to be set (via CLI injection)
    """
    model_source = merged.get("model_source", {})
    if not isinstance(model_source, dict):
        raise TypeError("model_source must be set for finetune/eval phases")

    # Resolve source run directory
    src_run_dir, src_run_id_for_yaml = _resolve_model_source_dir(model_source, phase=phase)

    # Load source run config
    src_cfg = _load_source_run_config_yaml(src_run_dir)

    # Validate checkpoints exist
    ckpt_root = src_run_dir / "checkpoints"
    best_dir = ckpt_root / "best"
    latest_dir = ckpt_root / "latest"
    if not best_dir.is_dir() and not latest_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoints found in {src_run_dir}/checkpoints/\n"
            f"Expected: {best_dir} or {latest_dir}"
        )

    # Inherit model config
    if "model" not in src_cfg:
        raise KeyError(f"Source config missing 'model' key: {src_run_dir}")

    if phase == "finetune":
        # Inherit model + preprocess, allow overrides
        model_override = copy.deepcopy(merged.get("model", {}))
        merged["model"] = copy.deepcopy(src_cfg["model"])
        if model_override:
            merged["model"] = _deep_merge(merged["model"], model_override)
        _inherit_preprocess_chunk_trim(merged, src_cfg, allow_override=True)

    else:  # eval
        # Inherit model + embeddings + preprocess, no overrides
        if "embeddings" not in src_cfg:
            raise KeyError(f"Source config missing 'embeddings' key: {src_run_dir}")
        merged["model"] = copy.deepcopy(src_cfg["model"])
        merged["embeddings"] = copy.deepcopy(src_cfg["embeddings"])
        merged["embeddings_profile"] = src_cfg.get("embeddings_profile", merged.get("embeddings_profile"))
        _inherit_preprocess_chunk_trim(merged, src_cfg, allow_override=False)

    # Store resolved paths in model_source
    merged["model_source"]["run_dir"] = str(src_run_dir)
    if src_run_id_for_yaml:
        merged["model_source"]["run_id"] = src_run_id_for_yaml


def _finalize_and_save_config(
    merged: Dict[str, Any],
    *,
    phase: str,
    configs_root_path: Path,
    tasks_overrides_dir: Path,
) -> None:
    """Compute output paths, synchronize IDs, and save configuration snapshot.

    This function performs the final steps of config loading:
    1. Computes all output paths (run_dir, log_dir, checkpoint_dir, etc.)
    2. Syncs top-level run_id/eval_id with computed paths
    3. Saves the complete merged config as a YAML snapshot

    Config naming conventions:
        - pretrain/finetune: saves as {run_id}.yaml in run_dir
        - eval: saves as {eval_id}.yaml in model's eval/ subdirectory

    Parameters
    ----------
    merged : Dict[str, Any]
        Configuration dictionary to finalize and save
    phase : str
        Training phase (pretrain, finetune, or eval)
    configs_root_path : Path
        Absolute path to configs root directory
    tasks_overrides_dir : Path
        Absolute path to task-specific overrides directory

    Notes
    -----
    - Modifies merged dict in-place (adds 'paths', syncs IDs)
    - Creates output directories if they don't exist
    - Logs the saved config path
    """
    # Compute output paths
    merged["paths"] = _compute_paths(
        merged, configs_root=configs_root_path, task_dir=tasks_overrides_dir
    )

    # Sync top-level ids with paths
    if phase in ("pretrain", "finetune"):
        merged["run_id"] = merged["paths"]["run_id"]
    if phase == "eval":
        merged["eval_id"] = merged["paths"]["eval_id"]

    # Save merged config snapshot
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

    logger.info(f"Saved config snapshot: {config_path}")


def load_experiment_config(
    *,
    task: str,
    phase: str,
    configs_root: str | Path = "scripts_mast/configs",
    embeddings_profile: str = "dct3d",
    model: str | None = None,
    run_id: str | None = None,
    tag: str | None = None,
) -> ExperimentConfig:
    """Load and merge YAML configuration for a given task + phase.

    Parameters
    ----------
    task:
        Task folder name under ``scripts_mast/configs/tasks_overrides/<task>/``.
    phase:
        One of: ``pretrain``, ``finetune``, ``eval``.
    configs_root:
        Root folder containing the convention-based config tree.
    embeddings_profile:
        Embedding profile key. The loader merges the corresponding task-level
        embedding overrides file:

            tasks_overrides/<task>/embeddings_overrides/<embeddings_profile>.yaml (pretrain/finetune only)

        Default: ``dct3d``.
    model:
        Model source for finetune/eval (run_id or path). Required for finetune and eval phases.
        Auto-generates run_id/eval_id based on task + model + tag.
    run_id:
        Explicit run_id for pretrain phase. If not provided, uses --tag or task name.
    tag:
        Optional experiment tag for versioning (pretrain/finetune).
        - Pretrain: generates {task}_{tag} if no run_id provided
        - Finetune: generates ft-{task}-{tag}-{model_id}

    Returns
    -------
    ExperimentConfig
        A dynamic wrapper around the merged raw dict (``cfg.raw``).
    """
    # Validate inputs
    if phase not in ("pretrain", "finetune", "eval"):
        raise ValueError(f"Unsupported phase: {phase}")

    emb_profile = str(embeddings_profile).strip()
    if not emb_profile:
        raise ValueError("embeddings_profile must be a non-empty string")

    # Resolve paths
    configs_root_path = _resolve_from_repo_root(str(configs_root))
    tasks_overrides_dir = configs_root_path / "tasks_overrides" / task

    # Load and merge base configs
    merged = _load_and_merge_base_configs(
        task=task,
        phase=phase,
        embeddings_profile=emb_profile,
        configs_root_path=configs_root_path,
        tasks_overrides_dir=tasks_overrides_dir,
    )

    # Enforce task + phase from CLI
    task_in_yaml = merged.get("task", None)
    if task_in_yaml is not None and str(task_in_yaml) != str(task):
        raise ValueError(
            f"Task mismatch: requested task={task!r} but an overrides file defines task={task_in_yaml!r}."
        )
    merged["task"] = task
    merged["phase"] = phase
    merged["embeddings_profile"] = emb_profile

    # Inject CLI overrides for model selection
    _inject_cli_model_overrides(
        merged,
        phase=phase,
        task=task,
        model=model,
        run_id=run_id,
        tag=tag,
    )

    # Inherit from source model (finetune/eval only)
    if phase in ("finetune", "eval"):
        _inherit_from_source_model(merged, phase=phase)

    # Finalize: compute paths and save config snapshot
    _finalize_and_save_config(
        merged,
        phase=phase,
        configs_root_path=configs_root_path,
        tasks_overrides_dir=tasks_overrides_dir,
    )

    return ExperimentConfig(raw=merged)
