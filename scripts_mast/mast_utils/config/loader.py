"""
Top-level config loader orchestration for scripts_mast.

Pipeline:
1. Merge base YAML configs
2. Inject CLI model/run overrides
3. Apply phase semantics and optional source inheritance
4. Finalize paths and persist the config snapshot
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from mmt.utils.config.schema import ExperimentConfig

from .cli_overrides import inject_cli_model_overrides
from .finalize import finalize_and_save_config
from .inheritance import apply_finetune_model_semantics, inherit_from_source_model
from .merge import load_and_merge_base_configs, resolve_from_repo_root


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.ConfigLoader")


# ----------------------------------------------------------------------------------------------------------------------
def load_experiment_config(
    *,
    task: str,
    phase: Literal["pretrain", "finetune", "eval"],
    configs_root: str | Path = "scripts_mast/configs",
    embeddings_profile: str = "dct3d",
    model: str | None = None,
    run_id: str | None = None,
    tag: str | None = None,
    finetune_init: Literal["warmstart", "scratch"] | None = None,
) -> ExperimentConfig:
    """
    Load, merge, and persist experiment config for a task+phase run.

    Merge pipeline:
    1. Base YAML merge (common + task overrides + embeddings profile)
    2. CLI injection (`--model`, `--run-id`, `--tag`, `--init`)
    3. Phase semantics:
      - finetune scratch: build `model` from `model_scratch` + shared overrides
      - finetune warmstart: inherit source model, keep current preprocess config
      - eval: inherit from source run config
    4. Path finalization and config snapshot write

    Parameters
    ----------
    task : str
        Task identifier.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".
    configs_root : str | Path
        Path to the root directory for configuration files. It could be either a relative or absolute path.
        Optional. Default: "scripts_mast/configs".
    embeddings_profile : str
        Path to YAML file with embeddings profile.
        Optional. Default: "dct3d".
    model : str | None
        Source model for finetune/warmstart and eval only, ignored for pretrain.
        Optional. Default: None.
    run_id : str | None
        Explicit run identifier, or None for auto-generation.
        Optional. Default: None
    tag : str | None
        Optional experiment tag.
        Optional. Default: None
    finetune_init : Literal["warmstart", "scratch"] | None
        Finetune initialization mode, either "warmstart" or "scratch", required if `phase` is "finetune".
        Optional. Default: None.

    Returns
    -------
    ExperimentConfig
        Resulting experiment configuration object.

    Raises
    ------
    ValueError
        If `phase` not in Literal["pretrain", "finetune", "eval"].
        If `embeddings_profile` is an empty string.
        If `task` is passed and `merged["task"]` defines an override task.
        If unexpected value in `merged["cli"]["init"]`.

    """

    if phase not in ["pretrain", "finetune", "eval"]:
        raise ValueError(f"Unsupported `phase` {phase}.")

    emb_profile = str(embeddings_profile).strip()
    if not emb_profile:
        raise ValueError("`embeddings_profile` must be a non-empty string.")

    configs_root_path = resolve_from_repo_root(str(configs_root))
    tasks_overrides_dir = configs_root_path / "tasks_overrides" / task

    merged = load_and_merge_base_configs(
        task=task,
        phase=phase,
        embeddings_profile=emb_profile,
        configs_root_path=configs_root_path,
        tasks_overrides_dir=tasks_overrides_dir,
    )

    task_in_yaml = merged.get("task")
    if (task_in_yaml is not None) and (str(task_in_yaml) != str(task)):
        raise ValueError(
            f"Task mismatch: requested `task` {task!r} but `merged['task']` defines an override task={task_in_yaml!r}."
        )

    merged["task"] = task
    merged["phase"] = phase
    merged["embeddings_profile"] = emb_profile

    inject_cli_model_overrides(
        merged=merged,
        phase=phase,
        task=task,
        model=model,
        run_id=run_id,
        tag=tag,
        finetune_init=finetune_init,
    )

    if phase == "finetune":
        init_mode: str = str((merged.get("cli") or {}).get("init", "warmstart")).lower()
        if init_mode not in ["warmstart", "scratch"]:
            raise ValueError(f"Unexpected finetune init_mode {init_mode!r} in `merged['cli']['init']`.")
        apply_finetune_model_semantics(merged=merged, init_mode=init_mode)
        if init_mode == "warmstart":
            inherit_from_source_model(merged=merged, phase=phase)
        else:
            logger.warning("Finetune init=scratch: skipping warm-start inheritance from source model.")

    elif phase == "eval":
        inherit_from_source_model(merged=merged, phase=phase)

    else:  # -> I.e., phase is "pretrain"
        pass  # Nothing to do for pretrain.

    finalize_and_save_config(
        merged=merged,
        phase=phase,
        configs_root_path=configs_root_path,
        tasks_overrides_dir=tasks_overrides_dir,
    )

    return ExperimentConfig(raw=merged)
