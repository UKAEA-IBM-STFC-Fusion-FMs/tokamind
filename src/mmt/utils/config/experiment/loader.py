"""
Dataset-agnostic top-level config loader orchestration.

Pipeline:
1. Merge base YAML configs
2. Inject CLI model/run overrides
3. Apply phase semantics and optional source inheritance
4. Finalize paths and persist the config snapshot
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, MutableMapping
from pathlib import Path
from typing import Any, Literal

from mmt.utils.config.schema import ExperimentConfig

from .cli_overrides import inject_cli_model_overrides
from .finalize import finalize_and_save_config
from .inheritance import (
    apply_finetune_model_semantics,
    infer_embeddings_profile_from_source,
    infer_model_profile_from_source,
    inherit_from_source_model,
)
from .merge import load_and_merge_base_configs, resolve_from_repo_root


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.ConfigLoader")


def _normalize_preprocess_chunks(merged: MutableMapping[str, Any]) -> None:
    """Materialize canonical role-specific chunks while accepting saved legacy configs.

    New configs should define ``preprocess.chunks`` directly.  Existing run
    snapshots and local overrides may still carry ``chunk`` plus
    ``trim_chunks``; those are translated at the loader boundary so source-run
    inheritance remains usable during the migration.
    """

    preprocess = merged.get("preprocess")
    if not isinstance(preprocess, MutableMapping) or "chunks" in preprocess:
        return
    chunk = preprocess.get("chunk")
    trim = preprocess.get("trim_chunks")
    if not isinstance(chunk, Mapping) or not isinstance(trim, Mapping):
        return
    length = chunk.get("chunk_length")
    if length is None:
        return
    stride = chunk.get("stride") if chunk.get("stride") is not None else length
    max_chunks = trim.get("max_chunks", 1)
    preprocess["chunks"] = {
        "input": {"chunk_length": length, "stride": stride, "max_chunks": max_chunks},
        "actuator": {"chunk_length": length, "stride": stride, "max_chunks": max_chunks},
        "output": {"max_chunks": 1},
    }
    logger.warning("Normalized legacy preprocess.chunk/trim_chunks into preprocess.chunks.")


# ----------------------------------------------------------------------------------------------------------------------
def _model_source_cfg_from_cli(model_source: str | None) -> dict[str, Any]:
    """
    Convert a CLI ``--model_source`` value into the internal source mapping.

    Parameters
    ----------
    model_source : str | None
        Source model run id or path.

    Returns
    -------
    dict[str, Any]
        Mapping with ``run_id`` or ``model_path`` set.

    Raises
    ------
    ValueError
        If ``model_source`` is missing or empty.
    """

    if model_source is None:
        raise ValueError("Eval phase requires --model_source <run_id_or_path> before model_profile can be inferred.")

    source = str(model_source).strip()
    if not source:
        raise ValueError("Eval phase requires a non-empty --model_source before model_profile can be inferred.")

    model_is_path = ("/" in source) or ("\\" in source)
    if model_is_path:
        return {"model_path": str(Path(source).resolve()), "run_id": None}
    return {"run_id": source, "model_path": None}


# ----------------------------------------------------------------------------------------------------------------------
def load_experiment_config(
    *,
    task: str,
    phase: Literal["pretrain", "finetune", "eval"],
    configs_root: str | Path = "scripts_mast/configs",
    model_profile: str | None = None,
    embeddings_profile: str = "dct3d",
    model_source: str | None = None,
    run_id: str | None = None,
    tag: str | None = None,
    tag_date: bool = False,
    finetune_init: Literal["warmstart", "scratch"] | None = None,  # noqa - Ignore expected type warning
    integration_hook: Callable[[MutableMapping[str, Any], str], None] | None = None,
) -> ExperimentConfig:
    """
    Load, merge, and persist experiment config for a task+phase run.

    Merge pipeline:
    1. Base YAML merge (common + task overrides + embeddings profile)
    2. CLI injection (`--model_source`, `--run-id`, `--tag`, `--tag-date`, `--init`)
    3. Phase semantics:
      - finetune scratch: build `model` directly from `model_scratch`
      - finetune warmstart: inherit source model, keep current preprocess config
      - eval: inherit from source run config
    4. Optional integration-owned post-inheritance hook
    5. Path finalization and config snapshot write

    Parameters
    ----------
    task : str
        Task identifier.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".
    configs_root : str | Path
        Path to the root directory for configuration files. It could be either a relative or absolute path.
        Optional. Default: "scripts_mast/configs".
    model_profile : str | None
        Model profile name; its folder ``configs/<model_profile>/`` holds ``phases/``, ``tasks/`` and ``embeddings/``.
        For eval, this only selects the generic eval defaults before source-run inheritance; when omitted for eval, it
        is inferred from ``model_source`` before loading defaults.
        Optional. Default: None.
    embeddings_profile : str
        Embedding profile name under ``configs/<model_profile>/embeddings/``.
        Optional. Default: "dct3d".
    model_source : str | None
        Source model for finetune/warmstart and eval only, ignored for pretrain.
        Optional. Default: None.
    run_id : str | None
        Explicit run identifier, or None for auto-generation.
        Optional. Default: None
    tag : str | None
        Optional experiment tag.
        Optional. Default: None
    tag_date : bool
        Whether to append a UTC timestamp to the experiment or evaluation tag.
        Optional. Default: False.
    finetune_init : Literal["warmstart", "scratch"] | None
        Finetune initialization mode, either "warmstart" or "scratch", required if `phase` is "finetune".
        Optional. Default: None.
    integration_hook : Callable[[MutableMapping[str, Any], str], None] | None
        Optional integration-owned callback invoked after source inheritance and before path finalization. This allows
        dataset layers to inherit their own config fields without adding dataset semantics to MMT.
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
    FileNotFoundError
        If required phase or embedding profile config files are missing.
    TypeError
        If aggregated task overrides or embedding profile blocks have invalid types.

    """

    if phase not in ["pretrain", "finetune", "eval"]:
        raise ValueError(f"Unsupported `phase` {phase}.")

    emb_profile = str(embeddings_profile).strip()
    if not emb_profile:
        raise ValueError("`embeddings_profile` must be a non-empty string.")

    model_profile_explicit = model_profile is not None
    if model_profile_explicit:
        model_profile_norm = str(model_profile).strip()
        if not model_profile_norm:
            raise ValueError("`model_profile` must be a non-empty string when provided.")
    elif phase == "eval":
        model_source_cfg = _model_source_cfg_from_cli(model_source=model_source)
        model_profile_norm = infer_model_profile_from_source(model_source=model_source_cfg, phase="eval")
        emb_profile = infer_embeddings_profile_from_source(model_source=model_source_cfg, phase="eval")
        logger.info(
            "Eval model_profile and embeddings_profile inferred from model_source=%r: %s / %s",
            model_source,
            model_profile_norm,
            emb_profile,
        )
    else:
        model_profile_norm = "mmt"

    if phase == "finetune":
        init_mode_for_merge = str(finetune_init or "").lower()
        if init_mode_for_merge not in ["warmstart", "scratch"]:
            raise ValueError(
                "Argument `finetune_init` is required for phase 'finetune' and must be one of "
                f"['warmstart', 'scratch'], got {finetune_init!r}."
            )
    else:
        init_mode_for_merge = None

    configs_root_path = resolve_from_repo_root(str(configs_root))

    merged = load_and_merge_base_configs(
        task=task,
        phase=phase,
        model_profile=model_profile_norm,
        embeddings_profile=emb_profile,
        configs_root_path=configs_root_path,
        finetune_init=init_mode_for_merge,
    )
    _normalize_preprocess_chunks(merged)

    task_in_yaml = merged.get("task")
    if (task_in_yaml is not None) and (str(task_in_yaml) != str(task)):
        raise ValueError(
            f"Task mismatch: requested `task` {task!r} but `merged['task']` defines an override task={task_in_yaml!r}."
        )

    merged["task"] = task
    merged["phase"] = phase
    merged["model_profile"] = model_profile_norm
    merged["embeddings_profile"] = emb_profile

    inject_cli_model_overrides(
        merged=merged,
        phase=phase,
        task=task,
        model_source=model_source,
        run_id=run_id,
        tag=tag,
        tag_date=tag_date,
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

    _normalize_preprocess_chunks(merged)

    if integration_hook is not None:
        integration_hook(merged, phase)

    finalize_and_save_config(
        merged=merged,
        phase=phase,
        configs_root_path=configs_root_path,
    )

    return ExperimentConfig(raw=merged)
