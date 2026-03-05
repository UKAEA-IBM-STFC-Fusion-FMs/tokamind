"""
CLI override injection for pretrain/finetune/eval phases.

This module handles command-line parameter injection into the merged config,
translating user-facing CLI arguments (--model, --init, --tag, --run_id) into
the internal config structure.

Key responsibilities:
- Validate CLI parameter combinations (e.g., warmstart requires --model)
- Inject model_source config for warmstart/eval
- Generate run_id if not explicitly provided
- Store CLI metadata in config['cli'] for debugging/logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .ids import generate_finetune_run_id, generate_pretrain_run_id

logger = logging.getLogger("mmt.ConfigLoader")


def inject_cli_overrides_pretrain(
    merged: Dict[str, Any],
    *,
    task: str,
    run_id: str | None,
    tag: str | None,
) -> None:
    """
    Inject CLI overrides for pretrain phase.

    Generates or uses explicit run_id and stores CLI metadata.

    Parameters
    ----------
    merged : Dict[str, Any]
        Merged config dictionary (modified in-place)
    task : str
        Task identifier
    run_id : str | None
        Explicit run identifier, or None for auto-generation
    tag : str | None
        Optional experiment tag

    Side Effects
    ------------
    Modifies merged dict in-place:
    - Sets merged['run_id']
    - Sets merged['cli'] with CLI metadata
    """
    merged["run_id"] = generate_pretrain_run_id(task, run_id, tag)
    merged["cli"] = {"run_id": run_id, "tag": tag, "phase": "pretrain"}


def inject_cli_overrides_finetune(
    merged: Dict[str, Any],
    *,
    model: str | None,
    tag: str | None,
    init_mode: str | None,
) -> None:
    """
    Inject CLI overrides for finetune phase.

    Handles both warmstart and scratch initialization modes:
    - Warmstart: requires --model, sets model_source config
    - Scratch: ignores --model, sets model_source to None

    Parameters
    ----------
    merged : Dict[str, Any]
        Merged config dictionary (modified in-place)
    model : str | None
        Source model for warmstart (run_id or path). Required for warmstart,
        ignored for scratch.
    tag : str | None
        Optional experiment tag
    init_mode : str | None
        Initialization mode: "warmstart" (default) or "scratch"

    Raises
    ------
    ValueError
        If init_mode invalid or warmstart missing --model
    TypeError
        If model_source config structure invalid

    Side Effects
    ------------
    Modifies merged dict in-place:
    - Sets merged['run_id'] (auto-generated if not in config)
    - Sets merged['model_source'] (warmstart) or None (scratch)
    - Sets merged['cli'] with CLI metadata

    Notes
    -----
    Model source resolution:
    - If model contains '/' or backslash, treated as path (sets model_path)
    - Otherwise, treated as run_id (sets run_id, looks in runs/{run_id}/)
    """
    mode = "warmstart" if init_mode is None else str(init_mode).strip().lower()
    if mode not in ("warmstart", "scratch"):
        raise ValueError(
            "Finetune init mode must be one of {'warmstart', 'scratch'} "
            f"(got {init_mode!r})."
        )

    model_norm = None
    if model is not None:
        m = str(model).strip()
        if m:
            model_norm = m

    if mode == "warmstart":
        if model_norm is None:
            raise ValueError(
                "Finetune init=warmstart requires --model <run_id_or_path>.\n"
                "Example: python run_finetune.py --task task_1-1 --init warmstart "
                "--model tokamind_base_v1"
            )

        model_is_path = "/" in model_norm or "\\" in model_norm

        model_source_cfg = merged.get("model_source")
        if model_source_cfg is None:
            model_source_cfg = {}
            merged["model_source"] = model_source_cfg
        if not isinstance(model_source_cfg, dict):
            raise TypeError(
                "Config key 'model_source' must be a mapping (dict) or null."
            )

        if model_is_path:
            model_source_cfg["model_path"] = str(Path(model_norm).resolve())
            model_source_cfg["run_id"] = None
        else:
            model_source_cfg["run_id"] = model_norm
            model_source_cfg["model_path"] = None
    else:
        if model_norm is not None:
            logger.warning(
                "Finetune init=scratch ignores --model=%s (no warm-start will be used).",
                model_norm,
            )
        merged["model_source"] = None

    if merged.get("run_id") is None:
        merged["run_id"] = generate_finetune_run_id(
            task=merged["task"],
            model=model_norm,
            tag=tag,
            init_mode=mode,
        )

    merged["cli"] = {
        "init": mode,
        "model": model_norm,
        "tag": tag,
        "phase": "finetune",
    }


def inject_cli_overrides_eval(
    merged: Dict[str, Any],
    *,
    model: str | None,
) -> None:
    """
    Inject CLI overrides for eval phase.

    Evaluation always requires a source model to evaluate. This function
    validates --model is provided and sets up model_source config.

    Parameters
    ----------
    merged : Dict[str, Any]
        Merged config dictionary (modified in-place)
    model : str | None
        Source model to evaluate (run_id or path). Required.

    Raises
    ------
    ValueError
        If model is None (--model required for eval)
    TypeError
        If model_source config structure invalid

    Side Effects
    ------------
    Modifies merged dict in-place:
    - Sets merged['model_source'] with run_id or model_path
    - Sets merged['cli'] with CLI metadata

    Notes
    -----
    Model source resolution:
    - If model contains '/' or backslash, treated as path (sets model_path)
    - Otherwise, treated as run_id (sets run_id, looks in runs/{run_id}/)
    """
    if model is None:
        raise ValueError(
            "Eval phase requires --model <run_id_or_path> to specify which model to evaluate.\n"
            "Example: python run_eval.py --task task_1-1 --model ft-task_1-1-ws-base-v1"
        )

    model_is_path = "/" in model or "\\" in model

    model_source_cfg = merged.get("model_source")
    if model_source_cfg is None:
        model_source_cfg = {}
        merged["model_source"] = model_source_cfg
    if not isinstance(model_source_cfg, dict):
        raise TypeError("Config key 'model_source' must be a mapping (dict) or null.")

    if model_is_path:
        model_source_cfg["model_path"] = str(Path(model).resolve())
        model_source_cfg["run_id"] = None
    else:
        model_source_cfg["run_id"] = model
        model_source_cfg["model_path"] = None

    merged["cli"] = {"model": model, "phase": "eval"}


def inject_cli_model_overrides(
    merged: Dict[str, Any],
    *,
    phase: str,
    task: str,
    model: str | None,
    run_id: str | None,
    tag: str | None,
    finetune_init: str | None,
) -> None:
    """
    Inject CLI model selection and run-id/eval-id metadata by phase.

    Top-level dispatcher that routes to phase-specific CLI injection functions.

    Parameters
    ----------
    merged : Dict[str, Any]
        Merged config dictionary (modified in-place)
    phase : str
        Training phase: "pretrain", "finetune", or "eval"
    task : str
        Task identifier
    model : str | None
        Source model (finetune warmstart/eval only)
    run_id : str | None
        Explicit run identifier (pretrain only)
    tag : str | None
        Optional experiment tag (pretrain/finetune)
    finetune_init : str | None
        Finetune initialization mode: "warmstart" or "scratch"

    Raises
    ------
    ValueError
        If phase unsupported or phase-specific validation fails

    Side Effects
    ------------
    Modifies merged dict in-place via phase-specific injection functions.
    """
    if phase == "pretrain":
        inject_cli_overrides_pretrain(
            merged,
            task=task,
            run_id=run_id,
            tag=tag,
        )
    elif phase == "finetune":
        inject_cli_overrides_finetune(
            merged,
            model=model,
            tag=tag,
            init_mode=finetune_init,
        )
    elif phase == "eval":
        inject_cli_overrides_eval(
            merged,
            model=model,
        )
    else:
        raise ValueError(f"Unsupported phase for CLI override injection: {phase}")
