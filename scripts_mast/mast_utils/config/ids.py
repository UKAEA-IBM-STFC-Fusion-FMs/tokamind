"""
run-id and model-id naming conventions for experiment tracking.

This module provides utilities for generating consistent, descriptive run identifiers that encode key experiment
metadata (task, model source, init mode, tags) into the directory name.

Naming conventions:
- Pretrain: {task}[_{tag}] or explicit run_id
- Finetune warmstart: ft-{task}-ws-{model}[-{tag}]
- Finetune scratch: ft-{task}-scratch[-{tag}]
- Eval: eval/ subdirectory under model run_dir
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal


# ----------------------------------------------------------------------------------------------------------------------
def extract_model_id(model: str) -> str:
    """
    Extract model ID from a path, or return the value unchanged.

    Parameters
    ----------
    model : str
        Model name or path.

    Returns
    -------
    str
        Extracted model ID.

    """

    if ("/" in model) or ("\\" in model):
        return Path(model).name
    return model


# ----------------------------------------------------------------------------------------------------------------------
def generate_finetune_run_id(
    *,
    task: str,
    model: str | None,
    tag: str | None,
    init_mode: Literal["warmstart", "scratch"],
) -> str:
    """
    Generate finetune run ID with explicit init marker.

    Parameters
    ----------
    task : str
        Task identifier.
    model : str | None
        Model name or path, or None.
    tag : str | None
        Optional experiment tag.
    init_mode : Literal["warmstart", "scratch"]
        Finetune initialization mode, either "warmstart" or "scratch".

    Returns
    -------
    str
        Resulting run ID for finetune.

    Raises
    ------
    ValueError
        Run ID generation with `init_mode=warmstart` requires a non-empty `model`.
        Value of `init_mode` not in ["warmstart", "scratch"].

    """

    if init_mode == "warmstart":
        if not model:
            raise ValueError("Run ID generation with `init_mode=warmstart` requires a non-empty `model`.")

        base = f"ft-{task}-ws-{extract_model_id(model=model)}"
    elif init_mode == "scratch":
        base = f"ft-{task}-scratch"
    else:
        raise ValueError(f"Unsupported `init_mode` '{init_mode!r}' for finetune.")

    if tag:
        return f"{base}-{tag}"
    return base


# ----------------------------------------------------------------------------------------------------------------------
def generate_pretrain_run_id(task: str, run_id: str | None, tag: str | None) -> str:
    """
    Generate pretrain run_id with priority: explicit ID > tag > task.

    Parameters
    ----------
    task : str
        Task identifier.
    run_id : str | None
        Run ID, or None.
    tag : str | None
        Optional experiment tag.

    Returns
    -------
    str
        Resulting ID for pretrain.

    Raises
    ------
        If `task` is empty string when `run_id` and `tag` are both None.

    """

    if run_id:
        return run_id
    if tag:
        return f"{task}_{tag}"

    if not task:
        raise ValueError("Argument `task` must be a non-empty string when `run_id` and `tag` are both None.")

    return task
