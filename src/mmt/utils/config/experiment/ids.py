"""
run-id and model-id naming conventions for experiment tracking.

This module provides utilities for generating consistent, descriptive run identifiers that encode key experiment
metadata (task, model source, init mode, tags) into the directory name.

Naming conventions:
- Pretrain: {task}[_{tag}] or explicit run_id
- Finetune warmstart: ft-{task}-ws-{model}-{model_profile}[-{tag}]
- Finetune scratch: ft-{task}-scratch-{model_profile}[-{tag}]
- Eval: eval[_tag]/ subdirectory under model run_dir
"""

from __future__ import annotations

from datetime import datetime, timezone
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
def normalize_run_tag(tag: str | None) -> str | None:
    """
    Normalize and validate an optional run/eval tag.

    Parameters
    ----------
    tag : str | None
        Optional tag.

    Returns
    -------
    str | None
        Stripped tag, or None when no tag is provided.

    Raises
    ------
    ValueError
        If the tag contains path separators or unsupported characters.

    """

    if tag is None:
        return None

    tag_norm = str(tag).strip()
    if not tag_norm:
        return None

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    if any(ch not in allowed for ch in tag_norm):
        raise ValueError(f"Run/eval tags may contain only letters, numbers, '-', '_', and '.'. Got tag={tag_norm!r}.")
    if tag_norm in {".", ".."}:
        raise ValueError(f"Run/eval tag must not be {tag_norm!r}.")

    return tag_norm


# ----------------------------------------------------------------------------------------------------------------------
def resolve_run_tag(
    tag: str | None,
    *,
    tag_date: bool = False,
    now: datetime | None = None,
) -> str | None:
    """Normalize an optional tag and append a UTC timestamp when requested.

    Parameters
    ----------
    tag : str | None
        User-provided run or evaluation tag.
    tag_date : bool
        Whether to append a UTC timestamp to the tag. When no text tag is supplied, the timestamp becomes the tag.
        Optional. Default: False.
    now : datetime | None
        Optional time used for deterministic callers and tests. Naive values are interpreted as UTC. Defaults to the
        current UTC time.

    Returns
    -------
    str | None
        Normalized literal tag, a timestamp-appended tag, a ``YYYYMMDDTHHMMSSZ`` timestamp, or ``None`` when no tag
        was provided.

    Notes
    -----
    Timestamp suffixes have whole-second precision. Concurrent launches within the same second can therefore resolve
    to the same identifier.
    """

    tag_norm = normalize_run_tag(tag=tag)
    if not tag_date:
        return tag_norm

    timestamp = now if now is not None else datetime.now(timezone.utc)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp_tag = timestamp.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{tag_norm}_{timestamp_tag}" if tag_norm else timestamp_tag


# ----------------------------------------------------------------------------------------------------------------------
def generate_eval_id(tag: str | None, *, tag_date: bool = False) -> str:
    """
    Generate the eval output directory name.

    Parameters
    ----------
    tag : str | None
        Optional evaluation tag.
    tag_date : bool
        Whether to append a UTC timestamp to the evaluation tag.
        Optional. Default: False.

    Returns
    -------
    str
        ``"eval"`` when no tag is provided, otherwise ``"eval_<tag>"``.

    """

    tag_norm = resolve_run_tag(tag=tag, tag_date=tag_date)
    if tag_norm:
        return f"eval_{tag_norm}"
    return "eval"


# ----------------------------------------------------------------------------------------------------------------------
def generate_finetune_run_id(
    *,
    task: str,
    model: str | None,
    model_profile: str,
    tag: str | None,
    tag_date: bool = False,
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
    model_profile : str
        Model profile selected for the run.
    tag : str | None
        Optional experiment tag.
    tag_date : bool
        Whether to append a UTC timestamp to the experiment tag.
        Optional. Default: False.
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

    model_profile_norm = normalize_run_tag(tag=model_profile)
    if not model_profile_norm:
        raise ValueError("Run ID generation for finetune requires a non-empty `model_profile`.")

    if init_mode == "warmstart":
        if not model:
            raise ValueError("Run ID generation with `init_mode=warmstart` requires a non-empty `model`.")

        base = f"ft-{task}-ws-{extract_model_id(model=model)}-{model_profile_norm}"
    elif init_mode == "scratch":
        base = f"ft-{task}-scratch-{model_profile_norm}"
    else:
        raise ValueError(f"Unsupported `init_mode` '{init_mode!r}' for finetune.")

    tag_norm = resolve_run_tag(tag=tag, tag_date=tag_date)
    if tag_norm:
        return f"{base}-{tag_norm}"
    return base


# ----------------------------------------------------------------------------------------------------------------------
def generate_pretrain_run_id(
    task: str,
    run_id: str | None,
    tag: str | None,
    *,
    tag_date: bool = False,
) -> str:
    """
    Generate a pretrain run ID from an explicit ID, tag, or task name.

    Parameters
    ----------
    task : str
        Task identifier.
    run_id : str | None
        Run ID, or None.
    tag : str | None
        Optional experiment tag.
    tag_date : bool
        Whether to append a UTC timestamp to the experiment tag.
        Optional. Default: False.

    Returns
    -------
    str
        Resulting ID for pretrain.

    Raises
    ------
        If `task` is empty string when `run_id` and `tag` are both None.

    """

    tag_norm = resolve_run_tag(tag=tag, tag_date=tag_date)
    if run_id:
        return f"{run_id}_{tag_norm}" if tag_norm else run_id
    if tag_norm:
        return f"{task}_{tag_norm}"

    if not task:
        raise ValueError("Argument `task` must be a non-empty string when `run_id` and `tag` are both None.")

    return task
