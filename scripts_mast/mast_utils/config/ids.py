"""
Run-id and model-id naming conventions for experiment tracking.

This module provides utilities for generating consistent, descriptive run
identifiers that encode key experiment metadata (task, model source, init mode,
tags) into the directory name.

Naming conventions:
- Pretrain: {task}[_{tag}] or explicit run_id
- Finetune warmstart: ft-{task}-ws-{model}[-{tag}]
- Finetune scratch: ft-{task}-scratch[-{tag}]
- Eval: eval/ subdirectory under model run_dir
"""

from __future__ import annotations

from pathlib import Path


def extract_model_id(model: str) -> str:
    """Extract model id from a path, or return the value unchanged."""
    if "/" in model or "\\" in model:
        return Path(model).name
    return model


def generate_finetune_run_id(
    *,
    task: str,
    model: str | None,
    tag: str | None,
    init_mode: str,
) -> str:
    """Generate finetune run_id with explicit init marker."""
    if init_mode == "warmstart":
        if not model:
            raise ValueError("warmstart run_id generation requires a non-empty model.")
        base = f"ft-{task}-ws-{extract_model_id(model)}"
    elif init_mode == "scratch":
        base = f"ft-{task}-scratch"
    else:
        raise ValueError(f"Unsupported finetune init mode: {init_mode!r}")

    if tag:
        return f"{base}-{tag}"
    return base


def generate_pretrain_run_id(task: str, run_id: str | None, tag: str | None) -> str:
    """Generate pretrain run_id with priority: explicit id > tag > task."""
    if run_id:
        return run_id
    if tag:
        return f"{task}_{tag}"
    return task
