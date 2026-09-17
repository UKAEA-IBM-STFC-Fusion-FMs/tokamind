"""
Checkpoint management API for the Multi-Modal Transformer (MMT).

This module provides the public entry points for saving, loading, resuming, and warm-starting model checkpoints. It
orchestrates lower-level utilities from sibling modules (io, rng, blocks, warmstart) and defines the supported
checkpointing workflows.

Supported workflows
-------------------
1) Strict resume of the same run
   - Restores model weights, optimizer, scheduler, scaler, RNG state, and training metadata from checkpoints/latest.
   - Used when resuming interrupted training.

2) Warm-start / partial loading
   - Loads only compatible subsets of model parameters from a previous run (key + shape match), leaving others
     initialized.
   - Intended for pretraining → finetuning or cross-task initialization.
   - Implemented via load_parts_from_run_dir(...).

3) Load best weights for evaluation
   - Loads model weights strictly from checkpoints/best (or latest as fallback), without optimizer or RNG state.

Checkpoint layout
-----------------
run_dir/
  checkpoints/
    latest/   # full training resume state
    best/     # best-performing model weights only

Model requirements
------------------
Models used with this API must expose ``get_named_blocks()`` with stable block names.
"""

from __future__ import annotations

import json
from json import JSONDecodeError
import os
import time
from typing import Any, cast
from collections.abc import Callable, Mapping

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.amp.grad_scaler import GradScaler

from .block_io import save_model_blocks, load_model_blocks
from .io import atomic_save, atomic_json_save, torch_load_full, best_or_latest_dir
from .rng import capture_rng_state, restore_rng_state
from .strict import inspect_resume, write_manifest
from mmt.models.blocks import get_named_model_blocks


# ----------------------------------------------------------------------------------------------------------------------

META_FILENAME = "meta.json"


# ======================================================================================================================
# Public API: save BEST / save LATEST / resume / load BEST
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def save_best(
    run_dir: str,
    model: nn.Module,
    *,
    epoch: int,
    best_val: float,
    extra_meta: Mapping[str, Any] | None = None,
) -> None:
    """
    Save a strict best snapshot with one ``<block_name>.pt`` file per model-declared block.

    Parameters
    ----------
    run_dir : str
        Run directory for best model checkpoint to be saved.
    model : nn.Module
        Best model checkpoint to be saved.
    epoch : int
        Epoch number.
    best_val : float
        Best value.
    extra_meta : Mapping[str, Any] | None
        Additional metadata as mapping (dict).
        Optional. Default: None.

    Returns
    -------
    None

    """

    best_dir = os.path.join(run_dir, "checkpoints", "best")
    os.makedirs(best_dir, exist_ok=True)

    save_model_blocks(model=model, subdir=best_dir)

    meta = {
        "epoch_best": int(epoch),
        "best_val": float(best_val),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    atomic_json_save(obj=meta, path=os.path.join(best_dir, META_FILENAME))


# ----------------------------------------------------------------------------------------------------------------------
def save_latest(
    run_dir: str,
    model: nn.Module,
    *,
    optimizer: Optimizer | None,
    scheduler: LRScheduler | None,
    scaler: GradScaler | None,
    epoch: int,
    global_step: int,
    best_val_so_far: float,
    bad_epochs: int,
    extra_meta: Mapping[str, Any] | None = None,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """
    Save a strict "resume point" with model blocks, optimizer/scheduler/scaler, RNG state, and metadata.

    Parameters
    ----------
    run_dir : str
        Run directory for latest model checkpoint to be saved.
    model : nn.Module
        Latest model checkpoint to be saved.
    optimizer : Optimizer | None
        Optional optimizer to be saved.
    scheduler : LRScheduler | None
        Optional scheduler to be saved.
    scaler : GradScaler | None,
        Optional scaler to be saved.
    epoch : int
        Epoch number.
    global_step : int
        Global step.
    best_val_so_far : float
        Best achieved loss value.
    bad_epochs : int
        Number of bad epochs.
    extra_meta : Mapping[str, Any] | None
        Optional mapping (dict) with extra metadata.
        Optional. Default: None.

    training_state : Mapping[str, Any] | None
        Configuration signature, history and loader RNG states from the training loop. When provided, all
        optimizer/scheduler/scaler states are required and a versioned integrity manifest is written last.
        Optional. Default: None. Checkpoints without this state cannot be used for strict resume.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If training_state is provided without an optimizer, scheduler or scaler.
    OSError
        If a checkpoint file or its final manifest cannot be written.

    """

    if training_state is not None and any(value is None for value in (optimizer, scheduler, scaler)):
        raise ValueError("Verified resume checkpoints require optimizer, scheduler and scaler state.")
    lat = os.path.join(run_dir, "checkpoints", "latest")
    os.makedirs(name=lat, exist_ok=True)

    save_model_blocks(model=model, subdir=lat)

    if optimizer is not None:
        atomic_save(obj=optimizer.state_dict(), path=os.path.join(lat, "optimizer.pt"))
    if (scheduler is not None) and hasattr(scheduler, "state_dict"):
        atomic_save(obj=scheduler.state_dict(), path=os.path.join(lat, "scheduler.pt"))
    if (scaler is not None) and hasattr(scaler, "state_dict"):
        atomic_save(obj=scaler.state_dict(), path=os.path.join(lat, "scaler.pt"))

    atomic_save(obj=capture_rng_state(), path=os.path.join(lat, "rng.pt"))

    meta = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_so_far": float(best_val_so_far),
        "bad_epochs": int(bad_epochs),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    atomic_json_save(obj=meta, path=os.path.join(lat, META_FILENAME))
    if training_state is not None:
        atomic_save(obj=training_state, path=os.path.join(lat, "training_state.pt"))
        files = [f"{name}.pt" for name in get_named_model_blocks(model)]
        files += ["meta.json", "optimizer.pt", "scheduler.pt", "scaler.pt", "rng.pt", "training_state.pt"]
        # Written last: partial overwrites fail verification instead of mixing epochs.
        write_manifest(lat, files, [f"{name}.pt" for name in get_named_model_blocks(model)] + ["meta.json"])


# ----------------------------------------------------------------------------------------------------------------------
def resume_from_latest(
    run_dir: str,
    model: nn.Module,
    *,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: GradScaler | None = None,
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
    load_model: bool = True,
    signature: str | None = None,
):
    """
    Strict resume of the *same* run.

    The train loop decides whether resume is allowed (train.resume = true).

    Restores:
      - model-declared blocks [if load_model=True]
      - optimizer / scheduler / scaler states
      - RNG state
      - meta.json (epoch, best_val_so_far, etc.)

    Parameters
    ----------

    run_dir : str
        Target run directory.
    model : nn.Module
        Model to be resumed from latest.
    optimizer : Optimizer | None
        Optional optimizer to be loaded.
        Optional. Default: None.
    scheduler : LRScheduler | None
        Optional scheduler to be loaded.
        Optional. Default: None.
    scaler : GradScaler | None,
        Optional scaler to be loaded.
        Optional. Default: None.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as `map_location` parameter of `torch.load()`.
        Optional. Default: "cpu".
    load_model : bool
        If False, skip loading model weights (useful when model was already loaded and only optimizer/scheduler/scaler
        state needs to be restored).
        Optional. Default: True.
    signature : str | None
        Expected training/configuration digest checked before model weights are changed.
        Optional. Default: None.

    Returns
    -------
    tuple[int, float, dict[str, Any]]
        Next global epoch, best validation value and saved metadata. Metadata includes ``_training_state``
        containing the verified history and loader RNG states for the caller to restore.

    Raises
    ------
    FileNotFoundError
        If no 'latest' checkpoint found under `run_dir`/checkpoints.
        If no metadata file 'meta.json' found under `run_dir`/checkpoints/latest.
    ValueError
        If failed to parse resume metadata 'meta.json' found under `run_dir`/checkpoints/latest.
        If loaded metadata from `run_dir`/checkpoints/latest/meta.json is not a dictionary.

    Notes
    -----
    A versioned manifest verifies every latest and best file before loading any model weights.
    Missing/corrupt state or a changed training signature fails explicitly; legacy checkpoints
    require a new run. The returned metadata includes `_training_state` for history and loader RNGs.

    """

    meta, _state = inspect_resume(run_dir, signature=signature, block_names=get_named_model_blocks(model))
    lat = os.path.join(run_dir, "checkpoints", "latest")

    # Now restore the model + training state.
    if load_model:
        load_model_blocks(
            model=model,
            subdir=lat,
            map_location=map_location,
            strict=True,
        )

    for obj, filename in ((optimizer, "optimizer.pt"), (scheduler, "scheduler.pt"), (scaler, "scaler.pt")):
        if obj is not None:
            state = torch_load_full(os.path.join(lat, filename), map_location=map_location)
            obj.load_state_dict(state_dict=state)

    rng_file = os.path.join(lat, "rng.pt")
    restore_rng_state(torch_load_full(path=rng_file, map_location="cpu"))

    start_epoch = int(meta.get("epoch", 0)) + 1
    best_val = float(meta.get("best_val_so_far", float("inf")))

    return start_epoch, best_val, {**meta, "_training_state": _state}


# ----------------------------------------------------------------------------------------------------------------------
def load_best_weights(
    run_dir: str,
    model: nn.Module,
    *,
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
) -> tuple[int, float, dict[Any, Any]]:
    """
    Load the best checkpoint for evaluation.

    Parameters
    ----------
    run_dir : str
        Run directory for best weights to be loaded.
    model : nn.Module
        Target model.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as `map_location` parameter of `torch.load()`.
        Optional. Default: "cpu".

    Returns
    -------
    tuple[int, float, dict[Any, Any]]
        Tuple (epoch_best, best_val, meta). If both checkpoints/best and latest exist, best is preferred.

    """

    ckpt = best_or_latest_dir(run_dir=run_dir)
    if ckpt is None:
        return -1, float("inf"), {}

    ckpt = cast(str, ckpt)  # -> To avoid type mismatch warning
    load_model_blocks(
        model=model,
        subdir=ckpt,
        map_location=map_location,
        strict=True,
    )

    meta_path = os.path.join(ckpt, META_FILENAME)
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except (OSError, JSONDecodeError):
            meta = {}

    epoch_best = int(meta.get("epoch_best", -1))
    best_val = float(meta.get("best_val", float("inf")))

    return epoch_best, best_val, meta
