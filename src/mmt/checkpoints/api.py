"""
Checkpoint management API for the Multi-Modal Transformer (MMT).

This module provides the public entry points for saving, loading, resuming,
and warm-starting model checkpoints. It orchestrates lower-level utilities
from sibling modules (io, rng, blocks, warmstart) and defines the supported
checkpointing workflows.

Supported workflows
-------------------
1) Strict resume of the same run
   - Restores model weights, optimizer, scheduler, scaler, RNG state,
     and training metadata from checkpoints/latest.
   - Used when resuming interrupted training.

2) Warm-start / partial loading
   - Loads only compatible subsets of model parameters from a previous run
     (key + shape match), leaving others initialized.
   - Intended for pretraining → finetuning or cross-task initialization.
   - Implemented via load_parts_from_run_dir(...).

3) Load best weights for evaluation
   - Loads model weights strictly from checkpoints/best
     (or latest as fallback), without optimizer or RNG state.

Checkpoint layout
-----------------
run_dir/
  checkpoints/
    latest/   # full training resume state
    best/     # best-performing model weights only

Model requirements
------------------
Models used with this API must expose explicit get_*_state_dict() and
load_*_state_dict(state, strict) methods for each learnable block
(token encoder, backbone, modality heads, output adapters).
"""

from __future__ import annotations
import json
from json import JSONDecodeError
import os
import time

import torch.nn as nn

from .blocks import save_model_quadruplet, load_model_quadruplet
from .io import (
    atomic_save,
    atomic_json_save,
    torch_load_full,
    best_or_latest_dir,
)
from .rng import capture_rng_state, restore_rng_state

# ------------------------------------------------------------------
# Public API: save BEST / save LATEST / resume
# ------------------------------------------------------------------


def save_best(
    run_dir: str, model: nn.Module, *, epoch: int, best_val: float, extra_meta=None
) -> None:
    """
    Save a strict best snapshot:
      run_dir/checkpoints/best/
        backbone.pt
        modality_heads.pt
        output_adapters.pt
        meta.json
    """

    best_dir = os.path.join(run_dir, "checkpoints", "best")
    os.makedirs(best_dir, exist_ok=True)

    save_model_quadruplet(model, best_dir)

    meta = {
        "epoch_best": int(epoch),
        "best_val": float(best_val),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    atomic_json_save(meta, os.path.join(best_dir, "meta.json"))


def save_latest(
    run_dir: str,
    model: nn.Module,
    *,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    best_val_so_far: float,
    bad_epochs: int,
    extra_meta=None,
) -> None:
    """
    Save a strict "resume point":
      run_dir/checkpoints/latest/
        backbone.pt
        modality_heads.pt
        output_adapters.pt
        optimizer.pt
        scheduler.pt
        scaler.pt
        rng.pt
        meta.json
    """

    lat = os.path.join(run_dir, "checkpoints", "latest")
    os.makedirs(lat, exist_ok=True)

    save_model_quadruplet(model, lat)

    if optimizer is not None:
        atomic_save(optimizer.state_dict(), os.path.join(lat, "optimizer.pt"))
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        atomic_save(scheduler.state_dict(), os.path.join(lat, "scheduler.pt"))
    if scaler is not None and hasattr(scaler, "state_dict"):
        atomic_save(scaler.state_dict(), os.path.join(lat, "scaler.pt"))

    atomic_save(capture_rng_state(), os.path.join(lat, "rng.pt"))

    meta = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_so_far": float(best_val_so_far),
        "bad_epochs": int(bad_epochs),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    atomic_json_save(meta, os.path.join(lat, "meta.json"))


def resume_from_latest(
    run_dir: str,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
    load_model=True,
):
    """
    Strict resume of the *same* run.

    The train loop decides whether resume is allowed
    (train.resume = true).

    Restores:
      - model quadruplet (token encoder, backbone, heads, adapters) [if load_model=True]
      - optimizer / scheduler / scaler states
      - RNG state
      - meta.json (epoch, best_val_so_far, etc.)

    Parameters
    ----------
    load_model : bool, default=True
        If False, skip loading model weights (useful when model was already loaded
        and only optimizer/scheduler/scaler state needs to be restored).

    Notes
    -----
    This function expects run_dir/checkpoints/latest/meta.json to be valid JSON.
    If it is missing or corrupted, resume fails explicitly.
    """

    lat = os.path.join(run_dir, "checkpoints", "latest")
    if not os.path.isdir(lat):
        raise FileNotFoundError(f"No 'latest' checkpoint found: {lat}")

    # Load metadata FIRST so that a corrupted meta.json cannot leave the model
    # partially resumed while the caller falls back to "start from scratch".
    meta_path = os.path.join(lat, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing resume metadata file: {meta_path}")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(
            f"Failed to parse resume metadata (expected JSON): {meta_path}"
        ) from e

    if not isinstance(meta, dict):
        raise ValueError(f"Invalid resume metadata (expected a dict): {meta_path}")

    # Now restore the model + training state.
    if load_model:
        load_model_quadruplet(
            model,
            lat,
            map_location=map_location,
            strict_token=True,
            strict_backbone=True,
            strict_heads=True,
            strict_adapters=True,
        )

    def _maybe_load(obj, filename):
        if obj is None:
            return
        p = os.path.join(lat, filename)
        if os.path.exists(p):
            state = torch_load_full(p, map_location)
            obj.load_state_dict(state)

    _maybe_load(optimizer, "optimizer.pt")
    _maybe_load(scheduler, "scheduler.pt")
    _maybe_load(scaler, "scaler.pt")

    rng_file = os.path.join(lat, "rng.pt")
    if os.path.exists(rng_file):
        restore_rng_state(torch_load_full(rng_file, map_location))

    start_epoch = int(meta.get("epoch", 0)) + 1
    best_val = float(meta.get("best_val_so_far", float("inf")))

    return start_epoch, best_val, meta


def load_best_weights(run_dir: str, model: nn.Module, *, map_location="cpu"):
    """
    Load the best checkpoint for evaluation.

    Returns (epoch_best, best_val, meta).
    If both checkpoints/best and latest exist, best is preferred.
    """

    ckpt = best_or_latest_dir(run_dir)
    if ckpt is None:
        return -1, float("inf"), {}

    load_model_quadruplet(
        model,
        ckpt,
        map_location=map_location,
        strict_token=True,
        strict_backbone=True,
        strict_heads=True,
        strict_adapters=True,
    )

    meta_path = os.path.join(ckpt, "meta.json")
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
