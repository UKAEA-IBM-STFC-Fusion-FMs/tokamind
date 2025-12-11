from __future__ import annotations
import json
from json import JSONDecodeError
import os
import random
import tempfile
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import logging

logger = logging.getLogger("mmt.WarmStart")

"""
Checkpoint I/O utilities for the Multi-Modal Transformer (MMT).

This module implements three clearly separated workflows:

======================================================================
1) Strict resume of the *same* run
----------------------------------------------------------------------
Triggered when train.resume=true and checkpoints/latest exists.
Restores:
  • model weights (token_encoder, backbone, modality_heads, output_adapters)
  • optimizer state
  • scheduler state
  • scaler state
  • RNG state
  • epoch/global_step/best_val_so_far/bad_epochs from meta.json

======================================================================
2) Warm-start from a previous run (pretrain → finetune)
----------------------------------------------------------------------
Implemented by:
    load_parts_from_run_dir(model, run_dir, load_parts={...})

For each requested block:
  • loads the checkpoint state_dict
  • intersects with current state_dict (key AND shape must match)
  • loads the filtered subset (strict=False)

Blocks:
  - token_encoder
  - backbone
  - modality_heads
  - output_adapters

======================================================================
3) Loading best weights for evaluation
----------------------------------------------------------------------
Loads strictly from checkpoints/best (or fallback to latest).

======================================================================
Directory layout
----------------------------------------------------------------------
run_dir/
  checkpoints/
    latest/
      token_encoder.pt
      backbone.pt
      modality_heads.pt
      output_adapters.pt
      optimizer.pt
      scheduler.pt
      scaler.pt
      rng.pt
      meta.json

    best/
      token_encoder.pt
      backbone.pt
      modality_heads.pt
      output_adapters.pt
      meta.json

======================================================================
Model API required by this module:
----------------------------------------------------------------------
The model must expose:

  - get_token_encoder_state_dict()
  - load_token_encoder_state_dict(state, strict)

  - get_backbone_state_dict()
  - load_backbone_state_dict(state, strict)

  - get_modality_heads_state_dict()
  - load_modality_heads_state_dict(state, strict)

  - get_output_adapters_state_dict()
  - load_output_adapters_state_dict(state, strict)

"""


# ======================================================================
# Low-level atomic save/load
# ======================================================================


def _atomic_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_ext = os.path.splitext(path)[1] or ".pt"

    with tempfile.NamedTemporaryFile(
        suffix=final_ext + ".tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp = f.name
        torch.save(obj, tmp)

    os.replace(tmp, path)


def _atomic_json_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp = f.name
        json.dump(obj, f)
    os.replace(tmp, path)


def _tload(path: str, map_location="cpu") -> Any:
    return torch.load(path, map_location=map_location)


def _torch_load_full(path: str, map_location="cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ======================================================================
# RNG
# ======================================================================


def _capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, and Torch RNG states (incl. CUDA if available)."""
    state: Dict[str, Any] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "time": time.time(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """
    Restore RNG states if present (best-effort).
    Errors during restore are ignored, but only expected ones.
    """
    if not isinstance(state, dict):
        return

    try:
        if "py" in state:
            random.setstate(state["py"])
    except (TypeError, ValueError):
        pass

    try:
        if "np" in state:
            np.random.set_state(state["np"])
    except (TypeError, ValueError):
        pass

    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
    except (RuntimeError, TypeError, ValueError):
        pass

    try:
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except (RuntimeError, TypeError, ValueError):
        pass


# ======================================================================
# Save/load model blocks (NOW FOUR BLOCKS)
# ======================================================================


def _save_model_quadruplet(model: nn.Module, subdir: str) -> None:
    """
    Save the four learnable model blocks:
      • token_encoder
      • backbone
      • modality_heads
      • output_adapters
    """
    _atomic_save(
        model.get_token_encoder_state_dict(),
        os.path.join(subdir, "token_encoder.pt"),
    )
    _atomic_save(
        model.get_backbone_state_dict(),
        os.path.join(subdir, "backbone.pt"),
    )
    _atomic_save(
        model.get_modality_heads_state_dict(),
        os.path.join(subdir, "modality_heads.pt"),
    )
    _atomic_save(
        model.get_output_adapters_state_dict(),
        os.path.join(subdir, "output_adapters.pt"),
    )


def _load_model_quadruplet(
    model: nn.Module,
    subdir: str,
    *,
    map_location="cpu",
    strict_token=True,
    strict_backbone=True,
    strict_heads=True,
    strict_adapters=True,
) -> None:
    """
    Strictly load all four model blocks (used for strict resume/eval).
    """
    fn_token = os.path.join(subdir, "token_encoder.pt")
    fn_backb = os.path.join(subdir, "backbone.pt")
    fn_heads = os.path.join(subdir, "modality_heads.pt")
    fn_adapt = os.path.join(subdir, "output_adapters.pt")

    for fn in (fn_token, fn_backb, fn_heads, fn_adapt):
        if not os.path.exists(fn):
            raise FileNotFoundError(
                f"Checkpoint directory '{subdir}' missing required file: {os.path.basename(fn)}"
            )

    model.load_token_encoder_state_dict(
        _tload(fn_token, map_location), strict=strict_token
    )
    model.load_backbone_state_dict(
        _tload(fn_backb, map_location), strict=strict_backbone
    )
    model.load_modality_heads_state_dict(
        _tload(fn_heads, map_location), strict=strict_heads
    )
    model.load_output_adapters_state_dict(
        _tload(fn_adapt, map_location), strict=strict_adapters
    )


# ======================================================================
# Warm-start overlap loading
# ======================================================================


def _filter_overlap_state(
    loaded: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    out = {}
    for k, v in loaded.items():
        if k not in current:
            continue
        if not (isinstance(v, torch.Tensor) and isinstance(current[k], torch.Tensor)):
            continue
        if v.shape != current[k].shape:
            continue
        out[k] = v
    return out


def _best_or_latest_dir(run_dir: str) -> Optional[str]:
    best = os.path.join(run_dir, "checkpoints", "best")
    latest = os.path.join(run_dir, "checkpoints", "latest")
    if os.path.isdir(best):
        return best
    if os.path.isdir(latest):
        return latest
    return None


def load_parts_from_run_dir(
    model: nn.Module,
    run_dir: str,
    *,
    load_parts: Optional[Dict[str, bool]] = None,
    map_location="cpu",
) -> None:
    """
    Overlap-load selected parts of `model` from a previous run_dir.

    This function is meant for *initialising a new run from pretraining*,
    not for strict resume. Optimizer/scheduler/scaler/RNG are NOT touched.

    It looks for either:
        run_dir/checkpoints/best/
    or
        run_dir/checkpoints/latest/
    (prefers best if it exists).

    Args
    ----
    model:
        MultiModalTransformer instance exposing:
          - get_backbone_state_dict()
          - get_modality_heads_state_dict()
          - get_output_adapters_state_dict()
          - load_backbone_state_dict(sd, strict=False)
          - load_modality_heads_state_dict(sd, strict=False)
          - load_output_adapters_state_dict(sd, strict=False)

    run_dir:
        Path to a *previous* run directory.

    load_parts:
        Dict with optional boolean flags:
            {
              "backbone": True/False,
              "modality_heads": True/False,
              "output_adapters": True/False,
            }
        If None, defaults to loading all three with overlap:
            {"backbone": True, "modality_heads": True, "output_adapters": True}

        For each part with True:
          - loads its state_dict from checkpoint
          - intersects with current state_dict (key+shape overlap)
          - loads that filtered dict with strict=False

    map_location:
        Device to map tensors to when loading.

    Raises
    ------
    FileNotFoundError
        If no 'checkpoints/best' or 'checkpoints/latest' directory is found
        under run_dir, or if required .pt files are missing.
    """
    ckpt = _best_or_latest_dir(run_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoints/best or checkpoints/latest found under '{run_dir}'."
        )

    if load_parts is None:
        load_parts = {
            "token_encoder": True,
            "backbone": True,
            "modality_heads": True,
            "output_adapters": True,
        }

    def _count(sd):
        return sum(v.numel() for v in sd.values() if isinstance(v, torch.Tensor))

    stats = {}

    def _load(blk, get_fn, load_fn, filename):
        if not load_parts.get(blk, False):
            return

        path = os.path.join(ckpt, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing '{filename}' in checkpoint {ckpt}.")

        loaded_sd = _tload(path, map_location)
        current_sd = get_fn()

        overlap_sd = _filter_overlap_state(loaded_sd, current_sd)
        if overlap_sd:
            load_fn(overlap_sd, strict=False)

        stats[blk] = (_count(overlap_sd), _count(current_sd))

    # Load four blocks
    _load(
        "token_encoder",
        model.get_token_encoder_state_dict,
        model.load_token_encoder_state_dict,
        "token_encoder.pt",
    )

    _load(
        "backbone",
        model.get_backbone_state_dict,
        model.load_backbone_state_dict,
        "backbone.pt",
    )

    _load(
        "modality_heads",
        model.get_modality_heads_state_dict,
        model.load_modality_heads_state_dict,
        "modality_heads.pt",
    )

    _load(
        "output_adapters",
        model.get_output_adapters_state_dict,
        model.load_output_adapters_state_dict,
        "output_adapters.pt",
    )

    # Summary
    summary = []
    for block in ("token_encoder", "backbone", "modality_heads", "output_adapters"):
        if load_parts.get(block, False):
            if block in stats:
                L, T = stats[block]
                summary.append(f"{block}: {L}/{T} params matched")
            else:
                summary.append(f"{block}: loaded (no overlapping params found)")
        else:
            summary.append(f"{block}: skipped (load_parts=False)")

    logger.info(f"Loaded from {ckpt}: " + " | ".join(summary))


# ======================================================================
# Public API: save BEST / save LATEST / resume
# ======================================================================


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

    _save_model_quadruplet(model, best_dir)

    meta = {
        "epoch_best": int(epoch),
        "best_val": float(best_val),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    _atomic_json_save(meta, os.path.join(best_dir, "meta.json"))


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

    _save_model_quadruplet(model, lat)

    if optimizer is not None:
        _atomic_save(optimizer.state_dict(), os.path.join(lat, "optimizer.pt"))
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        _atomic_save(scheduler.state_dict(), os.path.join(lat, "scheduler.pt"))
    if scaler is not None and hasattr(scaler, "state_dict"):
        _atomic_save(scaler.state_dict(), os.path.join(lat, "scaler.pt"))

    _atomic_save(_capture_rng_state(), os.path.join(lat, "rng.pt"))

    meta = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_so_far": float(best_val_so_far),
        "bad_epochs": int(bad_epochs),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    _atomic_json_save(meta, os.path.join(lat, "meta.json"))


def resume_from_latest(
    run_dir: str,
    model: nn.Module,
    *,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
):
    """
    Strict resume of the *same* run.

    The train loop decides whether resume is allowed
    (train.resume = true).

    Restores:
      - model triplet
      - optimizer / scheduler / scaler states
      - RNG state
      - meta.json (epoch, best_val_so_far, etc.)
    """

    lat = os.path.join(run_dir, "checkpoints", "latest")
    if not os.path.isdir(lat):
        raise FileNotFoundError(f"No 'latest' checkpoint found: {lat}")

    _load_model_quadruplet(
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
            state = _torch_load_full(p, map_location)
            obj.load_state_dict(state)

    _maybe_load(optimizer, "optimizer.pt")
    _maybe_load(scheduler, "scheduler.pt")
    _maybe_load(scaler, "scaler.pt")

    rng_file = os.path.join(lat, "rng.pt")
    if os.path.exists(rng_file):
        _restore_rng_state(_torch_load_full(rng_file, map_location))

    meta_path = os.path.join(lat, "meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except (OSError, JSONDecodeError):
            meta = {}

    start_epoch = int(meta.get("epoch", 0)) + 1
    best_val = float(meta.get("best_val_so_far", float("inf")))

    return start_epoch, best_val, meta


def load_best_weights(run_dir: str, model: nn.Module, *, map_location="cpu"):
    """
    Load the best checkpoint for evaluation.

    Returns (epoch_best, best_val, meta).
    If both checkpoints/best and latest exist, best is preferred.
    """

    ckpt = _best_or_latest_dir(run_dir)
    if ckpt is None:
        return -1, float("inf"), {}

    _load_model_quadruplet(
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
