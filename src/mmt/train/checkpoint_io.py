"""
Checkpoint I/O utilities for the Multi-Modal Transformer (MMT).

This module implements three clearly separated workflows that correspond
to three distinct phases of the MMT lifecycle:

======================================================================
1) Strict resume of the *same* run
----------------------------------------------------------------------
Triggered when `train.resume: true` and a `checkpoints/latest/`
directory exists under the current run_dir.

    resume_from_latest(run_dir, model, optimizer, scheduler, scaler)

Restores:
  • model weights (backbone, modality_heads, output_adapters)
  • optimizer state
  • scheduler state
  • GradScaler state (if provided)
  • RNG state (Python, NumPy, Torch CPU/CUDA)
  • epoch, global_step, best_val_so_far, bad_epochs (from meta.json)

This is intended only for *continuing the same run after interruption*.
Architecture must match exactly (strict load).

======================================================================
2) Warm-start from a previous run (pretrain → finetune, transfer)
----------------------------------------------------------------------
Controlled by `model_init.run_dir` and `model_init.load_parts`
in the experiment config, and implemented via:

    load_parts_from_run_dir(model, run_dir, load_parts={...})

Behaviour:
  • Finds either `checkpoints/best/` or `checkpoints/latest/` in the
    specified run_dir (prefers best).
  • For each block where load_parts[block] == True:
        - loads the corresponding state_dict
        - intersects with the current model (key AND tensor shape must match)
        - loads the filtered state_dict with strict=False

This "overlap loading" allows:
  • adding or removing outputs (adapters)
  • adding or removing modalities (heads)
  • changing internal shapes

Warm-start does *not* restore optimizer/scheduler/scaler/RNG or counters.
It always begins a *new* run with fresh optimizer state.

======================================================================
3) Loading best weights for evaluation
----------------------------------------------------------------------
Used by evaluation scripts:

    epoch_best, best_val, meta = load_best_weights(run_dir, model)

Loads strictly from:
    run_dir/checkpoints/best/
or if missing:
    run_dir/checkpoints/latest/

Only loads model weights; used for inference or evaluation pipelines.

======================================================================
Checkpoint directory layout
----------------------------------------------------------------------
run_dir/
  checkpoints/
    latest/
      backbone.pt
      modality_heads.pt
      output_adapters.pt
      optimizer.pt        (optional)
      scheduler.pt        (optional)
      scaler.pt           (optional)
      rng.pt
      meta.json

    best/
      backbone.pt
      modality_heads.pt
      output_adapters.pt
      meta.json

======================================================================
Model API requirements
----------------------------------------------------------------------
The model must provide:

  - get_backbone_state_dict()
  - get_modality_heads_state_dict()
  - get_output_adapters_state_dict()

  - load_backbone_state_dict(sd, strict=True/False)
  - load_modality_heads_state_dict(sd, strict=True/False)
  - load_output_adapters_state_dict(sd, strict=True/False)

These allow strict resume, warm-start, and evaluation to work safely and
independently of optimizer/scheduler structures.
"""

from __future__ import annotations
import json
from json import JSONDecodeError
import os
import random
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import logging

logger = logging.getLogger("mmt.Train")

# ======================================================================
# Low-level atomic save / load helpers
# ======================================================================


def _atomic_save(obj: Any, path: str) -> None:
    """
    Atomically save a PyTorch object:
      - write to temporary file
      - rename over the final file
    Prevents corruption if a crash occurs during write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_ext = os.path.splitext(path)[1] or ".pt"

    with tempfile.NamedTemporaryFile(
        suffix=final_ext + ".tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp_path = f.name
        torch.save(obj, tmp_path)

    os.replace(tmp_path, path)


def _atomic_json_save(obj: Dict[str, Any], path: str) -> None:
    """Atomically save a JSON dict."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp_path = f.name
        json.dump(obj, f)
    os.replace(tmp_path, path)


def _tload(path: str, map_location: str | torch.device = "cpu") -> Any:
    """Simple torch.load wrapper with default map_location."""
    return torch.load(path, map_location=map_location)


def _torch_load_full(path: str, map_location: str | torch.device = "cpu") -> Any:
    """
    Load optimizer/scheduler/scaler/RNG blobs in a PyTorch-version-safe way.
    PyTorch ≥ 2.6 supports weights_only=False.
    """
    try:
        return torch.load(
            path, map_location=map_location, weights_only=False
        )  # PyTorch 2.6+
    except TypeError:
        return torch.load(path, map_location=map_location)


# ======================================================================
# RNG utilities
# ======================================================================


def _capture_rng_state() -> Dict[str, Any]:
    """Capture Python, NumPy, Torch CPU/CUDA RNG states."""
    state = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "time": time.time(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    """Restore RNG states (best effort)."""
    if not isinstance(state, dict):
        return

    try:
        if "py" in state:
            random.setstate(state["py"])
    except Exception:
        pass

    try:
        if "np" in state:
            np.random.set_state(state["np"])
    except Exception:
        pass

    try:
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
    except Exception:
        pass

    try:
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        pass


# ======================================================================
# Save/load the "model triplet"
# ======================================================================


def _save_model_triplet(model: nn.Module, subdir: str) -> None:
    """
    Save backbone, modality_heads, and output_adapters state_dicts.

    These are always saved separately so the train loop can:
      - warm-start only selected blocks,
      - resume strictly,
      - evaluate safely even if optimizer is absent.
    """
    _atomic_save(model.get_backbone_state_dict(), os.path.join(subdir, "backbone.pt"))
    _atomic_save(
        model.get_modality_heads_state_dict(), os.path.join(subdir, "modality_heads.pt")
    )
    _atomic_save(
        model.get_output_adapters_state_dict(),
        os.path.join(subdir, "output_adapters.pt"),
    )


def _load_model_triplet(
    model: nn.Module,
    subdir: str,
    *,
    map_location: str | torch.device = "cpu",
    strict_backbone: bool = True,
    strict_heads: bool = True,
    strict_adapters: bool = True,
) -> None:
    """
    Strictly load the model triplet.
    Used for strict resume and strict evaluation ("best" checkpoint).
    """
    b_path = os.path.join(subdir, "backbone.pt")
    h_path = os.path.join(subdir, "modality_heads.pt")
    a_path = os.path.join(subdir, "output_adapters.pt")

    if not (
        os.path.exists(b_path) and os.path.exists(h_path) and os.path.exists(a_path)
    ):
        raise FileNotFoundError(
            f"Checkpoint directory '{subdir}' missing required model files "
            "(backbone.pt, modality_heads.pt, output_adapters.pt)."
        )

    b_sd = _tload(b_path, map_location)
    h_sd = _tload(h_path, map_location)
    a_sd = _tload(a_path, map_location)

    model.load_backbone_state_dict(b_sd, strict=strict_backbone)
    model.load_modality_heads_state_dict(h_sd, strict=strict_heads)
    model.load_output_adapters_state_dict(a_sd, strict=strict_adapters)


# ======================================================================
# Overlap-loading for warm-start (pretrain → finetune)
# ======================================================================


def _filter_overlap_state(
    loaded: Dict[str, Any], current: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Keep only keys whose name AND shape match.
    Makes warm-start safe when output adapters / modalities differ.
    """
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
    """
    Prefer checkpoints/best over checkpoints/latest.
    Used for evaluation and warm-start.
    """
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
    map_location: str | torch.device = "cpu",
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
    # Locate checkpoint directory (prefer 'best', else 'latest')
    ckpt = _best_or_latest_dir(run_dir)
    if ckpt is None:
        raise FileNotFoundError(
            f"No checkpoints/best or checkpoints/latest found under '{run_dir}'."
        )

    # Default: load all three parts unless explicitly overridden
    if load_parts is None:
        load_parts = {
            "backbone": True,
            "modality_heads": True,
            "output_adapters": True,
        }

    # Helper: count total number of tensor parameters in a state_dict
    def _count_params(sd: Dict[str, Any]) -> int:
        n = 0
        for v in sd.values():
            if isinstance(v, torch.Tensor):
                n += v.numel()
        return n

    # Collect load statistics: {block_name: (loaded_params, total_params)}
    stats: Dict[str, tuple[int, int]] = {}

    # Load a specific block (backbone / modality_heads / output_adapters)
    def _load(block_name: str, get_fn, load_fn, filename: str) -> None:
        # Skip if user disabled loading for this block
        if not load_parts.get(block_name, False):
            return

        path = os.path.join(ckpt, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing '{filename}' in checkpoint {ckpt}.")

        loaded_sd = _tload(path, map_location)
        current_sd = get_fn()

        # Keep only keys present in both checkpoint and current model
        overlap_sd = _filter_overlap_state(loaded_sd, current_sd)
        if overlap_sd:
            load_fn(overlap_sd, strict=False)

        # Store how many parameters were reused vs total in the current block
        stats[block_name] = (_count_params(overlap_sd), _count_params(current_sd))

    # Try loading each block independently
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

    # Summary log: show loaded vs skipped blocks for user clarity
    all_parts = ["backbone", "modality_heads", "output_adapters"]
    summary_lines = []

    for part in all_parts:
        if load_parts.get(part, False):
            if part in stats:
                n_loaded, n_total = stats[part]
                summary_lines.append(f"{part}: {n_loaded}/{n_total} params matched")
            else:
                summary_lines.append(f"{part}: loaded (no overlapping params found)")
        else:
            summary_lines.append(f"{part}: skipped (load_parts=False)")

    summary_text = " | ".join(summary_lines)
    logger.info(f"[WarmStart] Loaded from {ckpt}: {summary_text}")


# ======================================================================
# Public API: strict save/load for best / latest checkpoints
# ======================================================================


def save_best(
    run_dir: str,
    model: nn.Module,
    *,
    epoch: int,
    best_val: float,
    extra_meta: Optional[Dict[str, Any]] = None,
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

    _save_model_triplet(model, best_dir)

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
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_val_so_far: float,
    bad_epochs: int,
    extra_meta: Optional[Dict[str, Any]] = None,
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

    _save_model_triplet(model, lat)

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
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, float, Dict[str, Any]]:
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
        raise FileNotFoundError(f"No 'latest' checkpoint found at: {lat}")

    # Strict model restore
    _load_model_triplet(
        model,
        lat,
        map_location=map_location,
        strict_backbone=True,
        strict_heads=True,
        strict_adapters=True,
    )

    # Restore optimizer/scheduler
    def _maybe_load(obj, filename):
        if obj is None:
            return
        path = os.path.join(lat, filename)
        if os.path.exists(path):
            state = _torch_load_full(path, map_location)
            obj.load_state_dict(state)

    _maybe_load(optimizer, "optimizer.pt")
    _maybe_load(scheduler, "scheduler.pt")
    _maybe_load(scaler, "scaler.pt")

    # RNG
    rng_path = os.path.join(lat, "rng.pt")
    if os.path.exists(rng_path):
        _restore_rng_state(_torch_load_full(rng_path, map_location))

    # Meta
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


def load_best_weights(
    run_dir: str,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Load the best checkpoint for evaluation.

    Returns (epoch_best, best_val, meta).
    If both checkpoints/best and latest exist, best is preferred.
    """
    ckpt = _best_or_latest_dir(run_dir)
    if ckpt is None:
        return -1, float("inf"), {}

    _load_model_triplet(
        model,
        ckpt,
        map_location=map_location,
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
