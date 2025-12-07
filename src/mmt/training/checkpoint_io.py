"""
Checkpoint I/O utilities for the Multi-Modal Transformer.

This module supports three distinct workflows:

1) Strict resume of an existing run
   --------------------------------
   Used when `training.resume: true` and a `checkpoints/latest/` directory
   exists under the current run_dir.

   - Call `resume_from_latest(run_dir, model, optimizer, scheduler, scaler, ...)`
   - Restores:
       * model weights (backbone, modality_heads, output_adapters)
       * optimizer state
       * scheduler state
       * GradScaler state (if provided)
       * RNG state (Python, NumPy, Torch CPU/CUDA)
       * epoch/global_step/best_val/bad_epochs from meta.json
   - Intended for continuing the *same* run after an interruption.

2) Warm-start from a previous run (pretrain → finetune, transfer)
   ----------------------------------------------------------------
   Controlled by `model_init.run_dir` and `model_init.load_parts` in the
   experiment config, and implemented by `load_parts_from_run_dir`.

   Typical config:

       model_init:
         run_dir: "runs/pretrain_task_2-1/some_run"
         load_parts:
           backbone: true
           modality_heads: true
           output_adapters: false

   Behaviour:

   - Finds either `checkpoints/best/` or `checkpoints/latest/` in the given
     run_dir (prefers best if present).
   - For each block with load_parts[block_name] == True:
       * loads the corresponding state_dict from the checkpoint
       * computes the *overlap* with the current model (only keys that exist
         in the current block AND have the same tensor shape)
       * loads this filtered state_dict into the current model with strict=False
     All other parameters keep their current (fresh) initialisation.

   This *overlap loading* lets you:
   - add or remove outputs (new adapters start from scratch)
   - add or remove modalities (new heads start from scratch)
   - change some internal shapes
   while still reusing all matching parameters from pretraining.

   NOTE: optimizer/scheduler/scaler/RNG are NOT touched here;
   warm-start always begins a *new* run with fresh optimizer state.

3) Loading best weights for evaluation
   ------------------------------------
   Used by evaluation scripts to obtain the best snapshot ("best" if present,
   otherwise "latest"):

       epoch_best, best_val, meta = load_best_weights(run_dir, model)

   - Strictly loads backbone, modality_heads, and output_adapters.
   - Expects the architecture to match; will raise if shapes/keys differ.

File layout per run_dir
-----------------------
We follow a simple, opinionated structure:

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

The model is expected to expose:

    - get_backbone_state_dict()
    - get_modality_heads_state_dict()
    - get_output_adapters_state_dict()

    - load_backbone_state_dict(state_dict, strict: bool = True)
    - load_modality_heads_state_dict(state_dict, strict: bool = True)
    - load_output_adapters_state_dict(state_dict, strict: bool = True)
"""

from __future__ import annotations

import json
import os
import random
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ======================================================================
# Low-level helpers
# ======================================================================


def _atomic_save(obj: Any, path: str) -> None:
    """
    Safely save a PyTorch object (or any pickle-able object) to disk using
    a temp file then rename, to avoid partial writes on crashes.
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
    """Safely save a small JSON dict atomically."""
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
    """Thin wrapper around torch.load with a default map_location."""
    return torch.load(path, map_location=map_location)


def _torch_load_full(path: str, map_location: str | torch.device = "cpu") -> Any:
    """
    Load pickled blobs (optimizer/scheduler/scaler/RNG) in a way that works
    with PyTorch 2.6+ (weights_only=False) but degrades gracefully on older
    versions where that argument doesn't exist.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # type: ignore[arg-type]
    except TypeError:
        # Older PyTorch without weights_only
        return torch.load(path, map_location=map_location)


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
    """Restore RNG states if present (best-effort, silently skips on errors)."""
    if not isinstance(state, dict):
        return
    try:
        if "py" in state:
            random.setstate(state["py"])
        if "np" in state:
            np.random.set_state(state["np"])
        if "torch_cpu" in state:
            torch.set_rng_state(state["torch_cpu"])
        if "torch_cuda" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])
    except Exception:
        # best-effort: we don't want RNG restore to kill resume
        pass


# ======================================================================
# DRY save/load for the standard model triplet
# ======================================================================


def _save_model_triplet(model: nn.Module, subdir: str) -> None:
    """
    Save backbone / modality_heads / output_adapters state_dicts to subdir.

    The model is expected to implement:
        - get_backbone_state_dict()
        - get_modality_heads_state_dict()
        - get_output_adapters_state_dict()
    """
    _atomic_save(model.get_backbone_state_dict(), os.path.join(subdir, "backbone.pt"))
    _atomic_save(
        model.get_modality_heads_state_dict(),
        os.path.join(subdir, "modality_heads.pt"),
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
    Strictly load backbone / modality_heads / output_adapters from subdir.

    Used by:
      - resume_from_latest (model part)
      - load_best_weights (eval)
    """
    b_path = os.path.join(subdir, "backbone.pt")
    h_path = os.path.join(subdir, "modality_heads.pt")
    a_path = os.path.join(subdir, "output_adapters.pt")

    if not (
        os.path.exists(b_path) and os.path.exists(h_path) and os.path.exists(a_path)
    ):
        raise FileNotFoundError(
            f"Checkpoint directory {subdir!r} is missing one of the required "
            f"files: backbone.pt / modality_heads.pt / output_adapters.pt."
        )

    b_sd = _tload(b_path, map_location=map_location)
    h_sd = _tload(h_path, map_location=map_location)
    a_sd = _tload(a_path, map_location=map_location)

    _ = model.load_backbone_state_dict(b_sd, strict=strict_backbone)
    _ = model.load_modality_heads_state_dict(h_sd, strict=strict_heads)
    _ = model.load_output_adapters_state_dict(a_sd, strict=strict_adapters)


def _filter_overlap_state(
    loaded: Dict[str, Any],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Keep only parameters whose keys and shapes match between loaded and current.

    This is the core of "overlap loading" used in warm-start:
    - It lets us reuse all compatible parameters from a previous run
      while safely ignoring new or changed parameters in the current model.
    """
    out: Dict[str, Any] = {}
    for k, v in loaded.items():
        if k not in current:
            continue
        if not isinstance(v, torch.Tensor) or not isinstance(current[k], torch.Tensor):
            continue
        if v.shape != current[k].shape:
            continue
        out[k] = v
    return out


def _best_or_latest_dir(run_dir: str) -> Optional[str]:
    """
    Helper: return path to checkpoints/best or checkpoints/latest.

    Prefers 'best' if present, otherwise 'latest'.
    Returns None if neither exists.
    """
    p_best = os.path.join(run_dir, "checkpoints", "best")
    p_lat = os.path.join(run_dir, "checkpoints", "latest")

    if os.path.isdir(p_best):
        return p_best
    if os.path.isdir(p_lat):
        return p_lat
    return None


# ======================================================================
# Public API: warm-start (pretrain → finetune)
# ======================================================================


def load_parts_from_run_dir(
    model: nn.Module,
    run_dir: str,
    *,
    load_parts: Optional[Dict[str, bool]] = None,
    map_location: str | torch.device = "cpu",
    verbose: bool = True,
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

    verbose:
        If True, prints a short summary of how many params were reused.

    Raises
    ------
    FileNotFoundError
        If no 'checkpoints/best' or 'checkpoints/latest' directory is found
        under run_dir, or if required .pt files are missing.
    """
    ckpt_dir = _best_or_latest_dir(run_dir)
    if ckpt_dir is None:
        raise FileNotFoundError(
            f"No checkpoints found under {run_dir!r}. "
            f"Expected 'checkpoints/best' or 'checkpoints/latest'."
        )

    if load_parts is None:
        load_parts = {
            "backbone": True,
            "modality_heads": True,
            "output_adapters": True,
        }

    stats: Dict[str, Tuple[int, int]] = {}

    def _overlap_load_one_block(
        block_name: str,
        get_sd_fn,
        load_sd_fn,
        filename: str,
    ) -> None:
        if not load_parts.get(block_name, False):
            return

        path = os.path.join(ckpt_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Expected {filename!r} for block {block_name!r} in {ckpt_dir}, "
                f"but file does not exist."
            )

        loaded_sd = _tload(path, map_location=map_location)
        current_sd = get_sd_fn()

        overlap_sd = _filter_overlap_state(loaded_sd, current_sd)
        load_sd_fn(overlap_sd, strict=False)

        stats[block_name] = (len(overlap_sd), len(current_sd))

    _overlap_load_one_block(
        "backbone",
        model.get_backbone_state_dict,
        model.load_backbone_state_dict,
        "backbone.pt",
    )
    _overlap_load_one_block(
        "modality_heads",
        model.get_modality_heads_state_dict,
        model.load_modality_heads_state_dict,
        "modality_heads.pt",
    )
    _overlap_load_one_block(
        "output_adapters",
        model.get_output_adapters_state_dict,
        model.load_output_adapters_state_dict,
        "output_adapters.pt",
    )

    if verbose and stats:
        msg_parts = []
        for block, (n_loaded, n_total) in stats.items():
            msg_parts.append(f"{block}: {n_loaded}/{n_total} params reused")
        print(f"[WarmStart] Loaded from {ckpt_dir}: " + ", ".join(msg_parts))


# ======================================================================
# Public API: best / latest / resume / eval
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
    Save the best snapshot for the current run:

        run_dir/checkpoints/best/
            backbone.pt
            modality_heads.pt
            output_adapters.pt
            meta.json
    """
    best_dir = os.path.join(run_dir, "checkpoints", "best")
    os.makedirs(best_dir, exist_ok=True)

    _save_model_triplet(model, best_dir)

    meta: Dict[str, Any] = {
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
    Save a 'latest' resume point for the current run:

        run_dir/checkpoints/latest/
            backbone.pt
            modality_heads.pt
            output_adapters.pt
            optimizer.pt           (if optimizer is not None)
            scheduler.pt           (if scheduler is not None)
            scaler.pt              (if scaler is not None)
            rng.pt
            meta.json
    """
    lat_dir = os.path.join(run_dir, "checkpoints", "latest")
    os.makedirs(lat_dir, exist_ok=True)

    # Model weights
    _save_model_triplet(model, lat_dir)

    # Optimizer / scheduler / scaler states
    if optimizer is not None:
        _atomic_save(optimizer.state_dict(), os.path.join(lat_dir, "optimizer.pt"))

    if scheduler is not None and hasattr(scheduler, "state_dict"):
        _atomic_save(scheduler.state_dict(), os.path.join(lat_dir, "scheduler.pt"))

    if scaler is not None and hasattr(scaler, "state_dict"):
        _atomic_save(scaler.state_dict(), os.path.join(lat_dir, "scaler.pt"))

    # RNG state
    _atomic_save(_capture_rng_state(), os.path.join(lat_dir, "rng.pt"))

    # Meta
    meta: Dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_so_far": float(best_val_so_far),
        "bad_epochs": int(bad_epochs),
        "saved_at": time.time(),
    }
    if extra_meta:
        meta.update(extra_meta)

    _atomic_json_save(meta, os.path.join(lat_dir, "meta.json"))


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
    Resume training from checkpoints/latest:

      - strictly loads model weights (backbone, modality_heads, output_adapters)
      - loads optimizer/scheduler/scaler (if provided)
      - restores RNG
      - returns (start_epoch, best_val_so_far, meta_dict)

    Intended for continuing the *same* run after interruption.
    """
    lat_dir = os.path.join(run_dir, "checkpoints", "latest")
    if not os.path.isdir(lat_dir):
        raise FileNotFoundError(f"No 'latest' checkpoint directory at {lat_dir}")

    # Model weights: strict on resume (if arch changed, fail fast)
    _load_model_triplet(
        model,
        lat_dir,
        map_location=map_location,
        strict_backbone=True,
        strict_heads=True,
        strict_adapters=True,
    )

    # Optimizer / Scheduler / Scaler
    def _maybe_load_state(obj: Any, filename: str) -> None:
        if obj is None:
            return
        path = os.path.join(lat_dir, filename)
        if os.path.exists(path) and hasattr(obj, "load_state_dict"):
            state = _torch_load_full(path, map_location=map_location)
            obj.load_state_dict(state)

    _maybe_load_state(optimizer, "optimizer.pt")
    _maybe_load_state(scheduler, "scheduler.pt")
    _maybe_load_state(scaler, "scaler.pt")

    # RNG
    rng_path = os.path.join(lat_dir, "rng.pt")
    if os.path.exists(rng_path):
        rng_state = _torch_load_full(rng_path, map_location=map_location)
        _restore_rng_state(rng_state)

    # Meta
    meta_path = os.path.join(lat_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    start_epoch = int(meta.get("epoch", 0)) + 1
    best_val_so_far = float(meta.get("best_val_so_far", float("inf")))

    return start_epoch, best_val_so_far, meta


def load_best_weights(
    run_dir: str,
    model: nn.Module,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[int, float, Dict[str, Any]]:
    """
    Load the best snapshot (or latest if best is missing) for evaluation.

    Returns:
        (epoch_best, best_val, meta_dict)
    or (-1, inf, {}) if no checkpoint is found.

    Uses strict loading: assumes the architecture matches the checkpoint.
    """
    best_dir = _best_or_latest_dir(run_dir)
    if best_dir is None:
        return -1, float("inf"), {}

    _load_model_triplet(
        model,
        best_dir,
        map_location=map_location,
        strict_backbone=True,
        strict_heads=True,
        strict_adapters=True,
    )

    meta_path = os.path.join(best_dir, "meta.json")
    meta: Dict[str, Any] = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    epoch_best = int(meta.get("epoch_best", -1))
    best_val = float(meta.get("best_val", float("inf")))

    return epoch_best, best_val, meta
