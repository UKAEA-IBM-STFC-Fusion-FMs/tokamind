"""
rng.py — Global and DataLoader random-state capture, validation and restoration.

This module owns the random state needed to continue training at a completed-epoch boundary:

    • Python, NumPy and Torch CPU generators.
    • Torch CUDA generators, when present in the saved state.
    • Independent DataLoader and sampler generators, including wrapped CGPO loaders.

Validation uses temporary generators and does not change the global random streams. Restoration is strict: missing
or incompatible state raises an error instead of silently continuing with newly initialized generators. CUDA state
requires the original device topology. Streaming dataset state and persistent-worker RNG state are not checkpointed;
loader capture records that limitation and resume verification rejects such checkpoints.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any
import numpy as np

import torch
from torch.utils.data import IterableDataset


# ======================================================================================================================
# Random number generation (RNG)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def capture_rng_state() -> dict[str, Any]:
    """
    Capture Python, NumPy, and Torch RNG states (incl. CUDA if available).

    Returns
    -------
    dict[str, Any]
        Dictionary with captured random states.

    """

    state: dict[str, Any] = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "time": time.time(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()

    return state


# ----------------------------------------------------------------------------------------------------------------------
def validate_rng_state(state: Mapping[str, Any]) -> None:
    """
    Validate saved global RNG state without changing the current random streams.

    Parameters
    ----------
    state : Mapping[str, Any]
        State produced by ``capture_rng_state()``, with required Python, NumPy and Torch CPU entries and optional
        per-device CUDA entries.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a required generator state is missing.
    TypeError, ValueError, RuntimeError
        If a generator state is malformed or the saved CUDA device count differs from the current topology.
    """

    random.Random().setstate(state["py"])
    np.random.RandomState().set_state(state["np"])
    torch.Generator().set_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        if not isinstance(state["torch_cuda"], list) or not state["torch_cuda"]:
            raise ValueError("Invalid CUDA RNG state.")
        if not torch.cuda.is_available() or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the original CUDA device topology.")
        for value in state["torch_cuda"]:
            if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.numel() == 0:
                raise ValueError("Invalid CUDA RNG state.")


# ----------------------------------------------------------------------------------------------------------------------
def restore_rng_state(state: Mapping[str, Any]) -> None:
    """
    Restore all saved global RNG streams after validating the complete state.

    Parameters
    ----------
    state : Mapping[str, Any]
        Global RNG state produced by ``capture_rng_state()``.

    Returns
    -------
    None

    Raises
    ------
    KeyError, TypeError, ValueError, RuntimeError
        If a required state is missing, invalid or incompatible with the current runtime. Errors are propagated;
        restoration never falls back to newly initialized generators.
    """

    validate_rng_state(state)
    random.setstate(state["py"])
    np.random.set_state(state["np"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["torch_cuda"]])


# ----------------------------------------------------------------------------------------------------------------------
def _loader_generators(loader: Any) -> dict[str, torch.Generator]:
    """
    Locate the independent generators of a loader that supports epoch-boundary resume.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader | Any
        Map-style DataLoader or a wrapper exposing its underlying loader through ``checkpoint_loader``.

    Returns
    -------
    dict[str, torch.Generator]
        Available generators keyed by ``loader`` and ``sampler``. Both keys may refer to the same generator.
        Missing independent generators are omitted; global Torch RNG state is handled separately.

    Raises
    ------
    ValueError
        If the dataset is iterable or workers persist across epochs, since their internal state is not captured.
    """

    loader = getattr(loader, "checkpoint_loader", loader)
    if isinstance(getattr(loader, "dataset", None), IterableDataset):
        raise ValueError("Exact resume requires a map-style dataset; streaming state is not checkpointed.")
    if getattr(loader, "persistent_workers", False):
        raise ValueError("Exact resume requires persistent_workers=false.")
    candidates = {
        "loader": getattr(loader, "generator", None),
        "sampler": getattr(getattr(loader, "sampler", None), "generator", None),
    }
    return {key: value for key, value in candidates.items() if isinstance(value, torch.Generator)}


# ----------------------------------------------------------------------------------------------------------------------
def capture_loader_state(loader: Any) -> dict[str, torch.Tensor | str]:
    """
    Capture independent loader/sampler RNG state, or record why exact resume is unsupported.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader | Any
        DataLoader or wrapper exposing ``checkpoint_loader``.

    Returns
    -------
    dict[str, torch.Tensor | str]
        Generator states keyed by generator name. Unsupported loaders return an ``unsupported`` reason instead,
        allowing ordinary streaming training to continue while making subsequent resume fail explicitly.

    Notes
    -----
    The training loop logs an unsupported-loader warning at the first checkpoint save, once per loader per
    invocation. This function only captures status; repeated captures do not emit duplicate warnings.
    """

    try:
        return {name: generator.get_state() for name, generator in _loader_generators(loader).items()}
    except ValueError as error:
        # Other entry points still support streaming training, but cannot promise exact resume.
        return {"unsupported": str(error)}


# ----------------------------------------------------------------------------------------------------------------------
def restore_loader_state(loader: Any, state: Mapping[str, torch.Tensor]) -> None:
    """
    Restore the loader and sampler generators saved at the end of an epoch.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader | Any
        Reconstructed DataLoader or wrapper exposing ``checkpoint_loader``.
    state : Mapping[str, torch.Tensor]
        Previously captured generator states for a supported loader.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the loader cannot support exact resume or its generator names differ from the checkpoint.
    RuntimeError
        If a saved generator state is invalid.
    """

    generators = _loader_generators(loader)
    if generators.keys() != state.keys():
        raise ValueError("Resume DataLoader generator configuration changed.")
    for name, generator in generators.items():
        generator.set_state(state[name].cpu())
