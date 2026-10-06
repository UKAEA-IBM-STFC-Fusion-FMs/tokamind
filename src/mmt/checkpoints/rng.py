"""
Random number generator (RNG) state utilities.

This module provides helpers to capture and restore RNG states for Python, NumPy, and PyTorch (CPU and CUDA when
available), enabling reproducible checkpoint save, resume, and warm-start workflows.
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any
import numpy as np

import torch


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
def restore_rng_state(state: Mapping[str, Any]) -> None:
    """
    Restore RNG states if present (best-effort). Errors during restore are ignored, but only expected ones.

    Parameters
    ----------
    state : Mapping[str, Any]
        Mapping (dict) with states in ["py", "np", "torch_cpu", "torch_cuda"] to be restored.

    Returns
    -------
    None

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
