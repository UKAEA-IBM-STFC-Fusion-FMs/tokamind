"""
Standardization utilities for MMT signals.

This module provides functions to undo signal standardization (mean/std normalization) in both
numpy and torch. The two implementations share the same broadcasting conventions and are
intended to be used together:

    • `destandardize_numpy`  — for eval/export paths after detaching from the compute graph.
    • `destandardize_torch`  — for training losses that operate in physical-unit space (e.g., a future Grad-Shafranov
      loss) and for eval paths that share the same forward pass as training.

Broadcasting conventions (both functions)
------------------------------------------
Given an array `x` of shape (B, ...) and scalar or array statistics:
  - Scalar mean/std          → broadcast directly.
  - Per-channel (C,)         → broadcast along axis 1 for arrays of shape (B, C, ...).
  - Spatial (H, W)           → broadcast along axes 1-2 for arrays of shape (B, H, W, T).
  - Exact match (same shape as x[0]) → broadcast with a leading singleton batch dim.
"""

from __future__ import annotations

import numpy as np
from torch import Tensor


# ======================================================================================================================
# NumPy (eval / export)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def destandardize_numpy(arr: np.ndarray, mean: float | np.ndarray, std: float | np.ndarray) -> np.ndarray:
    """
    Undo standardization in numpy: `x = x_std * std + mean`.

    Parameters
    ----------
    arr : np.ndarray
        Standardized input array of shape (B, ...).
    mean : float | np.ndarray
        Mean used during standardization. Scalar, per-channel (C,), spatial (H, W),
        or exact match with `arr.shape[1:]`.
    std : float | np.ndarray
        Standard deviation used during standardization. Same shape conventions as `mean`.

    Returns
    -------
    np.ndarray
        Destandardized array of the same shape as `arr`.

    Raises
    ------
    ValueError
        If `mean` or `std` shape cannot be broadcast to `arr`.

    """

    # ..................................................................................................................
    def _broadcast(stat: np.ndarray) -> np.ndarray:
        stat = np.asarray(stat)

        if stat.shape == arr.shape[1:]:
            return stat.reshape((1,) + stat.shape)

        # Spatial stats: (H, W) for arr (B, H, W, T)
        if (arr.ndim >= 4) and (stat.shape == arr.shape[1:-1]):
            return stat.reshape((1,) + stat.shape + (1,))

        # Per-channel stats: (C,) for arr (B, C, ...)
        if (arr.ndim >= 2) and (stat.ndim == 1) and (stat.shape[0] == arr.shape[1]):
            shape = [1] * arr.ndim
            shape[1] = stat.shape[0]
            return stat.reshape(shape)

        raise ValueError(f"Cannot broadcast stats shape {stat.shape} to arr shape {arr.shape}.")

    # ..................................................................................................................

    mean_arr = np.asarray(mean)
    std_arr = np.asarray(std)

    if mean_arr.ndim == 0 or mean_arr.size == 1:
        return arr * std_arr + mean_arr

    mean_b = _broadcast(mean_arr)
    std_b = _broadcast(std_arr) if (std_arr.ndim > 0) and (std_arr.size > 1) else std_arr

    return arr * std_b + mean_b


# ======================================================================================================================
# Torch (training losses + eval shared path)
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def destandardize_torch(x: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """
    Undo standardization in torch: `x = x_std * std + mean`.

    Gradients are preserved with respect to `x`. `mean` and `std` are treated as fixed constants (no gradient required).

    Parameters
    ----------
    x : Tensor
        Standardized input tensor of shape (B, ...).
    mean : Tensor
        Mean used during standardization. Must be broadcastable to `x`.
    std : Tensor
        Standard deviation used during standardization. Must be broadcastable to `x`.

    Returns
    -------
    Tensor
        Destandardized tensor of the same shape as `x`.

    """

    return x * std + mean
