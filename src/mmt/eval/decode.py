"""
Decoding and de-standardization utilities for MMT evaluation.

This module converts model outputs from standardized coefficient space (backbone / adapter outputs) into native
physical units by:

1) decoding coefficients using signal-specific codecs,
2) inverting the standardization (mean / std).

All functions operate on NumPy arrays (CPU) and are intended for evaluation and trace saving, not training.
"""

from collections.abc import Mapping
from typing import Any
import numpy as np
import logging


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Eval")


# ======================================================================================================================
# De-standardize
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def apply_stats(arr: np.ndarray, mean: float | np.ndarray, std: float | np.ndarray) -> np.ndarray:
    """

    Undo standardization:  x = x_std * std + mean.

    Supports different stats formats:
      - Scalar mean/std
      - Per-channel: mean/std shape (C,) for arr shaped (B, C, ...)
      - Spatial/video: mean/std shape (H, W) for arr shaped (B, H, W, T)
      - Exact match: mean/std shape == arr.shape[1:]

    Parameters
    ----------
    arr : np.ndarray
        Input array to be destandardized.
    mean : float | np.ndarray
        Mean value for destandardization.
    std : float | np.ndarray
        STD value for destandardization

    Returns
    -------
    np.ndarray
        Destandardized input array.

    """

    # ..................................................................................................................
    def _broadcast(stat: np.ndarray) -> np.ndarray:
        """
        Broadcast (reshape) passed statistics.

        Parameters
        ----------
        stat : np.ndarray
            Statistics to be broadcast.

        Returns
        -------
        np.ndarray
            Broadcast (reshaped) statistics.

        Raises
        ------
        ValueError
            If incompatibilities between `arr.shape={arr.shape}` and `stat.shape={stat.shape}`) are detected.

        """

        stat = np.asarray(stat)

        # Exact match (no batch)
        if stat.shape == arr.shape[1:]:
            return stat.reshape((1,) + stat.shape)

        # Video/map stats: (H, W) -> (1, H, W, 1) for arr (B, H, W, T)
        if (arr.ndim >= 4) and (stat.shape == arr.shape[1:-1]):
            return stat.reshape((1,) + stat.shape + (1,))

        # Per-channel stats: (C,) -> (1, C, 1, 1, ...)
        if (arr.ndim >= 2) and (stat.ndim == 1) and (stat.shape[0] == arr.shape[1]):
            shape = [1] * arr.ndim
            shape[1] = stat.shape[0]
            return stat.reshape(shape)

        raise ValueError(f"Incompatible stats shape: `arr.shape={arr.shape}`, `stat.shape={stat.shape}`.")

    # ..................................................................................................................

    mean_arr: np.ndarray = np.asarray(mean)
    std_arr: np.ndarray = np.asarray(std)

    # Scalar stats (or effectively scalar)
    if mean_arr.ndim == 0 or mean_arr.size == 1:
        return arr * std_arr + mean_arr

    mean_b = _broadcast(stat=mean_arr)
    std_b = _broadcast(stat=std_arr) if (std_arr.ndim > 0) and (std_arr.size > 1) else std_arr

    return arr * std_b + mean_b


# ======================================================================================================================
# Decode first, then destandardize
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def decode_and_destandardize(
    y_pred_std: Mapping[str, np.ndarray],
    y_true_std: Mapping[str, np.ndarray],
    stats: Mapping[str, Mapping[str, float]],
    codecs: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    """
    Decode model outputs from coefficient space and destandardize them in native physical space.

    All inputs are NumPy arrays on CPU:

      - y_pred_std[name] : (B, D)  standardized coefficients
      - y_true_std[name] : (B, ...) standardized native values (only used here to infer the original native shape per
        sample)

    Parameters
    ----------
    y_pred_std : Mapping[str, np.ndarray]
        Mapping with encoded/standardized predicted signal values keyed by signal name.
    y_true_std : Mapping[str, np.ndarray]
        Mapping with encoded/standardized true signal values keyed by signal name.
    stats, codecs : Mapping
        Output decoding / de-standardization inputs (native-space evaluation).

    Returns
    -------
    y_native : dict[name, np.ndarray]
        Decoded and destandardized predictions in native units.

    Raises
    ------
    ValueError
        If `y_pred_std` has an item with incompatible shape.

    """

    y_native: dict[str, np.ndarray] = {}

    for name, pred_std in y_pred_std.items():
        if (name not in stats) or (name not in codecs) or (name not in y_true_std):
            continue

        if pred_std.ndim != 2:
            raise ValueError(f"`y_pred_std[{name!r}]` expected shape (B, D), got {pred_std.shape}.")

        codec = codecs[name]
        true_arr = y_true_std[name]
        B = pred_std.shape[0]
        original_shape = true_arr.shape[1:]  # (...,)

        # Decode each sample separately
        decoded = np.stack(
            [codec.decode(z=pred_std[b], original_shape=original_shape) for b in range(B)],
            axis=0,
        )  # (B, ...)

        y_native[name] = apply_stats(arr=decoded, mean=stats[name]["mean"], std=stats[name]["std"])

    return y_native

    # ..................................................................................................................
