"""
Decoding and de-standardisation utilities for MMT evaluation.

This module converts model outputs from standardised coefficient space
(backbone / adapter outputs) into native physical units by:

1) decoding coefficients using signal-specific codecs,
2) inverting the standardisation (mean / std).

All functions operate on NumPy arrays (CPU) and are intended for
evaluation and trace saving, not training.
"""

from typing import Dict, Any
import numpy as np

import logging

logger = logging.getLogger("mmt.Eval")


# ============================================================================
# De-standardize
# ============================================================================


def apply_stats(arr: np.ndarray, mean, std) -> np.ndarray:
    """
    Undo standardisation:  x = x_std * std + mean

    Supports different stats formats:
      - scalar mean/std
      - per-channel: mean/std shape (C,) for arr shaped (B, C, ...)
      - spatial/video: mean/std shape (H, W) for arr shaped (B, H, W, T)
      - exact match: mean/std shape == arr.shape[1:]
    """
    mean = np.asarray(mean)
    std = np.asarray(std)

    # scalar stats (or effectively scalar)
    if mean.ndim == 0 or mean.size == 1:
        return arr * std + mean

    def _broadcast(stat: np.ndarray) -> np.ndarray:
        stat = np.asarray(stat)

        # exact match (no batch)
        if stat.shape == arr.shape[1:]:
            return stat.reshape((1,) + stat.shape)

        # video/map stats: (H, W) -> (1, H, W, 1) for arr (B, H, W, T)
        if arr.ndim >= 4 and stat.shape == arr.shape[1:-1]:
            return stat.reshape((1,) + stat.shape + (1,))

        # per-channel stats: (C,) -> (1, C, 1, 1, ...)
        if arr.ndim >= 2 and stat.ndim == 1 and stat.shape[0] == arr.shape[1]:
            shape = [1] * arr.ndim
            shape[1] = stat.shape[0]
            return stat.reshape(shape)

        raise ValueError(
            f"Incompatible stats shape: arr.shape={arr.shape}, stat.shape={stat.shape}"
        )

    mean_b = _broadcast(mean)
    std_b = (
        _broadcast(std)
        if np.asarray(std).ndim > 0 and np.asarray(std).size > 1
        else std
    )
    return arr * std_b + mean_b


# ============================================================================
# Decode first, then destandardise
# ============================================================================


def decode_and_destandardize(
    y_pred_std: Dict[str, np.ndarray],
    y_true_std: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Decode model outputs from coefficient space and destandardise them in
    native physical space.

    All inputs are NumPy arrays on CPU:

      - y_pred_std[name] : (B, D)  standardized coefficients
      - y_true_std[name] : (B, ...) standardized native values
        (only used here to infer the original native shape per sample)

    Returns
    -------
    y_native : dict[name, np.ndarray]
        Decoded and destandardised predictions in native units.
    """
    y_native: Dict[str, np.ndarray] = {}

    for name, pred_std in y_pred_std.items():
        if name not in stats or name not in codecs or name not in y_true_std:
            continue

        if pred_std.ndim != 2:
            raise ValueError(
                f"y_pred_std[{name!r}] expected shape (B, D), got {pred_std.shape}."
            )

        codec = codecs[name]
        true_arr = y_true_std[name]
        B = pred_std.shape[0]
        original_shape = true_arr.shape[1:]  # (...,)

        # Decode each sample separately
        decoded = np.stack(
            [codec.decode(pred_std[b], original_shape) for b in range(B)],
            axis=0,
        )  # (B, ...)

        y_native[name] = apply_stats(
            decoded, mean=stats[name]["mean"], std=stats[name]["std"]
        )

    return y_native
