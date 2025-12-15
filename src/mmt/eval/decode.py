"""
Decoding and de-standardisation utilities for MMT evaluation.

This module converts model outputs from standardised coefficient space
(backbone / adapter outputs) into native physical units by:

1) decoding coefficients using signal-specific codecs,
2) inverting the baseline standardisation (mean / std).

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
    Invert the standardisation used in the baseline:

        values_std = (values - mean[..., None]) / std[..., None]

    Here `arr` is batch-first, e.g. (B, C, T, ...) or (B, T, ...).
    `mean` / `std` can be:
      - scalar
      - shape (1,)
      - shape (C,)
    """
    mean = np.asarray(mean)
    std = np.asarray(std)

    # Scalar or effectively scalar
    if mean.ndim == 0 or (mean.ndim == 1 and mean.shape[0] == 1):
        return arr * std + mean

    # Per-channel (C,)
    if arr.ndim < 2:
        raise ValueError(
            f"Cannot apply per-channel stats: arr.shape={arr.shape}, mean.shape={mean.shape}"
        )
    C = arr.shape[1]
    if mean.shape[0] != C:
        raise ValueError(
            f"Incompatible shapes: mean.shape={mean.shape}, arr.shape={arr.shape} "
            f"(expected mean.shape[0] == arr.shape[1] == {C})"
        )

    shape = [1] * arr.ndim
    shape[1] = C  # (1, C, 1, 1, ...)
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return arr * std + mean


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
