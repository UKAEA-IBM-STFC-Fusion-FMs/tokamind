"""
Decoding and de-standardization utilities for MMT evaluation.

This module converts model outputs from standardized coefficient space (backbone / adapter outputs) into native
physical units by:

1) decoding coefficients back to native shape via the same ``TorchDecoder`` used during training,
   wrapped in ``torch.no_grad()`` and detached to CPU numpy — no gradient computation,
2) inverting the standardization using ``destandardize_numpy`` from ``mmt.data.standardization``.

All functions return NumPy arrays (CPU) and are intended for evaluation and trace saving, not training.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from mmt.data.standardization import destandardize_numpy

if TYPE_CHECKING:
    from mmt.data.embeddings.torch_decoder import TorchDecoder


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Eval")


# ======================================================================================================================
# Decode then destandardize
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def decode_and_destandardize(
    y_pred_std: Mapping[str, np.ndarray],
    y_true_std: Mapping[str, np.ndarray],
    stats: Mapping[str, Mapping[str, float]],
    decoders: Mapping[str, TorchDecoder],
) -> dict[str, np.ndarray]:
    """
    Decode model outputs from coefficient space and destandardize them to native physical units.

    Decoding uses the pre-built ``TorchDecoder`` instances, executed under ``torch.no_grad()`` and detached
    to CPU numpy.

    Parameters
    ----------
    y_pred_std : Mapping[str, np.ndarray]
        Predicted embeddings in coefficient space, keyed by signal name. Shape: ``(B, D)``.
    y_true_std : Mapping[str, np.ndarray]
        Ground-truth tensors in native standardized space, keyed by signal name. Shape: ``(B, *native_shape)``.
        Values are not decoded here; used only for presence checks.
    stats : Mapping[str, Mapping[str, float]]
        Per-signal stats dict with ``"mean"`` and ``"std"`` keys.
    decoders : Mapping[str, TorchDecoder]
        Pre-built per-signal ``TorchDecoder`` instances keyed by signal name.

    Returns
    -------
    dict[str, np.ndarray]
        Decoded and destandardized predictions in native units, keyed by signal name.

    Raises
    ------
    ValueError
        If a prediction array does not have shape ``(B, D)``.

    """

    y_native: dict[str, np.ndarray] = {}

    for name, pred_std in y_pred_std.items():
        if (name not in stats) or (name not in decoders) or (name not in y_true_std):
            continue

        if pred_std.ndim != 2:
            raise ValueError(f"`y_pred_std[{name!r}]` expected shape (B, D), got {pred_std.shape}.")

        z = torch.from_numpy(pred_std)  # (B, D)
        with torch.no_grad():
            decoded_t = decoders[name](z)  # (B, *original_shape)

        decoded = decoded_t.detach().cpu().float().numpy()

        y_native[name] = destandardize_numpy(arr=decoded, mean=stats[name]["mean"], std=stats[name]["std"])

    return y_native


# ----------------------------------------------------------------------------------------------------------------------
def decode_samples_and_destandardize(
    y_pred_samples_std: Mapping[str, np.ndarray],
    y_true_std: Mapping[str, np.ndarray],
    stats: Mapping[str, Mapping[str, float]],
    decoders: Mapping[str, TorchDecoder],
) -> dict[str, np.ndarray]:
    """
    Decode an ensemble of predictive samples from coefficient space and destandardize to native physical units.

    Mirrors :func:`decode_and_destandardize` but carries an extra leading sample axis ``S``. Each sample is decoded
    through the same per-signal ``TorchDecoder``; decoding runs under ``torch.no_grad()`` and is detached to CPU numpy.

    Parameters
    ----------
    y_pred_samples_std : Mapping[str, np.ndarray]
        Predictive samples in coefficient space, keyed by signal name. Shape: ``(B, S, D)``.
    y_true_std : Mapping[str, np.ndarray]
        Ground-truth tensors in native standardized space, keyed by signal name. Used only for presence checks.
    stats : Mapping[str, Mapping[str, float]]
        Per-signal stats dict with ``"mean"`` and ``"std"`` keys.
    decoders : Mapping[str, TorchDecoder]
        Pre-built per-signal ``TorchDecoder`` instances keyed by signal name.

    Returns
    -------
    dict[str, np.ndarray]
        Decoded and destandardized samples in native units, keyed by signal name. Shape: ``(B, S, *native_shape)``.

    Raises
    ------
    ValueError
        If a sample array does not have shape ``(B, S, D)``.

    """

    y_native: dict[str, np.ndarray] = {}

    for name, samples_std in y_pred_samples_std.items():
        if (name not in stats) or (name not in decoders) or (name not in y_true_std):
            continue

        if samples_std.ndim != 3:
            raise ValueError(f"`y_pred_samples_std[{name!r}]` expected shape (B, S, D), got {samples_std.shape}.")

        b, s, d = samples_std.shape
        z = torch.from_numpy(samples_std).reshape(b * s, d)  # (B*S, D)
        with torch.no_grad():
            decoded_t = decoders[name](z)  # (B*S, *native_shape)

        decoded = decoded_t.detach().cpu().float().numpy()
        decoded = decoded.reshape(b, s, *decoded.shape[1:])  # (B, S, *native_shape)

        y_native[name] = destandardize_numpy(arr=decoded, mean=stats[name]["mean"], std=stats[name]["std"])

    return y_native
