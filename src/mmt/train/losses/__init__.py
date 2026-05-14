"""
Loss package for the Multi-Modal Transformer.

This package provides a composable loss system consisting of individual loss terms combined by a
``LossAggregator``. Each term is an instance of ``BaseLoss`` and declares what batch fields it needs.

Available loss terms
--------------------
• ``EmbedMSELoss``        — MSE in embedding (coeff) space. Default term, no decoding required.
• ``NativeSparseMSELoss`` — MSE in native standardized space. Requires decoders and ``keep_output_native=True``.

Config schema (``train.loss``)
-------------------------------
Old format (single embed-MSE term, still supported)::

    loss:
      output_weights: {}   # optional per-signal weights

New format (explicit terms)::

    loss:
      output_weights: {}   # applied to embed_mse term when using old-style weight lookup
      terms:
        - type: embed_mse
          weight: 1.0
        - type: native_sparse_mse
          weight: 0.5

Notes
-----
For orthonormal encoders (DCT/FPCA) using all coefficients, coeff-space MSE and native-space MSE differ only by a
constant factor ``K / N``:

    native_MSE ≈ (K / N) * coeff_MSE

where K = number of coefficients and N = number of native samples. This implementation does **not** apply the K/N
scaling; losses are expressed in their respective space's MSE units.
"""

from __future__ import annotations

from .aggregator import LossAggregator, build_loss_aggregator
from .base import BaseLoss
from .embed import EmbedMSELoss
from .native_sparse import NativeSparseMSELoss


__all__ = [
    "BaseLoss",
    "EmbedMSELoss",
    "NativeSparseMSELoss",
    "LossAggregator",
    "build_loss_aggregator",
]
