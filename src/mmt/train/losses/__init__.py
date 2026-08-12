"""
Loss package for the Multi-Modal Transformer.

This package provides a composable loss system consisting of individual loss terms combined by a
``LossAggregator``. Each term is an instance of ``BaseLoss`` and declares what batch fields it needs.

Available loss terms
--------------------
• ``EmbedMSELoss``        — MSE in embedding (coeff) space. Default term, no decoding required.
• ``NativeSparseMSELoss`` — MSE in native standardized space. Requires decoders and ``keep_output_native=True``.
• ``GradShafranovResidualLoss`` — Grad-Shafranov residual in destandardized native space.

Config schema (``train.loss``)
-------------------------------
Old format (single embed-MSE term, still supported)::

    loss:
      output_weights: {}   # optional per-signal weights

New format (explicit terms)::

    loss:
      output_weights: {}   # optional per-signal weights applied inside each term
      terms:
        - type: embed_mse
          weight: 1.0
          outputs:
            include: [output_a, output_b]
        - type: native_sparse_mse
          weight: 0.5
          outputs:
            exclude: [output_a, output_b]
        - type: grad_shafranov_residual
          weight: 0.1
          rhs_input:
            origin: predicted_j_tor
            calculation_method: GS_operator  # REMARK: Only used when "built" origin is used. TODO: Improve wording.
          outputs:
            include: [equilibrium-psi, equilibrium-j_tor]

Notes
-----
For orthonormal encoders (DCT/FPCA) using all coefficients, coeff-space MSE and native-space MSE differ only by a
constant factor ``K / N``:

    native_MSE ≈ (K / N) * coeff_MSE

where K = number of coefficients and N = number of native samples. This implementation does **not** apply the K/N
scaling; losses are expressed in their respective space's MSE units.
"""

from __future__ import annotations

from .constants import DEFAULT_LOSS_TERMS

from .aggregator import LossAggregator, build_loss_aggregator
from .base import BaseLoss, LossComputeContext
from .embed_mse import EmbedMSELoss
from .embed_gaussian_crps import EmbedGaussianCRPSLoss
from .filters import resolve_loss_output_filters, resolve_native_loss_output_names
from .grad_shafranov import GradShafranovResidualLoss, WeakFormGradShafranovLoss
from .native_sparse_mse import NativeSparseMSELoss
from .native_sparse_sample_crps import NativeSparseSampleCRPSLoss
from .registry import LOSS_REGISTRY, get_loss_class


__all__ = [
    "BaseLoss",
    "LossComputeContext",
    "EmbedMSELoss",
    "EmbedGaussianCRPSLoss",
    "NativeSparseMSELoss",
    "NativeSparseSampleCRPSLoss",
    "GradShafranovResidualLoss",
    "LossAggregator",
    "DEFAULT_LOSS_TERMS",
    "build_loss_aggregator",
    "resolve_loss_output_filters",
    "resolve_native_loss_output_names",
    "LOSS_REGISTRY",
    "get_loss_class",
    "WeakFormGradShafranovLoss",
]
