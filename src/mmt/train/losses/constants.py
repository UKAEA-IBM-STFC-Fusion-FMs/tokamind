"""
Loss-term string vocabulary.

This module centralizes the lightweight string identifiers shared by configuration validation, output-filter
resolution, and loss construction. It is intentionally dependency-free (standard library only) so that lightweight
consumers — such as the config validator — can import it without pulling in heavy numerical dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# ======================================================================================================================
# Loss term types
# ======================================================================================================================

EMBED_MSE_LOSS_TYPE = "embed_mse"
EMBED_GAUSSIAN_CRPS_LOSS_TYPE = "embed_gaussian_crps"
NATIVE_SPARSE_MSE_LOSS_TYPE = "native_sparse_mse"
NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE = "native_sparse_sample_crps"
GRAD_SHAFRANOV_LOSS_TYPE = "grad_shafranov_residual"
GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE = "grad_shafranov_weak_form"

# Embedding-space terms operate on output_emb and require no decoder. Probabilistic terms additionally consume the
# Gaussian output head's pred_dist (mu/sigma) rather than only the mean.
EMBED_SPACE_LOSS_TYPES = frozenset({EMBED_MSE_LOSS_TYPE, EMBED_GAUSSIAN_CRPS_LOSS_TYPE})
NATIVE_SPACE_LOSS_TYPES = frozenset(
    {
        NATIVE_SPARSE_MSE_LOSS_TYPE,
        NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE,
        GRAD_SHAFRANOV_LOSS_TYPE,
        GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE,
    }
)
ALL_LOSS_TYPES = frozenset({*EMBED_SPACE_LOSS_TYPES, *NATIVE_SPACE_LOSS_TYPES})

# Terms that consume the output head's pred_dist and therefore require a probabilistic (gaussian) output adapter.
# Config-time mirror of BaseLoss.requires_pred_dist (terms are not instantiated during validation); keep in sync.
PRED_DIST_LOSS_TYPES = frozenset({EMBED_GAUSSIAN_CRPS_LOSS_TYPE, NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE})

# Sample-based probabilistic terms draw reparameterized samples from pred_dist and therefore require an `n_samples`
# config field (>= 2). Closed-form / deterministic terms must not set it. (SAMPLE_LOSS_TYPES ⊆ PRED_DIST_LOSS_TYPES.)
SAMPLE_LOSS_TYPES = frozenset({NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE})

# Default number of reparameterized samples for sample-based CRPS terms when `n_samples` is omitted in config.
DEFAULT_CRPS_N_SAMPLES = 16

DEFAULT_LOSS_TERMS: tuple[Mapping[str, Any], ...] = ({"type": EMBED_MSE_LOSS_TYPE, "weight": 1.0},)


# ======================================================================================================================
# Grad-Shafranov loss options
# ======================================================================================================================

GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR = "predicted_j_tor"
GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR = "derived_j_tor"
GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES = "predicted_profiles"

GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR = "GS_operator"
GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX = "parametric_approx"

GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY = "origin"
GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY = "calculation_method"
GRAD_SHAFRANOV_RHS_KEYS = frozenset(
    {GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY, GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY}
)
GRAD_SHAFRANOV_RHS_INPUT_ORIGINS = frozenset(
    {
        GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR,
        GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR,
        GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES,
    }
)
GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS = frozenset(
    {GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR, GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX}
)
