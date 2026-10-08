"""
Loss-term string vocabulary.

This module centralizes the lightweight string identifiers shared by configuration validation, output-filter
resolution, and loss construction. It is intentionally dependency-free (standard library only) so that lightweight
consumers — such as the config validator — can import it without pulling in heavy numerical dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, get_args

# ======================================================================================================================
# Loss term types
# ======================================================================================================================

EMBED_MSE_LOSS_TYPE = "embed_mse"
NATIVE_SPARSE_MSE_LOSS_TYPE = "native_sparse_mse"
STRONG_GRAD_SHAFRANOV_LOSS_TYPE = "strong_grad_shafranov"
WEAK_GRAD_SHAFRANOV_LOSS_TYPE = "weak_grad_shafranov"

# Embedding-space terms operate on output_emb and require no decoder.
EMBED_SPACE_LOSS_TYPES = frozenset({EMBED_MSE_LOSS_TYPE})
NATIVE_SPACE_LOSS_TYPES = frozenset(
    {
        NATIVE_SPARSE_MSE_LOSS_TYPE,
        STRONG_GRAD_SHAFRANOV_LOSS_TYPE,
        WEAK_GRAD_SHAFRANOV_LOSS_TYPE,
    }
)
ALL_LOSS_TYPES = frozenset({*EMBED_SPACE_LOSS_TYPES, *NATIVE_SPACE_LOSS_TYPES})

DEFAULT_LOSS_TERMS: tuple[Mapping[str, Any], ...] = ({"type": EMBED_MSE_LOSS_TYPE, "weight": 1.0},)

LOSS_ACRONYM_MAP = {"NSM": "data"}

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

GRAD_SHAFRANOV_WEIGHT_MAP = {"no_gt": r"$\omega_{NA}$", "lhs_gt": r"$\omega_{LA}$", "rhs_gt": r"$\omega_{RA}$"}

# ======================================================================================================================
# Plotting options
# ======================================================================================================================

ALLOWED_PLOT_FORMATS_TYPE = Literal["png", "pdf", "svg"]
ALLOWED_PLOT_FORMATS = get_args(ALLOWED_PLOT_FORMATS_TYPE)
ALLOWED_PLOT_CHECK_TYPES = frozenset({None, "show_plots", "save_png", "save_pdf", "save_svg"})
