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
NATIVE_SPARSE_MSE_LOSS_TYPE = "native_sparse_mse"
CONTINUOUS_GPO_LOSS_TYPE = "continuous_gpo"

# Embedding-space terms operate on output_emb and require no decoder.
EMBED_SPACE_LOSS_TYPES = frozenset({EMBED_MSE_LOSS_TYPE, CONTINUOUS_GPO_LOSS_TYPE})
NATIVE_SPACE_LOSS_TYPES = frozenset({NATIVE_SPARSE_MSE_LOSS_TYPE})
ALL_LOSS_TYPES = frozenset({*EMBED_SPACE_LOSS_TYPES, *NATIVE_SPACE_LOSS_TYPES})

DEFAULT_LOSS_TERMS: tuple[Mapping[str, Any], ...] = ({"type": EMBED_MSE_LOSS_TYPE, "weight": 1.0},)
