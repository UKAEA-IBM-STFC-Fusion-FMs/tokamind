"""
Loss-term registry.

The registry maps lightweight config ``type`` strings to the concrete loss classes that implement construction,
runtime computation, and optional term-specific config validation.
"""

from __future__ import annotations

from mmt.train.losses.constants import (
    ALL_LOSS_TYPES,
    EMBED_GAUSSIAN_CRPS_LOSS_TYPE,
    EMBED_MSE_LOSS_TYPE,
    GRAD_SHAFRANOV_LOSS_TYPE,
    GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE,
    NATIVE_SPARSE_MSE_LOSS_TYPE,
    NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE,
)

from .base import BaseLoss
from .embed_gaussian_crps import EmbedGaussianCRPSLoss
from .embed_mse import EmbedMSELoss
from .grad_shafranov import GradShafranovResidualLoss, WeakFormGradShafranovLoss
from .native_sparse_mse import NativeSparseMSELoss
from .native_sparse_sample_crps import NativeSparseSampleCRPSLoss


# ======================================================================================================================

LOSS_REGISTRY: dict[str, type[BaseLoss]] = {
    EMBED_MSE_LOSS_TYPE: EmbedMSELoss,
    EMBED_GAUSSIAN_CRPS_LOSS_TYPE: EmbedGaussianCRPSLoss,
    NATIVE_SPARSE_MSE_LOSS_TYPE: NativeSparseMSELoss,
    NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE: NativeSparseSampleCRPSLoss,
    GRAD_SHAFRANOV_LOSS_TYPE: GradShafranovResidualLoss,
    GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE: WeakFormGradShafranovLoss,
}


def _check_registry_sync() -> None:
    """
    Ensure the registry covers exactly the configured loss-term vocabulary.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If a loss type is declared but not registered, or registered but not declared.

    """

    missing = ALL_LOSS_TYPES - set(LOSS_REGISTRY)
    extra = set(LOSS_REGISTRY) - ALL_LOSS_TYPES
    if missing or extra:
        raise RuntimeError(
            f"LOSS_REGISTRY is out of sync with ALL_LOSS_TYPES. Missing: {sorted(missing)} | Extra: {sorted(extra)}."
        )


_check_registry_sync()


# ----------------------------------------------------------------------------------------------------------------------
def get_loss_class(term_type: str) -> type[BaseLoss]:
    """
    Return the registered loss class for a configured term type.

    Parameters
    ----------
    term_type : str
        Configured loss term type.

    Returns
    -------
    type[BaseLoss]
        Concrete loss class.

    Raises
    ------
    KeyError
        If ``term_type`` is not registered.

    """

    try:
        return LOSS_REGISTRY[term_type]
    except KeyError as exc:
        raise KeyError(f"Unknown loss term type {term_type!r}. Registered: {sorted(LOSS_REGISTRY)}.") from exc
