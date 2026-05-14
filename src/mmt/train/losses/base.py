"""
Base class for MMT loss terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Hashable

from torch import Tensor


# ======================================================================================================================
class BaseLoss(ABC):
    """
    Abstract base class for individual loss terms.

    Each loss term operates on a batch and returns a scalar loss together with per-output logs. Multiple terms
    can be combined by a ``LossAggregator`` with optional per-term weights.

    Subclasses declare their data requirements via the three boolean properties below. These are read by
    ``LossAggregator`` to decide what batch fields to extract and whether the dataset must carry native outputs.

    Properties
    ----------
    requires_native_target : bool
        True if ``compute()`` needs ``y_native`` from ``batch['output_native']``.
        Implies ``data.keep_output_native=True`` must be set at dataset build time.
    requires_decode : bool
        True if the loss decodes predictions from embedding space to native space internally.
    requires_destandardize : bool
        True if the loss undoes standardization (requires per-signal stats at init).

    """

    # ------------------------------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def requires_native_target(self) -> bool:
        """True if this term needs ``batch['output_native']``."""

    # ------------------------------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def requires_decode(self) -> bool:
        """True if this term decodes embeddings to native space."""

    # ------------------------------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def requires_destandardize(self) -> bool:
        """True if this term undoes standardization."""

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: Mapping[Hashable, Tensor],
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute the loss for one batch.

        Parameters
        ----------
        preds:
            Prediction tensors in embedding (coeff) space, keyed by signal_id. Shape: ``(B, D)``.
        y_emb:
            Ground-truth tensors in embedding space, keyed by signal_id. Shape: ``(B, D)``.
        y_native:
            Ground-truth tensors in native standardized space, keyed by signal_id. Shape: ``(B, *native_shape)``.
            ``None`` when ``requires_native_target=False`` or ``keep_output_native=False``.
        output_mask:
            Boolean mask tensors of shape ``(B,)``, True for supervised samples.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            ``(loss_scalar, per_output_logs)``

        """
