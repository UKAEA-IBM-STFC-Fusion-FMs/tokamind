"""
Base class for MMT loss terms.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Hashable

from torch import Tensor

GENERIC_LOSS_TERM_CFG_KEYS = frozenset({"type", "weight", "outputs"})


# ======================================================================================================================
@dataclass(frozen=True)
class LossComputeContext:
    """
    Optional metadata passed to loss terms during computation.

    This context is metadata-only: loss terms may use it for logging or plots, but it must not be required to compute
    the scalar loss.

    Attributes
    ----------
    stage_index : int | None
        Current training stage index.
    stage_name : str | None
        Current training stage name.
    epoch_global : int | None
        Global epoch number.
    epoch_in_stage : int | None
        Epoch number within the current stage.
    global_step : int | None
        Optimizer step at which the loss is being computed.
    batch_index : int | None
        Batch index within the current epoch.
    train : bool | None
        Whether the current pass is training or validation.
    term_index : int | None
        Loss term index inside the aggregator.
    term_name : str | None
        Runtime loss term class name.
    term_weight : float | None
        Aggregator-level weight for this term.
    term_config : Mapping[str, Any] | None
        User-facing config block for this term, when available.
    stage_loss_terms : tuple[Mapping[str, Any], ...]
        User-facing config blocks for all terms in the current stage loss, when available.
    run_dir : str | None
        Current run directory, when available. Intended only for metadata side effects such as diagnostic plots.

    """

    stage_index: int | None = None
    stage_name: str | None = None
    epoch_global: int | None = None
    epoch_in_stage: int | None = None
    global_step: int | None = None
    batch_index: int | None = None
    train: bool | None = None
    term_index: int | None = None
    term_name: str | None = None
    term_weight: float | None = None
    term_config: Mapping[str, Any] | None = None
    stage_loss_terms: tuple[Mapping[str, Any], ...] = ()
    run_dir: str | None = None


# ======================================================================================================================
class BaseLoss(ABC):
    """
    Abstract base class for individual loss terms.

    Each loss term operates on a batch and returns a scalar loss together with per-output logs. Multiple terms can be
    combined by a `LossAggregator` with optional per-term weights.

    Subclasses declare their data requirements via the boolean properties below. These are read by `LossAggregator`
    to decide what batch fields to extract, whether the dataset must carry native outputs, and whether to forward the
    per-output predictive distribution.

    Properties
    ----------
    requires_native_target : bool
        True if `compute()` needs `y_native` from `batch['output_native']`.
        Implies `data.keep_output_native=True` must be set at dataset build time.
    requires_decode : bool
        True if the loss decodes predictions from embedding space to native space internally.
    requires_destandardize : bool
        True if the loss undoes standardization (requires per-signal stats at init).
    requires_pred_dist : bool
        True if `compute()` needs `pred_dist` — the per-output predictive distribution parameters produced by a
        Gaussian output head. Implies the model must be built with a probabilistic (gaussian) output adapter.
        Defaults to False, so existing deterministic terms are never passed `pred_dist`.

    """

    # ------------------------------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def requires_native_target(self) -> bool:
        """True if this term needs batch['output_native']`."""

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
    @property
    def requires_pred_dist(self) -> bool:
        """
        True if this term needs the per-output predictive distribution (`pred_dist`).

        Concrete (non-abstract) with a default of False so existing deterministic terms need not declare it and are
        never passed `pred_dist`. Probabilistic terms (e.g. CRPS) override this with a class attribute set to True.
        """
        return False

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_term_cfg(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate loss-specific config fields for one ``train.loss.terms`` item.

        Generic config checks such as term type, probabilistic-head compatibility, and ``n_samples`` handling live in
        the config validator. Subclasses override this hook for fields they own, so application-specific options can
        fail early without hard-coding those details in the generic validator.

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.

        Returns
        -------
        None

        """

        return None

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _validate_known_term_keys(
        cls,
        term_def: Mapping[str, Any],
        path: str,
        *,
        allowed_specific_keys: set[str] | frozenset[str],
    ) -> None:
        """
        Validate that a term config contains only generic keys plus the subclass-owned keys.

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.
        allowed_specific_keys : set[str] | frozenset[str]
            Keys owned by the concrete loss class.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If an unknown key is present.

        """

        allowed = GENERIC_LOSS_TERM_CFG_KEYS | set(allowed_specific_keys)
        unknown = sorted(str(key) for key in term_def.keys() if str(key) not in allowed)
        if unknown:
            raise KeyError(f"Unknown {path} keys: {unknown}. Supported keys are {sorted(allowed)}.")

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def _validate_weight_mapping(
        cls,
        weights: Mapping[str, Any],
        path: str,
        *,
        allowed_keys: set[str] | frozenset[str],
    ) -> None:
        """
        Validate a named non-negative numeric weight mapping.

        Parameters
        ----------
        weights : Mapping[str, Any]
            Weight mapping to validate.
        path : str
            Human-readable config path used in error messages.
        allowed_keys : set[str] | frozenset[str]
            Supported weight keys.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If an unknown key is present.
        TypeError
            If a value is not numeric.
        ValueError
            If a value is negative.

        """

        unknown_weight_keys = sorted(str(key) for key in weights if str(key) not in allowed_keys)
        if unknown_weight_keys:
            raise KeyError(f"Unknown {path} keys: {unknown_weight_keys}. Supported keys are {sorted(allowed_keys)}.")
        for key, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise TypeError(f"{path}.{key} must be a number.")
            if float(value) < 0.0:
                raise ValueError(f"{path}.{key} must be non-negative.")

    # ------------------------------------------------------------------------------------------------------------------
    @abstractmethod
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: Mapping[Hashable, Tensor],
        pred_dist: Mapping[Hashable, Mapping[str, Any]] | None = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute the loss for one batch.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Prediction tensors in embedding (coeff) space, keyed by signal_id. Shape: `(B, D)`.
            For probabilistic output heads this is the predictive mean (`mu`).
        y_emb : Mapping[Hashable, Tensor]
            Ground-truth tensors in embedding space, keyed by signal_id. Shape: `(B, D)`.
        y_native : Mapping[Hashable, Tensor] | None
            Ground-truth tensors in native standardized space, keyed by signal_id. Shape: `(B, *native_shape)`.
            None when `requires_native_target=False` or `keep_output_native=False`.
        output_mask : Mapping[Hashable, Tensor]
            Boolean mask tensors of shape `(B,)`, True for supervised samples.
        pred_dist : Mapping[Hashable, Mapping[str, Any]] | None
            Per-output predictive distribution parameters from a Gaussian output head, keyed by signal_id. Each value
            is a mapping with keys such as `"mu"`, `"sigma"`, `"raw_scale"`, `"log_sigma"`, each shaped `(B, D)`.
            Passed by `LossAggregator` only to terms with `requires_pred_dist=True`; otherwise None.
        context : LossComputeContext | None
            Optional metadata about the current stage, epoch, and aggregator term. Losses may use it for logging or
            plots, but should not require it for core loss computation.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            `(loss_scalar, per_output_logs)`

        """
