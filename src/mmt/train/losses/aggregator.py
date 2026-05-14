"""
Loss aggregator for the Multi-Modal Transformer.

This module provides:
    • ``LossAggregator``        — combines multiple ``BaseLoss`` terms into a single weighted scalar loss.
    • ``build_loss_aggregator`` — factory that instantiates an aggregator from the ``train.loss`` config block.

Each term contributes an independent scalar loss; the aggregator combines them as a normalized weighted sum across
terms. Per-output weights (if any) are handled inside each individual term.

The ``build_loss_aggregator`` factory supports the following term types:
    • ``embed_mse``         — MSE in embedding (coeff) space; no decoding required.
    • ``native_sparse_mse`` — MSE in native standardized space; requires pre-built torch decoders.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Hashable

import torch
from torch import Tensor

from .base import BaseLoss
from .embed import EmbedMSELoss
from .native_sparse import NativeSparseMSELoss

if TYPE_CHECKING:
    from mmt.data.embeddings.torch_decoder import TorchDecoder


# ======================================================================================================================
class LossAggregator:
    """
    Combine multiple loss terms into a single weighted scalar.

    Each term is weighted by its own term-level weight before aggregation. Per-output weights (if any) are
    handled inside each term.

    Parameters
    ----------
    terms : list[tuple[BaseLoss, float]]
        List of ``(loss_term, term_weight)`` pairs. Weights do not need to sum to 1.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, terms: list[tuple[BaseLoss, float]]) -> None:
        if not terms:
            raise ValueError("LossAggregator requires at least one loss term.")

        self._terms = terms

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def requires_native_target(self) -> bool:
        """True if any term needs ``batch['output_native']``."""
        return any(t.requires_native_target for t, _ in self._terms)

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        batch: Mapping[str, Any],
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Compute the aggregated loss for one batch.

        Parameters
        ----------
        preds:
            Model predictions in embedding space, keyed by signal_id. Shape: ``(B, D)`` per key.
        batch:
            Collated batch dict. Expected keys: ``output_emb``, ``output_mask``, and optionally
            ``output_native`` (when any term has ``requires_native_target=True``).

        Returns
        -------
        tuple[Tensor, dict]
            ``(total_loss, logs)`` where logs contains per-term and per-output loss values.

        """

        y_emb: Mapping[Hashable, Tensor] = batch.get("output_emb", {})
        output_mask: Mapping[Hashable, Tensor] = batch.get("output_mask", {})
        y_native: Mapping[Hashable, Tensor] | None = batch.get("output_native", None)

        if not preds:
            return torch.tensor(0.0, dtype=torch.float32), {}

        ref = next(iter(preds.values()))
        device = ref.device

        term_losses: list[Tensor] = []
        term_weights: list[float] = []
        logs: dict[str, Any] = {}

        for i, (term, weight) in enumerate(self._terms):
            term_loss, term_logs = term.compute(
                preds=preds,
                y_emb=y_emb,
                y_native=y_native,
                output_mask=output_mask,
            )

            term_name = type(term).__name__
            key_prefix = f"{term_name}_{i}" if len(self._terms) > 1 else term_name

            if not math.isfinite(float(term_loss.detach().cpu())):
                logs[f"{key_prefix}/total"] = float("nan")
            else:
                logs[f"{key_prefix}/total"] = float(term_loss.detach().cpu())
            for out_key, out_val in term_logs.items():
                logs[f"{key_prefix}/{out_key}"] = out_val

            term_losses.append(term_loss)
            term_weights.append(weight)

        stacked = torch.stack(term_losses)
        weights_t = torch.tensor(term_weights, device=device, dtype=torch.float32)

        w_sum = float(weights_t.sum())
        if w_sum > 0.0:
            total = (stacked * weights_t).sum() / w_sum
        else:
            total = stacked.mean()

        logs["total"] = float(total.detach().cpu())

        return total, logs


# ======================================================================================================================
# Factory
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_loss_aggregator(
    loss_cfg: Mapping[str, Any],
    output_weights_by_id: Mapping[Hashable, float] | None = None,
    decoders: dict[Hashable, TorchDecoder] | None = None,
) -> LossAggregator:
    """
    Build a ``LossAggregator`` from the ``train.loss`` config block.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        The ``train.loss`` config dict. If ``terms`` is absent, defaults to a single ``embed_mse`` term.
    output_weights_by_id : Mapping[Hashable, float] | None
        Per-output weights keyed by signal_id (int). Applied to embed_mse terms.
    decoders : dict[Hashable, TorchDecoder] | None
        Per-signal torch decoders, required for ``native_sparse_mse`` terms.

    Returns
    -------
    LossAggregator

    Raises
    ------
    ValueError
        If an unknown term type is encountered.
        If a ``native_sparse_mse`` term is requested but ``decoders`` is None or empty.

    """

    terms_cfg: list[dict[str, Any]] = loss_cfg.get("terms", [{"type": "embed_mse", "weight": 1.0}])
    ow = dict(output_weights_by_id) if output_weights_by_id else {}

    built: list[tuple[BaseLoss, float]] = []

    for term_def in terms_cfg:
        term_type = str(term_def.get("type", "embed_mse"))
        term_weight = float(term_def.get("weight", 1.0))

        if term_type == "embed_mse":
            built.append((EmbedMSELoss(output_weights=ow if ow else None), term_weight))

        elif term_type == "native_sparse_mse":
            if not decoders:
                raise ValueError(
                    "Loss term 'native_sparse_mse' requires decoders to be provided, but got None or empty dict. "
                    "Build and pass a dict[signal_id, TorchDecoder] when using this term."
                )
            built.append((NativeSparseMSELoss(decoders=decoders, output_weights=ow if ow else None), term_weight))

        else:
            raise ValueError(f"Unknown loss term type '{term_type}'. Supported: 'embed_mse', 'native_sparse_mse'.")

    return LossAggregator(terms=built)
