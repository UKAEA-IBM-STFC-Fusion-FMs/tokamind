"""
Loss aggregator for the Multi-Modal Transformer.

This module provides:
    • `LossAggregator`        — combines multiple `BaseLoss` terms into a single weighted scalar loss.
    • `build_loss_aggregator` — factory that instantiates an aggregator from the `train.loss` config block.

Each term contributes an independent scalar loss; the aggregator combines them as a normalized weighted sum across
active terms for the current batch. Per-output weights (if any) are handled inside each individual term.

The `build_loss_aggregator` factory supports the following term types:
    • `embed_mse`         — MSE in embedding (coeff) space; no decoding required.
    • `native_sparse_mse` — MSE in native standardized space; requires pre-built torch decoders.
    • `continuous_gpo`    — Continuous preference optimization in embedding space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Hashable

import torch
from torch import Tensor

from mmt.train.losses.constants import (
    ALL_LOSS_TYPES,
    CONTINUOUS_GPO_LOSS_TYPE,
    DEFAULT_LOSS_TERMS,
    EMBED_MSE_LOSS_TYPE,
    NATIVE_SPARSE_MSE_LOSS_TYPE,
)

from .base import BaseLoss, LossComputeContext
from .continuous_gpo import ContinuousGPOLoss, gpo_preference_accuracy
from .embed_mse import EmbedMSELoss
from .native_sparse_mse import NativeSparseMSELoss

if TYPE_CHECKING:
    from mmt.data.embeddings.torch_decoder import TorchDecoder


# ======================================================================================================================
class LossAggregator:
    """
    Combine multiple loss terms into a single weighted scalar.

    Each term is weighted by its own term-level weight before aggregation. Terms that are inactive for the current batch
    are excluded from normalization. Per-output weights (if any) are handled inside each term.

    Parameters
    ----------
    terms : list[tuple[BaseLoss, float]]
        List of `(loss_term, term_weight)` pairs. Weights do not need to sum to 1.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        terms: list[tuple[BaseLoss, float]],
        term_configs: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        if not terms:
            raise ValueError("LossAggregator requires at least one loss term.")

        self._terms = terms
        self._term_configs = list(term_configs or [{} for _ in terms])
        if len(self._term_configs) != len(self._terms):
            raise ValueError(f"Expected {len(self._terms)} term configs, got {len(self._term_configs)}.")
        self._stage_loss_terms = tuple(self._term_configs)

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def requires_native_target(self) -> bool:
        """True if any term needs `batch['output_native']`."""
        return any(t.requires_native_target for t, _ in self._terms)

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        batch: Mapping[str, Any],
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Compute the aggregated loss for one batch.

        Parameters
        ----------
        preds:
            Model predictions in embedding space, keyed by signal_id. Shape: `(B, D)` per key.
        batch:
            Collated batch dict. Expected keys: `output_emb`, `output_mask`, and optionally `output_native` (when
            any term has `requires_native_target=True`).
        context:
            Optional metadata for logging/plotting. The aggregator augments it with per-term weight/config before
            forwarding it to each loss term.

        Returns
        -------
        tuple[Tensor, dict]
            `(total_loss, logs)` where logs contains per-term and per-output loss values.

        Raises
        ------
        RuntimeError
            If ``preds`` is empty, which indicates that the model produced no output predictions.

        """

        y_emb: dict[Hashable, Tensor] = batch.get("output_emb", {})
        output_mask: dict[Hashable, Tensor] = batch.get("output_mask", {})
        y_native: dict[Hashable, Tensor] | None = batch.get("output_native", None)
        # GPO pair embeddings and per-row pair masks — present only when a GPO
        # DataLoader is active.
        y_w_emb: dict[Hashable, Tensor] | None = batch.get("y_w_emb", None)
        y_l_emb: dict[Hashable, Tensor] | None = batch.get("y_l_emb", None)
        gpo_pair_mask: dict[Hashable, Tensor] | None = batch.get("gpo_pair_mask", None)
        # Reference model predictions — present only when use_reference_model=true.
        ref_preds: dict[Hashable, Tensor] | None = batch.get("ref_preds", None)

        if not preds:
            raise RuntimeError("LossAggregator received empty predictions from the model.")

        ref = next(iter(preds.values()))
        device = ref.device

        # A GPO term is inactive when the batch contains no actual preference
        # pairs.  Exclude that term from normalization so an unpaired batch
        # receives the configured full-strength reconstruction loss rather than
        # embed_mse / (gpo_weight + embed_mse_weight).  Retain the legacy
        # y_l_emb check when no explicit pair mask is supplied.
        if gpo_pair_mask is None:
            has_gpo_pairs = bool(y_l_emb)
        else:
            has_gpo_pairs = any(bool(mask.any()) for mask in gpo_pair_mask.values())
        active_weights = [
            weight if (not isinstance(term, ContinuousGPOLoss) or has_gpo_pairs) else 0.0
            for term, weight in self._terms
        ]
        active_weight_sum = sum(active_weights)
        n_terms = len(self._terms)

        term_losses: list[Tensor] = []
        logs: dict[str, Any] = {}

        for i, (term, weight) in enumerate(self._terms):
            term_name = type(term).__name__
            base_context = context or LossComputeContext()
            term_context = replace(
                base_context,
                term_index=i,
                term_name=term_name,
                term_weight=float(weight),
                term_config=self._term_configs[i],
                stage_loss_terms=self._stage_loss_terms,
            )

            if isinstance(term, ContinuousGPOLoss):
                # GPO terms receive y_w_emb / y_l_emb from the batch in addition to
                # the standard compute arguments.
                term_loss, term_logs = term.compute(
                    preds=preds,
                    y_emb=y_emb,
                    y_native=y_native,
                    output_mask=output_mask,
                    context=term_context,
                    y_w_emb=y_w_emb,
                    y_l_emb=y_l_emb,
                    gpo_pair_mask=gpo_pair_mask,
                    ref_preds=ref_preds,
                )
                # Preference accuracy: fraction of windows where the current model
                # prediction is closer to y_w than to y_l.  Pure diagnostic — no
                # gradient.  Only computed when GPO pairs are present in the batch.
                if y_w_emb and y_l_emb:
                    for sig_id, acc in gpo_preference_accuracy(
                        preds=preds,
                        y_w_emb=y_w_emb,
                        y_l_emb=y_l_emb,
                        output_mask=output_mask,
                        pair_mask=gpo_pair_mask,
                    ).items():
                        term_logs[f"pref_acc/{sig_id}"] = acc
            else:
                term_loss, term_logs = term.compute(
                    preds=preds,
                    y_emb=y_emb,
                    y_native=y_native,
                    output_mask=output_mask,
                    context=term_context,
                )

            key_prefix = f"{term_name}_{i}" if (n_terms > 1) else term_name

            raw = float(term_loss.detach().cpu())
            # Weighted contribution uses only terms active for this batch.  In
            # particular, no-pair batches normalize embed_mse by its own weight.
            active_weight = active_weights[i]
            logs[f"{key_prefix}/weighted"] = (
                raw * active_weight / active_weight_sum if active_weight_sum > 0.0 else (raw / n_terms)
            )
            for out_key, out_val in term_logs.items():
                logs[f"{key_prefix}/{out_key}"] = out_val

            term_losses.append(term_loss)

        stacked = torch.stack(term_losses)
        weights_t = torch.tensor(active_weights, device=device, dtype=torch.float32)
        total = (stacked * weights_t).sum() / active_weight_sum if active_weight_sum > 0.0 else stacked.mean()

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
    term_output_filters: Sequence[set[Hashable] | None] | None = None,
    output_name_to_id: Mapping[str, Hashable] | None = None,
    signal_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> LossAggregator:
    """
    Build a `LossAggregator` from the `train.loss` config block.

    Parameters
    ----------
    loss_cfg : Mapping[str, Any]
        The `train.loss` config dict. If `terms` is absent, defaults to a single `embed_mse` term.
    output_weights_by_id : Mapping[Hashable, float] | None
        Per-output weights keyed by signal_id (int). Applied to embed_mse terms.
    decoders : dict[Hashable, TorchDecoder] | None
        Per-signal torch decoders, required for ``native_sparse_mse`` terms.
    term_output_filters : Sequence[set[Hashable] | None] | None
        Optional per-term signal ID filters, aligned with ``loss_cfg["terms"]``.
    output_name_to_id : Mapping[str, Hashable] | None
        Mapping from configured output signal names to runtime prediction keys. Reserved metadata for loss extensions.
    signal_stats : Mapping[str, Mapping[str, Any]] | None
        Per-signal mean/std metadata. Reserved metadata for loss extensions.

    Returns
    -------
    LossAggregator

    Raises
    ------
    ValueError
        If an unknown term type is encountered.
        If a native-space term is requested but `decoders` is None or empty.

    """

    terms_cfg: list[Mapping[str, Any]] = list(loss_cfg.get("terms", DEFAULT_LOSS_TERMS))
    ow = dict(output_weights_by_id) if output_weights_by_id else {}
    output_filters = [None] * len(terms_cfg) if term_output_filters is None else list(term_output_filters)
    if len(output_filters) != len(terms_cfg):
        raise ValueError(f"Expected {len(terms_cfg)} loss output filters, got {len(output_filters)}.")

    built: list[tuple[BaseLoss, float]] = []

    for term_def, output_filter in zip(terms_cfg, output_filters, strict=True):
        term_type = str(term_def.get("type", EMBED_MSE_LOSS_TYPE))
        term_weight = float(term_def.get("weight", 1.0))

        # ..............................................................................................................
        # Embed MSE loss term

        if term_type == EMBED_MSE_LOSS_TYPE:
            built.append((EmbedMSELoss(output_weights=ow if ow else None, output_filter=output_filter), term_weight))

        # ..............................................................................................................
        # Continuous GPO loss term

        elif term_type == CONTINUOUS_GPO_LOSS_TYPE:
            _clip = term_def.get("mse_gap_clip", None)
            built.append(
                (
                    ContinuousGPOLoss(
                        beta=float(term_def.get("beta", 0.1)),
                        loss_type=str(term_def.get("loss_type", "dpo")),
                        sft_weight=float(term_def.get("sft_weight", 0.0)),
                        mse_gap_clip=float(_clip) if _clip is not None else None,
                        log_mse_gap=bool(term_def.get("log_mse_gap", False)),
                        output_filter=output_filter,
                    ),
                    term_weight,
                )
            )

        # ..............................................................................................................
        # Native sparse MSE loss term

        elif term_type == NATIVE_SPARSE_MSE_LOSS_TYPE:
            if not decoders:
                raise ValueError(
                    f"Loss term '{NATIVE_SPARSE_MSE_LOSS_TYPE}' requires decoders to be provided, "
                    "but got None or empty dict. "
                    "Build and pass a dict[signal_id, TorchDecoder] when using this term."
                )
            built.append(
                (
                    NativeSparseMSELoss(
                        decoders=decoders,
                        output_weights=ow if ow else None,
                        output_filter=output_filter,
                    ),
                    term_weight,
                )
            )

        else:
            raise ValueError(f"Unknown loss term type '{term_type}'. Supported: {sorted(ALL_LOSS_TYPES)}.")

    return LossAggregator(terms=built, term_configs=terms_cfg)
