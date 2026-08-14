"""
Loss aggregator for the Multi-Modal Transformer.

This module provides:
    • `LossAggregator`        — combines multiple `BaseLoss` terms into a single weighted scalar loss.
    • `build_loss_aggregator` — factory that instantiates an aggregator from the `train.loss` config block.

Each term contributes an independent scalar loss; the aggregator combines them as a normalized weighted sum across
terms. Per-output weights (if any) are handled inside each individual term.

The `build_loss_aggregator` factory supports the following term types:
    • `embed_mse`         — MSE in embedding (coeff) space; no decoding required.
    • `native_sparse_mse` — MSE in native standardized space; requires pre-built torch decoders.
    • `native_sparse_sample_crps` — sample-CRPS in native standardized space; requires decoders + gaussian head.
    • `grad_shafranov_residual` — Grad-Shafranov residual norm in native destandardized space.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Hashable, Literal, cast

import torch
from torch import Tensor

from mmt.train.losses.constants import (
    ALL_LOSS_TYPES,
    DEFAULT_CRPS_N_SAMPLES,
    DEFAULT_LOSS_TERMS,
    EMBED_GAUSSIAN_CRPS_LOSS_TYPE,
    EMBED_MSE_LOSS_TYPE,
    GRAD_SHAFRANOV_LOSS_TYPE,
    GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY,
    GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE,
    NATIVE_SPARSE_MSE_LOSS_TYPE,
    NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE,
)

from .base import BaseLoss, LossComputeContext
from .embed_mse import EmbedMSELoss
from .embed_gaussian_crps import EmbedGaussianCRPSLoss
from .grad_shafranov import GradShafranovResidualLoss, WeakFormGradShafranovLoss
from .native_sparse_mse import NativeSparseMSELoss
from .native_sparse_sample_crps import NativeSparseSampleCRPSLoss

if TYPE_CHECKING:
    from mmt.data.embeddings.torch_decoder import TorchDecoder


# ======================================================================================================================
class LossAggregator:
    """
    Combine multiple loss terms into a single weighted scalar.

    Each term is weighted by its own term-level weight before aggregation. Per-output weights (if any) are handled
    inside each term.

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
    @property
    def requires_pred_dist(self) -> bool:
        """True if any term needs the per-output predictive distribution (`pred_dist`)."""
        return any(t.requires_pred_dist for t, _ in self._terms)

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        batch: Mapping[str, Any],
        pred_dist: Mapping[Hashable, Mapping[str, Any]] | None = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[str, Any]]:
        """
        Compute the aggregated loss for one batch.

        Parameters
        ----------
        preds:
            Model predictions in embedding space, keyed by signal_id. Shape: `(B, D)` per key.
            For probabilistic output heads this is the predictive mean (`mu`).
        batch:
            Collated batch dict. Expected keys: `output_emb`, `output_mask`, and optionally `output_native` (when
            any term has `requires_native_target=True`).
        pred_dist:
            Per-output predictive distribution parameters from a Gaussian output head, keyed by signal_id (each value a
            mapping with `"mu"`, `"sigma"`, ...). Forwarded only to terms with `requires_pred_dist=True`. None for
            deterministic output heads.
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
            If a term requires `pred_dist` but none was provided (model lacks a Gaussian output head).

        """

        y_emb: dict[Hashable, Tensor] = batch.get("output_emb", {})
        output_mask: dict[Hashable, Tensor] = batch.get("output_mask", {})
        y_native: dict[Hashable, Tensor] | None = batch.get("output_native", None)

        if not preds:
            raise RuntimeError("LossAggregator received empty predictions from the model.")

        ref = next(iter(preds.values()))
        device = ref.device

        w_sum = sum(w for _, w in self._terms)
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

            # Forward pred_dist only to terms that declare they need it (e.g. CRPS); deterministic terms keep their
            # original signature and are never passed it.
            if term.requires_pred_dist:
                if pred_dist is None:
                    raise RuntimeError(
                        f"Loss term '{type(term).__name__}' requires pred_dist, but the model produced none. "
                        "Build the model with a probabilistic (gaussian) output adapter."
                    )
                term_loss, term_logs = term.compute(
                    preds=preds,
                    y_emb=y_emb,
                    y_native=y_native,
                    output_mask=output_mask,
                    pred_dist=pred_dist,
                    context=term_context,
                )
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
            # Weighted contribution: w_i * L_i / Σw_j — gradient share after term weights.
            logs[f"{key_prefix}/weighted"] = (raw * weight / w_sum) if w_sum > 0.0 else (raw / n_terms)
            for out_key, out_val in term_logs.items():
                logs[f"{key_prefix}/{out_key}"] = out_val

            term_losses.append(term_loss)

        stacked = torch.stack(term_losses)
        weights_t = torch.tensor([w for _, w in self._terms], device=device, dtype=torch.float32)
        total = (stacked * weights_t).sum() / w_sum if (w_sum > 0.0) else stacked.mean()

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
        Mapping from configured output signal names to runtime prediction keys. Required for Grad-Shafranov terms.
    signal_stats : Mapping[str, Mapping[str, Any]] | None
        Per-signal mean/std metadata. Required for Grad-Shafranov terms.

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
        # Embed gaussian CRPS loss term

        elif term_type == EMBED_GAUSSIAN_CRPS_LOSS_TYPE:
            built.append(
                (
                    EmbedGaussianCRPSLoss(
                        output_weights=ow if ow else None,
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

        # ..............................................................................................................
        # Native sparse sample CRPS loss term

        elif term_type == NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE:
            if not decoders:
                raise ValueError(
                    f"Loss term '{NATIVE_SPARSE_SAMPLE_CRPS_LOSS_TYPE}' requires decoders to be provided, "
                    "but got None or empty dict. "
                    "Build and pass a dict[signal_id, TorchDecoder] when using this term."
                )
            n_samples = int(term_def.get("n_samples", DEFAULT_CRPS_N_SAMPLES))
            built.append(
                (
                    NativeSparseSampleCRPSLoss(
                        decoders=decoders,
                        n_samples=n_samples,
                        output_weights=ow if ow else None,
                        output_filter=output_filter,
                    ),
                    term_weight,
                )
            )

        # ..............................................................................................................
        # Strong Grad-Shafranov loss term

        elif term_type == GRAD_SHAFRANOV_LOSS_TYPE:
            if not decoders:
                raise ValueError(
                    f"Loss term '{GRAD_SHAFRANOV_LOSS_TYPE}' requires decoders to be provided, "
                    "but got None or empty dict. "
                    "Build and pass a dict[signal_id, TorchDecoder] when using this term."
                )
            if output_name_to_id is None:
                raise ValueError(f"Loss term '{GRAD_SHAFRANOV_LOSS_TYPE}' requires output_name_to_id.")

            if signal_stats is None:
                raise ValueError(f"Loss term '{GRAD_SHAFRANOV_LOSS_TYPE}' requires signal_stats.")

            # Config shape (params file presence/type, rhs_input keys/values) is validated up-front in
            # validator._validate_loss_terms, and the constructor backstops direct instantiation; here we just read.
            rhs_input_cfg = term_def.get("rhs_input") or {}
            plot_check_cfg = term_def.get("plot_check") or {}
            all_losses_weights = {term_["type"]: term_["weight"] for term_ in terms_cfg}
            plot_check_cfg["all_losses_weights"] = all_losses_weights

            built.append(
                (
                    GradShafranovResidualLoss(
                        decoders=decoders,
                        signal_stats=signal_stats,
                        output_name_to_id=output_name_to_id,
                        grad_shafranov_params_file=term_def.get("grad_shafranov_params_file"),
                        grad_shafranov_weights=term_def.get("grad_shafranov_weights"),
                        rhs_input=rhs_input_cfg.get(GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY),
                        j_tor_calculation_method=rhs_input_cfg.get(GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY),
                        loss_type=term_def.get("loss_type", "mse"),
                        output_weights=ow if ow else None,
                        output_filter=output_filter,
                        plot_check_cfg=plot_check_cfg,
                    ),
                    term_weight,
                )
            )
        # ..............................................................................................................
        # Weak Grad-Shafranov loss term

        elif term_type == GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE:
            if not decoders:
                raise ValueError(f"Loss term '{GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE}' requires decoders.")

            if output_name_to_id is None:
                raise ValueError(f"Loss term '{GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE}' requires output_name_to_id.")

            if signal_stats is None:
                raise ValueError(f"Loss term '{GRAD_SHAFRANOV_WEAK_FORM_LOSS_TYPE}' requires signal_stats.")

            plot_check_cfg = term_def.get("plot_check") or {}
            rhs_input_cfg = term_def.get("rhs_input") or {}
            weak_loss_type_raw = term_def.get("loss_type") or "mse"
            if weak_loss_type_raw not in {"l2", "mse"}:
                raise ValueError(f"Invalid weak-form GS loss_type={weak_loss_type_raw!r}. Expected 'l2' or 'mse'.")

            weak_loss_type = cast(Literal["l2", "mse"], weak_loss_type_raw)
            weak_gs_weights = cast(dict[str, float] | None, term_def.get("grad_shafranov_weights"))
            built.append(
                (
                    WeakFormGradShafranovLoss(
                        decoders=decoders,
                        signal_stats=signal_stats,
                        output_name_to_id=output_name_to_id,
                        grad_shafranov_params_file=term_def.get("grad_shafranov_params_file"),
                        grad_shafranov_weights=weak_gs_weights,
                        rhs_input=rhs_input_cfg.get(GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY),
                        loss_type=weak_loss_type,
                        output_weights=ow if ow else None,
                        output_filter=output_filter,
                        plot_check_type=plot_check_cfg.get("type"),
                        plot_check_probability=plot_check_cfg.get("probability"),
                    ),
                    term_weight,
                )
            )

        else:
            raise ValueError(f"Unknown loss term type '{term_type}'. Supported: {sorted(ALL_LOSS_TYPES)}.")

    return LossAggregator(terms=built, term_configs=terms_cfg)
