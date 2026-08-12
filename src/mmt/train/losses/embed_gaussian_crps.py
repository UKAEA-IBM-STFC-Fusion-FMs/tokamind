"""
Embedding-space closed-form Gaussian CRPS loss term.

`EmbedGaussianCRPSLoss` scores a Gaussian output head with the Continuous Ranked Probability Score (CRPS) directly in
embedding (coeff) space, against `output_emb`. CRPS is a proper scoring rule for probabilistic forecasts; for a
Gaussian predictive distribution it has a closed form, so this term needs no sampling and no decoder.

Each embedding coordinate is treated as an independent univariate Gaussian `N(mu, sigma^2)`, matching the diagonal
Gaussian produced by `ProbabilisticOutputAdapter`. The closed-form per-coordinate CRPS is

    CRPS(N(mu, sigma^2), y) = sigma * [ omega * (2 * Phi(omega) - 1) + 2 * phi(omega) - 1 / sqrt(pi) ]

with omega = (y - mu) / sigma, where Phi and phi are the standard-normal CDF and PDF. The term averages CRPS over
embedding coordinates and over supervised samples, mirroring `EmbedMSELoss`'s reduction and weighting so the two are
drop-in comparable.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Hashable

import torch
from torch import Tensor

from mmt.constants import FLOAT_STABILITY_EPS
from mmt.distributions import GAUSSIAN_DIST_FAMILY, PRED_DIST_FAMILY_KEY
from mmt.scoring.crps import gaussian_crps

from .base import BaseLoss, LossComputeContext


# ======================================================================================================================
class EmbedGaussianCRPSLoss(BaseLoss):
    """
    Closed-form Gaussian CRPS computed in prediction (embedding / coeff) space.

    Consumes the per-output Gaussian parameters (`pred_dist`) produced by a probabilistic output head and scores them
    against the embedding-space targets `output_emb`. No sampling and no decoding are involved, so this is the
    preferred embedding-space probabilistic objective and works for any embedding type.

    Parameters
    ----------
    output_weights : dict[Hashable, float] | None
        Optional per-output scalar weights (signal_id → weight). If provided and their sum is > 0, the per-output
        losses are combined as a normalized weighted average. Otherwise, a uniform mean is used.
    output_filter : set[Hashable] | None
        Optional set of signal IDs supervised by this term.

    """

    requires_native_target: bool = False
    requires_decode: bool = False
    requires_destandardize: bool = False
    requires_pred_dist: bool = True

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        output_weights: dict[Hashable, float] | None = None,
        output_filter: set[Hashable] | None = None,
    ) -> None:
        self._output_weights = output_weights or {}
        self._output_filter = set(output_filter) if output_filter is not None else None

    # ------------------------------------------------------------------------------------------------------------------
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
        Compute masked closed-form Gaussian CRPS in embedding space.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Predictive means (`mu`) in coeff space, keyed by signal_id. Shape: `(B, D)`. Used only to anchor the
            output device; the CRPS reads `mu`/`sigma` from `pred_dist`.
        y_emb : Mapping[Hashable, Tensor]
            Ground-truth tensors in coeff space, keyed by signal_id. Shape: `(B, D)`.
        y_native : Mapping[Hashable, Tensor] | None
            Unused. May be None.
        output_mask : Mapping[Hashable, Tensor]
            Boolean mask tensors of shape `(B,)`, True for supervised samples.
        pred_dist : Mapping[Hashable, Mapping[str, Any]] | None
            Per-output Gaussian parameters with at least `"mu"` and `"sigma"`, each shaped `(B, D)`. Required.
        context : LossComputeContext | None
            Unused metadata context. May be None.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            `(loss_scalar, per_output_logs)`

        Raises
        ------
        RuntimeError
            If `preds` is empty, which indicates that the model produced no output predictions.
            If `pred_dist` is None, which indicates the model lacks a Gaussian output head.
            If `output_mask` is not a bool tensor.
            If there is a shape mismatch between `pred_dist` parameters and the corresponding `y_emb` item.

        """

        if not preds:
            raise RuntimeError("EmbedGaussianCRPSLoss received empty predictions from the model.")

        if pred_dist is None:
            raise RuntimeError(
                "EmbedGaussianCRPSLoss requires pred_dist, but none was provided. "
                "Build the model with a probabilistic (gaussian) output adapter."
            )

        ref = next(iter(preds.values()))
        device = ref.device

        per_out_losses: list[Tensor] = []
        per_out_weights: list[float] = []
        logs: dict[Hashable, float] = {}

        for out_key, dist in pred_dist.items():
            if (self._output_filter is not None) and (out_key not in self._output_filter):
                continue

            if (out_key not in y_emb) or (out_key not in output_mask):
                continue

            mask = output_mask[out_key]
            if mask.dtype != torch.bool:
                raise RuntimeError(f"[{out_key!r}] output_mask must be bool tensor, got {mask.dtype}.")

            if not bool(mask.any()):
                continue

            family = dist.get(PRED_DIST_FAMILY_KEY)
            if family != GAUSSIAN_DIST_FAMILY:
                raise RuntimeError(
                    f"[{out_key!r}] EmbedGaussianCRPSLoss scores a Gaussian distribution but pred_dist has "
                    f"family={family!r}. This term is only valid for a Gaussian output head."
                )

            mu = dist["mu"].to(torch.float32)
            sigma = dist["sigma"].to(torch.float32)
            y_t = y_emb[out_key].to(torch.float32)
            if mu.shape != y_t.shape:
                raise RuntimeError(f"[{out_key!r}] pred/label shape mismatch: {tuple(mu.shape)} vs {tuple(y_t.shape)}.")

            crps = gaussian_crps(mu=mu, sigma=sigma, y=y_t)  # (B, D)
            L_o = crps.mean(dim=1)[mask].mean()  # NOSONAR - Ignore lowercase warning

            w_o = float(self._output_weights.get(out_key, 1.0))
            per_out_losses.append(L_o)
            per_out_weights.append(w_o)
            logs[out_key] = float(L_o.detach().cpu())

        if not per_out_losses:
            # ref.sum() * 0.0 is zero but stays in the computation graph, so backward() does not crash when
            # embed_gaussian_crps is the only active loss term.
            return ref.sum() * 0.0, logs

        per_out = torch.stack(per_out_losses)
        weights = torch.tensor(per_out_weights, device=device, dtype=torch.float32)

        if float(weights.sum()) > 0.0:
            loss = (per_out * (weights / (weights.sum() + FLOAT_STABILITY_EPS))).sum()
        else:
            loss = per_out.mean()

        return loss, logs
