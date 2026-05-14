"""
Embedding-space MSE loss term.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Hashable, Optional

import torch
from torch import Tensor

from mmt.constants import FLOAT_STABILITY_EPS

from .base import BaseLoss


# ======================================================================================================================
class EmbedMSELoss(BaseLoss):
    """
    Masked MSE loss computed in prediction (embedding / coeff) space.

    This is the default loss term. It does not require native-space outputs and works directly on the encoded
    representations produced by the embedding pipeline.

    For orthonormal encoders (DCT/FPCA) using all coefficients, coeff-space MSE and native-space MSE differ only by a
    constant factor ``K / N`` (see ``mmt.train.losses`` module docstring for details).

    Parameters
    ----------
    output_weights : dict[Hashable, float] | None
        Optional per-output scalar weights (signal_id → weight). If provided and their sum is > 0,
        the per-output losses are combined as a normalized weighted average. Otherwise, a uniform mean is used.

    """

    requires_native_target: bool = False
    requires_decode: bool = False
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, output_weights: Optional[dict[Hashable, float]] = None) -> None:
        self._output_weights = output_weights or {}

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Optional[Mapping[Hashable, Tensor]],
        output_mask: Mapping[Hashable, Tensor],
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute masked MSE in embedding space.

        Parameters
        ----------
        preds:
            Prediction tensors in coeff space, keyed by signal_id. Shape: ``(B, D)``.
        y_emb:
            Ground-truth tensors in coeff space, keyed by signal_id. Shape: ``(B, D)``.
        y_native:
            Unused. May be ``None``.
        output_mask:
            Boolean mask tensors of shape ``(B,)``, True for supervised samples.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            ``(loss_scalar, per_output_logs)``

        """

        if not preds:
            return torch.tensor(0.0, dtype=torch.float32), {}

        ref = next(iter(preds.values()))
        device = ref.device

        per_out_losses: list[Tensor] = []
        per_out_weights: list[float] = []
        logs: dict[Hashable, float] = {}

        for out_key, y_pred in preds.items():
            if (out_key not in y_emb) or (out_key not in output_mask):
                continue

            y_t = y_emb[out_key]
            mask = output_mask[out_key]

            if y_pred.shape != y_t.shape:
                raise RuntimeError(
                    f"[{out_key!r}] pred/label shape mismatch: {tuple(y_pred.shape)} vs {tuple(y_t.shape)}."
                )

            if mask.dtype != torch.bool:
                raise RuntimeError(f"[{out_key!r}] output_mask must be bool tensor, got {mask.dtype}.")

            if not bool(mask.any()):
                continue

            y_pred_f = y_pred.to(torch.float32)
            y_t_f = y_t.to(torch.float32)
            L_o = (y_pred_f - y_t_f).square().mean(dim=1)[mask].mean()  # NOSONAR

            w_o = float(self._output_weights.get(out_key, 1.0))
            per_out_losses.append(L_o)
            per_out_weights.append(w_o)
            logs[out_key] = float(L_o.detach().cpu())

        if not per_out_losses:
            return torch.zeros((), device=device, dtype=torch.float32), logs

        per_out = torch.stack(per_out_losses)
        weights = torch.tensor(per_out_weights, device=device, dtype=torch.float32)

        if float(weights.sum()) > 0.0:
            loss = (per_out * (weights / (weights.sum() + FLOAT_STABILITY_EPS))).sum()
        else:
            loss = per_out.mean()

        return loss, logs
