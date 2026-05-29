"""
Native-space sparse MSE loss term.

`NativeSparseMSELoss` computes MSE in native (standardized) signal space, skipping NaN positions in the target.
Native signals can be spatially or temporally sparse (e.g. bolometry profiles, partial Thomson scattering grids) where
missing positions are represented as NaN. The loss is computed only over valid (non-NaN) positions, matching the
sparse evaluation convention in TokaMark.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Hashable

import torch
from torch import Tensor

from mmt.constants import FLOAT_STABILITY_EPS
from mmt.data.embeddings.torch_decoder import TorchDecoder

from .base import BaseLoss


# ======================================================================================================================
class NativeSparseMSELoss(BaseLoss):
    """
    Masked MSE loss computed in native (standardized) signal space, with NaN-position masking.

    Predictions arrive in embedding space and are decoded to native space via per-signal `TorchDecoder` instances
    before the MSE is computed. Ground-truth native targets come from `batch['output_native']`, which requires
    `data.keep_output_native=True` in the dataset config.

    NaN values in the target represent missing measurements (sparse signals). The loss is computed only over
    valid (non-NaN) positions within each supervised sample, consistent with the TokaMark sparse evaluation.

    Gradients flow through the decoders into the model predictions. Decoder parameters (e.g. frozen VAE weights,
    fixed scatter indices) are kept fixed.

    Parameters
    ----------
    decoders : dict[Hashable, TorchDecoder]
        Per-signal differentiable decoders, keyed by signal_id.
    output_weights : dict[Hashable, float] | None
        Optional per-output scalar weights (signal_id → weight).

    """

    requires_native_target: bool = True
    requires_decode: bool = True
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        decoders: dict[Hashable, TorchDecoder],
        output_weights: dict[Hashable, float] | None = None,
    ) -> None:
        if not decoders:
            raise ValueError("NativeSparseMSELoss requires at least one decoder.")

        self._decoders = decoders
        self._output_weights = output_weights or {}

    # ------------------------------------------------------------------------------------------------------------------
    def compute(  # NOSONAR - Ignore cognitive complexity
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: Mapping[Hashable, Tensor],
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Decode predictions to native space and compute sparse MSE against native targets.

        NaN positions in `y_native` are excluded from the MSE — only valid measurements contribute.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Prediction tensors in embedding space, keyed by signal_id. Shape: `(B, D)`.
        y_emb : Mapping[Hashable, Tensor]
            Unused. May be None.
        y_native : Mapping[Hashable, Tensor] | None
            Ground-truth tensors in native standardized space, keyed by signal_id. Shape: `(B, *native_shape)`.
            Must be non-None when any supervised output is present.
        output_mask : Mapping[Hashable, Tensor]
            Boolean mask tensors of shape `(B,)`, True for supervised samples.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            `(loss_scalar, per_output_logs)`

        Raises
        ------
        RuntimeError
            If ``preds`` is empty, which indicates that the model produced no output predictions.
            If `output_mask` is not a bool tensor.
            If `y_native` is None when a supervised output is encountered.

        """

        if not preds:
            raise RuntimeError("NativeSparseMSELoss received empty predictions from the model.")

        ref = next(iter(preds.values()))
        device = ref.device

        per_out_losses: list[Tensor] = []
        per_out_weights: list[float] = []
        logs: dict[Hashable, float] = {}

        for out_key, y_pred in preds.items():
            if (out_key not in self._decoders) or (out_key not in output_mask):
                continue

            mask = output_mask[out_key]
            if mask.dtype != torch.bool:
                raise RuntimeError(f"[{out_key!r}] output_mask must be bool tensor, got {mask.dtype}.")

            if not bool(mask.any()):
                continue

            if (y_native is None) or (out_key not in y_native):
                raise RuntimeError(
                    f"[{out_key!r}] NativeSparseMSELoss requires batch['output_native'] but it is missing. "
                    "Ensure data.keep_output_native=True is set in the dataset config."
                )

            # Decode predictions: (B, D) → (B, *native_shape)
            decoder = self._decoders[out_key]
            pred_native = decoder(y_pred.to(torch.float32))  # Gradients flow through here

            y_nat = y_native[out_key].to(torch.float32)

            if pred_native.shape != y_nat.shape:
                raise RuntimeError(
                    f"[{out_key!r}] decoded pred shape {tuple(pred_native.shape)} != "
                    f"native target shape {tuple(y_nat.shape)}."
                )

            # MSE over valid (non-NaN) positions only, per supervised sample. NaN positions in y_nat represent missing
            # measurements (sparse signals).
            B = y_pred.shape[0]
            valid = ~torch.isnan(y_nat)  # (B, *native_shape)
            valid_flat = valid.reshape(B, -1)  # (B, N)

            # Zero out NaN positions so squaring is safe; they are excluded via n_valid below.
            diff = (pred_native - y_nat.nan_to_num(0.0)).reshape(B, -1)  # (B, N)
            sq = diff.square().masked_fill(~valid_flat, 0.0)
            n_valid = valid_flat.float().sum(dim=1).clamp(min=1.0)  # (B,) — clamp avoids 0/0
            per_sample = sq.sum(dim=1) / n_valid  # (B,)

            # Exclude samples that are supervised (mask=True) but have no valid native points (all-NaN targets). Such
            # samples have per_sample=0 not because the prediction is perfect but because there is nothing to supervise
            # — averaging them in would dilute the gradient signal.
            sample_has_valid = valid_flat.any(dim=1)  # (B,)
            active = mask & sample_has_valid
            if not bool(active.any()):
                continue

            L_o = per_sample[active].mean()  # NOSONAR

            w_o = float(self._output_weights.get(out_key, 1.0))
            per_out_losses.append(L_o)
            per_out_weights.append(w_o)
            logs[out_key] = float(L_o.detach().cpu())

        if not per_out_losses:
            # ref.sum() * 0.0 is zero but stays in the computation graph, so backward() does not crash when
            # native_sparse_mse is the only active loss term.
            return ref.sum() * 0.0, logs

        per_out = torch.stack(per_out_losses)
        weights = torch.tensor(per_out_weights, device=device, dtype=torch.float32)

        if float(weights.sum()) > 0.0:
            loss = (per_out * (weights / (weights.sum() + FLOAT_STABILITY_EPS))).sum()
        else:
            loss = per_out.mean()

        return loss, logs
