"""
Native-space sparse sample-CRPS loss term.

`NativeSparseSampleCRPSLoss` is the primary native-space probabilistic objective. It draws reparameterized samples
from the Gaussian output head in embedding (coeff) space, decodes each sample through the per-signal differentiable
`TorchDecoder`, and scores the resulting native-space ensemble against the native target with the fair sample-CRPS
estimator (`mmt.scoring.crps.sample_crps`).

Because scoring happens after decoding, this term is decoder-agnostic: it works for DCT3D, VAE, and identity outputs
alike, and it does not require propagating predictive variances analytically through the decoder. CRPS is computed
marginally per native position, over valid (non-NaN) positions only, using the same NaN sparse-masking convention as
`NativeSparseMSELoss`.

Gradients flow through the reparameterized samples and the decoders into the model's `mu`/`sigma`. Decoder parameters
(e.g. frozen VAE weights, fixed scatter indices) are kept fixed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Hashable

import torch
from torch import Tensor

from mmt.constants import FLOAT_STABILITY_EPS
from mmt.data.embeddings.torch_decoder import TorchDecoder
from mmt.distributions import GAUSSIAN_DIST_FAMILY, PRED_DIST_FAMILY_KEY, sample_gaussian_dist

from mmt.scoring.crps import sample_crps

from .base import BaseLoss, LossComputeContext


# ======================================================================================================================
class NativeSparseSampleCRPSLoss(BaseLoss):
    """
    Sample-CRPS loss computed in native (standardized) signal space, with NaN-position masking.

    Per output, `n_samples` reparameterized draws `mu + sigma * eps` are taken in embedding space from the Gaussian
    head's `pred_dist`, decoded to native space via the per-signal `TorchDecoder`, and scored against the native target
    with the fair (unbiased) marginal CRPS estimator. NaN positions in the target represent missing measurements and
    are excluded, consistent with `NativeSparseMSELoss` and the TokaMark sparse evaluation.

    Parameters
    ----------
    decoders : dict[Hashable, TorchDecoder]
        Per-signal differentiable decoders, keyed by signal_id.
    n_samples : int
        Number of reparameterized samples per output. Must be >= 2 (the CRPS cross-term is undefined for one sample).
    output_weights : dict[Hashable, float] | None
        Optional per-output scalar weights (signal_id → weight).
    output_filter : set[Hashable] | None
        Optional set of signal IDs supervised by this term.
    generator : torch.Generator | None
        Optional random generator for the sampling noise. Must live on the model's device. Default None (fresh
        randomness each step).

    """

    requires_native_target: bool = True
    requires_decode: bool = True
    requires_destandardize: bool = False
    requires_pred_dist: bool = True

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        decoders: dict[Hashable, TorchDecoder],
        n_samples: int,
        output_weights: dict[Hashable, float] | None = None,
        output_filter: set[Hashable] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if not decoders:
            raise ValueError("NativeSparseSampleCRPSLoss requires at least one decoder.")
        if int(n_samples) < 2:
            raise ValueError(f"NativeSparseSampleCRPSLoss requires n_samples >= 2, got {n_samples}.")

        self._decoders = decoders
        self._n_samples = int(n_samples)
        self._output_weights = output_weights or {}
        self._output_filter = set(output_filter) if output_filter is not None else None
        self._generator = generator

    # ------------------------------------------------------------------------------------------------------------------
    def compute(  # NOSONAR - Ignore cognitive complexity
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: Mapping[Hashable, Tensor],
        pred_dist: Mapping[Hashable, Mapping[str, Any]] | None = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Sample from pred_dist, decode each sample to native space, and compute sparse marginal CRPS.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Predictive means (`mu`) in coeff space, keyed by signal_id. Shape: `(B, D)`. Used only to anchor the
            output device.
        y_emb : Mapping[Hashable, Tensor]
            Unused. May be None.
        y_native : Mapping[Hashable, Tensor] | None
            Ground-truth tensors in native standardized space, keyed by signal_id. Shape: `(B, *native_shape)`.
            Must be non-None when any supervised output is present.
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
            If `y_native` is None or missing for a supervised output.
            If a decoded sample's native shape does not match the native target shape.

        """

        if not preds:
            raise RuntimeError("NativeSparseSampleCRPSLoss received empty predictions from the model.")

        if pred_dist is None:
            raise RuntimeError(
                "NativeSparseSampleCRPSLoss requires pred_dist, but none was provided. "
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

            if (out_key not in self._decoders) or (out_key not in output_mask):
                continue

            mask = output_mask[out_key]
            if mask.dtype != torch.bool:
                raise RuntimeError(f"[{out_key!r}] output_mask must be bool tensor, got {mask.dtype}.")

            if not bool(mask.any()):
                continue

            if (y_native is None) or (out_key not in y_native):
                raise RuntimeError(
                    f"[{out_key!r}] NativeSparseSampleCRPSLoss requires batch['output_native'] but it is missing. "
                    "Ensure data.keep_output_native=True is set in the dataset config."
                )

            family = dist.get(PRED_DIST_FAMILY_KEY)
            if family != GAUSSIAN_DIST_FAMILY:
                raise RuntimeError(
                    f"[{out_key!r}] NativeSparseSampleCRPSLoss samples a Gaussian distribution but pred_dist has "
                    f"family={family!r}. This term is only valid for a Gaussian output head."
                )

            S = self._n_samples  # noqa - Ignore lowercase warning

            # Sample in float32 (pred_dist may be lower precision under autocast; the loss runs outside it) using the
            # shared Gaussian sampler, so training and eval draw mu + sigma * eps identically. Gradients flow to
            # mu/sigma. Reparameterized samples in embedding space: (B, S, K).
            dist_f32 = {**dist, "mu": dist["mu"].to(torch.float32), "sigma": dist["sigma"].to(torch.float32)}
            samples_emb = sample_gaussian_dist(dist_f32, n_samples=S, generator=self._generator)  # (B, S, K)
            B = samples_emb.shape[0]  # noqa - Ignore lowercase warning

            # Decode each sample to native space, looping over S to bound peak memory. Each decode is differentiable.
            decoder = self._decoders[out_key]
            y_nat = y_native[out_key].to(torch.float32)  # (B, *native_shape)
            decoded_samples = []
            for s in range(S):
                pred_native_s = decoder(samples_emb[:, s, :])  # (B, *native_shape)
                if pred_native_s.shape != y_nat.shape:
                    raise RuntimeError(
                        f"[{out_key!r}] decoded sample shape {tuple(pred_native_s.shape)} != "
                        f"native target shape {tuple(y_nat.shape)}."
                    )
                decoded_samples.append(pred_native_s)
            samples_native = torch.stack(decoded_samples, dim=1)  # (B, S, *native_shape)

            # Flatten native dims and compute fair marginal CRPS per position over valid (non-NaN) targets only.
            samples_flat = samples_native.reshape(B, S, -1)  # (B, S, N)
            valid_flat = (~torch.isnan(y_nat)).reshape(B, -1)  # (B, N)
            y_flat = y_nat.nan_to_num(0.0).reshape(B, -1)  # (B, N) — NaN→0; excluded via mask below

            crps_pos = sample_crps(samples_flat, y_flat, sample_dim=1)  # (B, N)
            crps_pos = crps_pos.masked_fill(~valid_flat, 0.0)
            n_valid = valid_flat.float().sum(dim=1).clamp(min=1.0)  # (B,) — clamp avoids 0/0
            per_sample = crps_pos.sum(dim=1) / n_valid  # (B,)

            # Exclude supervised samples with no valid native points (all-NaN targets), matching NativeSparseMSELoss.
            sample_has_valid = valid_flat.any(dim=1)  # (B,)
            active = mask & sample_has_valid
            if not bool(active.any()):
                continue

            L_o = per_sample[active].mean()  # NOSONAR - Ignore lowercase warning

            w_o = float(self._output_weights.get(out_key, 1.0))
            per_out_losses.append(L_o)
            per_out_weights.append(w_o)
            logs[out_key] = float(L_o.detach().cpu())

        if not per_out_losses:
            # ref.sum() * 0.0 is zero but stays in the computation graph, so backward() does not crash when this is the
            # only active loss term.
            return ref.sum() * 0.0, logs

        per_out = torch.stack(per_out_losses)
        weights = torch.tensor(per_out_weights, device=device, dtype=torch.float32)

        if float(weights.sum()) > 0.0:
            loss = (per_out * (weights / (weights.sum() + FLOAT_STABILITY_EPS))).sum()
        else:
            loss = per_out.mean()

        return loss, logs
