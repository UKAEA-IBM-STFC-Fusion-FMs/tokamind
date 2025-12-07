# mmt/training/losses.py

from __future__ import annotations

from typing import Dict, Hashable, Mapping, Optional, Tuple

import torch
from torch import Tensor


"""
Loss utilities for the Multi-Modal Transformer.

We compute loss **in prediction (coeff) space**, but scale it so that it
approximates native-space MSE for orthonormal encoders (e.g. DCT/FPCA).

Inputs expected by the main function:

  • preds:  {output_key -> Tensor(B, K)}
      Model predictions in coeff space, one (B, K) tensor per output.

  • y_true: {output_key -> Tensor(B, K)}
      Encoded labels in the same coeff space. The pipeline is expected to
      produce these via the embedding transforms (we do *not* re-encode
      here).

  • outputs_mask: {output_key -> BoolTensor(B,)}
      Per-output presence mask (True where the sample has a label for that
      output and it was not dropped by collate).

  • outputs_native_sizes: {output_key -> int}
      Native number of points N for each output, computed from the original
      shape in MMTCollate (product of dims). Used to rescale coeff-space
      MSE to native-space MSE:

          scale = K / N

      For orthonormal encoders, SSE in coeff-space == SSE in native space,
      so dividing by N (instead of K) corresponds to MSE over native points.

  • output_weights: {output_key -> float} (optional)
      Optional per-output weights.
      If provided and sum(weights) > 0, we compute a weighted
      average across outputs, normalised by sum(weights). If not provided
      (or sum=0), we fall back to a simple mean over outputs.

By default (no weights), each supervised output contributes equally to the
total loss, regardless of dimensionality or how many samples in the batch
carry that output.
"""


def compute_loss_pred_space(
    preds: Mapping[Hashable, Tensor],
    y_true: Mapping[Hashable, Tensor],
    outputs_mask: Mapping[Hashable, Tensor],
    outputs_native_sizes: Optional[Mapping[Hashable, int]] = None,
    output_weights: Optional[Mapping[Hashable, float]] = None,
) -> Tuple[Tensor, Dict[Hashable, float]]:
    """
    Compute masked MSE in prediction space, normalised to approximate
    native-space MSE, and aggregate across outputs.

    Parameters
    ----------
    preds:
        Mapping from output_key -> prediction tensor of shape (B, K).

    y_true:
        Mapping from output_key -> label tensor of shape (B, K), already
        in the same coeff/prediction space as preds.

    outputs_mask:
        Mapping from output_key -> BoolTensor(B,), True where that output
        is supervised (present and not dropped) for that sample.

    outputs_native_sizes:
        Optional mapping from output_key -> native size N (product of the
        original shape dims). If provided, each per-output loss L_o is
        multiplied by (K / N) so that it approximates native-space MSE.

        If not provided or a key is missing, we fall back to N = K for
        that output (i.e. no scaling effect).

    output_weights:
        Optional mapping from output_key -> scalar weight (>=0). If given
        and the sum of weights over all supervised outputs in this batch
        is > 0, we compute:

            loss = sum_o L_o * (w_o / sum_o w_o)

        Otherwise, we fall back to a simple mean over outputs:

            loss = mean_o L_o

    Returns
    -------
    loss : Tensor (scalar)
        Aggregated loss on the same device/dtype as preds.

    logs : Dict[output_key, float]
        Per-output loss values (after native scaling and weights), detached
        to Python floats for logging.
    """
    if not preds:
        return torch.tensor(0.0), {}

    # Use first prediction as reference for device/dtype
    ref = next(iter(preds.values()))
    device, dtype = ref.device, ref.dtype

    per_out_losses = []
    per_out_weights = []
    logs: Dict[Hashable, float] = {}

    for out_key, y_pred in preds.items():
        if out_key not in y_true or out_key not in outputs_mask:
            continue

        y_t = y_true[out_key]
        mask = outputs_mask[out_key]

        if y_pred.shape != y_t.shape:
            raise RuntimeError(
                f"[{out_key!r}] pred/label shape mismatch: "
                f"{tuple(y_pred.shape)} vs {tuple(y_t.shape)}"
            )

        if mask.dtype != torch.bool:
            raise RuntimeError(
                f"[{out_key!r}] outputs_mask must be bool tensor, got {mask.dtype}."
            )

        if not bool(mask.any()):
            # No supervised samples for this output in this batch
            continue

        # Per-sample MSE in coeff space
        # y_pred, y_t: (B, K) → per-sample: (B,)
        per_sample = ((y_pred - y_t) ** 2).mean(dim=1)
        L_o = per_sample[mask].mean()

        # Native-space scaling: scale coeff-mean to native-mean
        # K from last dimension, N from outputs_native_sizes (if available)
        B, K = y_t.shape
        if outputs_native_sizes is not None and out_key in outputs_native_sizes:
            N = int(outputs_native_sizes[out_key])
        else:
            # Fallback: treat N=K (no scaling effect)
            N = K
        if N <= 0:
            N = K
        scale_native = K / float(N)
        L_o = L_o * scale_native

        # Per-output weight (if provided)
        w_o = 1.0
        if output_weights is not None and out_key in output_weights:
            w_o = float(output_weights[out_key])

        per_out_losses.append(L_o)
        per_out_weights.append(w_o)
        logs[out_key] = float(L_o.detach().cpu())

    if not per_out_losses:
        # No supervised outputs in this batch
        return torch.zeros((), device=device, dtype=dtype), logs

    per_out = torch.stack(per_out_losses)  # (num_outputs_supervised,)
    weights = torch.tensor(per_out_weights, device=device, dtype=dtype)

    if float(weights.sum()) > 0.0:
        # Normalised weighted average across outputs (matches old behaviour)
        loss = (per_out * (weights / (weights.sum() + 1e-8))).sum()
    else:
        # Fallback: uniform average over outputs
        loss = per_out.mean()

    return loss, logs
