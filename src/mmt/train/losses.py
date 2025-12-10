# mmt/train/losses.py

from __future__ import annotations

from typing import Dict, Hashable, Mapping, Optional, Tuple

import torch
from torch import Tensor


"""
Loss utilities for the Multi-Modal Transformer.

We compute a **single masked MSE loss in prediction (coeff) space**.

The train pipeline is expected to provide:

  • preds:  {output_key -> Tensor(B, K)}
      Model predictions in coeff space, one (B, K) tensor per output.

  • y_true: {output_key -> Tensor(B, K)}
      Encoded labels in the same coeff space. The embedding transforms are
      responsible for producing these; we do *not* re-encode in the loss.

  • outputs_mask: {output_key -> BoolTensor(B,)}
      Per-output presence mask:
        - True  → this sample carries a supervised label for that output
                  (and it was not dropped by collate),
        - False → ignore this sample for that output in the loss.

  • output_weights: {output_key -> float} (optional)
      Optional per-output scalar weights.
      If provided and sum(weights) > 0, we compute a weighted average across
      outputs, normalised by sum(weights). If not provided (or sum=0), we
      fall back to a simple mean over outputs.

Notes on native vs prediction space
-----------------------------------
For orthonormal encoders (e.g. DCT / FPCA) and when using all coefficients,
the sum of squared errors in coeff space is equal to the sum of squared
errors in native space. The difference between "MSE over coeffs" and "MSE
over native points" is just a constant factor:

    native_MSE ≈ (K / N) * coeff_MSE

where:
  • K = number of coefficients per output,
  • N = number of native points (product of the original shape dims).

In this implementation we **do not** apply this K/N scaling. The loss is
therefore expressed in "coeff-space MSE" units. If you ever need a loss
numerically comparable to a native-space MSE (e.g. for cross-model
comparison in a paper), you can reintroduce a per-output scale factor:

    L_o_native ≈ (K / N) * L_o_coeff

using N derived from the original output shape (e.g. via metadata or a
collate-side computation).
"""


def compute_loss_pred_space(
    preds: Mapping[Hashable, Tensor],
    y_true: Mapping[Hashable, Tensor],
    outputs_mask: Mapping[Hashable, Tensor],
    output_weights: Optional[Mapping[Hashable, float]] = None,
) -> Tuple[Tensor, Dict[Hashable, float]]:
    """
    Compute masked MSE in prediction space and aggregate across outputs.

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
        Per-output loss values (coeff-space MSE), detached to Python
        floats for logging.
    """
    if not preds:
        # No predictions → loss = 0 by convention
        return torch.tensor(0.0), {}

    # Use first prediction as reference for device/dtype
    ref = next(iter(preds.values()))
    device, dtype = ref.device, ref.dtype

    per_out_losses: list[Tensor] = []
    per_out_weights: list[float] = []
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
        # Normalised weighted average across outputs
        loss = (per_out * (weights / (weights.sum() + 1e-8))).sum()
    else:
        # Uniform average over outputs
        loss = per_out.mean()

    return loss, logs
