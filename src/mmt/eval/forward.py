"""
Run a forward pass for evaluation and return outputs in native physical units.

This helper:
- runs the model on a single batch (no grad),
- extracts predictions in standardized coefficient space,
- converts ID-keyed outputs to name-keyed arrays,
- decodes predictions via signal-specific codecs,
- de-standardizes both predictions and ground truth,
- returns per-window metadata (shot_id, window_index).

It is used by evaluation and trace-saving routines and assumes window-level batches produced by TokaMarkDataset +
collate.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Any
import numpy as np
import logging

import torch

from mmt.data.standardization import destandardize_numpy
from mmt.distributions import sample_gaussian_dist
from mmt.train.loop_utils import move_batch_to_device
from mmt.utils.amp_utils import amp_ctx_for_model
from .decode import decode_and_destandardize, decode_samples_and_destandardize

if True:  # TYPE_CHECKING:  # FIXME: This does not work for ROH
    from mmt.data.embeddings.torch_decoder import TorchDecoder


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Eval")


# ======================================================================================================================
# Forward + decode + destandardize
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def forward_decode_native(  # NOSONAR - Ignore cognitive complexity
    batch: MutableMapping[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    stats: Mapping[str, Mapping[str, float]],
    decoders: Mapping[str, TorchDecoder],
    id_to_name: Mapping[int, str],
    amp_enabled: bool = True,
    n_samples: int = 0,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
]:
    """
    Run one evaluation step and return outputs in native physical units.

    Steps
    -----
    1. Move the batch to `device` and run the model.
    2. Extract standardized coefficient predictions from out["pred"] (ID-keyed).
    3. Convert ID-keyed dicts (true outputs, masks, preds) into name-keyed dicts.
    4. Move everything to CPU NumPy.
    5. Decode + destandardize predictions (coeff → native).
    6. Destandardize ground truth using the same stats.
    7. If `n_samples > 0` and the model has a Gaussian head, decode predictive samples to native space too.

    Parameters
    ----------
    batch : MutableMapping[str, Any]
        Passed batch.
    model : torch.nn.Module
        Standard evaluation model input.
    device : torch.device
        Standard evaluation device input.
    stats : Mapping[str, Mapping[str, float]]
        Per-signal stats dict with ``"mean"`` and ``"std"`` keys.
    decoders : Mapping[str, TorchDecoder]
        Pre-built per-signal ``TorchDecoder`` instances keyed by signal name.
    id_to_name : Mapping[int, str]
        Mapping from signal_id to signal name.
    amp_enabled : bool
        Whether to enable AMP in the forward pass.
        Optional. Default: True.
    n_samples : int
        If > 0 and the model emits ``pred_dist`` (a Gaussian head), draw this many reparameterized predictive samples
        from ``pred_dist`` (in fp32, outside autocast) and decode them to native space (returned as the sixth
        element). For deterministic heads or ``n_samples == 0`` no sampling is performed and the sixth element is
        ``None``. Optional. Default: 0.

    Returns
    -------
    tuple
        ``(y_true_native, y_pred_native, y_mask, shot_ids, window_indices, y_pred_samples_native, output_time)``.
        ``y_pred_samples_native`` is ``dict[name -> (B, S, *native_shape)]`` of decoded native samples, or ``None``
        when no samples were requested/available. ``output_time`` is ``dict[name -> (B, T)]`` of per-output-slice
        physical times (from ``batch["output_time"]``, ID→name keyed), or ``None`` when the batch carries no
        ``output_time`` (e.g. collate did not populate it). These times are used by eval-only reference metrics
        (e.g. the LCFS/X-point topology metric) to align each predicted slice to the source equilibrium time axis.

    Raises
    ------
    KeyError
        If `batch` does not have required key "shot_id".
        If `batch` does not have required key "window_index".
    ValueError
        If `len(batch["window_index"])` does not match `len(batch["shot_id"]).

    """

    # ..................................................................................................................
    def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
        """Tensor to numpy conversion."""

        t = t.detach().cpu()

        # NumPy does not support bfloat16; cast to float32 for safe export.
        if t.dtype == torch.bfloat16:
            t = t.float()

        return t.numpy()

    # ..................................................................................................................

    # ..................................................................................................................
    # 1) Move batch to device and run model
    # ..................................................................................................................

    batch = move_batch_to_device(batch=batch, device=device)

    # Metadata: shot_id and window_index must be present
    if "shot_id" not in batch:
        raise KeyError("`batch` is missing 'shot_id' field required for evaluation.")
    if "window_index" not in batch:
        raise KeyError("`batch` is missing 'window_index' field required for evaluation.")

    shot_ids = np.asarray(batch["shot_id"])
    window_indices = np.asarray(batch["window_index"])

    if len(window_indices) != len(shot_ids):
        raise ValueError(
            f"`batch['window_index']` length {len(window_indices)} does not match `batch['shot_id'] length "
            f"{len(shot_ids)} in evaluation batch."
        )

    y_true_id = batch["output_native"]  # dict[int, Tensor] (standardized)
    y_mask_id = batch["output_mask"]  # dict[int, Tensor] (bool per window)
    y_true_emb_id = batch["output_emb"]
    output_time_id = batch.get("output_time")  # dict[int, Tensor (B, T)] of per-slice physical times, or None

    model.eval()
    y_pred_samples_std_id: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        with amp_ctx_for_model(model=model, enable=amp_enabled):
            out = model(batch)

        y_pred_std_id = out.get("pred", {})  # dict[int, Tensor] (standardized coeffs)

        # Sampling is the consumer's responsibility: draw from pred_dist (a Gaussian head) in fp32, outside autocast.
        # Deterministic heads produce no pred_dist, so no samples are drawn.
        pred_dist = out.get("pred_dist", {})
        if (int(n_samples) > 0) and bool(pred_dist):
            for sid, dist in pred_dist.items():
                dist_f32 = {**dist, "mu": dist["mu"].to(torch.float32), "sigma": dist["sigma"].to(torch.float32)}
                y_pred_samples_std_id[sid] = sample_gaussian_dist(dist_f32, n_samples=int(n_samples))  # (B, S, K)

    # ..................................................................................................................
    # Optional debug: MSE in *standardized coeff space*, same space as training loss (pred vs output_emb).
    # ..................................................................................................................

    if logger.isEnabledFor(logging.DEBUG):
        if y_true_emb_id is not None:
            for sig_id, pred_std in y_pred_std_id.items():
                if (sig_id not in y_true_emb_id) or (sig_id not in y_mask_id):
                    continue

                target_std = y_true_emb_id[sig_id]  # (B, D)
                mask = y_mask_id[sig_id].bool()  # (B,) or (B, 1, ...)

                # Collapse any extra dims in the mask (e.g. (B,1) → (B,))
                if mask.ndim > 1:
                    mask = mask.view(mask.shape[0], -1).any(dim=1)

                if not mask.any():
                    continue

                diff2 = (pred_std[mask] - target_std[mask]) ** 2
                mse_coeff = diff2.mean().item()

                name = id_to_name.get(sig_id, f"id={sig_id}")
                logger.debug(f"Coeff-space MSE [{name}]: {mse_coeff:.6f}")

    # ..................................................................................................................
    # 2) ID-keyed → name-keyed (torch)
    # ..................................................................................................................

    y_true_t = {id_to_name[sid]: tens for sid, tens in y_true_id.items() if sid in id_to_name}
    y_mask_t = {id_to_name[sid]: tens for sid, tens in y_mask_id.items() if sid in id_to_name}
    y_pred_std_t = {id_to_name[sid]: tens for sid, tens in y_pred_std_id.items() if sid in id_to_name}
    y_pred_samples_std_t = {id_to_name[sid]: tens for sid, tens in y_pred_samples_std_id.items() if sid in id_to_name}

    # ..................................................................................................................
    # 3) torch → NumPy (CPU)
    # ..................................................................................................................

    y_true_std = {k: _tensor_to_numpy(v) for k, v in y_true_t.items()}
    y_mask = {k: v.detach().cpu().bool().numpy() for k, v in y_mask_t.items()}
    y_pred_std = {k: _tensor_to_numpy(v) for k, v in y_pred_std_t.items()}
    y_pred_samples_std = {k: _tensor_to_numpy(v) for k, v in y_pred_samples_std_t.items()}

    # Per-output physical times (ID → name keyed), used by eval-only reference metrics to align each predicted
    # slice to the source equilibrium time axis. None when the batch carries no output_time.
    output_time: dict[str, np.ndarray] | None = None
    if output_time_id is not None:
        output_time = {
            id_to_name[sid]: _tensor_to_numpy(tens) for sid, tens in output_time_id.items() if sid in id_to_name
        }

    # ..................................................................................................................
    # 4) Decode + destandardize predictions (mean, and optionally the predictive samples)
    # ..................................................................................................................

    y_pred_native = decode_and_destandardize(
        y_pred_std=y_pred_std, y_true_std=y_true_std, stats=stats, decoders=decoders
    )

    y_pred_samples_native: dict[str, np.ndarray] | None = None
    if y_pred_samples_std:
        y_pred_samples_native = decode_samples_and_destandardize(
            y_pred_samples_std=y_pred_samples_std, y_true_std=y_true_std, stats=stats, decoders=decoders
        )

    # ..................................................................................................................
    # 5) Destandardize ground truth
    # ..................................................................................................................

    y_true_native: dict[str, np.ndarray] = {}
    for name, arr in y_true_std.items():
        if name not in stats:
            # No stats → leave as-standardized (should be rare)
            y_true_native[name] = arr
            continue

        y_true_native[name] = destandardize_numpy(arr=arr, mean=stats[name]["mean"], std=stats[name]["std"])

    if logger.isEnabledFor(logging.DEBUG):
        for name in y_pred_native:
            yt = y_true_native[name]
            yp = y_pred_native[name]
            logger.debug(
                f"min-max [{name}]: "
                f"true min/max=({yt.min():.3f}, {yt.max():.3f}), "
                f"pred min/max=({yp.min():.3f}, {yp.max():.3f})"
            )

    return y_true_native, y_pred_native, y_mask, shot_ids, window_indices, y_pred_samples_native, output_time

    # ..................................................................................................................
