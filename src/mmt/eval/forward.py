"""
Run a forward pass for evaluation and return outputs in native physical units.

This helper:
- runs the model on a single batch (no grad),
- extracts predictions in standardised coefficient space,
- converts id-keyed outputs to name-keyed arrays,
- decodes predictions via signal-specific codecs,
- de-standardises both predictions and ground truth,
- returns per-window metadata (shot_id, window_index).

It is used by evaluation and trace-saving routines and assumes
window-level batches produced by TaskModelTransformWrapper + collate.
"""

from typing import Dict, Any, Tuple

import numpy as np
import torch

from mmt.train.loop_utils import move_batch_to_device
from mmt.utils.amp_utils import amp_ctx_for_model
from .decode import decode_and_destandardize, apply_stats

import logging

logger = logging.getLogger("mmt.Eval")


# ------------------------------------------------------------------
# Forward + decode + destandardize
# ------------------------------------------------------------------


def forward_decode_native(
    batch: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
    id_to_name: Dict[int, str],
    amp_enabled: bool = True,
) -> Tuple[
    Dict[str, np.ndarray],  # y_true_native
    Dict[str, np.ndarray],  # y_pred_native
    Dict[str, np.ndarray],  # y_mask (bool)
    np.ndarray,  # shot_ids
    np.ndarray,  # window_indices
]:
    """
    Run one evaluation step and return outputs in native physical units.

    Steps
    -----
    1. Move the batch to `device` and run the model.
    2. Extract standardised coefficient predictions from out["pred"] (id-keyed).
    3. Convert id-keyed dicts (true outputs, masks, preds) into name-keyed dicts.
    4. Move everything to CPU NumPy.
    5. Decode + destandardise predictions (coeff → native).
    6. Destandardise ground truth using the same stats.
    """

    # 1) Move batch to device and run model
    batch = move_batch_to_device(batch, device)

    y_true_id = batch["output_native"]  # Dict[int, Tensor] (standardised)
    y_mask_id = batch["output_mask"]  # Dict[int, Tensor] (bool per window)
    y_true_emb_id = batch["output_emb"]

    # Metadata: shot_id and window_index must be present
    if "shot_id" not in batch:
        raise KeyError("Batch is missing 'shot_id' field required for evaluation.")
    if "window_index" not in batch:
        raise KeyError("Batch is missing 'window_index' field required for evaluation.")

    shot_ids = np.asarray(batch["shot_id"])
    window_indices = np.asarray(batch["window_index"])

    if len(window_indices) != len(shot_ids):
        raise ValueError(
            f"'window_index' length {len(window_indices)} does not match "
            f"'shot_id' length {len(shot_ids)} in evaluation batch."
        )

    model.eval()
    with torch.no_grad():
        with amp_ctx_for_model(model, enable=amp_enabled):
            out = model(batch)

    y_pred_std_id = out.get("pred", {})  # Dict[int, Tensor] (standardised coeffs)

    # ------------------------------------------------------------------
    # Optional debug: MSE in *standardised coeff space*,
    # same space as training loss (pred vs output_emb).
    # ------------------------------------------------------------------
    if logger.isEnabledFor(logging.DEBUG):
        if y_true_emb_id is not None:
            for sig_id, pred_std in y_pred_std_id.items():
                if sig_id not in y_true_emb_id or sig_id not in y_mask_id:
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

    # 2) id-keyed → name-keyed (torch)
    y_true_t = {
        id_to_name[sid]: tens for sid, tens in y_true_id.items() if sid in id_to_name
    }
    y_mask_t = {
        id_to_name[sid]: tens for sid, tens in y_mask_id.items() if sid in id_to_name
    }
    y_pred_std_t = {
        id_to_name[sid]: tens
        for sid, tens in y_pred_std_id.items()
        if sid in id_to_name
    }

    # 3) torch → NumPy (CPU)
    def _tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
        t = t.detach().cpu()
        # NumPy does not support bfloat16; cast to float32 for safe export.
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()

    y_true_std = {k: _tensor_to_numpy(v) for k, v in y_true_t.items()}
    y_mask = {k: v.detach().cpu().bool().numpy() for k, v in y_mask_t.items()}
    y_pred_std = {k: _tensor_to_numpy(v) for k, v in y_pred_std_t.items()}

    # 4) Decode + destandardise predictions
    y_pred_native = decode_and_destandardize(
        y_pred_std=y_pred_std,
        y_true_std=y_true_std,
        stats=stats,
        codecs=codecs,
    )

    # 5) Destandardise ground truth
    y_true_native: Dict[str, np.ndarray] = {}
    for name, arr in y_true_std.items():
        if name not in stats:
            # No stats → leave as-standardised (should be rare)
            y_true_native[name] = arr
            continue

        y_true_native[name] = apply_stats(
            arr, mean=stats[name]["mean"], std=stats[name]["std"]
        )

    if logger.isEnabledFor(logging.DEBUG):
        for name in y_pred_native:
            yt = y_true_native[name]
            yp = y_pred_native[name]
            logger.debug(
                f"min-max [{name}]: "
                f"true min/max=({yt.min():.3f}, {yt.max():.3f}), "
                f"pred min/max=({yp.min():.3f}, {yp.max():.3f})"
            )

    return y_true_native, y_pred_native, y_mask, shot_ids, window_indices
