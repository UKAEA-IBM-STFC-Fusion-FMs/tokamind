"""
Evaluation utilities for the Multi-Modal Transformer (MMT).

What this module does
---------------------
- Runs the model on a window-level dataloader.
- Decodes predictions from coefficient space to native space via codecs.
- Destandardises both predictions and ground truth using baseline mean/std.
- Computes simple native-space MSE metrics and (optionally) saves traces.

Public entry points
-------------------
- evaluate_metrics(...)
    Compute per-window MSE in native units and save:
      <run_dir>/metrics/metrics_full.csv
      <run_dir>/metrics/metrics_summary.csv

- save_traces_for_subset(...)
    Save per-window native-space traces for a limited set of shots:
      <run_dir>/traces/<shot_id>__<output>.npz

Both functions are streaming-safe and make no assumptions about dataset size.
"""

import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch

from mmt.train.loop_utils import move_batch_to_device
from mmt.utils.amp_utils import amp_ctx_for_model

import logging

logger = logging.getLogger("mmt.Eval")


# ============================================================================
# Helpers
# ============================================================================


def _apply_stats(arr: np.ndarray, mean, std) -> np.ndarray:
    """
    Invert the standardisation used in the baseline:

        values_std = (values - mean[..., None]) / std[..., None]

    Here `arr` is batch-first, e.g. (B, C, T, ...) or (B, T, ...).
    `mean` / `std` can be:
      - scalar
      - shape (1,)
      - shape (C,)
    """
    mean = np.asarray(mean)
    std = np.asarray(std)

    # Scalar or effectively scalar
    if mean.ndim == 0 or (mean.ndim == 1 and mean.shape[0] == 1):
        return arr * std + mean

    # Per-channel (C,)
    if arr.ndim < 2:
        raise ValueError(
            f"Cannot apply per-channel stats: arr.shape={arr.shape}, mean.shape={mean.shape}"
        )
    C = arr.shape[1]
    if mean.shape[0] != C:
        raise ValueError(
            f"Incompatible shapes: mean.shape={mean.shape}, arr.shape={arr.shape} "
            f"(expected mean.shape[0] == arr.shape[1] == {C})"
        )

    shape = [1] * arr.ndim
    shape[1] = C  # (1, C, 1, 1, ...)
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    return arr * std + mean


# ============================================================================
# Decode first, then destandardise
# ============================================================================


def decode_and_destandardize(
    y_pred_std: Dict[str, np.ndarray],
    y_true_std: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Decode model outputs from coefficient space and destandardise them in
    native physical space.

    All inputs are NumPy arrays on CPU:

      - y_pred_std[name] : (B, D)  standardized coefficients
      - y_true_std[name] : (B, ...) standardized native values
        (only used here to infer the original native shape per sample)

    Returns
    -------
    y_native : dict[name, np.ndarray]
        Decoded and destandardised predictions in native units.
    """
    y_native: Dict[str, np.ndarray] = {}

    for name, pred_std in y_pred_std.items():
        if name not in stats or name not in codecs or name not in y_true_std:
            continue

        if pred_std.ndim != 2:
            raise ValueError(
                f"y_pred_std[{name!r}] expected shape (B, D), got {pred_std.shape}."
            )

        codec = codecs[name]
        true_arr = y_true_std[name]
        B = pred_std.shape[0]
        original_shape = true_arr.shape[1:]  # (...,)

        # Decode each sample separately
        decoded = np.stack(
            [codec.decode(pred_std[b], original_shape) for b in range(B)],
            axis=0,
        )  # (B, ...)

        y_native[name] = _apply_stats(
            decoded, mean=stats[name]["mean"], std=stats[name]["std"]
        )

    return y_native


# ============================================================================
# Forward + decode + destandardize
# ============================================================================


def forward_decode_native(
    batch: Dict[str, Any],
    model: torch.nn.Module,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
    id_to_name: Dict[int, str],
    debug: bool = True,
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
        with amp_ctx_for_model(model, enable=True):
            out = model(batch)

    y_pred_std_id = out.get("pred", {})  # Dict[int, Tensor] (standardised coeffs)

    # ---------------------------------------------------------
    # Optional debug: MSE in *standardised coeff space*,
    # same space as training loss (pred vs output_emb).
    # ---------------------------------------------------------
    if debug:
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
                logger.info(f"[DEBUG] coeff-space MSE [{name}]: {mse_coeff:.6f}")

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
    y_true_std = {k: v.detach().cpu().numpy() for k, v in y_true_t.items()}
    y_mask = {k: v.detach().cpu().numpy().astype(bool) for k, v in y_mask_t.items()}
    y_pred_std = {k: v.detach().cpu().numpy() for k, v in y_pred_std_t.items()}

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

        y_true_native[name] = _apply_stats(
            arr, mean=stats[name]["mean"], std=stats[name]["std"]
        )

    if debug:
        for name in y_pred_native:
            yt = y_true_native[name]
            yp = y_pred_native[name]
            logger.info(
                f"[DEBUG] {name}: "
                f"true min/max=({yt.min():.3f}, {yt.max():.3f}), "
                f"pred min/max=({yp.min():.3f}, {yp.max():.3f})"
            )

    return y_true_native, y_pred_native, y_mask, shot_ids, window_indices


# ============================================================================
# METRICS
# ============================================================================


def evaluate_metrics(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
    id_to_name: Dict[int, str],
    run_dir: Path,
    debug: bool = False,
) -> Dict[str, float]:
    """
    Compute per-window native-space MSE for all outputs of the task.

    Writes:
      <run_dir>/metrics/metrics_full.csv   (per-shot, per-window, per-output)
      <run_dir>/metrics/metrics_summary.csv (per-output average MSE)
    """

    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    csv_full = metrics_dir / "metrics_full.csv"
    f_full = csv_full.open("w", newline="")
    wr = csv.writer(f_full)
    wr.writerow(["shot_id", "window_idx", "output", "mse"])

    # Accumulate mean over windows per output
    accum = {name: [0.0, 0.0] for name in stats.keys()}

    with torch.no_grad():
        for batch in dataloader:
            y_true, y_pred, y_mask, shot_ids, window_indices = forward_decode_native(
                batch=batch,
                model=model,
                device=device,
                stats=stats,
                codecs=codecs,
                id_to_name=id_to_name,
                debug=debug,  # or True if you want the min/max prints
            )

            B = len(shot_ids)

            for b in range(B):
                sid = int(shot_ids[b])
                widx = int(window_indices[b])  # true window index from baseline

                for out_name in stats.keys():
                    mask_b = bool(y_mask[out_name][b])
                    if not mask_b:
                        mse_b = float("nan")
                    else:
                        true_b = y_true[out_name][b]
                        pred_b = y_pred[out_name][b]
                        diff2 = (pred_b - true_b) ** 2
                        mse_b = float(diff2.mean())
                        accum[out_name][0] += mse_b
                        accum[out_name][1] += 1

                    wr.writerow([sid, widx, out_name, mse_b])

    f_full.close()

    # Summary CSV
    csv_sum = metrics_dir / "metrics_summary.csv"
    summary: Dict[str, float] = {}
    with csv_sum.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["output", "mse"])

        for out_name, (sum_mse, count) in accum.items():
            mse = sum_mse / count if count > 0 else float("nan")
            summary[out_name] = mse
            wr.writerow([out_name, mse])

    return summary


# ============================================================================
# TRACES
# ============================================================================


def save_traces_for_subset(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
    id_to_name: Dict[int, str],
    run_dir: Path,
    traces_cfg: Dict[str, Any],
) -> None:
    """
    Save native-space true/pred traces for up to `n_max` unique shots.

    Traces are ordered by window_index so they represent temporal
    evolution within each shot, independently of dataloader order.

    Config (traces_cfg)
    -------------------
    enable : bool
    n_max : int
        Maximum number of distinct shot_ids to save.
    signals : list[str] or None
        If not None, save only this subset of output names.
        If None, save all outputs present in y_pred.
    times_indexes : list[int] or None
        Optional index subset along the time dimension.
    """

    if not traces_cfg.get("enable", False):
        return

    traces_dir = Path(run_dir) / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    n_max = int(traces_cfg.get("n_max", 10))
    signals_filter = traces_cfg.get("signals", None)
    time_idx = traces_cfg.get("times_indexes", None)

    # shot_id -> {output_name -> [(window_index, true_arr, pred_arr), ...]}
    collected: Dict[int, Dict[str, list]] = {}
    seen_shots: set[int] = set()

    for batch in dataloader:
        (
            y_true,
            y_pred,
            y_mask,
            shot_ids,
            window_indices,
        ) = forward_decode_native(
            batch=batch,
            model=model,
            device=device,
            stats=stats,
            codecs=codecs,
            id_to_name=id_to_name,
            debug=False,
        )

        B = len(shot_ids)
        stop = False

        for b in range(B):
            sid = int(shot_ids[b])
            widx = int(window_indices[b])

            # Track distinct shots
            if sid not in seen_shots:
                seen_shots.add(sid)
                if len(seen_shots) > n_max:
                    stop = True
                    break

            collected.setdefault(sid, {})

            for out_name, pred_out in y_pred.items():
                # Apply signal subset filter (diagnostic only)
                if signals_filter is not None and out_name not in signals_filter:
                    continue

                # Skip windows where this output is not present
                if out_name not in y_mask or not bool(y_mask[out_name][b]):
                    continue

                true_arr = y_true[out_name][b]
                pred_arr = pred_out[b]

                # Optional time sub-sampling inside each window
                if time_idx is not None:
                    true_arr = true_arr[time_idx]
                    pred_arr = pred_arr[time_idx]

                collected[sid].setdefault(out_name, []).append(
                    (widx, true_arr, pred_arr)
                )

        if stop:
            break

    # Save NPZ files: one per (shot_id, output_name), ordered by window_index
    for sid, outputs in collected.items():
        for out_name, triples in outputs.items():
            if not triples:
                continue

            # sort windows temporally
            triples.sort(key=lambda x: x[0])

            window_idx_arr = np.asarray([w for w, _, _ in triples], dtype=np.int64)
            true_stack = np.stack([t for _, t, _ in triples], axis=0)
            pred_stack = np.stack([p for _, _, p in triples], axis=0)

            np.savez(
                traces_dir / f"{sid}__{out_name}.npz",
                true=true_stack,
                pred=pred_stack,
                window_index=window_idx_arr,
            )
