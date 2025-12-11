"""
Evaluation utilities for the Multi-Modal Transformer (MMT).

This module provides two evaluation flows:

---------------------------------------------------------
1) evaluate_metrics(...)
---------------------------------------------------------
    • Computes per-window Mean Squared Error (MSE) in NATIVE units.
    • Fully streaming-safe: never relies on len(dataloader).
    • Pipeline for each batch:
          - Move batch to device
          - Forward pass (standardized coefficient space)
          - Decode each output (inverse codec)
          - Destandardize using baseline mean/std
          - Compare decoded native predictions vs native ground truth
    • Produces:
          metrics_full.csv     – MSE per shot / per window / per output
          metrics_summary.csv  – aggregated MSE per output

---------------------------------------------------------
2) save_traces_for_subset(...)
---------------------------------------------------------
    • Saves true/pred native-unit traces for up to `n_max` shots.
    • Output format:
          <run_dir>/traces/<shot_id>__<output>.npz
    • Each NPZ contains arrays:
          true : (N_windows, ...)
          pred : (N_windows, ...)
      Shape depends on codec:
          timeseries → (N, T)
          profiles   → (N, C, T)
    • Optional filtering:
          outputs       – list of output names to save
          times_indexes – restrict time dimension (T)

---------------------------------------------------------
Shared utilities
---------------------------------------------------------
    • forward_decode_native(...)
          - Runs the model, moves batch to device,
            decodes outputs (inverse codec),
            destandardizes to native physical units,
            and extracts masks + shot_ids.

    • decode_and_destandardize(...)
          - Applies codec.decode() to each output,
            then multiplies by std and adds mean.

The evaluation pipeline mirrors the original FAIRMAST / baseline
procedure: model outputs exist in standardized coefficient space,
and must be decoded FIRST, then destandardized, to recover native
physical signals for metrics and trace analysis.
"""

import csv
from pathlib import Path
from typing import Dict, Any, Tuple

import torch
import numpy as np

from mmt.train.loop_utils import move_batch_to_device


# ============================================================================
# Helper: decode first, then destandardize
# ============================================================================


def decode_and_destandardize(
    y_pred_std: Dict[str, np.ndarray],
    y_true: Dict[str, np.ndarray],
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Decode model outputs from coefficient space and destandardise them
    in native physical space.

    All inputs are NumPy arrays on CPU:

      - y_pred_std[name] : (B, D)
          Standardised coefficients for output `name`.

      - y_true[name]     : (B, ...)
          Ground-truth in native shape. Only used to infer `original_shape`
          as y_true[name].shape[1:].

    For each output `name`:
      1) For each batch element b:
             decoded_b = codec.decode(y_pred_std[name][b], original_shape)
         where codec expects a 1D array (D,) and returns (...,).

      2) Stack all decoded_b along axis 0 → decoded with shape (B, ...).

      3) Apply destandardisation:
             decoded = decoded * std + mean

         where `mean` and `std` can be:
           - scalar (or shape (1,)) → global scaling/shift
           - vector of length C     → per-channel scaling/shift when
                                      decoded has shape (B, C, T, ...).
    """
    y_native: Dict[str, np.ndarray] = {}

    for out_name, pred_std in y_pred_std.items():
        # Skip outputs missing stats/codecs/ground-truth
        if out_name not in stats or out_name not in codecs or out_name not in y_true:
            continue

        codec = codecs[out_name]
        true_arr = y_true[out_name]  # (B, ...)

        if pred_std.ndim != 2:
            raise ValueError(
                f"y_pred_std[{out_name!r}] expected shape (B, D), got {pred_std.shape}."
            )

        B = pred_std.shape[0]
        original_shape = true_arr.shape[1:]  # (...,)

        # -----------------------------
        # 1) Decode each batch element
        # -----------------------------
        decoded_list = []
        for b in range(B):
            z_b = pred_std[b]  # (D,)
            decoded_b = codec.decode(z_b, original_shape)  # (...,)
            decoded_list.append(decoded_b)

        decoded = np.stack(decoded_list, axis=0)  # (B, ...)

        # -----------------------------
        # 2) Destandardise
        # -----------------------------
        mean = np.asarray(stats[out_name]["mean"])
        std = np.asarray(stats[out_name]["std"])

        # Case A: scalar or effectively scalar (shape (1,))
        if mean.ndim == 0 or (mean.ndim == 1 and mean.shape[0] == 1):
            # Broadcast over all dims of decoded
            decoded = decoded * std + mean

        # Case B: vector per-channel, e.g. mean.shape == (C,)
        else:
            if decoded.ndim < 2:
                raise ValueError(
                    f"Decoded array for {out_name!r} has shape {decoded.shape}, "
                    "cannot apply per-channel mean/std."
                )

            C = decoded.shape[1]
            if mean.shape[0] != C:
                raise ValueError(
                    f"stats[{out_name!r}].mean shape {mean.shape} is not "
                    f"compatible with decoded.shape {decoded.shape} "
                    f"(expected length C = decoded.shape[1] = {C})."
                )

            # Reshape mean/std to (1, C, 1, 1, ...) so they broadcast correctly
            shape = [1] * decoded.ndim
            shape[1] = C
            mean = mean.reshape(shape)
            std = std.reshape(shape)

            decoded = decoded * std + mean

        y_native[out_name] = decoded

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
) -> Tuple[
    Dict[str, np.ndarray],  # y_true_native
    Dict[str, np.ndarray],  # y_pred_native
    Dict[str, np.ndarray],  # y_mask (bool)
    np.ndarray,  # shot_ids
]:
    """
    Run one evaluation step and return outputs in native physical units.

    Pipeline:
      1. Move the batch to `device` and run the model forward.
      2. Extract predictions in standardised coefficient space from
         `out["pred"]`, keyed by integer signal_id.
      3. Convert all ID-keyed dicts (true outputs, masks, predictions)
         into name-keyed dicts via `id_to_name`.
      4. Move tensors to CPU and convert to NumPy arrays.
      5. Decode + destandardise predictions using per-output codecs + stats.
      6. Destandardise ground-truth outputs using the same stats.

    Returns
    -------
    y_true_native : dict[name, np.ndarray]
        Ground-truth outputs in native physical units.
    y_pred_native : dict[name, np.ndarray]
        Model predictions decoded and destandardised.
    y_mask : dict[name, np.ndarray]
        Boolean mask arrays indicating which windows are valid.
    shot_ids : np.ndarray
        Shot IDs for each batch element.
    """

    # -------------------------------------------------------------
    # 1) Move batch to device and run model
    # -------------------------------------------------------------
    batch = move_batch_to_device(batch, device)

    y_true_id = batch["output_native"]  # Dict[int, Tensor] (standardised)
    y_mask_id = batch["output_mask"]  # Dict[int, Tensor]
    shot_ids = batch["shot_id"]  # Tensor or array-like

    with torch.no_grad():
        out = model(batch)  # dict with key "pred"

    # DEBUG: compute MSE in *standardised coeff space* for one batch
    out = model(batch)

    y_pred_std_id = out.get("pred", {})  # Dict[int, Tensor] (standardised coeffs)

    # -------------------------------------------------------------
    # 2) Convert ID-keyed dicts → NAME-keyed dicts (still torch)
    # -------------------------------------------------------------
    y_true_torch = {
        id_to_name[sid]: tens for sid, tens in y_true_id.items() if sid in id_to_name
    }
    y_mask_torch = {
        id_to_name[sid]: tens for sid, tens in y_mask_id.items() if sid in id_to_name
    }
    y_pred_std_torch = {
        id_to_name[sid]: tens
        for sid, tens in y_pred_std_id.items()
        if sid in id_to_name
    }

    # -------------------------------------------------------------
    # 3) Torch → NumPy (CPU)
    # -------------------------------------------------------------
    y_true_std = {
        name: tens.detach().cpu().numpy() for name, tens in y_true_torch.items()
    }
    y_mask = {
        name: tens.detach().cpu().numpy().astype(bool)
        for name, tens in y_mask_torch.items()
    }
    y_pred_std = {
        name: tens.detach().cpu().numpy() for name, tens in y_pred_std_torch.items()
    }

    # -------------------------------------------------------------
    # 4) Decode + destandardise predictions (coeff → native)
    # -------------------------------------------------------------
    y_pred_native = decode_and_destandardize(
        y_pred_std=y_pred_std,
        y_true=y_true_std,  # only used for shapes
        stats=stats,
        codecs=codecs,
    )

    # -------------------------------------------------------------
    # 5) Destandardise ground-truth (already in native shape)
    # -------------------------------------------------------------
    y_true_native: Dict[str, np.ndarray] = {}
    for name, arr in y_true_std.items():
        if name not in stats:
            # No stats → leave as is (should not normally happen for outputs)
            y_true_native[name] = arr
            continue

        mean = np.asarray(stats[name]["mean"])
        std = np.asarray(stats[name]["std"])

        # Case A: global scalar or shape (1,)
        if mean.ndim == 0 or (mean.ndim == 1 and mean.shape[0] == 1):
            y_true_native[name] = arr * std + mean
        # Case B: per-channel stats, e.g. mean.shape == (C,)
        else:
            if arr.ndim < 2:
                raise ValueError(
                    f"y_true[{name!r}] has shape {arr.shape}, cannot apply "
                    "per-channel mean/std."
                )
            C = arr.shape[1]
            if mean.shape[0] != C:
                raise ValueError(
                    f"stats[{name!r}].mean shape {mean.shape} is not "
                    f"compatible with y_true[{name!r}].shape {arr.shape} "
                    f"(expected length C = {C})."
                )
            shape = [1] * arr.ndim
            shape[1] = C
            mean_reshaped = mean.reshape(shape)
            std_reshaped = std.reshape(shape)
            y_true_native[name] = arr * std_reshaped + mean_reshaped

    # -------------------------------------------------------------
    # Optional debug ranges (keep commented, enable if needed)
    # -------------------------------------------------------------
    for name in y_pred_native:
        yt = y_true_native[name]
        yp = y_pred_native[name]
        print(
            f"[DEBUG] {name}: "
            f"true min/max = ({yt.min():.3f}, {yt.max():.3f}), "
            f"pred min/max = ({yp.min():.3f}, {yp.max():.3f})"
        )

    return y_true_native, y_pred_native, y_mask, shot_ids


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
) -> Dict[str, float]:
    """
    Compute per-window native-space MSE for all outputs of the task.

    Outputs:
        <run_dir>/metrics/metrics_full.csv
        <run_dir>/metrics/metrics_summary.csv
    """

    metrics_dir = Path(run_dir) / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # CSV for all windows
    csv_full = metrics_dir / "metrics_full.csv"
    f_full = csv_full.open("w", newline="")
    wr = csv.writer(f_full)
    wr.writerow(["shot_id", "window_idx", "output", "mse"])

    # Accumulate weighted MSE per output
    accum = {name: [0.0, 0.0] for name in stats.keys()}

    window_counter = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            y_true, y_pred, y_mask, shot_ids = forward_decode_native(
                batch=batch,
                model=model,
                device=device,
                stats=stats,
                codecs=codecs,
                id_to_name=id_to_name,
            )

            B = len(shot_ids)

            for b in range(B):
                sid = shot_ids[b]

                for out_name in stats.keys():
                    true_b = y_true[out_name][b]
                    pred_b = y_pred[out_name][b]
                    mask_b = y_mask[out_name][b]

                    valid = mask_b.sum().item()
                    if valid == 0:
                        mse_b = float("nan")
                    else:
                        diff2 = (pred_b - true_b) ** 2
                        mse_b = (diff2 * mask_b).sum().item() / valid

                        # accumulate for summary
                        accum[out_name][0] += mse_b * valid
                        accum[out_name][1] += valid

                    wr.writerow([sid, window_counter, out_name, mse_b])

                window_counter += 1

    f_full.close()

    # summary CSV
    csv_sum = metrics_dir / "metrics_summary.csv"
    with csv_sum.open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["output", "mse"])

        summary = {}
        for out_name, (sum_w, cnt) in accum.items():
            mse = sum_w / cnt if cnt > 0 else float("nan")
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
):
    """
    Save native-space true/pred traces for up to `n_max` unique shots.

    Produces:
        <run_dir>/traces/<shot_id>__<output>.npz
    """

    if not traces_cfg.get("enable", False):
        return

    traces_dir = Path(run_dir) / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    n_max = traces_cfg.get("n_max", 10)
    outputs_filter = traces_cfg.get("outputs", None)
    time_idx = traces_cfg.get("times_indexes", None)

    collected: Dict[str, Dict[str, list]] = {}
    seen = set()

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            y_true, y_pred, _, shot_ids = forward_decode_native(
                batch=batch,
                model=model,
                device=device,
                stats=stats,
                codecs=codecs,
                id_to_name=id_to_name,
            )

            B = len(shot_ids)

            for b in range(B):
                sid = shot_ids[b]

                if sid not in collected:
                    collected[sid] = {}
                seen.add(sid)

                if len(seen) > n_max:
                    break

                for out_name, pred_out in y_pred.items():
                    if outputs_filter is not None and out_name not in outputs_filter:
                        continue

                    true_arr = y_true[out_name][b].cpu().numpy()
                    pred_arr = pred_out[b].cpu().numpy()

                    if time_idx is not None:
                        true_arr = true_arr[time_idx]
                        pred_arr = pred_arr[time_idx]

                    if out_name not in collected[sid]:
                        collected[sid][out_name] = []

                    collected[sid][out_name].append((true_arr, pred_arr))

            if len(seen) >= n_max:
                break

    # save files
    for sid, outputs in collected.items():
        for out_name, pairs in outputs.items():
            true_stack = np.stack([t for t, _ in pairs], axis=0)
            pred_stack = np.stack([p for _, p in pairs], axis=0)
            np.savez(
                traces_dir / f"{sid}__{out_name}.npz", true=true_stack, pred=pred_stack
            )
