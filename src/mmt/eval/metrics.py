"""
Evaluation utilities for the Multi-Modal Transformer (MMT).

This module provides the high-level evaluation entry points used during
validation and offline analysis. It operates on window-level dataloaders
and assumes batches produced by TaskModelTransformWrapper + MMTCollate.

Responsibilities
----------------
- Run the model in evaluation mode (no grad, AMP-enabled).
- Convert predictions from standardised coefficient space to native units.
- De-standardise ground truth using baseline statistics.
- Compute native-space MSE metrics per window and per output.
- Optionally save temporally ordered traces for selected shots.

Public API
----------
- evaluate_metrics(...)
    Compute per-window MSE and write CSV summaries under:
      <run_dir>/metrics/

- save_traces_for_subset(...)
    Save native-space true/pred traces ordered by window_index under:
      <run_dir>/traces/

Lower-level decoding and forward-pass logic is delegated to
`mmt.eval.forward` and `mmt.eval.decode`.
"""

import csv
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from .forward import forward_decode_native

import logging

logger = logging.getLogger("mmt.Eval")

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
