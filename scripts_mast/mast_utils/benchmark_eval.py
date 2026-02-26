"""scripts_mast.mast_utils.benchmark_eval

Benchmark-aligned evaluation utilities.

This module lives in the *scripts_mast* integration layer on purpose:

- The core library `mmt/` stays dataset / benchmark agnostic.
- The MAST benchmark repository is the source of truth for the *official*
  evaluation aggregation (window → shot → task), via its evaluator helpers.

What this module provides
-------------------------
One *single-pass* evaluation loop that can produce:

- Benchmark metrics:
    - per-window: ``windows_metrics.csv`` (optional)
    - per-task:   ``tasks_metrics.csv`` (optional)

  written under:

    ``<eval_run_dir>/benchmark/<task>/``

- Optional MMT-native diagnostics:
    - per-timestamp metrics CSV
    - qualitative traces (NPZ)

  written under:

    ``<eval_run_dir>/metrics/`` and ``<eval_run_dir>/traces/``

The loop uses ``mmt.eval.forward.forward_decode_native`` so decoding and
de-standardisation remain in `mmt/`.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional, cast

import numpy as np
import torch

from mmt.eval.forward import forward_decode_native

from .benchmark_imports import WindowMetricsAccumulator, compute_metrics


logger = logging.getLogger("mmt.Eval")

_LOG_INTERVAL = 50000


def _reduce_mask(mask: np.ndarray) -> np.ndarray:
    """Reduce a possibly high-rank mask to shape (B,) via OR over extra dims."""

    if mask.ndim == 1:
        return np.asarray(mask, dtype=bool)
    reduced = mask.reshape(mask.shape[0], -1).any(axis=1)
    return np.asarray(reduced, dtype=bool)


def evaluate_benchmark_and_diagnostics(
    *,
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    stats: Dict[str, Dict[str, float]],
    codecs: Dict[str, Any],
    id_to_name: Dict[int, str],
    run_dir: Path,
    task_name: str,
    amp_enabled: bool,
    compute_metrics_cfg: Optional[Dict[str, Any]] = None,
    traces_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run evaluation once and write configured outputs.

    Parameters
    ----------
    model, dataloader, device :
        Standard evaluation inputs.
    stats, codecs, id_to_name :
        Output decoding / de-standardisation inputs (native-space evaluation).
    run_dir : Path
        Eval run directory (``runs/<train_run>/<eval_id>/``).
    task_name : str
        Benchmark task name (e.g. ``task_2-1``) used for output folder naming.
    amp_enabled : bool
        Whether to enable AMP in the forward pass.
    compute_metrics_cfg : dict
        Supports keys:
          - per_task: bool (benchmark aggregation -> tasks_metrics.csv)
          - per_window: bool (keep windows_metrics.csv)
          - per_timestamp: bool (MMT-native per-timestamp CSV)
    traces_cfg : dict
        Same structure as in docs/evaluation.md.

    Returns
    -------
    dict
        Small summary of what was written (paths, and benchmark task metrics if available).
    """

    cfg = compute_metrics_cfg or {}
    cfg_traces = traces_cfg or {}

    per_task = bool(cfg.get("per_task", False))
    per_window = bool(cfg.get("per_window", False))
    per_timestamp = bool(cfg.get("per_timestamp", False))

    run_dir = Path(run_dir)

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    benchmark_dir = run_dir / "benchmark"
    metrics_dir = run_dir / "metrics"
    traces_dir = run_dir / "traces"

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    if per_timestamp:
        metrics_dir.mkdir(parents=True, exist_ok=True)
    if cfg_traces.get("enable", False):
        traces_dir.mkdir(parents=True, exist_ok=True)

    need_benchmark_metrics = bool(per_task or per_window)
    accumulator = WindowMetricsAccumulator(task_name) if need_benchmark_metrics else None

    # ------------------------------------------------------------------
    # Per-timestamp CSV writer (MMT-native diagnostic)
    # ------------------------------------------------------------------
    f_ts = None
    wr_ts = None
    if per_timestamp:
        csv_ts = metrics_dir / f"{task_name}_metrics_per_timestamp.csv"
        f_ts = csv_ts.open("w", newline="")
        wr_ts = csv.writer(f_ts)
        wr_ts.writerow(
            [
                "shot_id",
                "window_index",
                "time_id",
                "feature_name",
                "RMSE",
                "MSE",
                "MAE",
            ]
        )

    # ------------------------------------------------------------------
    # Traces collector (MMT-native diagnostic)
    # ------------------------------------------------------------------
    do_traces = bool(cfg_traces.get("enable", False))
    n_max = int(cfg_traces.get("n_max", 10))
    signals_filter = cfg_traces.get("signals", None)
    time_indexes = cfg_traces.get("times_indexes", None)

    selected_shots: set[int] = set()
    collected: Dict[int, Dict[str, list]] = {}

    # ------------------------------------------------------------------
    # Main evaluation loop (single pass)
    # ------------------------------------------------------------------
    n_windows = 0
    next_log_at = _LOG_INTERVAL

    with torch.no_grad():
        for batch in dataloader:
            y_true, y_pred, y_mask, shot_ids, window_indices = forward_decode_native(
                batch=batch,
                model=model,
                device=device,
                stats=stats,
                codecs=codecs,
                id_to_name=id_to_name,
                amp_enabled=amp_enabled,
            )

            B = len(shot_ids)
            n_windows += B

            # Log sparsely to avoid spam on large evaluations
            if n_windows >= next_log_at:
                logger.info("Evaluated %d windows so far", next_log_at)
                next_log_at += _LOG_INTERVAL

            # ------------------------------------------------------------------
            # Benchmark per-window metrics (buffered in memory)
            # ------------------------------------------------------------------
            if need_benchmark_metrics:
                for out_name in stats.keys():
                    if out_name not in y_true or out_name not in y_pred:
                        continue
                    if out_name not in y_mask:
                        continue

                    mask_b = _reduce_mask(y_mask[out_name])
                    if not mask_b.any():
                        continue

                    idx = np.where(mask_b)[0]
                    y_t = y_true[out_name][idx].reshape(len(idx), -1)
                    y_p = y_pred[out_name][idx].reshape(len(idx), -1)

                    if accumulator is not None:
                        accumulator.add_batch(
                            y_target=y_t,
                            y_pred=y_p,
                            shot_id=shot_ids[idx],
                            window_index=window_indices[idx],
                            feature_name=out_name,
                        )

            # ------------------------------------------------------------------
            # Per-timestamp metrics CSV
            # ------------------------------------------------------------------
            if wr_ts is not None:
                for out_name in stats.keys():
                    if out_name not in y_true or out_name not in y_pred:
                        continue
                    if out_name not in y_mask:
                        continue

                    mask_b = _reduce_mask(y_mask[out_name])
                    if not mask_b.any():
                        continue

                    idx = np.where(mask_b)[0]
                    for b in idx:
                        diff = y_pred[out_name][b] - y_true[out_name][b]
                        diff2 = diff.reshape(-1, diff.shape[-1])
                        mse_t = np.mean(diff2 * diff2, axis=0)
                        rmse_t = np.sqrt(mse_t)
                        mae_t = np.mean(np.abs(diff2), axis=0)

                        for t in range(mse_t.shape[0]):
                            wr_ts.writerow(
                                [
                                    int(shot_ids[b]),
                                    int(window_indices[b]),
                                    int(t),
                                    out_name,
                                    float(rmse_t[t]),
                                    float(mse_t[t]),
                                    float(mae_t[t]),
                                ]
                            )

            # ------------------------------------------------------------------
            # Traces collection
            # ------------------------------------------------------------------
            if do_traces:
                for b in range(B):
                    sid = int(shot_ids[b])

                    # Only collect for up to n_max distinct shots.
                    if sid not in selected_shots:
                        if len(selected_shots) >= n_max:
                            continue
                        selected_shots.add(sid)

                    collected.setdefault(sid, {})

                    # Decide which outputs to trace
                    out_names = (
                        signals_filter
                        if signals_filter is not None
                        else list(y_pred.keys())
                    )

                    for out_name in out_names:
                        if out_name not in y_true or out_name not in y_pred:
                            continue
                        if out_name not in y_mask:
                            continue

                        mask_b = _reduce_mask(y_mask[out_name])
                        if not bool(mask_b[b]):
                            continue

                        true_arr = y_true[out_name][b]
                        pred_arr = y_pred[out_name][b]

                        # Optional time sub-sampling inside each window
                        if time_indexes is not None:
                            true_arr = true_arr[..., time_indexes]
                            pred_arr = pred_arr[..., time_indexes]

                        collected[sid].setdefault(out_name, []).append(
                            (int(window_indices[b]), true_arr, pred_arr)
                        )

    if f_ts is not None:
        f_ts.close()

    # ------------------------------------------------------------------
    # Save traces
    # ------------------------------------------------------------------
    if do_traces:
        for sid, outputs in collected.items():
            for out_name, triples in outputs.items():
                if not triples:
                    continue
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

    # ------------------------------------------------------------------
    # Benchmark task aggregation
    # ------------------------------------------------------------------
    result: Dict[str, Any] = {
        "benchmark_dir": str(benchmark_dir),
        "task": task_name,
    }

    if need_benchmark_metrics:
        if accumulator is None or accumulator.is_empty():
            logger.warning("No benchmark windows were collected for task %s", task_name)
            return result

        df = cast(
            Any,
            compute_metrics(
                task=task_name,
                output_dir=str(benchmark_dir),
                window_metrics_accumulator=accumulator,
                save_windows_metrics=per_window,
                save_task_metrics=per_task,
            ),
        )
        # Return a small, JSON-friendly summary for logging.
        if per_task and task_name in df.index:
            result["task_metrics"] = {
                k: float(v) for k, v in df.loc[task_name].to_dict().items()
            }

    return result
