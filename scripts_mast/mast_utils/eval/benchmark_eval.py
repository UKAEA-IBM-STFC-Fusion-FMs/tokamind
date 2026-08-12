"""scripts_mast.mast_utils.eval.benchmark_eval

Benchmark-aligned evaluation utilities.

This module lives in the *scripts_mast* integration layer on purpose:

- The core library `mmt/` stays dataset / benchmark agnostic.
- The MAST benchmark repository is the source of truth for the *official* evaluation aggregation, via its evaluator
  helpers:
  window -> signal-within-shot -> task-within-shot -> task -> group
  with equal-weight means/stds across shots (for signals/tasks), and group summaries computed as mean(task means) and
  mean(task stds).

What this module provides
-------------------------
One *single-pass* evaluation loop that can produce:

- Task metrics:
    - per-window: `windows_metrics.csv` (optional)
    - per-shot:   `shots_metrics.csv` (optional)
    - per-task:   `task_metrics.csv` (optional)
    - per-timestamp: `timestamps_metrics.csv` (optional)

  written under:

    `<eval_run_dir>/metrics/<task>/`

- Optional traces diagnostics:
    - qualitative traces (NPZ)

  written under:

    `<eval_run_dir>/traces/`

The loop uses `mmt.eval.forward.forward_decode_native` so decoding and de-standardization remain in `mmt/`.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
import numpy as np

import torch
from torch.utils.data import DataLoader

from mmt.eval.forward import forward_decode_native

from ..benchmark_imports import WindowMetricsAccumulator, compute_metrics
from .crps_metrics import CrpsAccumulator, compute_crps_positions
from .grad_shafranov.constants import DEFAULT_GS_PARAMS_FILE
from .grad_shafranov.grid import GSEvalGrid
from .grad_shafranov.gs_metrics import GSMetrics
from .grad_shafranov.psi_error_map import PsiErrorMapMetric
from .grad_shafranov.smoothness_metric import SmoothnessMetric
from .grad_shafranov.topology_metric import GSTopologyMetric
from .grad_shafranov.topology_provider import TopologyProvider

# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Eval")

_LOG_INTERVAL = 50000
TIMESTAMPS_METRICS_FILE = "timestamps_metrics.csv"


# ----------------------------------------------------------------------------------------------------------------------
def _reduce_mask(mask: np.ndarray) -> np.ndarray:
    """
    Reduce a possibly high-rank mask to shape (B,) via OR over extra dims.

    Parameters
    ----------
    mask : np.ndarray
        Input mask to be reduced.

    Returns
    -------
    np.ndarray
        Reduced mask.

    """

    if mask.ndim == 1:
        return np.asarray(mask, dtype=bool)

    reduced = mask.reshape(mask.shape[0], -1).any(axis=1)

    return np.asarray(reduced, dtype=bool)


# ----------------------------------------------------------------------------------------------------------------------
def _combined_output_mask(masks: Mapping[str, np.ndarray], output_names: tuple[str, ...]) -> np.ndarray | None:
    """Return the intersection of available per-window masks for outputs consumed by one joint metric."""

    combined: np.ndarray | None = None
    for output_name in output_names:
        if output_name not in masks:
            continue
        output_mask = _reduce_mask(mask=masks[output_name])
        combined = output_mask if combined is None else (combined & output_mask)
    return combined


# ----------------------------------------------------------------------------------------------------------------------
def evaluate_benchmark_and_diagnostics(  # NOSONAR - Ignore cognitive complexity
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    stats: Mapping[str, Mapping[str, float]],
    decoders: Mapping[str, Any],
    id_to_name: Mapping[int, str],
    run_dir: Path,
    task_name: str,
    amp_enabled: bool,
    compute_metrics_cfg: Mapping[str, Any] | None = None,
    traces_cfg: Mapping[str, Any] | None = None,
    source_store_manager: Any | None = None,
    source_zarr_root: str | None = None,
    source_local: bool = True,
) -> dict[str, Any]:
    """Run evaluation once and write configured outputs.

    Parameters
    ----------
    model : torch.nn.Module
        Standard evaluation model input.
    dataloader : DataLoader
        Standard evaluation dataloader input.
    device : torch.device
        Standard evaluation device input.
    stats : Mapping[str, Mapping[str, float]]
        Per-signal stats dict with ``"mean"`` and ``"std"`` keys.
    decoders : Mapping[str, TorchDecoder]
        Pre-built per-signal ``TorchDecoder`` instances keyed by signal name,
        as returned by ``build_decoders()``.
    id_to_name : Mapping[int, str]
        Mapping from signal_id to signal name.
    run_dir : Path
        Eval run directory (`runs/<train_run>/<eval_id>/`).
    task_name : str
        Benchmark task name (e.g., `task_2-1`) used for output folder naming.
    amp_enabled : bool
        Whether to enable AMP in the forward pass.
    compute_metrics_cfg : Mapping[str, Any] | None
        Supports keys:
          - per_task: bool (benchmark aggregation -> task_metrics.csv)
          - per_shot: bool (benchmark aggregation -> shots_metrics.csv)
          - per_window: bool (keep windows_metrics.csv)
          - per_timestamp: bool (MMT-native per-timestamp CSV)
          - crps: {enable: bool, n_samples: int} (MMT-native native-space CRPS CSV; n_samples >= 2 for a Gaussian
            head, ignored (CRPS = MAE) for a deterministic head)
    traces_cfg : Mapping[str, Any] | None
        Same structure as in docs/evaluation.md.
    source_zarr_root : str | None
        Optional directory holding source ``<shot_id>.zarr`` equilibrium stores (``data.local_path``), used only as
        a standalone/local fallback when ``source_store_manager`` is unavailable.
    source_store_manager : Any | None
        The storage manager owned by the evaluation ``MastDataset``. Eval-only reference metrics use it to open the
        same local or remote stores as the data pipeline.
    source_local : bool
        Whether the source dataset uses local stores. Passed to ``source_store_manager`` when reference geometry is
        fetched remotely or locally.
        Optional. Default: True.

    Returns
    -------
    dict[str, Any]
        Small summary of what was written (paths, and benchmark task metrics if available).

    """

    cfg = compute_metrics_cfg or {}
    cfg_traces = traces_cfg or {}

    per_task = bool(cfg.get("per_task", False))
    per_shot = bool(cfg.get("per_shot", False))
    per_window = bool(cfg.get("per_window", False))
    per_timestamp = bool(cfg.get("per_timestamp", False))

    # Native-space CRPS (MMT-native diagnostic). Enabled explicitly via crps.enable; n_samples tunes the sampled
    # estimator for a Gaussian head (>= 2 required) and is ignored for a deterministic head (CRPS = MAE).
    crps_cfg = cfg.get("crps") or {}
    want_crps = bool(crps_cfg.get("enable", False))
    crps_n_samples = int(crps_cfg.get("n_samples", 0) or 0)
    gs_metrics_cfg = cfg.get("gs_metrics") or {}
    want_gs_metrics = bool(gs_metrics_cfg.get("enable", False))
    smoothness_cfg = cfg.get("smoothness") or {}
    want_smoothness = bool(smoothness_cfg.get("enable", False))
    smoothness_sigma = float(smoothness_cfg.get("sigma", 1.0))
    topology_cfg = cfg.get("gs_topology") or {}
    want_topology = bool(topology_cfg.get("enable", False))
    psi_error_map_cfg = cfg.get("psi_error_map") or {}
    want_psi_error_map = bool(psi_error_map_cfg.get("enable", False))

    run_dir = Path(run_dir)

    # ..................................................................................................................
    # Output directories
    # ..................................................................................................................

    metrics_root_dir = run_dir / "metrics"
    metrics_task_dir = metrics_root_dir / task_name
    traces_dir = run_dir / "traces"

    need_benchmark_metrics = bool(per_task or per_shot or per_window)
    need_metrics_task_dir = bool(
        need_benchmark_metrics
        or per_timestamp
        or want_crps
        or want_gs_metrics
        or want_smoothness
        or want_topology
        or want_psi_error_map
    )
    if need_metrics_task_dir:
        metrics_task_dir.mkdir(parents=True, exist_ok=True)
    if cfg_traces.get("enable", False):
        traces_dir.mkdir(parents=True, exist_ok=True)

    accumulator = WindowMetricsAccumulator(task=task_name) if need_benchmark_metrics else None

    # ..................................................................................................................
    # Shared MAST GS geometry/operator. Construct it once so GS diagnostics use identical region and stencils.
    # ..................................................................................................................
    gs_grid = None
    if want_gs_metrics or want_smoothness or want_topology or want_psi_error_map:
        try:
            gs_grid = GSEvalGrid(DEFAULT_GS_PARAMS_FILE)
        except Exception as exc:  # noqa: BLE001 - optional diagnostics must not break official eval
            logger.warning("Grad-Shafranov diagnostics disabled (grid initialization failed): %s", exc)

    # ..................................................................................................................
    # Grad-Shafranov equation and operator metrics (additive; companion CSVs only).
    # Enabled explicitly by eval.compute_metrics.gs_metrics.enable. Psi-only models receive the Delta* psi error;
    # predicted-j_tor models additionally receive GS self-consistency and RHS-current error diagnostics.
    # ..................................................................................................................
    gs_metric = None
    if want_gs_metrics and gs_grid is not None:
        try:
            gs_metric = GSMetrics(grid=gs_grid)
            metrics_task_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - metric must never break official eval
            logger.warning("GS metrics disabled (init failed): %s", exc)
            gs_metric = None

    # ..................................................................................................................
    # ..................................................................................................................
    # Smoothness metric (additive; companion CSVs only).
    # Enabled explicitly by eval.compute_metrics.smoothness.enable. Psi-only models receive psi and Delta* psi
    # roughness; j_tor roughness is additionally available when that output is predicted.
    # ..................................................................................................................
    sm_metric = None
    if want_smoothness and gs_grid is not None:
        try:
            sm_metric = SmoothnessMetric(
                sigma=smoothness_sigma,
                grid=gs_grid,
            )
            metrics_task_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - metric must never break official eval
            logger.warning("Smoothness metric disabled (init failed): %s", exc)
            sm_metric = None

    psi_error_map_metric = None
    if want_psi_error_map and gs_grid is not None:
        try:
            psi_error_map_metric = PsiErrorMapMetric(grid=gs_grid)
        except Exception as exc:  # noqa: BLE001 - optional diagnostic must not break official eval
            logger.warning("Psi error-map metric disabled (init failed): %s", exc)

    # ..................................................................................................................
    # LCFS / X-point flux-topology metric (eval-only; companion CSVs only).
    # Enabled explicitly by eval.compute_metrics.gs_topology.enable. It scores predicted psi against source EFIT
    # geometry fetched post-batch by a TopologyProvider, keyed by (shot_id, output_time). The provider reads only
    # the equilibrium keys backing the metric's declared required auxiliary fields, and is built only when a source
    # manager/root is available — so training (no enabled metric) opens no Zarr at all.
    # ..................................................................................................................
    topology_metric = None
    topology_provider = None
    if want_topology and gs_grid is not None:
        try:
            topology_metric = GSTopologyMetric(grid=gs_grid)
            metrics_task_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001 - metric must never break official eval
            logger.warning("Topology metric disabled (init failed): %s", exc)
            topology_metric = None

    if topology_metric is not None:
        # Union the required auxiliary signals over the enabled reference metrics (only topology today) and build
        # one provider for exactly that union. Guarded: a provider failure disables the metric, never eval.
        required_auxiliary: set[str] = set()
        for metric in (topology_metric,):
            required_auxiliary |= set(getattr(metric, "required_auxiliary_fields", frozenset()))
        if required_auxiliary and (source_store_manager is not None or source_zarr_root):
            try:
                topology_provider = TopologyProvider(
                    requested_fields=required_auxiliary,
                    store_manager=source_store_manager,
                    store_root=source_zarr_root,
                    local=source_local,
                )
            except Exception as exc:  # noqa: BLE001 - provider must never break official eval
                logger.warning("Topology provider disabled (init failed): %s", exc)
                topology_provider = None
        else:
            logger.warning(
                "Topology metric enabled but no source storage manager or Zarr root is available "
                "(source_store_manager=%s, source_zarr_root=%s); disabling it.",
                type(source_store_manager).__name__ if source_store_manager is not None else None,
                source_zarr_root,
            )
            topology_metric = None

    # ..................................................................................................................
    # Per-timestamp CSV writer (MMT-native diagnostic)
    # ..................................................................................................................

    f_ts = None
    wr_ts = None
    if per_timestamp:
        csv_ts = metrics_task_dir / TIMESTAMPS_METRICS_FILE
        f_ts = csv_ts.open(mode="w", newline="")
        wr_ts = csv.writer(f_ts)
        wr_ts.writerow(["shot_id", "window_index", "time_id", "feature_name", "RMSE", "MSE", "MAE"])

    # ..................................................................................................................
    # Native CRPS diagnostic. Scoring/aggregation/IO live in crps_metrics.CrpsAccumulator (this eval subpackage);
    # this loop only computes per-window per-position CRPS (it has the model samples) and feeds the accumulator.
    # ..................................................................................................................

    # For a deterministic model the predictive distribution is a point mass, so CRPS reduces exactly to the MAE
    # |pred - target|; we compute it directly (no sampling) rather than skipping the metric, so deterministic and
    # probabilistic models land on the same CRPS axis.
    crps_is_mae = want_crps and (getattr(model, "output_adapter_type", "deterministic") == "deterministic")
    if want_crps and (not crps_is_mae) and (crps_n_samples < 2):
        raise ValueError(
            "eval.compute_metrics.crps.enable=true for a probabilistic model requires crps.n_samples >= 2 "
            f"(the CRPS cross-term is undefined for fewer samples); got n_samples={crps_n_samples}."
        )

    # Per-signal std for NCRPS — the same benchmark-derived std tokamark normalizes NRMSE by (NRMSE = RMSE / std).
    crps_signal_std = {name: float(st.get("std", 0.0) or 0.0) for name, st in stats.items()}

    crps_acc = None
    if want_crps:
        crps_acc = CrpsAccumulator(
            metrics_dir=metrics_task_dir,
            task_name=task_name,
            signal_std=crps_signal_std,
            per_task=per_task,
            per_shot=per_shot,
            per_window=per_window,
            per_timestamp=per_timestamp,
        )
        if crps_is_mae:
            logger.warning(
                "eval.compute_metrics.crps.enable=true but the model is deterministic; CRPS reduces to MAE for a "
                "point forecast — computing |pred - target| (no sampling, n_samples ignored)."
            )
    # If crps.enable=true but no granularity flag is set, there is nothing to write — skip sampling/decoding entirely.
    crps_any_output = bool(crps_acc is not None and crps_acc.any_output)
    logger.info(
        "Metrics: per_task=%s | per_shot=%s | per_window=%s | per_timestamp=%s",
        per_task,
        per_shot,
        per_window,
        per_timestamp,
    )
    if want_crps:
        logger.info(
            "CRPS evaluation: mode=%s | n_samples=%d",
            "point" if crps_is_mae else "sample",
            int(crps_n_samples),
        )

    # ..................................................................................................................
    # Traces collector (MMT-native diagnostic)
    # ..................................................................................................................

    do_traces = bool(cfg_traces.get("enable", False))
    n_max = int(cfg_traces.get("n_max", 10))
    signals_filter: list[str] | None = cfg_traces.get("signals")
    time_indexes = cfg_traces.get("times_indexes")

    selected_shots: set[int] = set()
    collected: dict[int, dict[str, list]] = {}

    # ..................................................................................................................
    # Main evaluation loop (single pass)
    # ..................................................................................................................

    n_windows = 0
    next_log_at = _LOG_INTERVAL

    with torch.no_grad():
        for batch in dataloader:
            y_true, y_pred, y_mask, shot_ids, window_indices, y_pred_samples, output_time = forward_decode_native(
                batch=batch,
                model=model,
                device=device,
                stats=stats,
                decoders=decoders,
                id_to_name=id_to_name,
                amp_enabled=amp_enabled,
                n_samples=crps_n_samples if (crps_any_output and not crps_is_mae) else 0,
            )

            B = len(shot_ids)
            n_windows += B

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # GS equation/operator metrics, per-shot accumulation.
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if gs_metric is not None and "equilibrium-psi" in y_pred:
                gs_output_names = ["equilibrium-psi"]
                if "equilibrium-j_tor" in y_pred:
                    gs_output_names.append("equilibrium-j_tor")
                win_mask = _combined_output_mask(masks=y_mask, output_names=tuple(gs_output_names))
                try:
                    gs_metric.add_batch(
                        psi_native=y_pred["equilibrium-psi"],
                        shot_ids=shot_ids,
                        jtor_native=y_pred.get("equilibrium-j_tor"),
                        window_mask=win_mask,
                        psi_gt_native=y_true.get("equilibrium-psi"),
                        jtor_gt_native=y_true.get("equilibrium-j_tor"),
                    )
                except Exception as exc:  # noqa: BLE001 - never break official eval
                    logger.warning("GS metrics: skipped a batch (%s)", exc)
                    gs_metric = None  # disable for the rest of the run to avoid log spam

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Smoothness metric, per-shot accumulation.
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if sm_metric is not None and "equilibrium-psi" in y_pred:
                smoothness_output_names = ["equilibrium-psi"]
                if "equilibrium-j_tor" in y_pred:
                    smoothness_output_names.append("equilibrium-j_tor")
                win_mask_sm = _combined_output_mask(masks=y_mask, output_names=tuple(smoothness_output_names))
                try:
                    sm_metric.add_batch(
                        psi_native=y_pred["equilibrium-psi"],
                        shot_ids=shot_ids,
                        jtor_native=y_pred.get("equilibrium-j_tor"),
                        window_mask=win_mask_sm,
                        psi_gt_native=y_true.get("equilibrium-psi"),
                        jtor_gt_native=y_true.get("equilibrium-j_tor"),
                    )
                except Exception as exc:  # noqa: BLE001 - never break official eval
                    logger.warning("Smoothness metric: skipped a batch (%s)", exc)
                    sm_metric = None

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Aggregate full-grid psi reconstruction error map (one task-level artifact; no per-window grids kept).
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if psi_error_map_metric is not None and "equilibrium-psi" in y_pred:
                win_mask_psi = _combined_output_mask(masks=y_mask, output_names=("equilibrium-psi",))
                try:
                    psi_error_map_metric.add_batch(
                        psi_native=y_pred["equilibrium-psi"],
                        psi_gt_native=y_true.get("equilibrium-psi"),
                        window_mask=win_mask_psi,
                    )
                except Exception as exc:  # noqa: BLE001 - never break official eval
                    logger.warning("Psi error-map metric: skipped a batch (%s)", exc)
                    psi_error_map_metric = None

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # LCFS / X-point flux-topology metric (predicted psi vs reference geometry), per-shot accumulation.
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if (
                topology_metric is not None
                and topology_provider is not None
                and ("equilibrium-psi" in y_pred)
                and (output_time is not None)
                and ("equilibrium-psi" in output_time)
            ):
                win_mask_topo = _combined_output_mask(masks=y_mask, output_names=("equilibrium-psi",))
                try:
                    topo = topology_provider.query(shot_ids, output_time["equilibrium-psi"])
                    topology_metric.add_batch(
                        psi_native=y_pred["equilibrium-psi"],
                        shot_ids=shot_ids,
                        topology=topo,
                        window_mask=win_mask_topo,
                        psi_gt_native=y_true.get("equilibrium-psi"),
                    )
                except Exception as exc:  # noqa: BLE001 - never break official eval
                    logger.warning("Topology metric: skipped a batch (%s)", exc)
                    topology_metric = None  # disable for the rest of the run to avoid log spam

            # Log sparsely to avoid spam on large evaluations
            if n_windows >= next_log_at:
                logger.info("Evaluated %d windows so far", next_log_at)
                next_log_at += _LOG_INTERVAL

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Benchmark per-window metrics (buffered in memory)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            if need_benchmark_metrics:
                for out_name in stats.keys():
                    if (out_name not in y_true) or (out_name not in y_pred):
                        continue
                    if out_name not in y_mask:
                        continue

                    mask_b = _reduce_mask(mask=y_mask[out_name])
                    if not mask_b.any():
                        continue

                    idx = np.nonzero(mask_b)[0]
                    y_t = y_true[out_name][idx].reshape(len(idx), -1)
                    y_p = y_pred[out_name][idx].reshape(len(idx), -1)

                    if accumulator is not None:
                        accumulator.add_batch(
                            y_target=y_t,
                            y_pred=y_p,
                            shot_ids=shot_ids[idx],
                            window_indices=window_indices[idx],
                            feature_name=out_name,
                        )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Per-timestamp metrics CSV
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if wr_ts is not None:
                for out_name in stats.keys():
                    if (out_name not in y_true) or (out_name not in y_pred):
                        continue
                    if out_name not in y_mask:
                        continue

                    mask_b = _reduce_mask(mask=y_mask[out_name])
                    if not mask_b.any():
                        continue

                    idx = np.nonzero(mask_b)[0]
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

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Native CRPS. Gaussian head -> sampled estimator; deterministic head -> MAE (= point CRPS).
            # Streams per-timestamp / per-window rows and accumulates per-(signal, shot) for shot/task aggregation.
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

            if crps_any_output:
                for out_name in stats.keys():
                    if (out_name not in y_true) or (out_name not in y_mask):
                        continue
                    if crps_is_mae:
                        # MAE path reads the decoded mean y_pred[out_name]; skip signals without one.
                        if out_name not in y_pred:
                            continue
                    elif (y_pred_samples is None) or (out_name not in y_pred_samples):
                        continue

                    mask_b = _reduce_mask(mask=y_mask[out_name])
                    if not mask_b.any():
                        continue

                    idx = np.nonzero(mask_b)[0]
                    # Scoring (point vs sampled CRPS) is owned by crps_metrics; the loop just passes the arrays it has.
                    samples_sel = None if crps_is_mae else y_pred_samples[out_name][idx]  # (n_sel, S, *native)
                    crps_pos = compute_crps_positions(
                        y_true=y_true[out_name][idx],  # (n_sel, *native_shape)
                        y_pred=y_pred[out_name][idx] if crps_is_mae else None,
                        y_samples=samples_sel,
                    )  # (n_sel, *native_shape)

                    crps_acc.add_batch(
                        feature_name=out_name,
                        shot_ids=shot_ids[idx],
                        window_indices=window_indices[idx],
                        crps_pos=crps_pos,
                    )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Traces collection
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
                    out_names = signals_filter if signals_filter is not None else list(y_pred.keys())

                    for out_name in out_names:
                        if (out_name not in y_true) or (out_name not in y_pred):
                            continue
                        if out_name not in y_mask:
                            continue

                        mask_b = _reduce_mask(mask=y_mask[out_name])
                        if not bool(mask_b[b]):
                            continue

                        true_arr = y_true[out_name][b]
                        pred_arr = y_pred[out_name][b]

                        # Native predictive samples for confidence bands, when a Gaussian head produced them.
                        samples_arr = None
                        if (y_pred_samples is not None) and (out_name in y_pred_samples):
                            samples_arr = y_pred_samples[out_name][b]  # (S, *native_shape)

                        # Optional time subsampling inside each window (last axis = time)
                        if time_indexes is not None:
                            true_arr = true_arr[..., time_indexes]
                            pred_arr = pred_arr[..., time_indexes]
                            if samples_arr is not None:
                                samples_arr = samples_arr[..., time_indexes]

                        collected[sid].setdefault(out_name, []).append(
                            (int(window_indices[b]), true_arr, pred_arr, samples_arr)
                        )

    if f_ts is not None:
        f_ts.close()

    # Close CRPS streaming writers, write shot/task CSVs, and get the shot-weighted task summary.
    crps_summary = crps_acc.finalize() if crps_acc is not None else None

    if n_windows > 0:
        logger.info("Evaluated %d total windows", n_windows)

    # ..................................................................................................................
    # Save traces
    # ..................................................................................................................

    if do_traces:
        for sid, outputs in collected.items():
            for out_name, records in outputs.items():
                if not records:
                    continue
                records.sort(key=lambda x: x[0])
                window_idx_arr = np.asarray([w for w, _, _, _ in records], dtype=np.int64)
                true_stack = np.stack([t for _, t, _, _ in records], axis=0)
                pred_stack = np.stack([p for _, _, p, _ in records], axis=0)

                arrays = {
                    "true": true_stack,
                    "pred": pred_stack,
                    "window_index": window_idx_arr,
                }

                # Save native predictive samples (B_windows, S, *native) for confidence-band plotting, when every
                # traced window for this signal carries them (Gaussian head + crps enabled).
                samples_list = [s for _, _, _, s in records]
                if all(s is not None for s in samples_list):
                    arrays["pred_samples"] = np.stack(samples_list, axis=0)

                np.savez(traces_dir / f"{sid}__{out_name}.npz", **arrays)

    # ..................................................................................................................
    # Task aggregation
    # ..................................................................................................................

    result: dict[str, Any] = {
        "metrics_task_dir": str(metrics_task_dir),
        "metrics_root_dir": str(metrics_root_dir),
        "task": task_name,
    }
    if crps_summary is not None:
        result["crps_metrics"] = crps_summary

    if need_benchmark_metrics:
        if (accumulator is None) or accumulator.is_empty():
            logger.warning("No benchmark windows were collected for task %s", task_name)
        else:
            df = cast(
                Any,
                compute_metrics(
                    task=task_name,
                    output_dir=str(metrics_root_dir),
                    window_metrics_accumulator=accumulator,
                    save_windows_metrics=per_window,
                    save_shot_metrics=per_shot,
                    save_task_metrics=per_task,
                ),
            )
            if per_task and (task_name in df.index):
                result["task_metrics"] = {k: float(v) for k, v in df.loc[task_name].to_dict().items()}

    # ..................................................................................................................
    # GS equation/operator companion CSVs (independent of benchmark metrics).
    # ..................................................................................................................
    if gs_metric is not None and not gs_metric.is_empty():
        try:
            gs_summary = gs_metric.write_csvs(metrics_task_dir=metrics_task_dir, task_name=task_name)
            result["gs_metrics"] = gs_summary
            logger.info(
                "GS metrics (%s): self-consistency FD=%.4e | Delta* psi FD error=%.4e | "
                "RHS j_tor error=%.4e over %d shots",
                task_name,
                gs_summary.get("gs_self_consistency_fd_mean_abs_per_cell", float("nan")),
                gs_summary.get("dstar_psi_fd_error_mean_abs_per_cell", float("nan")),
                gs_summary.get("gs_rhs_jtor_error_mean_abs_per_cell", float("nan")),
                gs_summary.get("n_shots", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("GS metrics: failed to write CSVs (%s)", exc)

    # ..................................................................................................................
    # Smoothness companion CSVs (independent of benchmark metrics).
    # ..................................................................................................................
    if sm_metric is not None and not sm_metric.is_empty():
        try:
            sm_summary = sm_metric.write_csvs(metrics_task_dir=metrics_task_dir, task_name=task_name)
            result["smoothness"] = sm_summary
            logger.info(
                "Smoothness (%s, sigma=%.2f): psi=%.4e (gt %.4e) | j_tor=%.4e (gt %.4e) | "
                "dstar_fd=%.4e (gt %.4e) | dstar_A=%.4e (gt %.4e) over %d shots",
                task_name,
                sm_summary.get("sigma", float("nan")),
                sm_summary.get("roughness_psi_rel_pred", float("nan")),
                sm_summary.get("roughness_psi_rel_gt", float("nan")),
                sm_summary.get("roughness_jtor_rel_pred", float("nan")),
                sm_summary.get("roughness_jtor_rel_gt", float("nan")),
                sm_summary.get("roughness_dstar_psi_fd_rel_pred", float("nan")),
                sm_summary.get("roughness_dstar_psi_fd_rel_gt", float("nan")),
                sm_summary.get("roughness_dstar_psi_A_rel_pred", float("nan")),
                sm_summary.get("roughness_dstar_psi_A_rel_gt", float("nan")),
                sm_summary.get("n_shots", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Smoothness metric: failed to write CSVs (%s)", exc)

    if psi_error_map_metric is not None:
        if psi_error_map_metric.is_empty():
            logger.warning("Psi error-map metric: no finite ground-truth psi cells were accumulated.")
        else:
            try:
                psi_error_map_summary = psi_error_map_metric.write_npz(
                    metrics_task_dir=metrics_task_dir,
                    task_name=task_name,
                )
                result["psi_error_map"] = psi_error_map_summary
                logger.info(
                    "Psi error map (%s): %d valid grid cells across %d contributing fields -> %s",
                    task_name,
                    psi_error_map_summary["n_valid_cells"],
                    psi_error_map_summary["n_fields_contributed"],
                    psi_error_map_summary["psi_error_map_npz"],
                )
            except Exception as exc:  # noqa: BLE001 - optional diagnostic must not break official eval
                logger.warning("Psi error-map metric: failed to write NPZ (%s)", exc)

    # ..................................................................................................................
    # LCFS / X-point flux-topology companion CSVs (independent of benchmark metrics).
    # ..................................................................................................................
    # Write whenever the metric was enabled — even with zero usable slices. The coverage/skip counters are
    # exactly what explains an empty result, so suppressing the CSV in that case would hide the diagnosis.
    if topology_metric is not None:
        try:
            topology_summary = topology_metric.write_csvs(metrics_task_dir=metrics_task_dir, task_name=task_name)
            result["gs_topology"] = topology_summary
            logger.info(
                "Topology (%s): LCFS mean=%.4e max=%.4e | X-point=%.4e | near-X LCFS=%.4e | "
                "constancy pred=%.4e gt=%.4e over %d shots "
                "(attempted=%d, finite=%d, skipped_alignment=%d, skipped_geometry=%d)",
                task_name,
                topology_summary.get("topology_lcfs_mean_abs", float("nan")),
                topology_summary.get("topology_lcfs_max_abs", float("nan")),
                topology_summary.get("topology_x_point_abs", float("nan")),
                topology_summary.get("topology_near_x_point_lcfs_mean_abs", float("nan")),
                topology_summary.get("topology_constancy_pred_abs", float("nan")),
                topology_summary.get("topology_constancy_gt_abs", float("nan")),
                topology_summary.get("n_shots", 0),
                topology_summary.get("n_slices_attempted", 0),
                topology_summary.get("n_slices_finite", 0),
                topology_summary.get("n_slices_skipped_alignment", 0),
                topology_summary.get("n_slices_skipped_geometry", 0),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Topology metric: failed to write CSVs (%s)", exc)

    return result
