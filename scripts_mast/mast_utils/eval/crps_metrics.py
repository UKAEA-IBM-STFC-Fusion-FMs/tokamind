"""
MAST/tokamark-style native-space CRPS aggregation and reporting.

This is the MAST benchmark's CRPS *reporting convention* — the window -> shot -> signal -> task hierarchy, the
tokamark-style CSV schema, and the NRMSE-matched normalization. It is deliberately in the scripts/MAST layer (not in
core ``mmt``) because that convention is benchmark-specific; the generic, reusable CRPS scoring rules live in
``mmt.scoring.crps`` and are shared with the training losses.

The orchestration loop (``benchmark_eval``) feeds decoded native predictions/targets per batch; this module computes
per-position CRPS (via ``mmt.scoring.crps``), streams the requested per-timestamp / per-window rows, and aggregates
shot/task metrics.

CRPS is the Continuous Ranked Probability Score in native physical units. For a Gaussian head it is estimated from
decoded predictive samples (fair sample estimator); for a deterministic head a point forecast's CRPS is exactly the
MAE, so it reduces to ``|pred - target|``. Both land on one comparable axis.

CRPS/NCRPS are *linear*, so they aggregate by plain mean over the same shot-weighted hierarchy NRMSE uses
(window -> shot -> signal -> task), not RMSE's square/sqrt path. ``NCRPS = CRPS / signal_std`` uses the same
per-signal std that normalizes NRMSE, so NCRPS and NRMSE/NMAE share a scale.

File layout (each gated by the matching ``per_*`` flag, written only when CRPS is enabled):
- ``crps_timestamps_metrics.csv`` : per (shot, window, time, signal) raw CRPS/NCRPS (plotting source)
- ``crps_window_metrics.csv``     : per (shot, window, signal) mean CRPS/NCRPS over valid positions
- ``crps_shot_metrics.csv``       : per (signal, shot) mean CRPS/NCRPS over the shot's windows
- ``crps_task_metrics.csv``       : per-signal mean +/- population-std across shots, plus a task aggregate row
"""

from __future__ import annotations

import csv
import logging
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mmt.scoring.crps import sample_crps

# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Eval")

CRPS_TASK_METRICS_FILE = "crps_task_metrics.csv"
CRPS_SHOT_METRICS_FILE = "crps_shot_metrics.csv"
CRPS_WINDOW_METRICS_FILE = "crps_window_metrics.csv"
CRPS_TIMESTAMPS_METRICS_FILE = "crps_timestamps_metrics.csv"


# ======================================================================================================================
# Scoring
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def compute_crps_positions(
    y_true: np.ndarray,
    y_pred: np.ndarray | None = None,
    y_samples: np.ndarray | None = None,
) -> np.ndarray:
    """
    Per-position native-space CRPS for a batch of windows, dispatching on what the model produced.

    The caller passes the decoded native target plus *either* a point prediction (deterministic head) *or* an
    ensemble of decoded samples (Gaussian head); this function picks the right estimator, so the orchestration loop
    does not need to know the CRPS math:

    - samples given  → fair sample-CRPS estimator over the sample axis (probabilistic forecast),
    - point pred only → ``|y_pred - y_true|`` (a point forecast's CRPS is exactly its absolute error).

    Parameters
    ----------
    y_true : np.ndarray
        Native targets, shape ``(B, *native_shape)``. NaN at missing positions (left as NaN in the output).
    y_pred : np.ndarray | None
        Native point predictions (mean), shape ``(B, *native_shape)``. Used when ``y_samples`` is None.
    y_samples : np.ndarray | None
        Native predictive samples, shape ``(B, S, *native_shape)``. Takes precedence when provided.

    Returns
    -------
    np.ndarray
        Per-position CRPS, shape ``(B, *native_shape)``.

    Raises
    ------
    ValueError
        If neither ``y_samples`` nor ``y_pred`` is provided.
    """

    if y_samples is not None:
        # Fair marginal sample-CRPS over the sample axis (axis 1).
        return sample_crps(
            torch.from_numpy(y_samples).float(),
            torch.from_numpy(y_true).float(),
            sample_dim=1,
        ).numpy()
    if y_pred is not None:
        # Point forecast: CRPS = |pred - target|. np.abs is the numpy-path equivalent of mmt.scoring.crps.point_crps
        # (which is the torch version for tensor callers); no need to round-trip through torch here.
        return np.abs(y_pred - y_true)
    raise ValueError("compute_crps_positions requires either y_samples (probabilistic) or y_pred (point).")


# ======================================================================================================================
# CSV writers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def write_crps_shot_csv(
    path: Path,
    per_shot: Mapping[str, list[tuple[int, float]]],
    signal_std: Mapping[str, float],
) -> None:
    """
    Write per-(signal, shot) CRPS/NCRPS rows (each value is the mean per-window CRPS over that shot's windows).

    Parameters
    ----------
    path : Path
        Output CSV path.
    per_shot : Mapping[str, list[tuple[int, float]]]
        Mapping signal name -> list of (shot_id, mean-CRPS-over-windows).
    signal_std : Mapping[str, float]
        Per-signal std used to normalize CRPS into NCRPS.

    Returns
    -------
    None
    """

    with path.open(mode="w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["feature_name", "shot_id", "CRPS", "NCRPS"])
        for name in sorted(per_shot):
            std_o = float(signal_std.get(name, 0.0) or 0.0)
            for shot, crps in sorted(per_shot[name]):
                ncrps = (crps / std_o) if std_o > 0.0 else float("nan")
                wr.writerow([name, int(shot), float(crps), ncrps])


# ----------------------------------------------------------------------------------------------------------------------
def write_crps_task_csv(
    path: Path,
    task_name: str,
    per_signal: Mapping[str, Mapping[str, float]],
    task_mean: float,
    task_std: float,
    task_ncrps_mean: float,
    task_ncrps_std: float,
) -> None:
    """
    Write per-signal CRPS/NCRPS (mean +/- population-std across shots) plus the task-level aggregate row.

    Parameters
    ----------
    path : Path
        Output CSV path.
    task_name : str
        Task identifier (used for the task-scope row).
    per_signal : Mapping[str, Mapping[str, float]]
        Per-signal aggregates with CRPS_mean/CRPS_std_pop/NCRPS_mean/NCRPS_std_pop/n_shots.
    task_mean, task_std, task_ncrps_mean, task_ncrps_std : float
        Task-level equal-weight-over-signals aggregates.

    Returns
    -------
    None
    """

    with path.open(mode="w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["scope", "name", "n_shots", "CRPS_mean", "CRPS_std_pop", "NCRPS_mean", "NCRPS_std_pop"])
        for name in sorted(per_signal):
            s = per_signal[name]
            wr.writerow(
                [
                    "signal",
                    name,
                    int(s["n_shots"]),
                    s["CRPS_mean"],
                    s["CRPS_std_pop"],
                    s["NCRPS_mean"],
                    s["NCRPS_std_pop"],
                ]
            )
        wr.writerow(["task", task_name, len(per_signal), task_mean, task_std, task_ncrps_mean, task_ncrps_std])


# ======================================================================================================================
# Accumulator
# ======================================================================================================================


# ======================================================================================================================
class CrpsAccumulator:
    """
    Accumulate native-space CRPS across an evaluation pass and write the requested per-granularity outputs.

    The orchestration loop calls :meth:`add_batch` once per batch/signal with per-position CRPS values
    (already computed in native space, NaN where the target is missing). Per-timestamp and per-window rows are
    streamed to disk; per-shot/per-task aggregates are accumulated and written by :meth:`finalize`.

    Mirrors the role of tokamark's ``WindowMetricsAccumulator`` for the point metrics, keeping CRPS scoring/IO out of
    the orchestration layer.

    Parameters
    ----------
    metrics_dir : Path
        Directory to write the ``crps_*_metrics.csv`` files into.
    task_name : str
        Task identifier (task-scope row in the task CSV).
    signal_std : Mapping[str, float]
        Per-signal std for NCRPS normalization (same normalizer as NRMSE).
    per_task, per_shot, per_window, per_timestamp : bool
        Granularity flags (shared with the point metrics). Determine which CSVs are produced.
    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        metrics_dir: Path,
        task_name: str,
        signal_std: Mapping[str, float],
        *,
        per_task: bool,
        per_shot: bool,
        per_window: bool,
        per_timestamp: bool,
    ) -> None:
        self._dir = Path(metrics_dir)
        self._task_name = str(task_name)
        self._signal_std = dict(signal_std)
        self._per_task = bool(per_task)
        self._per_shot = bool(per_shot)
        self._per_window = bool(per_window)
        self._per_timestamp = bool(per_timestamp)
        self._need_shot_or_task = self._per_shot or self._per_task

        # Running per-(signal, shot): sum and count of per-window CRPS means.
        self._shot_sum: dict[tuple[str, int], float] = {}
        self._shot_cnt: dict[tuple[str, int], int] = {}

        # Streaming writers (opened lazily so an unused granularity produces no file).
        self._f_ts = None
        self._wr_ts = None
        self._f_win = None
        self._wr_win = None
        if self._per_timestamp:
            self._f_ts = (self._dir / CRPS_TIMESTAMPS_METRICS_FILE).open(mode="w", newline="")
            self._wr_ts = csv.writer(self._f_ts)
            self._wr_ts.writerow(["shot_id", "window_index", "time_id", "feature_name", "CRPS", "NCRPS"])
        if self._per_window:
            self._f_win = (self._dir / CRPS_WINDOW_METRICS_FILE).open(mode="w", newline="")
            self._wr_win = csv.writer(self._f_win)
            self._wr_win.writerow(["shot_id", "window_index", "feature_name", "CRPS", "NCRPS"])

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def any_output(self) -> bool:
        """True if any granularity is requested (else there is nothing to compute/write)."""
        return self._per_timestamp or self._per_window or self._need_shot_or_task

    # ------------------------------------------------------------------------------------------------------------------
    def add_batch(
        self,
        *,
        feature_name: str,
        shot_ids: np.ndarray,
        window_indices: np.ndarray,
        crps_pos: np.ndarray,
    ) -> None:
        """
        Add a batch of per-position CRPS values for one signal.

        This computes per-window reductions for the whole batch in one numpy operation, then only loops over the scalar
        rows that must be written or accumulated.

        Parameters
        ----------
        feature_name : str
            Output signal name.
        shot_ids : np.ndarray
            Shot identifiers, shape ``(B,)``.
        window_indices : np.ndarray
            Window indices within each shot, shape ``(B,)``.
        crps_pos : np.ndarray
            Per-position CRPS in native space, shape ``(B, *native_shape)`` with the last axis = time. NaN at
            positions where the target was missing (excluded via ``nanmean``).

        Returns
        -------
        None
        """

        if crps_pos.shape[0] == 0:
            return
        if crps_pos.ndim < 2:
            raise ValueError("CrpsAccumulator.add_batch expects crps_pos with shape (B, *native_shape).")

        std_o = float(self._signal_std.get(feature_name, 0.0) or 0.0)
        shot_ids_arr = np.asarray(shot_ids)
        window_indices_arr = np.asarray(window_indices)
        cp = crps_pos.reshape(crps_pos.shape[0], -1, crps_pos.shape[-1])  # (B, N_spatial, T)

        # nanmean over all-NaN slices (fully-missing timesteps/windows) is an expected NaN, not an error — silence the
        # numpy warning rather than spam it on sparse signals.
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Per-timestamp rows (mean over spatial positions; NaN targets excluded).
            if self._wr_ts is not None:
                crps_t = np.nanmean(cp, axis=1)  # (B, T)
                for i in range(crps_t.shape[0]):
                    for t in range(crps_t.shape[1]):
                        crps_val = float(crps_t[i, t])
                        ncrps_val = (crps_val / std_o) if std_o > 0.0 else float("nan")
                        self._wr_ts.writerow(
                            [
                                int(shot_ids_arr[i]),
                                int(window_indices_arr[i]),
                                int(t),
                                feature_name,
                                crps_val,
                                ncrps_val,
                            ]
                        )

            # Per-window value: mean CRPS over all valid positions in the window.
            win_crps = np.nanmean(cp, axis=(1, 2))  # (B,)

        finite = np.isfinite(win_crps)
        if not finite.any():
            return  # no windows with valid native positions

        if self._wr_win is not None:
            for i in np.nonzero(finite)[0]:
                crps_val = float(win_crps[i])
                ncrps_win = (crps_val / std_o) if std_o > 0.0 else float("nan")
                self._wr_win.writerow(
                    [int(shot_ids_arr[i]), int(window_indices_arr[i]), feature_name, crps_val, ncrps_win]
                )

        if self._need_shot_or_task:
            shots = shot_ids_arr[finite].astype(int, copy=False)
            vals = win_crps[finite].astype(float, copy=False)
            unique_shots, inverse = np.unique(shots, return_inverse=True)
            shot_sums = np.bincount(inverse, weights=vals)
            shot_counts = np.bincount(inverse)
            for shot, val_sum, cnt in zip(unique_shots, shot_sums, shot_counts, strict=True):
                key = (feature_name, int(shot))
                self._shot_sum[key] = self._shot_sum.get(key, 0.0) + float(val_sum)
                self._shot_cnt[key] = self._shot_cnt.get(key, 0) + int(cnt)

    # ------------------------------------------------------------------------------------------------------------------
    def finalize(self) -> dict[str, Any] | None:
        """
        Close streaming writers, write shot/task CSVs, and return a task-level summary.

        Returns
        -------
        dict[str, Any] | None
            ``{"CRPS_mean", "NCRPS_mean", "per_signal"}`` (shot-weighted task aggregate), or None when no shot/task
            data was accumulated.
        """

        if self._f_ts is not None:
            self._f_ts.close()
        if self._f_win is not None:
            self._f_win.close()

        if not (self._need_shot_or_task and self._shot_cnt):
            return None

        # per-(signal, shot): mean per-window CRPS over windows.
        per_shot: dict[str, list[tuple[int, float]]] = {}
        for (name, shot), cnt in self._shot_cnt.items():
            if cnt > 0:
                per_shot.setdefault(name, []).append((shot, self._shot_sum[(name, shot)] / cnt))

        # per-signal: mean +/- population-std across shots (shots equally weighted).
        per_signal: dict[str, dict[str, float]] = {}
        for name, shot_vals in per_shot.items():
            vals = np.asarray([v for _, v in shot_vals], dtype=float)
            std_o = float(self._signal_std.get(name, 0.0) or 0.0)
            crps_mean = float(vals.mean())
            crps_std = float(vals.std(ddof=0))
            per_signal[name] = {
                "CRPS_mean": crps_mean,
                "CRPS_std_pop": crps_std,
                "NCRPS_mean": (crps_mean / std_o) if std_o > 0.0 else float("nan"),
                "NCRPS_std_pop": (crps_std / std_o) if std_o > 0.0 else float("nan"),
                "n_shots": int(vals.size),
            }

        if self._per_shot and per_shot:
            write_crps_shot_csv(self._dir / CRPS_SHOT_METRICS_FILE, per_shot, self._signal_std)

        if not per_signal:
            return None

        # task row: equal-weight over signals (mean of signal means; mean of signal stds), matching tokamark's
        # cross-signal group aggregation.
        task_crps_mean = float(np.mean([s["CRPS_mean"] for s in per_signal.values()]))
        task_crps_std = float(np.mean([s["CRPS_std_pop"] for s in per_signal.values()]))
        ncrps_means = [s["NCRPS_mean"] for s in per_signal.values() if np.isfinite(s["NCRPS_mean"])]
        ncrps_stds = [s["NCRPS_std_pop"] for s in per_signal.values() if np.isfinite(s["NCRPS_std_pop"])]
        task_ncrps_mean = float(np.mean(ncrps_means)) if ncrps_means else float("nan")
        task_ncrps_std = float(np.mean(ncrps_stds)) if ncrps_stds else float("nan")

        if self._per_task:
            write_crps_task_csv(
                path=self._dir / CRPS_TASK_METRICS_FILE,
                task_name=self._task_name,
                per_signal=per_signal,
                task_mean=task_crps_mean,
                task_std=task_crps_std,
                task_ncrps_mean=task_ncrps_mean,
                task_ncrps_std=task_ncrps_std,
            )

        return {"CRPS_mean": task_crps_mean, "NCRPS_mean": task_ncrps_mean, "per_signal": per_signal}
