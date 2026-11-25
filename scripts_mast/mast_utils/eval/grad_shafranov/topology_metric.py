"""
Eval-only LCFS / X-point flux-topology metric for reconstructed ``psi``.

For each valid predicted timeslice this metric scores ``equilibrium-psi`` against
the source EFIT geometry supplied by a post-batch topology provider:

- interpolate both predicted and ground-truth ``psi`` at the LCFS boundary points
  and at the X-point ``(R_x, Z_x)``,
- report the mean and max absolute prediction-vs-truth difference along the LCFS
  and the absolute difference at the X-point, each normalized by the reference
  flux span ``|psi_sep - psi_axis|``,
- and a variant restricted to LCFS points near the X-point (the divertor leg
  region, where boundary fidelity matters most).
- and two separatrix-constancy diagnostics, ``mean |psi(LCFS) - psi(X-point)|``,
  computed per field. The ground-truth version is a validity check on the metric
  itself: EFIT's LCFS is by construction the ``psi = psi_sep`` level set of EFIT's
  ``psi``, so it must be ~0. A non-zero value indicates a mis-aligned timeslice, a
  transposed grid, or a contour/field inconsistency in the source — none of which
  the pred-vs-true diagnostics above would reveal.

The reference geometry (LCFS polyline, X-point, magnetic axis) is *reference
metadata for scoring only*: it never enters ``x``, ``y``, model normalization, or
the benchmark scores. The normalizer ``|psi_sep - psi_axis|`` is derived from the
ground-truth ``psi`` on the evaluation grid — the exact same psi space as the
``(psi_pred - psi_true)`` differences it scales — using the provider's magnetic-axis
and X-point anchors:

    psi_axis ≈ psi_true(magnetic_axis),   psi_sep ≈ mean psi_true(X-points).

This mirrors how the GS loss derives axis/separatrix flux and keeps the metric
self-consistent in units regardless of the source store's psi conventions.

The metric follows the ``SmoothnessMetric`` shape exactly (``__init__`` →
``add_batch`` → ``is_empty`` → ``write_csvs``, companion CSVs only) and never
touches the benchmark accumulator. It intentionally does not use the shared
limiter/plasma region conventions: topology is sampled at reference LCFS and
X-point geometry rather than reduced over a grid domain.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from .aggregation import RunningMean, format_value, task_stats
from .constants import DEFAULT_GS_PARAMS_FILE
from .grid import GSEvalGrid

logger = logging.getLogger("mmt.Eval")

# ======================================================================================================================
# Constants
# ======================================================================================================================

# Default radius (metres) defining the "near X-point" LCFS subset. MAST minor radius is ~0.5 m, so ~0.15 m
# selects the divertor-leg neighbourhood without collapsing to a single point.
_DEFAULT_NEAR_X_POINT_RADIUS = 0.15
_TOPOLOGY_COLUMNS = (
    "lcfs_mean_abs",
    "lcfs_max_abs",
    "x_point_abs",
    "near_x_point_lcfs_mean_abs",
    "constancy_pred_abs",
    "constancy_gt_abs",
)
_MODEL_TOPOLOGY_COLUMNS = _TOPOLOGY_COLUMNS[:-1]


def _as_finite_float(value: Any) -> float | None:
    """Return a finite float from untyped provider metadata, or ``None`` when unavailable or invalid."""

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


# ======================================================================================================================
# Per-slice and per-shot accumulators
# ======================================================================================================================


@dataclass
class _SliceResult:
    """The six normalized diagnostics for one timeslice plus the coverage flags used for reporting."""

    lcfs_mean: float
    lcfs_max: float
    x_point_err: float
    near_x_lcfs: float
    constancy_pred: float  # mean |psi_pred(LCFS) - psi_pred(X)| / flux_span  (shape diagnostic)
    constancy_gt: float  # mean |psi_true(LCFS) - psi_true(X)| / flux_span  (validity check: expect ~0)
    used_lcfs_fallback: bool  # psi_sep came from the mean-over-LCFS fallback (no usable X-point)
    n_x_points: int  # number of valid (finite, R>0) X-points in the slice

    def has_model_diagnostic(self) -> bool:
        """Return whether at least one model-dependent normalized diagnostic is finite.

        ``constancy_gt`` validates the reference geometry and is deliberately excluded: it must not make an invalid
        prediction look like a successfully scored topology slice.
        """

        return any(
            np.isfinite(v)
            for v in (
                self.lcfs_mean,
                self.lcfs_max,
                self.x_point_err,
                self.near_x_lcfs,
                self.constancy_pred,
            )
        )


# ======================================================================================================================
# Metric
# ======================================================================================================================
class GSTopologyMetric:
    """
    Score predicted ``psi`` against reference LCFS / X-point flux geometry.

    Parameters
    ----------
    grad_shafranov_params_file : str | Path
        Path to the MAST Grad-Shafranov grid asset when ``grid`` is not supplied.
    grid : GSEvalGrid | None
        Shared evaluation grid. Supplying the same instance as the other GS diagnostics avoids reloading the
        asset and guarantees an identical R/Z grid.
    near_x_point_radius : float
        Radius (metres) of the near-X-point LCFS subset.

    Attributes
    ----------
    required_auxiliary_fields : frozenset[str]
        Physical equilibrium fields this metric needs the topology provider to supply. ``psi_axis`` / ``psi_sep``
        are *not* requested: they are derived here from ``psi_true`` at the magnetic-axis and X-point anchors.
    _shot_acc : dict[int, dict[str, RunningMean]]
        Per-shot accumulators of the normalized topology diagnostics (only shots with a finite result are kept).

    Methods
    -------
    add_batch
        Accumulate per-shot topology diagnostics for one decoded evaluation batch.
    is_empty
        Return whether no finite diagnostic was accumulated.
    write_csvs
        Write per-shot and per-task companion CSV files.

    """

    required_auxiliary_fields: frozenset[str] = frozenset(
        {"lcfs_r", "lcfs_z", "x_point_r", "x_point_z", "magnetic_axis_r", "magnetic_axis_z"}
    )

    def __init__(
        self,
        grad_shafranov_params_file: str | Path = DEFAULT_GS_PARAMS_FILE,
        *,
        grid: GSEvalGrid | None = None,
        near_x_point_radius: float = _DEFAULT_NEAR_X_POINT_RADIUS,
    ) -> None:
        self._grid = grid or GSEvalGrid(grad_shafranov_params_file)
        self._near_radius = float(near_x_point_radius)
        if not np.isfinite(self._near_radius) or self._near_radius <= 0.0:
            raise ValueError(f"near_x_point_radius must be finite and positive, got {near_x_point_radius!r}.")
        # Strictly-increasing 1D axes (GSEvalGrid enforces positive dR/dZ), oriented (R, Z) to match to_fields.
        self._r_axis = np.asarray(self._grid.r[:, 0], dtype=np.float64)
        self._z_axis = np.asarray(self._grid.z[0, :], dtype=np.float64)
        self._shot_acc: dict[int, dict[str, RunningMean]] = {}
        # Task-level coverage counters (per masked slice attempted), forming a full accounting of every attempt:
        #   attempted = skipped_alignment + skipped_geometry + finite
        # "skipped_alignment" = provider returned None (unavailable shot / non-finite time / beyond tolerance);
        # "skipped_geometry"  = provider aligned a slice but its geometry yielded no finite diagnostic;
        # "finite"            = produced at least one finite normalized diagnostic (i.e. was scored).
        self._n_slices_attempted = 0
        self._n_slices_skipped_alignment = 0
        self._n_slices_skipped_geometry = 0
        self._n_slices_finite = 0
        # Regime counters among scored (finite) slices.
        self._n_slices_lcfs_fallback = 0  # psi_sep from mean-over-LCFS (no usable X-point)
        self._n_slices_multi_x_point = 0  # more than one valid X-point (double-null etc.)
        # Alignment-quality provenance accumulated from the provider (over scored slices).
        self._time_error = RunningMean()
        self._time_err_max = float("nan")
        self._match_tolerance_max = float("nan")

    @staticmethod
    def _empty_shot_accumulator() -> dict[str, RunningMean]:
        """Create finite-value accumulators for all topology diagnostics of one shot."""

        return {name: RunningMean() for name in _TOPOLOGY_COLUMNS}

    # ..................................................................................................................
    def _make_interpolator(self, field: np.ndarray) -> RegularGridInterpolator:
        """Return a bilinear interpolator for one ``(n_r, n_z)`` field (NaN outside the grid)."""

        if field.shape != (self._r_axis.size, self._z_axis.size):
            raise ValueError(
                f"topology metric expects (n_r, n_z)={self._r_axis.size, self._z_axis.size} fields, "
                f"got {field.shape}; check GSEvalGrid.to_fields orientation."
            )
        return RegularGridInterpolator(
            (self._r_axis, self._z_axis), field, method="linear", bounds_error=False, fill_value=np.nan
        )

    # ..................................................................................................................
    @staticmethod
    def _sample(interp: RegularGridInterpolator, points: np.ndarray) -> np.ndarray:
        """Sample a prepared interpolator at physical ``(R, Z)`` points (empty in, empty out)."""

        if points.size == 0:
            return np.empty(0, dtype=np.float64)
        return np.asarray(interp(points), dtype=np.float64)

    # ..................................................................................................................
    @staticmethod
    def _primary_x_point(x_point: np.ndarray) -> np.ndarray | None:
        """
        Select the single primary X-point: the finite X-point with the lowest ``Z``.

        MAST operates predominantly in lower-single-null, so the lowest-``Z`` X-point is the active divertor
        X-point. Using one explicit point (rather than averaging) makes ``psi_sep`` and the near-X LCFS region
        reproducible and order-independent for double-null slices.
        """

        if x_point.shape[0] == 0:
            return None
        valid = np.isfinite(x_point).all(axis=1) & (x_point[:, 0] > 0.0)
        if not bool(valid.any()):
            return None
        candidates = x_point[valid]
        return candidates[int(np.argmin(candidates[:, 1]))]

    # ..................................................................................................................
    @staticmethod
    def _count_valid_x_points(x_point: np.ndarray) -> int:
        """Number of physically valid (finite, ``R > 0``) X-points in one slice."""

        if x_point.shape[0] == 0:
            return 0
        return int((np.isfinite(x_point).all(axis=1) & (x_point[:, 0] > 0.0)).sum())

    # ..................................................................................................................
    def _slice_diagnostics(self, psi_pred: np.ndarray, psi_true: np.ndarray, topo: dict[str, Any]) -> _SliceResult:
        """
        Return the six normalized diagnostics for one timeslice, plus its coverage flags.

        Any diagnostic that cannot be computed (missing geometry, degenerate flux span, out-of-grid points) is
        ``NaN`` and is simply not accumulated. A single primary X-point (lowest ``Z``) is used consistently for
        ``psi_sep``, the X-point error, the near-X LCFS region, and the constancy reference. If the primary
        X-point's ground-truth flux is non-finite (e.g. it lies outside the evaluation grid), ``psi_sep`` falls
        back to the mean over the LCFS — psi is ~flat on the boundary — so a fringe X-point cannot defeat
        scoring the slice.

        The two constancy diagnostics measure ``mean |psi(LCFS) - psi(X-point)|`` for a *single* field. For
        ``psi_true`` this is a validity check on the metric itself rather than a model score: EFIT's LCFS is by
        construction the ``psi = psi_sep`` level set of EFIT's ``psi``, so the value should be ~0 (bounded by
        bilinear interpolation error). A materially non-zero ``constancy_gt`` indicates a mis-aligned timeslice,
        a transposed grid, or a contour/field inconsistency in the source — and every other diagnostic here is
        then suspect. For ``psi_pred`` it is a boundary-shape diagnostic, and must be read alongside
        ``lcfs_mean``: a collapsed constant field scores a perfect (zero) constancy.
        """

        lcfs = np.asarray(topo.get("lcfs"), dtype=np.float64).reshape(-1, 2)
        x_point = np.asarray(topo.get("x_point"), dtype=np.float64).reshape(-1, 2)
        magnetic_axis = np.asarray(topo.get("magnetic_axis"), dtype=np.float64).reshape(-1)
        primary_x = self._primary_x_point(x_point)
        n_x_points = self._count_valid_x_points(x_point)

        nan = float("nan")

        def result(
            lcfs_mean: float,
            lcfs_max: float,
            x_point_err: float,
            near_x_lcfs: float,
            constancy_pred: float,
            constancy_gt: float,
            fallback: bool,
        ) -> _SliceResult:
            return _SliceResult(
                lcfs_mean, lcfs_max, x_point_err, near_x_lcfs, constancy_pred, constancy_gt, fallback, n_x_points
            )

        # Build one interpolator per field and reuse it: every diagnostic below samples the same two fields at
        # several point sets, so constructing a RegularGridInterpolator per call would repeat that setup.
        interp_pred = self._make_interpolator(psi_pred)
        interp_true = self._make_interpolator(psi_true)

        # Interpolate each field once at each point set; the diagnostics are then pure arithmetic on these.
        pred_lcfs = self._sample(interp_pred, lcfs)
        true_lcfs = self._sample(interp_true, lcfs)
        pred_x = self._sample(interp_pred, primary_x[None, :]) if primary_x is not None else np.empty(0)
        true_x = self._sample(interp_true, primary_x[None, :]) if primary_x is not None else np.empty(0)

        # Reference flux span from ground truth: psi_axis at the magnetic axis.
        psi_axis = nan
        if magnetic_axis.shape[0] == 2 and np.isfinite(magnetic_axis).all():
            psi_axis = float(self._sample(interp_true, magnetic_axis[None, :])[0])

        # psi_sep at the primary X-point; fall back to the mean psi_true over the LCFS when there is no usable
        # X-point or its flux is non-finite (out-of-grid). Track which path was taken for coverage reporting.
        used_lcfs_fallback = False
        psi_sep = nan
        if true_x.size and np.isfinite(true_x[0]):
            psi_sep = float(true_x[0])
        if not np.isfinite(psi_sep) and np.isfinite(true_lcfs).any():
            psi_sep = float(np.nanmean(true_lcfs))
            used_lcfs_fallback = True

        flux_span = abs(psi_sep - psi_axis) if (np.isfinite(psi_axis) and np.isfinite(psi_sep)) else nan
        if not (np.isfinite(flux_span) and flux_span > 0.0):
            return result(nan, nan, nan, nan, nan, nan, used_lcfs_fallback)

        # (1) LCFS boundary error (predicted vs ground-truth psi at the reference boundary points).
        lcfs_mean = lcfs_max = nan
        if lcfs.shape[0] > 0:
            diff_lcfs = np.abs(pred_lcfs - true_lcfs) / flux_span
            if np.isfinite(diff_lcfs).any():
                lcfs_mean = float(np.nanmean(diff_lcfs))
                lcfs_max = float(np.nanmax(diff_lcfs))

        # (2) X-point error at the primary X-point.
        x_point_err = nan
        if pred_x.size and true_x.size:
            diff_x = abs(float(pred_x[0]) - float(true_x[0])) / flux_span
            if np.isfinite(diff_x):
                x_point_err = diff_x

        # (3) Near-X-point LCFS subset (divertor-leg region around the primary X-point).
        near_x_lcfs = nan
        near_sel: np.ndarray | None = None
        if primary_x is not None and lcfs.shape[0] > 0:
            distances = np.linalg.norm(lcfs - primary_x[None, :], axis=1)
            near_sel = distances <= self._near_radius
            if bool(near_sel.any()):
                diff_near = np.abs(pred_lcfs[near_sel] - true_lcfs[near_sel]) / flux_span
                if np.isfinite(diff_near).any():
                    near_x_lcfs = float(np.nanmean(diff_near))

        # (4) Separatrix constancy, per field: mean |psi(LCFS) - psi(X-point)| / flux_span. Only defined against
        # a real X-point anchor — under the LCFS fallback the reference *is* the LCFS mean, which would make the
        # ground-truth check trivially self-satisfying and therefore meaningless as a validity signal.
        constancy_pred = constancy_gt = nan
        if primary_x is not None and not used_lcfs_fallback and lcfs.shape[0] > 0:
            if pred_x.size and np.isfinite(pred_x[0]):
                dev_pred = np.abs(pred_lcfs - float(pred_x[0])) / flux_span
                if np.isfinite(dev_pred).any():
                    constancy_pred = float(np.nanmean(dev_pred))
            if true_x.size and np.isfinite(true_x[0]):
                dev_gt = np.abs(true_lcfs - float(true_x[0])) / flux_span
                if np.isfinite(dev_gt).any():
                    constancy_gt = float(np.nanmean(dev_gt))

        return result(lcfs_mean, lcfs_max, x_point_err, near_x_lcfs, constancy_pred, constancy_gt, used_lcfs_fallback)

    # ..................................................................................................................
    def add_batch(
        self,
        psi_native: np.ndarray,
        shot_ids: np.ndarray,
        topology: list[list[dict[str, Any] | None]],
        window_mask: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
    ) -> None:
        """
        Accumulate per-shot topology diagnostics for one decoded evaluation batch.

        Parameters
        ----------
        psi_native : np.ndarray
            Predicted decoded native ``psi`` shaped ``(B, H, W, T)`` or ``(B, H, W)``.
        shot_ids : np.ndarray
            Per-window source shot identifiers, length ``B``.
        topology : list[list[dict | None]]
            Provider output ``topo[b][t]`` with geometry keys ``"lcfs"``, ``"x_point"``, ``"magnetic_axis"``, or
            ``None`` for a slice the provider could not align (skipped and counted as such).
        window_mask : np.ndarray | None
            Optional per-window validity mask, length ``B``.
        psi_gt_native : np.ndarray | None
            Ground-truth decoded native ``psi`` matching ``psi_native``. Required — the normalizer and the
            reference boundary flux are both derived from it; the batch is skipped when it is absent.

        """

        if psi_gt_native is None:
            logger.warning("Topology metric: no ground-truth psi in batch; skipping (normalizer needs psi_true).")
            return

        psi_pred_fields = self._grid.to_fields(psi_native).astype(np.float64)
        psi_true_fields = self._grid.to_fields(psi_gt_native).astype(np.float64)
        if psi_pred_fields.shape != psi_true_fields.shape:
            raise ValueError(f"psi_pred {psi_pred_fields.shape} != psi_gt {psi_true_fields.shape}")

        shot_ids = np.asarray(shot_ids)
        batch_size = int(shot_ids.shape[0])
        n_fields = psi_pred_fields.shape[0]
        n_times = n_fields // max(batch_size, 1)
        if n_times * batch_size != n_fields:
            logger.warning("Topology: field count %d not divisible by batch %d; skipping.", n_fields, batch_size)
            return

        if window_mask is not None:
            window_mask = np.asarray(window_mask, dtype=bool)

        for field_index in range(n_fields):
            window_index = field_index // n_times
            t_index = field_index % n_times
            if window_mask is not None and not window_mask[window_index]:
                continue
            if window_index >= len(topology) or t_index >= len(topology[window_index]):
                continue

            self._n_slices_attempted += 1
            topo = topology[window_index][t_index]
            if topo is None:
                self._n_slices_skipped_alignment += 1
                continue

            result = self._slice_diagnostics(psi_pred_fields[field_index], psi_true_fields[field_index], topo)
            # ``constancy_gt`` only validates the reference geometry. A slice without a finite model-facing metric
            # remains unscorable even when that reference check succeeds.
            if not result.has_model_diagnostic():
                self._n_slices_skipped_geometry += 1
                continue

            self._n_slices_finite += 1
            if result.used_lcfs_fallback:
                self._n_slices_lcfs_fallback += 1
            if result.n_x_points > 1:
                self._n_slices_multi_x_point += 1
            self._accumulate_time_error(topo)

            shot_id = int(shot_ids[window_index])
            shot = self._shot_acc.setdefault(shot_id, self._empty_shot_accumulator())
            for name, value in (
                ("lcfs_mean_abs", result.lcfs_mean),
                ("lcfs_max_abs", result.lcfs_max),
                ("x_point_abs", result.x_point_err),
                ("near_x_point_lcfs_mean_abs", result.near_x_lcfs),
                ("constancy_pred_abs", result.constancy_pred),
                ("constancy_gt_abs", result.constancy_gt),
            ):
                shot[name].add(value)

    # ..................................................................................................................
    def _accumulate_time_error(self, topo: dict[str, Any]) -> None:
        """Fold one scored slice's alignment provenance (``abs_time_error`` / ``match_tolerance_s``) into totals."""

        value = _as_finite_float(topo.get("abs_time_error"))
        if value is not None:
            self._time_error.add(value)
            self._time_err_max = value if not np.isfinite(self._time_err_max) else max(self._time_err_max, value)

        tolerance = _as_finite_float(topo.get("match_tolerance_s"))
        if tolerance is not None:
            self._match_tolerance_max = (
                tolerance if not np.isfinite(self._match_tolerance_max) else max(self._match_tolerance_max, tolerance)
            )

    # ..................................................................................................................
    def is_empty(self) -> bool:
        """Return whether no shot accumulated any finite model-facing topology diagnostic."""

        return not any(
            any(sums[name].count > 0 for name in _MODEL_TOPOLOGY_COLUMNS) for sums in self._shot_acc.values()
        )

    # ..................................................................................................................
    def write_csvs(self, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write per-shot and per-task topology CSVs and return a compact summary."""

        metrics_task_dir = Path(metrics_task_dir)
        metrics_task_dir.mkdir(parents=True, exist_ok=True)

        # Each diagnostic keeps its own (sum, count) so per-diagnostic coverage is reported honestly; a single
        # "n_slices" as the max across the six would overstate the coverage of the sparsest diagnostic.
        per_shot: dict[str, list[float]] = {}

        shots_csv = metrics_task_dir / "topology_shots.csv"
        with shots_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(
                [
                    "shot_id",
                    *(f"topology_{name}" for name in _TOPOLOGY_COLUMNS),
                    *(f"n_slices_{name}" for name in _TOPOLOGY_COLUMNS),
                ]
            )
            for shot_id in sorted(self._shot_acc):
                sums = self._shot_acc[shot_id]
                values = {name: sums[name].mean for name in _TOPOLOGY_COLUMNS}
                for name in _TOPOLOGY_COLUMNS:
                    if np.isfinite(values[name]):
                        per_shot.setdefault(name, []).append(values[name])
                writer.writerow(
                    [
                        shot_id,
                        *(format_value(values[name]) for name in _TOPOLOGY_COLUMNS),
                        *(sums[name].count for name in _TOPOLOGY_COLUMNS),
                    ]
                )

        stats_by_name = {name: task_stats(per_shot.get(name, [])) for name in _TOPOLOGY_COLUMNS}
        time_err_mean = self._time_error.mean

        # Task-level coverage + alignment-quality columns, in one place so the CSV and summary stay in sync.
        coverage_cols: list[tuple[str, Any]] = [
            ("n_shots", len(self._shot_acc)),
            ("n_slices_attempted", self._n_slices_attempted),
            ("n_slices_finite", self._n_slices_finite),
            ("n_slices_skipped_alignment", self._n_slices_skipped_alignment),
            ("n_slices_skipped_geometry", self._n_slices_skipped_geometry),
            ("n_slices_lcfs_fallback", self._n_slices_lcfs_fallback),
            ("n_slices_multi_x_point", self._n_slices_multi_x_point),
            ("time_match_error_mean_s", time_err_mean),
            ("time_match_error_max_s", self._time_err_max),
            ("time_match_tolerance_s_max", self._match_tolerance_max),
            ("near_x_point_radius_m", self._near_radius),
        ]

        task_csv = metrics_task_dir / "topology_task.csv"
        task_header = ["task"]
        for name in _TOPOLOGY_COLUMNS:
            task_header.extend([f"topology_{name}", f"topology_{name}_std_pop", f"topology_{name}_n_shots"])
        task_header.extend(key for key, _ in coverage_cols)

        # Always emit exactly one task row, even with zero usable slices: the metric values are NaN but the
        # coverage/skip counters (attempted / skipped_alignment / skipped_geometry) explain the empty result.
        with task_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(task_header)
            row: list[Any] = [task_name]
            for name in _TOPOLOGY_COLUMNS:
                mean, std, count = stats_by_name[name]
                row.extend([format_value(mean), format_value(std), count])
            row.extend(format_value(value) if isinstance(value, float) else value for _, value in coverage_cols)
            writer.writerow(row)

        summary: dict[str, Any] = {
            "topology_shots_csv": str(shots_csv),
            "topology_task_csv": str(task_csv),
        }
        summary.update(coverage_cols)
        for name in _TOPOLOGY_COLUMNS:
            mean, _, count = stats_by_name[name]
            summary[f"topology_{name}"] = mean
            summary[f"topology_{name}_n_shots"] = count
        return summary
