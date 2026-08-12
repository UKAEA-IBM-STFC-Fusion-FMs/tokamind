"""Grad--Shafranov equation and operator diagnostics.

The evaluator reports independent, physically dimensioned mean-absolute errors
over the fixed eroded limiter interior and, when true ``j_tor`` is available,
the ground-truth plasma mask.  It deliberately keeps three effects separate:

* ``gs_self_consistency_*``: ``|L_pred - R_pred|``;
* ``dstar_psi_*_error``: ``|L_pred - L_true|``;
* ``gs_rhs_jtor_error``: ``|R_pred - R_true|``.

Here ``L = -Delta* psi`` and ``R = mu0 R j_tor``.  The finite-difference
(``fd``) operator is the loss-independent primary diagnostic.  The sparse
FreeGSNKE operator (``A``) is retained as a secondary diagnostic when it is
available.  A model predicting only ``psi`` still receives the ``dstar_psi``
diagnostics; self-consistency and RHS-current diagnostics require predicted
``j_tor`` and are therefore left as NaN.

Inputs are already-destandardized native fields.  No standardisation work is
performed here.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import RunningMean, format_value, task_stats, warn_once
from .constants import DEFAULT_GS_PARAMS_FILE
from .grid import GSEvalGrid
from .regions import EVAL_REGIONS, LIMITER, PLASMA_MASK, EvalRegion, PlasmaMaskCoverage

logger = logging.getLogger("mmt.Eval")

_METRICS = (
    "self_consistency_fd",
    "dstar_psi_fd_error",
    "rhs_jtor_error",
    "self_consistency_A",
    "dstar_psi_A_error",
)

_CSV_NAME = {
    "self_consistency_fd": "gs_self_consistency_fd_mean_abs_per_cell",
    "dstar_psi_fd_error": "dstar_psi_fd_error_mean_abs_per_cell",
    "rhs_jtor_error": "gs_rhs_jtor_error_mean_abs_per_cell",
    "self_consistency_A": "gs_self_consistency_A_mean_abs_per_cell",
    "dstar_psi_A_error": "dstar_psi_A_error_mean_abs_per_cell",
}

_CSV_STD_NAME = {
    "self_consistency_fd": "gs_self_consistency_fd_std_pop",
    "dstar_psi_fd_error": "dstar_psi_fd_error_std_pop",
    "rhs_jtor_error": "gs_rhs_jtor_error_std_pop",
    "self_consistency_A": "gs_self_consistency_A_std_pop",
    "dstar_psi_A_error": "dstar_psi_A_error_std_pop",
}


def _column(metric: str, region: EvalRegion) -> str:
    return region.column(metric)


def _csv_column(metric: str, region: EvalRegion) -> str:
    return f"{_CSV_NAME[metric]}{region.suffix}"


def _csv_std_column(metric: str, region: EvalRegion) -> str:
    return f"{_CSV_STD_NAME[metric]}{region.suffix}"


@dataclass(frozen=True)
class _MetricsBatch:
    """Per-field metric values and the validated plasma mask used to compute them."""

    values: dict[str, np.ndarray]
    plasma_mask: np.ndarray | None


class GSMetrics:
    """Accumulate Grad--Shafranov equation and operator diagnostics by shot."""

    def __init__(
        self,
        grad_shafranov_params_file: str | Path = DEFAULT_GS_PARAMS_FILE,
        *,
        grid: GSEvalGrid | None = None,
    ) -> None:
        self._grid = grid or GSEvalGrid(grad_shafranov_params_file)
        self._shot_acc: dict[int, dict[str, RunningMean]] = {}
        self._plasma_mask_coverage = PlasmaMaskCoverage()
        self._invalid_inputs: set[str] = set()

    def _warn_invalid_input(self, key: str, description: str, exc: ValueError) -> None:
        """Warn once when an optional prediction or reference cannot align with psi."""

        warn_once(
            self._invalid_inputs,
            key,
            logger,
            "GS metrics: ignoring invalid %s (%s); retaining valid diagnostics.",
            description,
            exc,
        )

    def _to_optional_fields(
        self, values: np.ndarray | None, name: str, shape: tuple[int, int, int]
    ) -> np.ndarray | None:
        if values is None:
            return None
        fields = self._grid.to_fields(values).astype(np.float64)
        if fields.shape != shape:
            raise ValueError(f"{name} {fields.shape} != psi_pred {shape}")
        return fields

    def _compute_per_field_metrics(
        self,
        psi_native: np.ndarray,
        *,
        jtor_native: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> _MetricsBatch:
        """Compute all available metrics once, then reduce them over each valid region."""

        psi = self._grid.to_fields(psi_native).astype(np.float64)
        n_fields = psi.shape[0]
        nan = np.full(n_fields, np.nan)

        try:
            jtor = self._to_optional_fields(jtor_native, "jtor_pred", psi.shape)
        except ValueError as exc:
            self._warn_invalid_input("jtor_pred", "predicted j_tor", exc)
            jtor = None

        psi_gt = None
        if psi_gt_native is not None:
            try:
                psi_gt = self._to_optional_fields(psi_gt_native, "psi_gt", psi.shape)
            except ValueError as exc:
                self._warn_invalid_input("psi_gt", "ground-truth psi", exc)

        jtor_gt = None
        if jtor_gt_native is not None:
            try:
                jtor_gt = self._to_optional_fields(jtor_gt_native, "jtor_gt", psi.shape)
            except ValueError as exc:
                self._warn_invalid_input("jtor_gt", "ground-truth j_tor", exc)

        lhs_fd = -self._grid.delta_star_fd(psi)
        lhs_A = -self._grid.delta_star_A(psi) if self._grid.operator is not None else None
        lhs_fd_gt = -self._grid.delta_star_fd(psi_gt) if psi_gt is not None else None
        lhs_A_gt = -self._grid.delta_star_A(psi_gt) if psi_gt is not None and self._grid.operator is not None else None

        rhs = self._grid.mu0 * self._grid.r[None, :, :] * jtor if jtor is not None else None
        rhs_gt = self._grid.mu0 * self._grid.r[None, :, :] * jtor_gt if jtor_gt is not None else None

        fields: dict[str, np.ndarray | None] = {
            "self_consistency_fd": lhs_fd - rhs if rhs is not None else None,
            "dstar_psi_fd_error": lhs_fd - lhs_fd_gt if lhs_fd_gt is not None else None,
            "rhs_jtor_error": rhs - rhs_gt if rhs is not None and rhs_gt is not None else None,
            "self_consistency_A": lhs_A - rhs if lhs_A is not None and rhs is not None else None,
            "dstar_psi_A_error": lhs_A - lhs_A_gt if lhs_A is not None and lhs_A_gt is not None else None,
        }

        plasma_mask = self._grid.plasma_region_from_fields(jtor_gt) if jtor_gt is not None else None
        regions: dict[str, np.ndarray | None] = {LIMITER.key: None}
        if plasma_mask is not None:
            regions[PLASMA_MASK.key] = plasma_mask

        values: dict[str, np.ndarray] = {}
        for region in EVAL_REGIONS:
            for metric in _METRICS:
                field = fields[metric]
                values[_column(metric, region)] = (
                    self._grid.mean_abs_over_region(field, region=regions[region.key])
                    if field is not None and region.key in regions
                    else nan.copy()
                )
        return _MetricsBatch(values=values, plasma_mask=plasma_mask)

    def per_field_metrics(
        self,
        psi_native: np.ndarray,
        *,
        jtor_native: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Return per-field GS metric arrays for all supported regions and diagnostics."""

        return self._compute_per_field_metrics(
            psi_native,
            jtor_native=jtor_native,
            psi_gt_native=psi_gt_native,
            jtor_gt_native=jtor_gt_native,
        ).values

    def add_batch(
        self,
        psi_native: np.ndarray,
        shot_ids: np.ndarray,
        *,
        jtor_native: np.ndarray | None = None,
        window_mask: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> None:
        """Accumulate finite metric values by shot for one decoded evaluation batch."""

        batch = self._compute_per_field_metrics(
            psi_native,
            jtor_native=jtor_native,
            psi_gt_native=psi_gt_native,
            jtor_gt_native=jtor_gt_native,
        )
        per_field = batch.values
        batch_size = int(np.asarray(shot_ids).shape[0])
        n_fields = next(iter(per_field.values())).shape[0]
        n_times = n_fields // max(batch_size, 1)
        if n_times * batch_size != n_fields:
            logger.warning("GS metrics: field count %d not divisible by batch %d; skipping.", n_fields, batch_size)
            return

        shot_ids = np.asarray(shot_ids)
        if window_mask is not None:
            window_mask = np.asarray(window_mask, dtype=bool)
            if window_mask.shape != (batch_size,):
                raise ValueError(f"window_mask {window_mask.shape} != batch {(batch_size,)}")
        for field_index in range(n_fields):
            window_index = field_index // n_times
            if window_mask is not None and not window_mask[window_index]:
                continue
            finite_values = {
                column: float(values[field_index])
                for column, values in per_field.items()
                if np.isfinite(values[field_index])
            }
            if not finite_values:
                continue
            shot_id = int(shot_ids[window_index])
            if batch.plasma_mask is not None:
                self._plasma_mask_coverage.observe(shot_id, batch.plasma_mask, field_index)
            shot = self._shot_acc.setdefault(shot_id, {})
            for column, value in finite_values.items():
                shot.setdefault(column, RunningMean()).add(value)

    def is_empty(self) -> bool:
        """Return whether no finite metric values have been accumulated."""

        return not any(entry.count > 0 for shot in self._shot_acc.values() for entry in shot.values())

    @staticmethod
    def _count_any_metric(shot: dict[str, RunningMean], region: EvalRegion) -> int:
        """Return the largest per-field count across metrics available in one region."""

        return max((shot.get(_column(metric, region), RunningMean()).count for metric in _METRICS), default=0)

    def write_csvs(self, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write per-shot and per-task GS diagnostic CSVs and return their summary."""

        metrics_task_dir = Path(metrics_task_dir)
        metrics_task_dir.mkdir(parents=True, exist_ok=True)
        ordered = [_column(metric, region) for region in EVAL_REGIONS for metric in _METRICS]
        per_shot_values: dict[str, list[float]] = {column: [] for column in ordered}

        shots_csv = metrics_task_dir / "gs_metrics_shots.csv"
        with shots_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(
                ["shot_id"]
                + [_csv_column(metric, region) for region in EVAL_REGIONS for metric in _METRICS]
                + [
                    "n_fields",
                    "n_fields_with_jtor_prediction",
                    "n_fields_with_jtor_reference",
                    "n_fields_plasma_mask",
                    "n_fields_with_plasma_reference",
                    "n_fields_empty_plasma_mask",
                ]
            )
            for shot_id in sorted(self._shot_acc):
                shot = self._shot_acc[shot_id]
                row: list[Any] = [shot_id]
                for column in ordered:
                    value = shot.get(column, RunningMean()).mean
                    if np.isfinite(value):
                        per_shot_values[column].append(value)
                    row.append(format_value(value))
                row.extend(
                    [
                        self._count_any_metric(shot, LIMITER),
                        shot.get(_column("self_consistency_fd", LIMITER), RunningMean()).count,
                        shot.get(_column("rhs_jtor_error", LIMITER), RunningMean()).count,
                        self._count_any_metric(shot, PLASMA_MASK),
                        *self._plasma_mask_coverage.per_shot(shot_id),
                    ]
                )
                writer.writerow(row)

        stats_by_column = {column: task_stats(per_shot_values[column]) for column in ordered}
        n_fields_with_plasma_reference, n_fields_empty_plasma_mask = self._plasma_mask_coverage.totals()
        task_csv = metrics_task_dir / "gs_metrics_task.csv"
        with task_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            header: list[str] = ["task"]
            for region in EVAL_REGIONS:
                for metric in _METRICS:
                    header.extend([_csv_column(metric, region), _csv_std_column(metric, region)])
            header.extend(
                [
                    "n_shots",
                    "n_shots_with_jtor_prediction",
                    "n_shots_with_jtor_reference",
                    "n_shots_plasma_mask",
                    "n_fields_with_plasma_reference",
                    "n_fields_empty_plasma_mask",
                ]
            )
            writer.writerow(header)
            row: list[Any] = [task_name]
            for column in ordered:
                mean, std, _ = stats_by_column[column]
                row.extend([format_value(mean), format_value(std)])
            row.extend(
                [
                    len(self._shot_acc),
                    stats_by_column[_column("self_consistency_fd", LIMITER)][2],
                    stats_by_column[_column("rhs_jtor_error", LIMITER)][2],
                    max(stats_by_column[_column(metric, PLASMA_MASK)][2] for metric in _METRICS),
                    n_fields_with_plasma_reference,
                    n_fields_empty_plasma_mask,
                ]
            )
            writer.writerow(row)

        summary: dict[str, Any] = {
            "gs_metrics_shots_csv": str(shots_csv),
            "gs_metrics_task_csv": str(task_csv),
            "n_shots": len(self._shot_acc),
            "n_shots_with_jtor_prediction": stats_by_column[_column("self_consistency_fd", LIMITER)][2],
            "n_shots_with_jtor_reference": stats_by_column[_column("rhs_jtor_error", LIMITER)][2],
            "n_shots_plasma_mask": max(stats_by_column[_column(metric, PLASMA_MASK)][2] for metric in _METRICS),
            "n_fields_with_plasma_reference": n_fields_with_plasma_reference,
            "n_fields_empty_plasma_mask": n_fields_empty_plasma_mask,
        }
        for region in EVAL_REGIONS:
            for metric in _METRICS:
                mean, _, count = stats_by_column[_column(metric, region)]
                if count > 0:
                    summary[_csv_column(metric, region)] = float(mean)
        return summary
