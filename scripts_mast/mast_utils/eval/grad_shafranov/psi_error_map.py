"""Task-level spatial error maps for decoded Grad-Shafranov ``equilibrium-psi`` fields."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .grid import GSEvalGrid


class PsiErrorMapMetric:
    """Stream full-grid psi errors and write one aggregate map per evaluated task.

    The metric stores only four ``(R, Z)`` accumulators, irrespective of the
    number of evaluation windows: absolute-error sum, signed-error sum,
    squared-error sum, and finite-value count. The aggregation deliberately
    covers the full grid (no limiter or plasma-region masking) so boundary and
    off-plasma reconstruction errors remain visible. This makes localization
    diagnostics inexpensive without retaining individual predictions.
    """

    def __init__(self, *, grid: GSEvalGrid) -> None:
        self._grid = grid
        shape = (grid.n_r, grid.n_z)
        self._sum_abs = np.zeros(shape, dtype=np.float64)
        self._sum_signed = np.zeros(shape, dtype=np.float64)
        self._sum_squared = np.zeros(shape, dtype=np.float64)
        self._count = np.zeros(shape, dtype=np.int64)
        self._n_fields_selected = 0
        self._n_fields_contributed = 0

    def add_batch(
        self,
        *,
        psi_native: np.ndarray,
        psi_gt_native: np.ndarray | None,
        window_mask: np.ndarray | None,
    ) -> None:
        """Accumulate finite full-grid errors for one decoded batch."""

        if psi_gt_native is None:
            return

        psi_pred = self._grid.to_fields(psi_native).astype(np.float64, copy=False)
        psi_gt = self._grid.to_fields(psi_gt_native).astype(np.float64, copy=False)
        if psi_pred.shape != psi_gt.shape:
            raise ValueError(f"Predicted psi shape {psi_pred.shape} != ground truth shape {psi_gt.shape}.")

        batch_size = int(np.asarray(psi_native).shape[0])
        if batch_size <= 0:
            raise ValueError("psi_native must contain at least one evaluation window.")
        if window_mask is None:
            selected = np.ones(batch_size, dtype=bool)
        else:
            selected = np.asarray(window_mask, dtype=bool)
            if selected.shape != (batch_size,):
                raise ValueError(f"window_mask shape {selected.shape} != ({batch_size},).")

        n_fields = psi_pred.shape[0]
        if n_fields % batch_size != 0:
            raise ValueError(f"Psi field count {n_fields} is not divisible by batch size {batch_size}.")
        n_fields_per_window = n_fields // batch_size
        field_selected_1d = np.repeat(selected, n_fields_per_window)
        field_selected = field_selected_1d[:, None, None]
        self._n_fields_selected += int(field_selected_1d.sum())
        valid = field_selected & np.isfinite(psi_pred) & np.isfinite(psi_gt)
        if not valid.any():
            return

        error = psi_pred - psi_gt
        self._sum_abs += np.where(valid, np.abs(error), 0.0).sum(axis=0)
        self._sum_signed += np.where(valid, error, 0.0).sum(axis=0)
        self._sum_squared += np.where(valid, error * error, 0.0).sum(axis=0)
        self._count += valid.sum(axis=0, dtype=np.int64)
        self._n_fields_contributed += int(valid.reshape(n_fields, -1).any(axis=1).sum())

    def is_empty(self) -> bool:
        """Return whether no finite psi cells have been accumulated."""

        return not bool(self._count.any())

    def write_npz(self, *, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write the aggregate map and return its compact metadata summary."""

        metrics_task_dir = Path(metrics_task_dir)
        metrics_task_dir.mkdir(parents=True, exist_ok=True)
        valid = self._count > 0
        mae = np.full(self._count.shape, np.nan, dtype=np.float64)
        bias = np.full(self._count.shape, np.nan, dtype=np.float64)
        rmse = np.full(self._count.shape, np.nan, dtype=np.float64)
        mae[valid] = self._sum_abs[valid] / self._count[valid]
        bias[valid] = self._sum_signed[valid] / self._count[valid]
        rmse[valid] = np.sqrt(self._sum_squared[valid] / self._count[valid])

        path = metrics_task_dir / "psi_error_map.npz"
        np.savez_compressed(
            path,
            task_name=np.asarray(task_name),
            r=self._grid.r,
            z=self._grid.z,
            mae=mae,
            bias=bias,
            rmse=rmse,
            count=self._count,
            full_grid=np.asarray(True),
            n_fields_selected=np.asarray(self._n_fields_selected, dtype=np.int64),
            n_fields_contributed=np.asarray(self._n_fields_contributed, dtype=np.int64),
        )
        return {
            "psi_error_map_npz": str(path),
            "n_fields_selected": self._n_fields_selected,
            "n_fields_contributed": self._n_fields_contributed,
            "n_valid_cells": int(valid.sum()),
            "max_count": int(self._count.max()),
        }
