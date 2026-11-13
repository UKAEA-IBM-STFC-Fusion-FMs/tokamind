"""Pointwise native-unit MSE over the full grid, limiter and its exterior.

Each window pools finite reference cells across its output times. Shot MSE is
the mean of scored window MSEs; task MSE is the equal-weight mean of shot MSEs.
The separate mse_pooled (squared-error sum/count) describes a pooled statistic, NOT that
hierarchical mean. No predictions or per-window grids are retained.
NRMSE follows TokaMark: sqrt(shot MSE) / metadata signal std, then an
equal-weight mean across shots. The same signal std is used for every region.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ...benchmark_imports import get_signals_metadata
from .aggregation import RunningMean, task_stats
from .grid import GSEvalGrid

SIGNALS = {"psi": "Wb/rad", "jtor": "A/m^2"}
FEATURES = {"psi": "equilibrium-psi", "jtor": "equilibrium-j_tor"}
COUNTERS = (
    "n_windows_selected",
    "n_windows_scored",
    "n_fields_selected",
    "n_fields_scored",
    "n_fields_empty_region",
    "n_cells",
    "n_reference_nonfinite",
)


@dataclass
class _ShotRegion:
    windows: RunningMean = field(default_factory=RunningMean)
    counters: Counter = field(default_factory=Counter)
    squared_error_sum: float = 0.0
    coverage: Any = field(default_factory=hashlib.sha256)


class RegionalMSEMetric:
    """Independent signal masks and fixed limiter geometry, without auxiliary data.

    Bad predictions on finite selected targets anywhere on the grid raise,
    rather than improve the MSE through exclusion. The limiter is not the LCFS.
    Coverage hashes record scored window/slice identities and finite region cells;
    compare hashes only with identical data ordering/protocol.
    Limiter and outside_limiter partition full_grid: their squared-error sums
    and cell counts add to the full-grid values. Never sum all three regions.
    Window, field and shot counts are not additive across regions.
    """

    def __init__(self, *, grid: GSEvalGrid) -> None:
        self.grid = grid
        self._regions = {
            "full_grid": np.ones_like(grid.limiter_region, dtype=bool),
            "limiter": grid.limiter_region.copy(),
            "outside_limiter": ~grid.limiter_region,
        }
        self._shots: dict[tuple[str, int, str], _ShotRegion] = {}

    def add_batch(
        self,
        *,
        signal: str,
        pred_native: np.ndarray,
        true_native: np.ndarray,
        shot_ids: np.ndarray,
        window_indices: np.ndarray,
        window_mask: np.ndarray,
    ) -> None:
        """Consume one signal's predictions and matching native targets."""
        if signal not in SIGNALS:
            raise ValueError(f"Unknown regional signal {signal}")
        ids = np.asarray(shot_ids)
        selected = np.asarray(window_mask, dtype=bool)
        indices = np.asarray(window_indices)
        batch = len(ids)
        if ids.shape != (batch,) or selected.shape != (batch,) or indices.shape != (batch,) or batch == 0:
            raise ValueError("Expected one shot ID, window index and mask entry per window")
        if np.asarray(pred_native).shape != np.asarray(true_native).shape or pred_native.shape[0] != batch:
            raise ValueError("Regional MSE prediction/reference/window shapes differ")
        pred = self.grid.to_fields(pred_native).astype(np.float64, copy=False)
        true = self.grid.to_fields(true_native).astype(np.float64, copy=False)
        if pred.shape[0] % batch:
            raise ValueError("Regional MSE field count is not divisible by batch size")
        n_times = pred.shape[0] // batch
        if n_times == 0:
            raise ValueError("Regional MSE needs at least one output time")
        valid = np.isfinite(true)
        active = np.repeat(selected, n_times)[:, None, None]
        if np.any(active & valid & ~np.isfinite(pred)):
            raise ValueError(f"Non-finite {signal} predictions on valid selected targets")
        with np.errstate(over="ignore", invalid="ignore"):
            errors = (pred - true) ** 2
        if np.any(active & valid & ~np.isfinite(errors)):
            raise ValueError(f"Non-finite squared errors for {signal}")

        for b in np.flatnonzero(selected):
            for region, mask in self._regions.items():
                state = self._shots.setdefault((signal, int(ids[b]), region), _ShotRegion())
                counts = state.counters
                counts["n_windows_selected"] += 1
                window_sse, window_cells = 0.0, 0
                for t in range(n_times):
                    idx = b * n_times + t
                    counts["n_fields_selected"] += 1
                    cells = mask & valid[idx]
                    counts["n_reference_nonfinite"] += int((mask & ~valid[idx]).sum())
                    n_cells = int(cells.sum())
                    if not n_cells:
                        counts["n_fields_empty_region"] += 1
                    if n_cells:
                        counts["n_fields_scored"] += 1
                        window_sse += float(errors[idx][cells].sum())
                        window_cells += n_cells
                        state.coverage.update(f"{int(indices[b])}:{t};".encode())
                        state.coverage.update(np.packbits(cells).tobytes())
                counts["n_cells"] += window_cells
                state.squared_error_sum += window_sse
                if window_cells:
                    if not np.isfinite(state.squared_error_sum):
                        raise ValueError(f"Regional squared-error sum overflow for {signal}")
                    counts["n_windows_scored"] += 1
                    state.windows.add(window_sse / window_cells)

    def write_csvs(self, *, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write long-form shot/task tables and a self-describing protocol manifest."""
        root = Path(metrics_task_dir)
        metadata = get_signals_metadata() if self._shots else {}
        normalization = {}
        for signal in sorted({key[0] for key in self._shots}):
            feature = FEATURES[signal]
            try:
                denominator = float(metadata[feature]["std"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Missing or invalid TokaMark normalization std for {feature}") from exc
            if not np.isfinite(denominator) or denominator <= 0:
                raise ValueError(f"TokaMark normalization std for {feature} must be finite and positive")
            normalization[signal] = denominator
        root.mkdir(parents=True, exist_ok=True)
        shot_rows, task_rows = [], []
        for (signal, shot, region), state in sorted(self._shots.items()):
            shot_rows.append(
                {
                    "task": task_name,
                    "signal": signal,
                    "shot_id": shot,
                    "region": region,
                    "mse_units": f"({SIGNALS[signal]})^2",
                    "mse": state.windows.mean,
                    "nrmse": float(np.sqrt(state.windows.mean) / normalization[signal]),
                    "normalization_std": normalization[signal],
                    "squared_error_sum": state.squared_error_sum,
                    **{key: state.counters[key] for key in COUNTERS},
                    "coverage_sha256": state.coverage.hexdigest(),
                }
            )
        for signal, region in sorted({(key[0], key[2]) for key in self._shots}):
            rows = [row for row in shot_rows if row["signal"] == signal and row["region"] == region]
            mse, std, n = task_stats(row["mse"] for row in rows if np.isfinite(row["mse"]))
            nrmse, nrmse_std, _ = task_stats(row["nrmse"] for row in rows if np.isfinite(row["nrmse"]))
            task_rows.append(
                {
                    "task": task_name,
                    "signal": signal,
                    "region": region,
                    "mse_units": f"({SIGNALS[signal]})^2",
                    "mse": mse,
                    "mse_std_pop": std,
                    "nrmse": nrmse,
                    "nrmse_std_pop": nrmse_std,
                    "normalization_std": normalization[signal],
                    "n_shots_scored": n,
                    "n_shots_selected": len(rows),
                    "squared_error_sum": sum(row["squared_error_sum"] for row in rows),
                    **{key: sum(row[key] for row in rows) for key in COUNTERS},
                }
            )
        paths = {}
        for row in shot_rows + task_rows:
            row["mse_pooled"] = row["squared_error_sum"] / row["n_cells"] if row["n_cells"] else float("nan")
        for name, rows, extra in (
            ("shots", shot_rows, ["shot_id", "coverage_sha256"]),
            ("task", task_rows, ["mse_std_pop", "nrmse_std_pop", "n_shots_scored", "n_shots_selected"]),
        ):
            path = root / f"regional_mse_{name}.csv"
            columns = [
                "task",
                "signal",
                "region",
                "mse_units",
                "mse",
                "mse_pooled",
                "nrmse",
                "normalization_std",
                *extra,
                "squared_error_sum",
                *COUNTERS,
            ]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            paths[f"regional_mse_{name}_csv"] = str(path)
        protocol = root / "regional_mse_protocol.json"
        protocol.write_text(
            json.dumps(
                {
                    "schema_version": 4,
                    "task": task_name,
                    "aggregation": "finite cells pooled within window; equal windows per shot; equal shots per task",
                    "mse_pooled": "squared_error_sum / n_cells; all valid elements equally weighted; NaN if empty",
                    "nrmse": "sqrt(shot mse) / signal std; equal mean across scored shots; NaN if empty",
                    "normalization": {
                        "source": "tokamark.tasks.get_signals_metadata()[feature]['std']",
                        "shared_across_regions": True,
                        "signals": {
                            signal: {"feature": FEATURES[signal], "std": value, "units": SIGNALS[signal]}
                            for signal, value in normalization.items()
                        },
                    },
                    "regions": {
                        "full_grid": "entire evaluation grid",
                        "limiter": "MAST_lim_mask_rz, no erosion",
                        "outside_limiter": "complement of MAST_lim_mask_rz on the evaluation grid",
                    },
                    "invalid_predictions": "raise on non-finite prediction at selected finite targets anywhere on grid",
                    "region_overlap": (
                        "limiter and outside_limiter partition full_grid; squared_error_sum and n_cells "
                        "add across these two parts only. Never sum all three regions. "
                        "Window, field and shot counts are not additive across regions."
                    ),
                    "coverage_sha256": "ordered window index, slice index and valid-region pixel bitmap per shot/signal/region",
                    "signals_reported": sorted({key[0] for key in self._shots}),
                    "grid_shape_rz": [self.grid.n_r, self.grid.n_z],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths["regional_mse_protocol_json"] = str(protocol)
        return {**paths, "rows": task_rows}
