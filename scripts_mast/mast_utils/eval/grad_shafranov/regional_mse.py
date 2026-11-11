"""Pointwise native-unit MSE on the full grid, un-eroded limiter and reference LCFS.

Each window pools finite reference cells across its output times. Shot MSE is
the mean of scored window MSEs; task MSE is the equal-weight mean of shot MSEs.
The separate squared-error sum/count describes a pooled statistic, NOT that
hierarchical mean. No predictions or per-window grids are retained.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import RunningMean, task_stats
from .grid import GSEvalGrid

REGIONS = ("full_grid", "limiter", "lcfs")
SIGNALS = {"psi": "Wb/rad", "jtor": "A/m^2"}
_GEOMETRY_TOLERANCE_M = 1e-10
COUNTERS = (
    "n_windows_selected",
    "n_windows_scored",
    "n_fields_selected",
    "n_fields_scored",
    "n_fields_missing_lcfs",
    "n_fields_invalid_lcfs",
    "n_fields_empty_region",
    "n_cells",
    "n_reference_nonfinite",
)


def _has_self_intersections(points: np.ndarray) -> bool:
    """Check non-adjacent edges for intersections or touching, including overlap.

    Expects at least three finite, normalized vertices, without a repeated closing
    vertex. The last edge closes the polygon implicitly.
    """
    end = np.roll(points, -1, axis=0)

    def cross(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    direction = end - points
    delta = points[None, :, :] - points[:, None, :]
    c1 = cross(direction[:, None, :], delta)
    c2 = cross(direction[:, None, :], end[None, :, :] - points[:, None, :])
    c3 = cross(direction[None, :, :], -delta)
    c4 = cross(direction[None, :, :], end[:, None, :] - points[None, :, :])
    bbox = np.all(
        np.maximum(np.minimum(points, end)[:, None, :], np.minimum(points, end)[None, :, :])
        <= np.minimum(np.maximum(points, end)[:, None, :], np.maximum(points, end)[None, :, :]) + _GEOMETRY_TOLERANCE_M,
        axis=-1,
    )
    pair = np.triu(np.ones((len(points), len(points)), dtype=bool), k=2)
    pair[0, -1] = False
    return bool(np.any(pair & bbox & (c1 * c2 <= 0) & (c3 * c4 <= 0)))


def lcfs_mask(grid: GSEvalGrid, points: np.ndarray) -> np.ndarray:
    """Rasterize a simple reference polygon in physical R/Z; include boundary centres.

    Closing the last edge is explicit. Reject non-finite, degenerate, self-crossing
    or out-of-grid contours rather than silently repair/clip them. No limiter
    intersection is applied. Edge tolerance is 1e-10 metres.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Non-finite or malformed LCFS contour")
    tol = _GEOMETRY_TOLERANCE_M
    if len(points) > 1:
        points = points[np.r_[True, np.any(np.abs(np.diff(points, axis=0)) > tol, axis=1)]]
    if len(points) > 1 and np.max(np.abs(points[-1] - points[0])) <= tol:
        points = points[:-1]
    if len(points) < 3:
        raise ValueError("LCFS has fewer than three distinct vertices")
    lo = np.array([grid.r.min(), grid.z.min()])
    hi = np.array([grid.r.max(), grid.z.max()])
    if np.any(points < lo - tol) or np.any(points > hi + tol):
        raise ValueError("LCFS extends outside the evaluation grid")
    end = np.roll(points, -1, axis=0)

    if abs(np.sum(points[:, 0] * end[:, 1] - points[:, 1] * end[:, 0])) <= tol**2:
        raise ValueError("LCFS has zero area")
    if _has_self_intersections(points):
        raise ValueError("LCFS is self-intersecting")

    x, y = grid.r, grid.z
    inside = np.zeros(x.shape, dtype=bool)
    boundary = np.zeros(x.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(points, end, strict=True):
        dx, dy = x2 - x1, y2 - y1
        boundary |= (
            (np.abs(dx * (y - y1) - dy * (x - x1)) <= tol * np.hypot(dx, dy))
            & (x >= min(x1, x2) - tol)
            & (x <= max(x1, x2) + tol)
            & (y >= min(y1, y2) - tol)
            & (y <= max(y1, y2) + tol)
        )
        if dy != 0:
            inside ^= ((y1 > y) != (y2 > y)) & (x < x1 + (y - y1) * dx / dy)
    return inside | boundary


@dataclass
class _ShotRegion:
    windows: RunningMean = field(default_factory=RunningMean)
    counters: Counter = field(default_factory=Counter)
    squared_error_sum: float = 0.0
    coverage: Any = field(default_factory=hashlib.sha256)


class RegionalMSEMetric:
    """Independent signal masks, reference-only geometry and explicit missing coverage.

    Bad predictions on finite selected targets raise, rather than improve the MSE
    through exclusion. Missing/invalid LCFS only excludes that slice from LCFS MSE.
    Coverage hashes record scored window/slice identities and finite region cells;
    compare hashes only with identical data ordering/protocol.
    """

    required_auxiliary_fields = frozenset({"lcfs_r", "lcfs_z"})

    def __init__(self, *, grid: GSEvalGrid) -> None:
        self.grid = grid
        self._full_grid_mask = np.ones((grid.n_r, grid.n_z), dtype=bool)
        self._shots: dict[tuple[str, int, str], _ShotRegion] = {}
        self._masks: OrderedDict[bytes, np.ndarray | None] = OrderedDict()

    def _mask(self, reference: dict | None) -> tuple[np.ndarray | None, str | None]:
        if reference is None:
            return None, "n_fields_missing_lcfs"
        points = np.asarray(reference.get("lcfs", []), dtype=np.float64)
        key = repr(points.shape).encode() + points.tobytes()
        if key not in self._masks:
            try:
                self._masks[key] = lcfs_mask(self.grid, points)
            except ValueError:
                self._masks[key] = None
            if len(self._masks) > 128:
                self._masks.popitem(last=False)
        self._masks.move_to_end(key)
        mask = self._masks[key]
        return mask, "n_fields_invalid_lcfs" if mask is None else None

    def add_batch(
        self,
        *,
        signal: str,
        pred_native: np.ndarray,
        true_native: np.ndarray,
        shot_ids: np.ndarray,
        window_indices: np.ndarray,
        window_mask: np.ndarray,
        topology: list[list[dict | None]] | None = None,
    ) -> None:
        """Consume one signal, with topology aligned to that signal's output times."""
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
        if topology is not None and (len(topology) != batch or any(len(row) != n_times for row in topology)):
            raise ValueError("Reference topology does not match signal windows/times")
        valid = np.isfinite(true)
        active = np.repeat(selected, n_times)[:, None, None]
        if np.any(active & valid & ~np.isfinite(pred)):
            raise ValueError(f"Non-finite {signal} predictions on valid selected targets")
        with np.errstate(over="ignore", invalid="ignore"):
            errors = (pred - true) ** 2
        if np.any(active & valid & ~np.isfinite(errors)):
            raise ValueError(f"Non-finite squared errors for {signal}")

        for b in np.flatnonzero(selected):
            masks = [self._mask(topology[b][t] if topology is not None else None) for t in range(n_times)]
            for region in REGIONS:
                state = self._shots.setdefault((signal, int(ids[b]), region), _ShotRegion())
                counts = state.counters
                counts["n_windows_selected"] += 1
                window_sse, window_cells = 0.0, 0
                for t in range(n_times):
                    idx = b * n_times + t
                    counts["n_fields_selected"] += 1
                    if region == "lcfs":
                        mask, reason = masks[t]
                        if reason is not None:
                            counts[reason] += 1
                            continue
                    else:
                        mask = self.grid.limiter_region if region == "limiter" else self._full_grid_mask
                    if not mask.any():
                        counts["n_fields_empty_region"] += 1
                        continue
                    cells = mask & valid[idx]
                    counts["n_reference_nonfinite"] += int((mask & ~valid[idx]).sum())
                    n_cells = int(cells.sum())
                    if n_cells:
                        counts["n_fields_scored"] += 1
                        window_sse += float(errors[idx][cells].sum())
                        window_cells += n_cells
                        state.coverage.update(f"{int(indices[b])}:{t};".encode())
                        state.coverage.update(np.packbits(cells).tobytes())
                counts["n_cells"] += window_cells
                state.squared_error_sum += window_sse
                if window_cells:
                    if not np.isfinite(window_sse):
                        raise ValueError(f"Regional squared-error sum overflow for {signal}")
                    counts["n_windows_scored"] += 1
                    state.windows.add(window_sse / window_cells)

    def write_csvs(self, *, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write long-form shot/task tables and a self-describing protocol manifest."""
        root = Path(metrics_task_dir)
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
                    "squared_error_sum": state.squared_error_sum,
                    **{key: state.counters[key] for key in COUNTERS},
                    "coverage_sha256": state.coverage.hexdigest(),
                }
            )
        for signal in sorted({key[0] for key in self._shots}):
            for region in REGIONS:
                rows = [row for row in shot_rows if row["signal"] == signal and row["region"] == region]
                mse, std, n = task_stats(row["mse"] for row in rows if np.isfinite(row["mse"]))
                task_rows.append(
                    {
                        "task": task_name,
                        "signal": signal,
                        "region": region,
                        "mse_units": f"({SIGNALS[signal]})^2",
                        "mse": mse,
                        "mse_std_pop": std,
                        "n_shots_scored": n,
                        "n_shots_selected": len(rows),
                        "squared_error_sum": sum(row["squared_error_sum"] for row in rows),
                        **{key: sum(row[key] for row in rows) for key in COUNTERS},
                    }
                )
        paths = {}
        for name, rows, extra in (
            ("shots", shot_rows, ["shot_id", "coverage_sha256"]),
            ("task", task_rows, ["mse_std_pop", "n_shots_scored", "n_shots_selected"]),
        ):
            path = root / f"regional_mse_{name}.csv"
            columns = ["task", "signal", "region", "mse_units", "mse", *extra, "squared_error_sum", *COUNTERS]
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
            paths[f"regional_mse_{name}_csv"] = str(path)
        protocol = root / "regional_mse_protocol.json"
        protocol.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "task": task_name,
                    "aggregation": "finite cells pooled within window; equal windows per shot; equal shots per task",
                    "regions": {
                        "full_grid": "all valid reference grid points",
                        "limiter": "MAST_lim_mask_rz, no erosion",
                        "lcfs": "reference EFIT polygon, boundary included, no limiter intersection",
                    },
                    "lcfs_boundary_tolerance_m": _GEOMETRY_TOLERANCE_M,
                    "lcfs_time_matching": "TopologyProvider nearest reference time, maximum 1.5 median reference cadence",
                    "missing_lcfs": "excluded only from LCFS MSE; missing/invalid counters reported",
                    "invalid_predictions": "raise on non-finite prediction at any selected finite target",
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
