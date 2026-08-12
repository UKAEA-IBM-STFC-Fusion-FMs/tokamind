"""
Post-batch topology reference provider for eval-only LCFS / X-point metrics.

Reference geometry (LCFS boundary polyline, X-point coordinates, magnetic-axis
coordinates) lives in the source equilibrium Zarr, **not** in the ML batch: it is
ragged (variable LCFS point count per timeslice), it must never enter ``x``, ``y``,
model normalization, or the benchmark scores, and it is needed only when an
eval-only reference metric is enabled.

This provider fetches that geometry *after* batching, keyed by ``(shot_id,
output_time)``. It reuses the model's exact window/output-time selection for time
alignment (nearest index on the equilibrium ``time`` axis), preserves the ragged
LCFS shape ``(n_points, 2)`` with no padding, opens each shot's Zarr once via a
per-shot cache, and costs nothing during training (no enabled metric ⇒ no provider
⇒ no I/O).

Field/orientation conventions mirror
``scripts_mast/grad_shafranov/grad_shafranov_params.py::load_gs_relevant_data_from_mast_shot``:
``equilibrium/{time,lcfs_r,lcfs_z,x_point_r,x_point_z,magnetic_axis_r,magnetic_axis_z}``.

When constructed with the ``MASTStorageManager`` already owned by the evaluation
dataset, the provider uses that manager to open the same local or remote Zarr
store as the normal data pipeline. A local root remains available as a small
standalone fallback.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("mmt.Eval")


# Physical equilibrium Zarr keys a topology metric may request. Metrics declare the physical fields they consume
# directly (no virtual psi_axis/psi_sep indirection); anything outside this set is ignored with a warning.
_KNOWN_TOPOLOGY_FIELDS: frozenset[str] = frozenset(
    {"lcfs_r", "lcfs_z", "x_point_r", "x_point_z", "magnetic_axis_r", "magnetic_axis_z"}
)

# Multiple of the local equilibrium time step used as the default nearest-time match tolerance when no absolute
# tolerance is supplied. A target output time farther than this from the nearest sample is treated as a missing /
# mismatched slice and skipped rather than silently matched.
_DEFAULT_TOLERANCE_CADENCE_FACTOR = 1.5


class TopologyProvider:
    """
    Lazy, per-shot-cached reader of source equilibrium topology aligned by physical time.

    Parameters
    ----------
    store_manager : Any | None
        The ``MASTStorageManager`` from the evaluation dataset. Its
        ``make_shot_store`` method is used with the configured ``local`` mode,
        so remote S3 and local Zarr evaluation use the same storage settings as
        normal data loading. Optional when ``store_root`` is supplied.
    store_root : str | Path | None
        Directory holding ``<shot_id>.zarr`` stores (``data.local_path``). This
        is a fallback for standalone/local callers that do not have a dataset
        storage manager.
    requested_fields : set[str]
        Physical equilibrium field names the enabled metrics asked for (e.g. ``lcfs_r``, ``x_point_z``,
        ``magnetic_axis_r``). Only these keys are read. Unknown names are ignored (with a warning) so a metric can
        never silently mis-request data.
    local : bool
        Whether the dataset source is local. This is passed through to
        ``store_manager.make_shot_store`` when a manager is available.
        Optional. Default: True.
    time_tolerance_s : float | None
        Maximum allowed absolute mismatch (seconds) between a requested output time and the nearest equilibrium
        sample. A slice whose best match exceeds this — or whose requested time is non-finite — is skipped
        (``None``) rather than scored against an unrelated equilibrium. When ``None``, an adaptive per-shot
        tolerance of ``1.5 x`` the median equilibrium time step is used; if that cadence cannot be inferred
        (fewer than two finite time samples), every slice for the shot is skipped rather than matched at an
        unbounded tolerance. Set an explicit value to force matching even when the cadence is indeterminable.
        Optional. Default: None.

    Attributes
    ----------
    zarr_keys : set[str]
        Resolved set of equilibrium Zarr keys actually read per shot.

    Methods
    -------
    query
        Return per-(window, timeslice) reference topology aligned to the requested physical times.

    """

    def __init__(
        self,
        requested_fields: set[str],
        *,
        store_manager: Any | None = None,
        store_root: str | Path | None = None,
        local: bool = True,
        time_tolerance_s: float | None = None,
    ) -> None:
        self._store_root = Path(store_root) if store_root is not None else None
        self._store_manager = store_manager
        self._local = bool(local)
        if time_tolerance_s is not None:
            time_tolerance_s = float(time_tolerance_s)
            if not np.isfinite(time_tolerance_s) or time_tolerance_s < 0.0:
                raise ValueError(f"time_tolerance_s must be finite and non-negative, got {time_tolerance_s!r}.")
        self._time_tolerance_s = time_tolerance_s

        keys: set[str] = set()
        for field in requested_fields:
            if field not in _KNOWN_TOPOLOGY_FIELDS:
                logger.warning("TopologyProvider: ignoring unknown requested topology field %r.", field)
                continue
            keys.add(field)
        self.zarr_keys = keys

        # Per-shot cache of the raw equilibrium arrays we read (opened once per shot).
        self._cache: dict[int, dict[str, np.ndarray] | None] = {}

        self._enabled = bool(self.zarr_keys and (self._store_manager is not None or self._store_root is not None))
        if not self._enabled:
            logger.warning(
                "TopologyProvider disabled (local=%s, store_manager=%s, store_root=%s, requested_keys=%s); "
                "topology queries return empty results.",
                self._local,
                type(self._store_manager).__name__ if self._store_manager is not None else None,
                self._store_root,
                sorted(self.zarr_keys),
            )

    # ..................................................................................................................
    def _load_shot(self, shot_id: int) -> dict[str, np.ndarray] | None:
        """Open one shot's Zarr once and cache the requested equilibrium arrays (``None`` on any failure)."""

        if shot_id in self._cache:
            return self._cache[shot_id]

        result: dict[str, np.ndarray] | None = None
        try:
            import zarr  # local import: only needed when a reference metric is active

            if self._store_manager is not None:
                store = self._store_manager.make_shot_store(shot_info={"shot_id": shot_id, "local": self._local})
                equilibrium = zarr.open_group(store=store, mode="r")["equilibrium"]
            else:
                shot_path = self._store_root / f"{shot_id}.zarr"  # type: ignore[operator]
                equilibrium = zarr.open_group(str(shot_path), mode="r")["equilibrium"]
            available = set(equilibrium.keys())

            arrays: dict[str, np.ndarray] = {}
            if "time" in available:
                arrays["time"] = np.asarray(equilibrium["time"][:], dtype=np.float64)
            for key in self.zarr_keys:
                if key in available:
                    arrays[key] = np.asarray(equilibrium[key][:], dtype=np.float64)
            if "time" not in arrays:
                logger.warning("TopologyProvider: shot %s has no equilibrium 'time' axis; cannot align.", shot_id)
            else:
                result = arrays
        except Exception as exc:  # noqa: BLE001 - reference metric must never break official eval
            logger.warning("TopologyProvider: failed to read shot %s (%s); topology unavailable for it.", shot_id, exc)
            result = None

        self._cache[shot_id] = result
        return result

    # ..................................................................................................................
    @staticmethod
    def _stack_ragged(r_over_time: np.ndarray | None, z_over_time: np.ndarray | None, t_idx: int) -> np.ndarray:
        """
        Return ``(n_pts, 2)`` valid ``(R, Z)`` points for time index ``t_idx`` (empty when unavailable).

        A point is retained only when both coordinates are finite **and** ``R > 0``, matching the GS-loss
        convention (``grad_shafranov.py::_make_lcfs_mask``). This drops zero-padded / placeholder rows (a
        missing X-point stored as ``(0, 0)`` is finite but unphysical) so they can never masquerade as real
        geometry downstream — e.g. become the "lowest-Z" primary X-point.
        """

        if r_over_time is None or z_over_time is None:
            return np.empty((0, 2), dtype=np.float64)
        # Source layout is (n_points, n_time); guard against a 1D (per-time scalar) array too.
        r_col = r_over_time[:, t_idx] if r_over_time.ndim == 2 else np.atleast_1d(r_over_time[t_idx])
        z_col = z_over_time[:, t_idx] if z_over_time.ndim == 2 else np.atleast_1d(z_over_time[t_idx])
        pts = np.stack([np.asarray(r_col, dtype=np.float64), np.asarray(z_col, dtype=np.float64)], axis=1)
        valid = np.isfinite(pts).all(axis=1) & (pts[:, 0] > 0.0)
        return pts[valid]

    # ..................................................................................................................
    def _slice_topology(
        self, arrays: dict[str, np.ndarray], t_idx: int, target_time: float, match_tolerance: float
    ) -> dict[str, Any]:
        """Assemble the reference topology dict (geometry + alignment provenance) for one time index."""

        lcfs = self._stack_ragged(arrays.get("lcfs_r"), arrays.get("lcfs_z"), t_idx)
        x_point = self._stack_ragged(arrays.get("x_point_r"), arrays.get("x_point_z"), t_idx)

        magnetic_axis = np.full(2, np.nan, dtype=np.float64)
        mag_r = arrays.get("magnetic_axis_r")
        mag_z = arrays.get("magnetic_axis_z")
        if mag_r is not None and mag_z is not None and t_idx < mag_r.shape[0] and t_idx < mag_z.shape[0]:
            magnetic_axis = np.asarray([mag_r[t_idx], mag_z[t_idx]], dtype=np.float64)

        source_time = float(arrays["time"][t_idx])
        return {
            "lcfs": lcfs,
            "x_point": x_point,
            "magnetic_axis": magnetic_axis,
            "source_time": source_time,
            "source_index": int(t_idx),
            "abs_time_error": abs(source_time - float(target_time)),
            "match_tolerance_s": float(match_tolerance),
        }

    # ..................................................................................................................
    def _tolerance_for(self, finite_times: np.ndarray) -> float:
        """
        Return the absolute time-match tolerance for a shot's *finite* time samples.

        Explicit ``time_tolerance_s`` wins; otherwise it is ``1.5x`` the median cadence. Returns ``inf`` when a
        cadence cannot be inferred (fewer than two finite samples or a degenerate spacing) — ``query`` treats
        that as "no safe tolerance" and skips the slice unless an explicit tolerance was configured.
        """

        if self._time_tolerance_s is not None:
            return self._time_tolerance_s
        if finite_times.size >= 2:
            cadence = float(np.median(np.abs(np.diff(finite_times))))
            if np.isfinite(cadence) and cadence > 0.0:
                return _DEFAULT_TOLERANCE_CADENCE_FACTOR * cadence
        return float("inf")

    # ..................................................................................................................
    def query(self, shot_ids: Sequence[Any], times: np.ndarray) -> list[list[dict[str, Any] | None]]:
        """
        Return reference topology for every ``(window, timeslice)`` in one eval batch.

        Parameters
        ----------
        shot_ids : Sequence[Any]
            Per-window source shot identifiers, length ``B``.
        times : np.ndarray
            Per-window per-slice physical output times, shape ``(B, T)`` (from ``output_time[signal]``).

        Returns
        -------
        list[list[dict[str, Any] | None]]
            ``topo[b][t]`` is ``None`` when the slice could not be aligned (shot unavailable, non-finite target
            time, or nearest equilibrium sample beyond the tolerance). Otherwise a dict with geometry keys
            ``"lcfs"`` ``(n_pts, 2)``, ``"x_point"`` ``(n_x, 2)``, ``"magnetic_axis"`` ``(2,)`` in physical
            ``(R, Z)`` metres, plus alignment provenance ``"source_time"``, ``"source_index"``,
            ``"abs_time_error"``.

        """

        times_arr = np.asarray(times, dtype=np.float64)
        if times_arr.ndim == 1:
            times_arr = times_arr[:, None]
        batch_size = len(shot_ids)
        n_times = int(times_arr.shape[1]) if times_arr.ndim == 2 else 1

        # Only an explicitly configured tolerance authorizes matching when no cadence can be inferred; otherwise
        # an indeterminate (inf) tolerance must reject rather than accept an arbitrarily distant equilibrium.
        allow_unbounded = self._time_tolerance_s is not None

        topo: list[list[dict[str, Any] | None]] = []
        for b in range(batch_size):
            row: list[dict[str, Any] | None] = []
            arrays = self._load_shot(int(shot_ids[b])) if self._enabled else None
            time_axis = arrays.get("time") if arrays is not None else None

            # Restrict alignment to finite time samples: a NaN in the axis otherwise wins np.argmin and, since
            # abs(nan - target) > tol is False, would be silently accepted as a (mis-timed) match.
            finite_idx: np.ndarray | None = None
            finite_times: np.ndarray | None = None
            tolerance = float("inf")
            if time_axis is not None and time_axis.size:
                finite_idx = np.where(np.isfinite(time_axis))[0]
                finite_times = time_axis[finite_idx]
                tolerance = self._tolerance_for(finite_times)

            for t in range(n_times):
                target = times_arr[b, t]
                if (
                    arrays is None
                    or finite_idx is None
                    or finite_idx.size == 0
                    or not np.isfinite(target)
                    or (not np.isfinite(tolerance) and not allow_unbounded)
                ):
                    row.append(None)
                    continue
                j = int(np.argmin(np.abs(finite_times - target)))
                t_idx = int(finite_idx[j])  # map back to the original (unfiltered) equilibrium index
                if abs(float(finite_times[j]) - float(target)) > tolerance:
                    row.append(None)  # nearest sample too far: missing data or a time-convention mismatch
                    continue
                row.append(self._slice_topology(arrays, t_idx, float(target), float(tolerance)))
            topo.append(row)
        return topo
