"""
Loss-independent smoothness diagnostics for reconstructed ``psi`` and optional ``j_tor`` fields.

For each native field, the metric compares the field with a Gaussian-smoothed
copy of itself. It reports both the fixed, eroded limiter-interior reduction
and, when ground-truth ``j_tor`` is available, the time-dependent plasma-mask
reduction. The latter is ``j_tor > 1e-6`` intersected with the limiter region.
Absolute roughness and relative roughness are reported, where the latter
divides by the field RMS to avoid rewarding a collapsed low-amplitude
prediction as artificially smooth.

The metric also evaluates the canonical positive ``Delta* psi`` fields from
the shared finite-difference and FreeGS operators. A model predicting only
``psi`` still receives the ``psi`` and ``Delta* psi`` diagnostics; its
``j_tor`` columns are NaN. Ground-truth columns are optional; fields containing
non-finite values are excluded rather than being silently imputed before
Gaussian smoothing.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from .aggregation import RunningMean, format_value, task_stats, warn_once
from .constants import DEFAULT_GS_PARAMS_FILE
from .grid import GSEvalGrid
from .regions import EVAL_REGIONS, LIMITER, PLASMA_MASK, PlasmaMaskCoverage

logger = logging.getLogger("mmt.Eval")

_DEFAULT_SIGMA = 1.0


@dataclass(frozen=True)
class _SmoothnessBatch:
    """Per-field roughness values and the optional validated plasma mask used to compute them."""

    values: dict[str, dict[str, np.ndarray]]
    plasma_mask: np.ndarray | None


class SmoothnessMetric:
    """
    Evaluate Gaussian-filter roughness for ``psi``, optional ``j_tor``, and ``Delta* psi``.

    Parameters
    ----------
    grad_shafranov_params_file : str | Path
        Path to the MAST Grad-Shafranov grid/operator asset when ``grid`` is not supplied.
    sigma : float
        Positive Gaussian filter width in grid cells.
    grid : GSEvalGrid | None
        Shared evaluation grid. Supplying the same instance as the GS residual metric avoids reloading the asset and
        guarantees identical region and operator definitions.

    Attributes
    ----------
    _shot_acc : dict[int, dict[str, dict[str, RunningMean]]]
        Per-shot roughness accumulators keyed by evaluated field.

    Methods
    -------
    add_batch
        Accumulate per-field roughness for one decoded evaluation batch.
    write_csvs
        Write per-shot and per-task companion CSV files.

    """

    _FIELDS = ("psi", "jtor", "dstar_psi_fd", "dstar_psi_A")

    def __init__(
        self,
        grad_shafranov_params_file: str | Path = DEFAULT_GS_PARAMS_FILE,
        sigma: float = _DEFAULT_SIGMA,
        *,
        grid: GSEvalGrid | None = None,
    ) -> None:
        self._sigma = float(sigma)
        if not np.isfinite(self._sigma) or self._sigma <= 0.0:
            raise ValueError(f"smoothness sigma must be finite and positive, got {sigma!r}.")
        self._grid = grid or GSEvalGrid(grad_shafranov_params_file)
        self._shot_acc: dict[int, dict[str, dict[str, RunningMean]]] = {}
        self._plasma_mask_coverage = PlasmaMaskCoverage()
        self._invalid_references: set[str] = set()

    def _warn_invalid_reference(self, name: str, exc: ValueError) -> None:
        """Warn once when an optional ground-truth reference cannot align with predictions."""

        warn_once(
            self._invalid_references,
            f"{name}_gt",
            logger,
            "Smoothness: ignoring invalid %s ground truth (%s); retaining diagnostics that do not require it.",
            name,
            exc,
        )

    @property
    def _accumulator_keys(self) -> tuple[str, ...]:
        return tuple(region.column(field) for region in EVAL_REGIONS for field in self._FIELDS)

    def _empty_shot_accumulator(self) -> dict[str, dict[str, RunningMean]]:
        """Create per-output finite accumulators for one shot."""

        return {
            key: {name: RunningMean() for name in ("abs", "rel", "abs_gt", "rel_gt")} for key in self._accumulator_keys
        }

    def _roughness_by_region(
        self,
        fields: np.ndarray,
        regions: dict[str, np.ndarray | None],
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute absolute and RMS-normalized roughness for each requested region.

        Fields with any non-finite values are excluded because Gaussian smoothing
        would otherwise propagate those values to nearby cells. A zero-amplitude
        field has valid absolute roughness but undefined relative roughness. The
        Gaussian filter is applied once per field collection; regions affect only
        the subsequent reductions.
        """

        n_fields = fields.shape[0]
        result = {tag: (np.full(n_fields, np.nan), np.full(n_fields, np.nan)) for tag in regions}
        finite_fields = np.isfinite(fields).all(axis=(1, 2))
        if not bool(finite_fields.any()):
            return result

        valid_fields = fields[finite_fields]
        smoothed = gaussian_filter(valid_fields, sigma=(0.0, self._sigma, self._sigma), mode="nearest")
        difference = valid_fields - smoothed

        for tag, region in regions.items():
            sub_region = region
            if region is not None and region.ndim == 3 and region.shape[0] > 1:
                sub_region = region[finite_fields]

            sq_error = self._grid.mean_square_over_region(difference, region=sub_region)
            sq_magnitude = self._grid.mean_square_over_region(valid_fields, region=sub_region)
            abs_valid = np.sqrt(sq_error)
            rms_valid = np.sqrt(sq_magnitude)
            rel_valid = np.full_like(abs_valid, np.nan)
            nonzero = rms_valid > 0.0
            rel_valid[nonzero] = abs_valid[nonzero] / rms_valid[nonzero]

            abs_roughness, rel_roughness = result[tag]
            abs_roughness[finite_fields] = abs_valid
            rel_roughness[finite_fields] = rel_valid
        return result

    def _compute_per_field_smoothness(
        self,
        psi_native: np.ndarray,
        jtor_native: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> _SmoothnessBatch:
        """Compute per-field roughness once and retain the validated plasma mask for batch coverage accounting."""

        psi = self._grid.to_fields(psi_native).astype(np.float64)
        jtor = None
        if jtor_native is not None:
            jtor = self._grid.to_fields(jtor_native).astype(np.float64)
            if psi.shape != jtor.shape:
                raise ValueError(f"psi {psi.shape} != j_tor {jtor.shape}")

        predicted_fields: dict[str, np.ndarray] = {
            "psi": psi,
            "dstar_psi_fd": self._grid.delta_star_fd(psi),
        }
        if jtor is not None:
            predicted_fields["jtor"] = jtor
        if self._grid.operator is not None:
            predicted_fields["dstar_psi_A"] = self._grid.delta_star_A(psi)

        ground_truth_fields: dict[str, np.ndarray] = {}
        if psi_gt_native is not None:
            try:
                psi_gt = self._grid.to_fields(psi_gt_native).astype(np.float64)
                if psi_gt.shape != psi.shape:
                    raise ValueError(f"psi_gt {psi_gt.shape} != psi_pred {psi.shape}")
            except ValueError as exc:
                self._warn_invalid_reference("psi", exc)
            else:
                ground_truth_fields["psi"] = psi_gt
                ground_truth_fields["dstar_psi_fd"] = self._grid.delta_star_fd(psi_gt)
                if self._grid.operator is not None:
                    ground_truth_fields["dstar_psi_A"] = self._grid.delta_star_A(psi_gt)

        plasma_mask = None
        if jtor_gt_native is not None:
            try:
                jtor_gt = self._grid.to_fields(jtor_gt_native).astype(np.float64)
                if jtor_gt.shape != psi.shape:
                    raise ValueError(f"jtor_gt {jtor_gt.shape} != psi_pred {psi.shape}")
                plasma_mask = self._grid.plasma_region_from_fields(jtor_gt)
            except ValueError as exc:
                self._warn_invalid_reference("jtor", exc)
            else:
                if jtor is not None:
                    ground_truth_fields["jtor"] = jtor_gt

        n_fields = psi.shape[0]
        nan_values = np.full(n_fields, np.nan)
        result: dict[str, dict[str, np.ndarray]] = {}
        regions: dict[str, np.ndarray | None] = {LIMITER.key: None}
        if plasma_mask is not None:
            regions[PLASMA_MASK.key] = plasma_mask

        for field_name in self._FIELDS:
            predicted_roughness = (
                self._roughness_by_region(predicted_fields[field_name], regions)
                if field_name in predicted_fields
                else {}
            )
            ground_truth_roughness = (
                self._roughness_by_region(ground_truth_fields[field_name], regions)
                if field_name in ground_truth_fields
                else {}
            )
            for region in EVAL_REGIONS:
                key = region.column(field_name)
                if region.key not in regions or field_name not in predicted_fields:
                    result[key] = {k: nan_values.copy() for k in ("abs", "rel", "abs_gt", "rel_gt")}
                    continue

                abs_pred, rel_pred = predicted_roughness[region.key]
                entry = {
                    "abs": abs_pred,
                    "rel": rel_pred,
                    "abs_gt": nan_values.copy(),
                    "rel_gt": nan_values.copy(),
                }
                if field_name in ground_truth_fields:
                    entry["abs_gt"], entry["rel_gt"] = ground_truth_roughness[region.key]
                result[key] = entry
        return _SmoothnessBatch(values=result, plasma_mask=plasma_mask)

    def per_field_smoothness(
        self,
        psi_native: np.ndarray,
        jtor_native: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Return per-field predicted and ground-truth roughness arrays.

        Parameters
        ----------
        psi_native : np.ndarray
            Destandardized predicted flux field.
        jtor_native : np.ndarray | None
            Optional destandardized predicted toroidal-current field.
        psi_gt_native, jtor_gt_native : np.ndarray | None
            Optional destandardized references for ground-truth roughness and
            the time-dependent plasma-mask region.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Absolute and relative roughness arrays for each field and domain.
            Optional references with incompatible shapes are ignored with a
            one-time warning; model-facing roughness remains available.
        """

        return self._compute_per_field_smoothness(
            psi_native,
            jtor_native,
            psi_gt_native=psi_gt_native,
            jtor_gt_native=jtor_gt_native,
        ).values

    def add_batch(
        self,
        psi_native: np.ndarray,
        jtor_native: np.ndarray | None,
        shot_ids: np.ndarray,
        window_mask: np.ndarray | None = None,
        psi_gt_native: np.ndarray | None = None,
        jtor_gt_native: np.ndarray | None = None,
    ) -> None:
        """Accumulate valid predicted and ground-truth smoothness values by shot."""

        batch = self._compute_per_field_smoothness(
            psi_native,
            jtor_native,
            psi_gt_native=psi_gt_native,
            jtor_gt_native=jtor_gt_native,
        )
        per_field = batch.values
        batch_size = int(np.asarray(shot_ids).shape[0])
        n_fields = per_field["psi"]["abs"].shape[0]
        n_times = n_fields // max(batch_size, 1)
        if n_times * batch_size != n_fields:
            logger.warning("Smoothness: field count %d not divisible by batch %d; skipping.", n_fields, batch_size)
            return

        shot_ids = np.asarray(shot_ids)
        if window_mask is not None:
            window_mask = np.asarray(window_mask, dtype=bool)
        for field_index in range(n_fields):
            window_index = field_index // n_times
            if window_mask is not None and not window_mask[window_index]:
                continue
            shot_id = int(shot_ids[window_index])
            has_model_value = any(
                np.isfinite(values[name][field_index]) for values in per_field.values() for name in ("abs", "rel")
            )
            if not has_model_value:
                continue
            if batch.plasma_mask is not None:
                self._plasma_mask_coverage.observe(shot_id, batch.plasma_mask, field_index)
            shot = self._shot_acc.setdefault(shot_id, self._empty_shot_accumulator())
            for field_name, values in per_field.items():
                if not any(np.isfinite(values[name][field_index]) for name in ("abs", "rel")):
                    continue
                for name in ("abs", "rel", "abs_gt", "rel_gt"):
                    shot[field_name][name].add(values[name][field_index])

    def is_empty(self) -> bool:
        """Return whether no shot has accumulated any finite smoothness values."""

        return not any(
            sums["abs"].count > 0 or sums["rel"].count > 0
            for shot_sums in self._shot_acc.values()
            for sums in shot_sums.values()
        )

    def write_csvs(self, metrics_task_dir: Path, task_name: str) -> dict[str, Any]:
        """Write per-shot and per-task smoothness CSVs and return a compact summary."""

        metrics_task_dir = Path(metrics_task_dir)
        metrics_task_dir.mkdir(parents=True, exist_ok=True)
        per_shot: dict[str, list[float]] = {}

        accumulator_keys = self._accumulator_keys

        # ..............................................................................................................
        # Per-shot CSV

        shots_csv = metrics_task_dir / "smoothness_shots.csv"
        header = ["shot_id"]
        for key in accumulator_keys:
            header.extend(
                [
                    f"roughness_{key}_abs_pred",
                    f"roughness_{key}_rel_pred",
                    f"roughness_{key}_abs_gt",
                    f"roughness_{key}_rel_gt",
                ]
            )
        header.extend(
            [
                "n_fields",
                "n_fields_with_gt",
                "n_fields_plasma_mask",
                "n_fields_with_plasma_reference",
                "n_fields_empty_plasma_mask",
            ]
        )

        with shots_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(header)
            for shot_id in sorted(self._shot_acc):
                row: list[Any] = [shot_id]
                n_fields = 0
                n_fields_with_gt = 0
                n_fields_plasma = 0
                for key in accumulator_keys:
                    sums = self._shot_acc[shot_id][key]
                    abs_mean = sums["abs"].mean
                    rel_mean = sums["rel"].mean
                    abs_gt_mean = sums["abs_gt"].mean
                    rel_gt_mean = sums["rel_gt"].mean

                    if key.endswith(PLASMA_MASK.suffix):
                        n_fields_plasma = max(n_fields_plasma, sums["abs"].count)
                    else:
                        n_fields = max(n_fields, sums["abs"].count)
                        n_fields_with_gt = max(n_fields_with_gt, sums["abs_gt"].count)

                    for name, value in (
                        (f"{key}_abs", abs_mean),
                        (f"{key}_rel", rel_mean),
                        (f"{key}_abs_gt", abs_gt_mean),
                        (f"{key}_rel_gt", rel_gt_mean),
                    ):
                        if np.isfinite(value):
                            per_shot.setdefault(name, []).append(value)
                    row.extend(format_value(value) for value in (abs_mean, rel_mean, abs_gt_mean, rel_gt_mean))
                row.extend([n_fields, n_fields_with_gt, n_fields_plasma, *self._plasma_mask_coverage.per_shot(shot_id)])
                writer.writerow(row)

        # ..............................................................................................................
        # Per-task CSV

        task_csv = metrics_task_dir / "smoothness_task.csv"
        task_header = ["task"]
        for key in accumulator_keys:
            task_header.extend(
                [
                    f"roughness_{key}_abs_pred",
                    f"roughness_{key}_abs_pred_std_pop",
                    f"roughness_{key}_rel_pred",
                    f"roughness_{key}_rel_pred_std_pop",
                    f"roughness_{key}_abs_gt",
                    f"roughness_{key}_rel_gt",
                ]
            )
        task_header.extend(
            [
                "n_shots",
                "n_shots_with_gt",
                "n_shots_plasma_mask",
                "n_fields_with_plasma_reference",
                "n_fields_empty_plasma_mask",
                "sigma",
            ]
        )

        n_shots = len(self._shot_acc)
        n_shots_with_gt = len(per_shot.get("psi_abs_gt", []))
        n_shots_plasma_mask = len(per_shot.get("psi_plasma_mask_abs", []))
        n_fields_with_plasma_reference, n_fields_empty_plasma_mask = self._plasma_mask_coverage.totals()
        stats_by_name = {
            f"{key}_{component}": task_stats(per_shot.get(f"{key}_{component}", []))
            for key in accumulator_keys
            for component in ("abs", "rel", "abs_gt", "rel_gt")
        }

        with task_csv.open("w", newline="") as file_handle:
            writer = csv.writer(file_handle)
            writer.writerow(task_header)
            row = [task_name]
            for key in accumulator_keys:
                abs_mean, abs_std, _ = stats_by_name[f"{key}_abs"]
                rel_mean, rel_std, _ = stats_by_name[f"{key}_rel"]
                abs_gt_mean, _, _ = stats_by_name[f"{key}_abs_gt"]
                rel_gt_mean, _, _ = stats_by_name[f"{key}_rel_gt"]
                row.extend(
                    format_value(value) for value in (abs_mean, abs_std, rel_mean, rel_std, abs_gt_mean, rel_gt_mean)
                )
            row.extend(
                [
                    n_shots,
                    n_shots_with_gt,
                    n_shots_plasma_mask,
                    n_fields_with_plasma_reference,
                    n_fields_empty_plasma_mask,
                    self._sigma,
                ]
            )
            writer.writerow(row)

        # ..............................................................................................................
        # Summary

        summary: dict[str, Any] = {
            "smoothness_shots_csv": str(shots_csv),
            "smoothness_task_csv": str(task_csv),
            "n_shots": n_shots,
            "n_shots_with_gt": n_shots_with_gt,
            "n_shots_plasma_mask": n_shots_plasma_mask,
            "n_fields_with_plasma_reference": n_fields_with_plasma_reference,
            "n_fields_empty_plasma_mask": n_fields_empty_plasma_mask,
            "sigma": self._sigma,
        }
        for key in accumulator_keys:
            summary[f"roughness_{key}_rel_pred"] = stats_by_name[f"{key}_rel"][0]
            summary[f"roughness_{key}_rel_gt"] = stats_by_name[f"{key}_rel_gt"][0]
        return summary
