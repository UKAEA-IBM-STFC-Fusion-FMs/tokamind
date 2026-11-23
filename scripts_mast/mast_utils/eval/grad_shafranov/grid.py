"""
Shared MAST Grad-Shafranov evaluation-grid utilities.

The helper owns the MAST-specific asset loading and the canonical positive
Grad-Shafranov operators:

    delta_star_fd(psi)  ~= Delta* psi
    delta_star_A(psi)   ~= Delta* psi

Residual metrics may negate these quantities to mirror a loss convention, but
the operator definitions themselves remain positive and are shared by all
evaluation diagnostics.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .constants import DEFAULT_GS_PARAMS_FILE, MU0, PLASMA_JTOR_THRESHOLD

logger = logging.getLogger("mmt.Eval")


class GSEvalGrid:
    """
    MAST Grad-Shafranov grid, limiter region, and canonical positive operators.

    Parameters
    ----------
    grad_shafranov_params_file : str | Path
        Path to the MAST Grad-Shafranov grid/operator asset.

    Attributes
    ----------
    r : np.ndarray
        Major-radius grid shaped ``(n_r, n_z)``.
    z : np.ndarray
        Vertical-coordinate grid shaped ``(n_r, n_z)``.
    operator : scipy.sparse.csr_matrix | None
        Sparse FreeGS operator approximating ``Delta*`` when available.
    region : np.ndarray
        Fixed, one-cell-eroded limiter region used for evaluation aggregation.

    Methods
    -------
    to_fields
        Convert decoded native arrays into ``(field, R, Z)`` layout.
    delta_star_fd
        Apply the canonical positive finite-difference ``Delta*`` operator.
    delta_star_A
        Apply the canonical positive FreeGS sparse ``Delta*`` operator.
    mean_abs_over_region
        Compute per-field mean absolute values over the fixed evaluation region.

    """

    def __init__(self, grad_shafranov_params_file: str | Path = DEFAULT_GS_PARAMS_FILE) -> None:
        asset = np.load(self._resolve_asset(grad_shafranov_params_file))

        self.r = np.asarray(asset["R_matrix"], dtype=np.float64)
        self.z = np.asarray(asset["Z_matrix"], dtype=np.float64)
        self.n_r = int(asset["n_r"])
        self.n_z = int(asset["n_z"])
        self.mu0 = float(asset["mu0"]) if "mu0" in asset else MU0

        if "r_axis_vector" in asset and "z_axis_vector" in asset:
            r_vec = np.asarray(asset["r_axis_vector"], dtype=np.float64).ravel()
            z_vec = np.asarray(asset["z_axis_vector"], dtype=np.float64).ravel()
            self.d_r = float(r_vec[1] - r_vec[0])
            self.d_z = float(z_vec[1] - z_vec[0])
        else:
            self.d_r = float(self.r[1, 0] - self.r[0, 0])
            self.d_z = float(self.z[0, 1] - self.z[0, 0])
        if self.d_r <= 0 or self.d_z <= 0:
            raise ValueError(f"Bad GS grid spacing dR={self.d_r}, dZ={self.d_z}; check R/Z axis orientation.")

        self.operator: sp.csr_matrix | None = None
        if all(key in asset for key in ("GS_op_coo_data", "GS_op_coo_row", "GS_op_coo_col", "GS_op_coo_shape")):
            try:
                self.operator = sp.coo_matrix(
                    (
                        np.asarray(asset["GS_op_coo_data"], dtype=np.float64),
                        (np.asarray(asset["GS_op_coo_row"]), np.asarray(asset["GS_op_coo_col"])),
                    ),
                    shape=tuple(int(size) for size in asset["GS_op_coo_shape"]),
                ).tocsr()
            except Exception as exc:  # noqa: BLE001
                logger.warning("GS evaluation: could not build FreeGS operator A (%s); A diagnostics disabled.", exc)

        limiter_mask = np.asarray(asset["MAST_lim_mask_rz"], dtype=bool)
        if limiter_mask.shape != (self.n_r, self.n_z):
            if limiter_mask.T.shape == (self.n_r, self.n_z):
                limiter_mask = limiter_mask.T
            else:
                raise ValueError(f"MAST_lim_mask_rz shape {limiter_mask.shape} != grid ({self.n_r},{self.n_z}).")
        self.region = self._erode_one_cell(limiter_mask)
        self.n_region_cells = int(self.region.sum())
        if self.n_region_cells == 0:
            raise RuntimeError("Static limiter region empty after erosion.")

    def _resolve_region(
        self,
        fields: np.ndarray,
        region: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the aggregation mask and per-field active-cell counts.

        Parameters
        ----------
        fields : np.ndarray
            Fields shaped ``(F, n_r, n_z)``.
        region : np.ndarray | None
            Optional ``(n_r, n_z)`` or ``(F, n_r, n_z)`` boolean region. When
            ``None`` the fixed eroded limiter region is used.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Broadcastable boolean mask and per-field counts shaped ``(F,)``.

        """

        n_fields = fields.shape[0]
        if region is None:
            return self.region[None, :, :], np.full(n_fields, float(self.n_region_cells))

        mask = np.asarray(region, dtype=bool)
        if mask.ndim == 2:
            mask = mask[None, :, :]
        if mask.ndim != 3 or mask.shape[1:] != (self.n_r, self.n_z):
            raise ValueError(f"region shape {mask.shape} incompatible with grid ({self.n_r},{self.n_z}).")
        if mask.shape[0] not in (1, n_fields):
            raise ValueError(f"region field count {mask.shape[0]} != {n_fields}.")

        counts = mask.reshape(mask.shape[0], -1).sum(axis=1).astype(np.float64)
        if mask.shape[0] == 1:
            counts = np.full(n_fields, counts[0])
        return mask, counts

    def mean_abs_over_region(self, fields: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
        """Return the per-field mean absolute value over the evaluation region."""

        mask, counts = self._resolve_region(fields, region)
        total = np.where(mask, np.abs(fields), 0.0).reshape(fields.shape[0], -1).sum(axis=1)
        return np.where(counts > 0, total / np.maximum(counts, 1.0), np.nan)

    def mean_square_over_region(self, fields: np.ndarray, region: np.ndarray | None = None) -> np.ndarray:
        """Return the per-field mean square value over the evaluation region."""

        mask, counts = self._resolve_region(fields, region)
        total = np.where(mask, fields**2, 0.0).reshape(fields.shape[0], -1).sum(axis=1)
        return np.where(counts > 0, total / np.maximum(counts, 1.0), np.nan)

    def plasma_region_from_fields(
        self,
        jtor_gt: np.ndarray,
        threshold: float = PLASMA_JTOR_THRESHOLD,
    ) -> np.ndarray:
        """
        Build the per-field ground-truth plasma region used by evaluation metrics.

        The evaluation mask is intersected with the fixed eroded limiter region
        so no derivative stencil evaluated inside it reaches the domain boundary.
        It intentionally does not claim to reproduce every training-loss mask:
        individual GS losses may use a different mask policy.

        Parameters
        ----------
        jtor_gt : np.ndarray
            Ground-truth ``j_tor`` fields shaped ``(F, n_r, n_z)`` in
            destandardized native units.
        threshold : float
            Positive-current threshold defining the plasma region.

        Returns
        -------
        np.ndarray
            Boolean mask shaped ``(F, n_r, n_z)``.

        """

        jtor = np.asarray(jtor_gt, dtype=np.float64)
        if jtor.ndim != 3 or jtor.shape[1:] != (self.n_r, self.n_z):
            raise ValueError(f"j_tor fields shape {jtor.shape} incompatible with grid ({self.n_r},{self.n_z}).")
        return np.isfinite(jtor) & (jtor > threshold) & self.region[None, :, :]

    def plasma_region(self, jtor_gt_native: np.ndarray, threshold: float = PLASMA_JTOR_THRESHOLD) -> np.ndarray:
        """Convert native ground-truth ``j_tor`` data and build its evaluation plasma region."""

        return self.plasma_region_from_fields(self.to_fields(jtor_gt_native), threshold=threshold)

    @staticmethod
    def _resolve_asset(filename: str | Path) -> Path:
        """Resolve a Grad-Shafranov asset path relative to the repository when needed."""

        path = Path(filename)
        if path.is_absolute():
            return path
        try:
            from mmt.utils.paths import REPO_ROOT

            return (Path(REPO_ROOT) / path).resolve()
        except Exception:  # noqa: BLE001
            return path.resolve()

    @staticmethod
    def _erode_one_cell(mask: np.ndarray) -> np.ndarray:
        """Return the interior cells of a boolean region after one orthogonal-cell erosion."""

        up = np.zeros_like(mask)
        up[1:, :] = mask[:-1, :]
        down = np.zeros_like(mask)
        down[:-1, :] = mask[1:, :]
        left = np.zeros_like(mask)
        left[:, 1:] = mask[:, :-1]
        right = np.zeros_like(mask)
        right[:, :-1] = mask[:, 1:]
        interior = np.zeros_like(mask)
        interior[1:-1, 1:-1] = True
        return mask & up & down & left & right & interior

    def to_fields(self, array: np.ndarray) -> np.ndarray:
        """
        Convert ``(B, H, W, T)`` or ``(B, H, W)`` native arrays into ``(B * T, n_r, n_z)`` fields.

        Parameters
        ----------
        array : np.ndarray
            Native decoded field data.

        Returns
        -------
        np.ndarray
            Fields in Grad-Shafranov ``(field, R, Z)`` orientation.

        """

        values = np.asarray(array)
        if values.ndim == 3:
            values = values[..., None]
        if values.ndim != 4:
            raise ValueError(f"Expected (B,H,W,T) or (B,H,W), got {values.shape}.")
        batch, height, width, n_times = values.shape
        fields = np.transpose(values, (0, 3, 2, 1)).reshape(batch * n_times, width, height)
        if fields.shape[1:] != (self.n_r, self.n_z):
            raise ValueError(f"Field grid {fields.shape[1:]} != ({self.n_r},{self.n_z}).")
        return fields

    def delta_star_fd(self, psi: np.ndarray) -> np.ndarray:
        """Apply the canonical positive centered finite-difference ``Delta*`` operator to ``(F, R, Z)`` fields."""

        result = np.zeros_like(psi)
        center = psi[:, 1:-1, 1:-1]
        r_plus = psi[:, 2:, 1:-1]
        r_minus = psi[:, :-2, 1:-1]
        z_plus = psi[:, 1:-1, 2:]
        z_minus = psi[:, 1:-1, :-2]
        radii = self.r[1:-1, 1:-1][None, :, :]
        d2_r = (r_plus - 2.0 * center + r_minus) / (self.d_r * self.d_r)
        d2_z = (z_plus - 2.0 * center + z_minus) / (self.d_z * self.d_z)
        d_r = (r_plus - r_minus) / (2.0 * self.d_r)
        result[:, 1:-1, 1:-1] = d2_z + d2_r - (1.0 / radii) * d_r
        return result

    def delta_star_A(self, psi: np.ndarray) -> np.ndarray:
        """Apply the canonical positive FreeGS sparse ``Delta*`` operator to ``(F, R, Z)`` fields."""

        if self.operator is None:
            raise RuntimeError("FreeGS operator A is unavailable in the Grad-Shafranov asset.")
        n_fields = psi.shape[0]
        psi_flat = psi.reshape(n_fields, self.n_r * self.n_z)
        result = (self.operator @ psi_flat.T).T
        return result.reshape(n_fields, self.n_r, self.n_z)
