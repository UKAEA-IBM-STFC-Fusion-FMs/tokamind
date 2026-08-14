"""
Weak-form Grad-Shafranov residual loss in native (destandardized) space.

Continuous weak formulation

    Find ψ such that

        ∫Ω (1/R) ∇ψ · ∇v dΩ - μ0 ∫Ω j_tor v dΩ = 0

    for all admissible test functions v.

After integration by parts, the boundary contribution is not assembled
explicitly. The residual is evaluated only over the masked plasma region,
so no explicit boundary condition is imposed by this loss.

Discrete weak residual

    The weak-form bilinear operator is discretized as an edge-based stiffness operator

        W ψ

    assembled from first-derivative edge fluxes with 1/R edge weighting.

    The source term uses a lumped (diagonal) mass approximation,

        μ0 j_tor,

    yielding the discrete residual

        r = W ψ − μ0 j_tor.

Supported RHS sources (`rhs_input.origin`)

    - ``predicted_j_tor`` (general case): both ψ and j_tor are specified model
      outputs. All three residual variants (`no_gt`, `lhs_gt`, `rhs_gt`) are
      available.

    - ``derived_j_tor`` (reduced case): ψ is the only specified model output.
      The reference current is derived from ground-truth ψ through the same
      weak operator,

          j_tor_true = W(psi_gt) / mu0,

      so the RHS reduces to W(psi_gt). In this mode the `no_gt` residual
      collapses onto the `lhs_gt` anchor and `rhs_gt` is degenerate; the
      recommended configuration is {no_gt: 0.0, lhs_gt: 1.0, rhs_gt: 0.0},
      giving the single-term loss R(W psi_pred - W psi_gt).

The loss minimizes a reduction (MSE by default) of this residual over
the masked plasma region.

Note: This loss evaluates the discrete weak residual rather than solving the
weak-form finite-element system.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Hashable, Literal

import numpy as np
import torch
from torch import Tensor

from mmt.data.embeddings.torch_decoder import TorchDecoder
from mmt.data.standardization import destandardize_torch
from mmt.train.losses.base import BaseLoss, LossComputeContext
from .helpers import (
    mu0,
    native_to_gs_fields,
    masked_reduce,
    output_masks_to_field_mask,
    parse_gs_grid_assets,
    prepare_target_field,
    resolve_gs_asset_path,
    resolve_output_key,
    runtime_tensor,
    select_diagnostic_plot_slice,
    training_plot_path,
    validate_plot_check_cfg,
)
from mmt.train.losses.constants import (
    GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR,
    GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR,
    GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY,
    GRAD_SHAFRANOV_RHS_KEYS,
)
from mmt.utils.paths import REPO_ROOT
from .plots import make_gs_plots

WEAK_FORM_RHS_INPUT_ORIGINS: frozenset[str] = frozenset(
    {GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR, GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR}
)

# ----------------------------------------------------------------------------------------------------------------------
# Preliminaries

DEFAULT_WEAK_FORM_GRAD_SHAFRANOV_WEIGHTS: dict[str, float] = {
    "no_gt": 0.5,
    "lhs_gt": 0.25,
    "rhs_gt": 0.25,
}


# ======================================================================================================================
class WeakFormGradShafranovLoss(BaseLoss):
    """
    Weak-form Grad-Shafranov residual loss.

    Parameters
    ----------
    decoders : dict[Hashable, TorchDecoder]
        Per-output differentiable decoders keyed by signal id.
    signal_stats : Mapping[str, Mapping[str, Any]]
        Per-signal standardization metadata containing ``mean`` and ``std``.
    output_name_to_id : Mapping[str, Hashable]
        Mapping from output signal names to signal ids.
    grad_shafranov_params_file : str | Path | None
        Path to the Grad-Shafranov grid/operator asset.
    grad_shafranov_weights : dict[str, float] | None
        Weights for the weak residual variants. ``no_gt`` scores ``W(psi_pred)`` against
        ``mu0 * j_tor_pred``. ``lhs_gt`` scores ``W(psi_pred)`` against the ground-truth-LHS reference
        ``W(psi_gt)``. ``rhs_gt`` scores ``mu0 * j_tor_pred`` against ``W(psi_gt)``.
    mask_to_plasma : bool
        Whether to restrict the residual norm to the ground-truth plasma region.
    rhs_input : str | None
        RHS source. ``predicted_j_tor`` (general case) uses the predicted current, and all three
        residual variants are available. ``derived_j_tor`` (reduced case) derives the reference
        current from ground-truth psi through W, so psi is the only required model output and
        ``rhs_gt`` must be zero. Optional. Default: ``predicted_j_tor``.
    output_weights : dict[Hashable, float] | None
        Optional per-output weights, accepted for loss-system consistency.
    output_filter : set[Hashable] | None
        Optional supervised output ids. When provided it must include all outputs required by
        ``rhs_input``: psi and j_tor for ``predicted_j_tor``, psi alone for ``derived_j_tor``.

    Attributes
    ----------
    requires_native_target : bool
        True because the term uses native output masks and optional diagnostic ground truth.
    requires_decode : bool
        True because predictions are decoded from embedding space.
    requires_destandardize : bool
        False for aggregator purposes; this class handles destandardization internally.

    """

    requires_native_target: bool = True
    requires_decode: bool = True
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_term_cfg(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate config fields owned by the weak form Grad-Shafranov loss.

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.

        Returns
        -------
        None

        """

        cls._validate_known_term_keys(
            term_def=term_def,
            path=path,
            allowed_specific_keys={
                "grad_shafranov_params_file",
                "grad_shafranov_weights",
                "loss_type",
                "plot_check",
                "rhs_input",
            },
        )

        grad_shafranov_params_file = term_def.get("grad_shafranov_params_file")
        if grad_shafranov_params_file is None:
            raise KeyError(f"{path}.grad_shafranov_params_file is required.")

        if not isinstance(grad_shafranov_params_file, str):
            raise TypeError(f"{path}.grad_shafranov_params_file must be a string.")

        rhs_input = term_def.get("rhs_input") or {}
        if not isinstance(rhs_input, Mapping):
            raise TypeError(f"{path}.rhs_input must be a mapping when provided.")

        unknown_rhs_keys = sorted(str(key) for key in rhs_input if key not in GRAD_SHAFRANOV_RHS_KEYS)
        if unknown_rhs_keys:
            raise KeyError(
                f"Unknown {path}.rhs_input keys: {unknown_rhs_keys}. Supported: {sorted(GRAD_SHAFRANOV_RHS_KEYS)}."
            )

        rhs_input_origin = str(
            rhs_input.get(GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY, GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR)
        )
        if rhs_input_origin not in WEAK_FORM_RHS_INPUT_ORIGINS:
            raise ValueError(
                f"{path}.rhs_input.origin={rhs_input_origin!r} is unsupported by the weak form. "
                f"Supported: {sorted(WEAK_FORM_RHS_INPUT_ORIGINS)}."
            )

        rhs_input_calculation_method = str(
            rhs_input.get(GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY, GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR)
        )
        if rhs_input_calculation_method != GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR:
            raise ValueError(
                f"{path}.rhs_input.calculation_method={rhs_input_calculation_method!r} is unsupported by the weak "
                f"form; only {GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR!r} (the discrete stiffness operator W) applies."
            )

        loss_type = term_def.get("loss_type")
        if loss_type is not None and loss_type not in {"l2", "mse"}:
            raise ValueError(f"{path}.loss_type must be 'l2' or 'mse'.")

        validate_plot_check_cfg(term_def.get("plot_check"), path)

        grad_shafranov_weights = term_def.get("grad_shafranov_weights") or {}
        if not isinstance(grad_shafranov_weights, Mapping):
            raise TypeError(f"{path}.grad_shafranov_weights must be a mapping when provided.")

        cls._validate_weight_mapping(
            weights=grad_shafranov_weights,
            path=f"{path}.grad_shafranov_weights",
            allowed_keys=set(DEFAULT_WEAK_FORM_GRAD_SHAFRANOV_WEIGHTS),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        decoders: dict[Hashable, TorchDecoder],
        signal_stats: Mapping[str, Mapping[str, Any]],
        output_name_to_id: Mapping[str, Hashable],
        grad_shafranov_params_file: str | Path | None,
        grad_shafranov_weights: dict[str, float] | None = None,
        mask_to_plasma: bool = True,
        rhs_input: str | None = None,
        loss_type: Literal["l2", "mse"] = "mse",
        output_weights: dict[Hashable, float] | None = None,
        output_filter: set[Hashable] | None = None,
        plot_check_type: str | None = None,
        plot_check_probability: float | None = None,
    ) -> None:
        if not decoders:
            raise ValueError("WeakFormGradShafranovLoss requires at least one decoder.")

        if grad_shafranov_params_file is None:
            raise ValueError("WeakFormGradShafranovLoss requires grad_shafranov_params_file.")

        self._decoders = decoders
        self.rhs_input = rhs_input or GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR
        if self.rhs_input not in WEAK_FORM_RHS_INPUT_ORIGINS:
            raise ValueError(
                f"Unsupported `rhs_input={self.rhs_input!r}`. Supported: {sorted(WEAK_FORM_RHS_INPUT_ORIGINS)}."
            )
        self._derives_j_tor = self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR
        self._output_weights = output_weights or {}
        self._output_filter = set(output_filter) if output_filter is not None else None
        self._output_name_to_id = {str(name): sid for name, sid in output_name_to_id.items()}
        self.signal_stats = {str(name): dict(stats) for name, stats in signal_stats.items()}
        self.gs_weights = {**DEFAULT_WEAK_FORM_GRAD_SHAFRANOV_WEIGHTS, **(grad_shafranov_weights or {})}
        if self._derives_j_tor and self.gs_weights["rhs_gt"] > 0.0:
            raise ValueError(
                "[WeakFormGradShafranovLoss] `rhs_gt` is degenerate when `rhs_input.origin='derived_j_tor'` "
                "(the derived RHS equals W(psi_gt), so the residual carries no gradient). Set rhs_gt: 0.0."
            )
        self.mask_to_plasma = bool(mask_to_plasma)
        if loss_type not in ("l2", "mse"):
            raise ValueError(
                f"[WeakFormGradShafranovLoss] Invalid `loss_type`: must be in ['l2', 'mse'], got '{loss_type}'."
            )
        self.loss_type = loss_type
        self.plot_check_type = plot_check_type  # TODO: Unify this into plot_check_cfg
        self.plot_check_probability = float(plot_check_probability or 0.0)  # TODO: Unify this into plot_check_cfg

        self._psi_key = resolve_output_key(
            self._output_name_to_id, "equilibrium-psi", loss_name="Weak form Grad-Shafranov loss"
        )
        if self._derives_j_tor:
            # Reduced case: j_tor is not a model output, so it may be absent from the output map entirely.
            self._j_tor_key = self._output_name_to_id.get("equilibrium-j_tor")
            self.required_output_keys = (self._psi_key,)
        else:
            self._j_tor_key = resolve_output_key(
                self._output_name_to_id, "equilibrium-j_tor", loss_name="Weak form Grad-Shafranov loss"
            )
            self.required_output_keys = (self._psi_key, self._j_tor_key)

        if self._output_filter is not None:
            missing = [key for key in self.required_output_keys if key not in self._output_filter]
            if missing:
                raise ValueError(
                    f"WeakFormGradShafranovLoss output filter must include all outputs required by "
                    f"`rhs_input.origin={self.rhs_input!r}`: {missing}."
                )

        self.n_r = None
        self.n_z = None
        self.R_matrix = None
        self.Z_matrix = None
        self.base_j_tor_limiter_mask_rz = None
        self.dr = None
        self.dz = None
        self.load_gs_params(filename=grad_shafranov_params_file)

    # ------------------------------------------------------------------------------------------------------------------
    def load_gs_params(self, filename: str | Path) -> None:
        """Load the weak-form grid geometry and derive its uniform R/Z spacings.

        Parameters
        ----------
        filename : str | Path
            Absolute or repository-relative path to the Grad-Shafranov ``.npz`` asset.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the loaded R/Z grids do not define positive uniform ``dr`` and ``dz`` spacings.

        Notes
        -----
        The method stores ``n_r``, ``n_z``, R/Z grids, the optional limiter mask, and the grid spacings used by the
        discrete stiffness operator.
        """

        with np.load(resolve_gs_asset_path(filename, REPO_ROOT)) as loaded:
            grid_assets = parse_gs_grid_assets(loaded, tensor_dtype=torch.float32)
        self.n_r = grid_assets.n_r
        self.n_z = grid_assets.n_z
        self.R_matrix = grid_assets.r_matrix  # (n_r, n_z): R varies along axis 0, Z along axis 1
        self.Z_matrix = grid_assets.z_matrix
        # R varies down rows (axis 0), Z across cols (axis 1) — confirmed from the asset.
        r_axis = self.R_matrix[:, 0]  # column 0, R changes down rows
        z_axis = self.Z_matrix[0, :]  # row 0, Z changes across cols
        self.dr = float((r_axis.max() - r_axis.min()) / (self.n_r - 1))
        self.dz = float((z_axis.max() - z_axis.min()) / (self.n_z - 1))
        if self.dr <= 0 or self.dz <= 0:
            raise ValueError(f"Bad GS grid spacing dr={self.dr}, dz={self.dz}; check R/Z axis orientation.")
        self.base_j_tor_limiter_mask_rz = grid_assets.base_j_tor_limiter_mask_rz

    # ------------------------------------------------------------------------------------------------------------------
    def _discrete_stiffness(self, psi_fields: Tensor, R_rz: Tensor) -> Tensor:
        """
        Apply the discrete stiffness operator associated with the weak-form
        bilinear form

           a(ψ,v)=∫Ω (1/R) ∇ψ·∇v dΩ.

        The operator is evaluated using edge-based flux differences with
        arithmetic averaging of 1/R across grid edges.

        This computes the action Wψ directly; no linear system is assembled
        or solved.
        """
        dR, dZ = self.dr, self.dz
        res = torch.zeros_like(psi_fields)

        # R-direction edges
        R_edge_R = 0.5 * (R_rz[1:, :] + R_rz[:-1, :])
        wR = (1.0 / R_edge_R) / (dR * dR)
        flux_R = wR.unsqueeze(0) * (psi_fields[:, 1:, :] - psi_fields[:, :-1, :])
        res[:, :-1, :] = res[:, :-1, :] - flux_R
        res[:, 1:, :] = res[:, 1:, :] + flux_R

        # Z-direction edges
        R_edge_Z = 0.5 * (R_rz[:, 1:] + R_rz[:, :-1])
        wZ = (1.0 / R_edge_Z) / (dZ * dZ)
        flux_Z = wZ.unsqueeze(0) * (psi_fields[:, :, 1:] - psi_fields[:, :, :-1])
        res[:, :, :-1] = res[:, :, :-1] - flux_Z
        res[:, :, 1:] = res[:, :, 1:] + flux_Z

        return res

    # ------------------------------------------------------------------------------------------------------------------
    def _j_tor_from_psi_via_weak_operator(self, psi_gt_fields: Tensor, R_rz: Tensor) -> Tensor:
        """
        Derive the reference toroidal current from ground-truth psi through the weak operator,

            j_tor_true = W(psi_gt) / mu0,

        clamped to non-negative values. This is the weak-form counterpart of
        `GradShafranovResidualLoss.j_tor_from_psi_via_operator`, which uses the strong-form
        relation j_tor_true = (mu0 * R)^-1 * (-Delta* psi_gt).

        Parameters
        ----------
        psi_gt_fields : Tensor
            Physical ground-truth poloidal-flux fields shaped ``(F, n_r, n_z)``.
        R_rz : Tensor
            Major-radius grid shaped ``(n_r, n_z)``.

        Returns
        -------
        Tensor
            Derived toroidal-current fields shaped ``(F, n_r, n_z)``.

        Notes
        -----
        The factor R that multiplies j_tor in the strong form is absorbed into the 1/R edge
        weights of W, so no explicit R appears here. The clamp mirrors the strong-form
        positivity constraint on plasma current; inside the plasma mask it is normally
        inactive, and the exact (unclamped) reduced residual is recovered by scoring with
        `lhs_gt` rather than `no_gt`, since `lhs_gt` compares W(psi_pred) against W(psi_gt)
        directly. Because this path consumes only ground truth, the derived current carries
        no gradient to the model and is used for the RHS and for diagnostics only.
        """

        return torch.clamp(self._discrete_stiffness(psi_gt_fields, R_rz) / mu0, min=0.0)

    # ------------------------------------------------------------------------------------------------------------------
    def _field_norm(self, field: Tensor, mask: Tensor | None = None) -> Tensor:  # FIXME: Replace "norm" by "metric"
        """
        Return the per-field residual norm for a stack of ``(n_r, n_z)`` residual fields.

        Parameters
        ----------
        field : Tensor
            Residual fields shaped ``(F, n_r, n_z)``.

        Returns
        -------
        Tensor
            Per-field norms shaped ``(F,)``.

        """
        return masked_reduce(residual=field, mask=mask, kind=self.loss_type)

    # ------------------------------------------------------------------------------------------------------------------
    def _decode_and_destandardize_predictions(
        self, preds: Mapping[Hashable, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Decode psi (and j_tor when predicted) and return them in physical GS field orientation."""

        if (self._psi_key not in self._decoders) or (self._psi_key not in preds):
            raise RuntimeError("Weak form Grad-Shafranov loss requires decoded 'equilibrium-psi'.")

        psi_bhwt = self._decoders[self._psi_key](preds[self._psi_key].to(torch.float32))
        psi_fields = destandardize_torch(
            x=native_to_gs_fields(psi_bhwt),
            mean=self.signal_stats["equilibrium-psi"]["mean"],
            std=self.signal_stats["equilibrium-psi"]["std"],
        )

        if self._derives_j_tor:
            return psi_bhwt, psi_fields, None

        if (self._j_tor_key not in self._decoders) or (self._j_tor_key not in preds):
            raise RuntimeError("Weak form Grad-Shafranov loss requires decoded 'equilibrium-j_tor'.")

        j_bhwt = self._decoders[self._j_tor_key](preds[self._j_tor_key].to(torch.float32))
        j_tor_fields = destandardize_torch(
            x=native_to_gs_fields(j_bhwt),
            mean=self.signal_stats["equilibrium-j_tor"]["mean"],
            std=self.signal_stats["equilibrium-j_tor"]["std"],
        )
        return psi_bhwt, psi_fields, j_tor_fields

    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_targets_and_cleaning_mask(
        self, y_native: Mapping[Hashable, Tensor], *, n_fields: int
    ) -> tuple[Tensor, Tensor | None, Tensor]:
        """Prepare finite target fields and the weak-form physical-domain mask."""

        if self._psi_key not in y_native:
            raise RuntimeError("WeakFormGradShafranovLoss requires native psi targets.")

        psi_gt_stdized = native_to_gs_fields(y_native[self._psi_key].to(torch.float32))
        psi_gt_fields, psi_finite_mask = prepare_target_field(psi_gt_stdized, self.signal_stats["equilibrium-psi"])

        jtor_gt_fields = None
        use_gt_jtor_mask = (not self._derives_j_tor) and (self._j_tor_key is not None) and (self._j_tor_key in y_native)
        if use_gt_jtor_mask:
            j_gt_stdized = native_to_gs_fields(y_native[self._j_tor_key].to(torch.float32))
            jtor_gt_fields, jtor_finite_mask = prepare_target_field(
                j_gt_stdized, self.signal_stats["equilibrium-j_tor"]
            )
            plasma_mask = jtor_finite_mask & (jtor_gt_fields > 1e-6)
        else:
            jtor_finite_mask = torch.ones_like(psi_finite_mask)
            if self.base_j_tor_limiter_mask_rz is None:
                raise RuntimeError("No GT j_tor and no `base_j_tor_lim_mask_rz` in the GS asset.")
            plasma_mask = self.base_j_tor_limiter_mask_rz.to(psi_finite_mask.device).expand(n_fields, -1, -1)

        # The weak residual is evaluated only over valid plasma nodes. No explicit boundary integral is assembled.
        cleaning_mask = psi_finite_mask & jtor_finite_mask
        if self.mask_to_plasma:
            cleaning_mask = cleaning_mask & plasma_mask
        return psi_gt_fields, jtor_gt_fields, cleaning_mask

    # ------------------------------------------------------------------------------------------------------------------
    def _run_plot_check(
        self,
        *,
        context: LossComputeContext | None,
        cleaning_mask: Tensor,
        lhs_pred: Tensor,
        rhs_pred: Tensor,
        lhs_gt: Tensor | None,
        psi_fields: Tensor,
        psi_gt_fields: Tensor,
        j_tor_fields: Tensor,
        jtor_gt_fields: Tensor | None,
        zero: Tensor,
    ) -> None:
        """Emit one optional weak-form diagnostic plot using the shared plot probability convention."""

        if self.plot_check_type is None:
            return

        field_index = select_diagnostic_plot_slice(
            context=context,
            probability=self.plot_check_probability,
            n_fields=psi_fields.shape[0],
            diagnostic_name="grad_shafranov_weak_form",
        )
        if field_index is None:
            return

        with torch.no_grad():
            rhs_ref = torch.where(cleaning_mask, mu0 * jtor_gt_fields, zero) if jtor_gt_fields is not None else None
            make_gs_plots(
                plot_data={
                    "grid_data": {
                        "R_for_x_data": self.R_matrix,
                        "Z_for_y_data": self.Z_matrix,
                        "z_lims": {
                            "gs_sides": [-2, 2],
                            "psi": [-0.2, 0.2],
                            "j_tor": [-0.2 * 1e6, 1e6],
                        },
                    },
                    "title": "Weak-form Grad-Shafranov: LHS vs RHS and signals",
                    "subplot_titles": [
                        "LHS = W psi (pred.)",
                        "LHS = W psi (real)",
                        "RHS = mu0 j_tor (derived)" if self._derives_j_tor else "RHS = mu0 j_tor (pred.)",
                        "RHS = mu0 j_tor (real)",
                        "Pred. psi",
                        "Real psi",
                        "Pred. j_tor",
                        "Real j_tor",
                    ],
                    "gs_data": {
                        "lhs_pred_data": lhs_pred[field_index].detach(),
                        "lhs_ref_data": lhs_gt[field_index].detach() if lhs_gt is not None else None,
                        "rhs_pred_data": rhs_pred[field_index].detach(),
                        "rhs_ref_data": rhs_ref[field_index].detach() if rhs_ref is not None else None,
                    },
                    "signal_data": {
                        "psi_pred_data": psi_fields[field_index].detach(),
                        "psi_ref_data": psi_gt_fields[field_index].detach(),
                        "j_tor_pred_data": j_tor_fields[field_index].detach(),
                        "j_tor_ref_data": jtor_gt_fields[field_index].detach() if jtor_gt_fields is not None else None,
                        "j_tor_case": "derived" if self._derives_j_tor else "predicted",
                    },
                },
                save_plots=(self.plot_check_type == "save_plots"),
                save_path=(
                    training_plot_path(context, slice_index=field_index)
                    if self.plot_check_type == "save_plots"
                    else None
                ),
            )

    # ------------------------------------------------------------------------------------------------------------------
    def compute(
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: None | Mapping[Hashable, Tensor] = None,
        pred_dist: Mapping[Hashable, Mapping[str, Tensor]] | None = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """Compute the masked weak-form Grad-Shafranov residual in physical native units."""

        # ..............................................................................................................
        # 0 - Preliminary checks and runtime grid tensors
        # ..............................................................................................................
        if not preds:
            raise RuntimeError("WeakFormGradShafranovLoss received empty predictions from the model.")
        if y_native is None:
            raise RuntimeError("WeakFormGradShafranovLoss requires native targets.")
        if self.R_matrix is None:
            raise RuntimeError("Weak-form Grad-Shafranov R matrix was not loaded.")

        ref = next(iter(preds.values()))
        logs: dict[Hashable, float] = {}
        # Keep R in float32 under AMP; weak-form stiffness uses reciprocal R weights.
        r_matrix = runtime_tensor(self.R_matrix, ref=ref, dtype=torch.float32)  # already (n_r, n_z)

        # ..............................................................................................................
        # 1 - Decode prediction fields and prepare finite native targets / physical-domain mask
        # ..............................................................................................................
        psi_bhwt, psi_fields, j_tor_fields = self._decode_and_destandardize_predictions(preds)
        n_fields = psi_fields.shape[0]
        psi_gt_fields, jtor_gt_fields, cleaning_mask = self._prepare_targets_and_cleaning_mask(
            y_native, n_fields=n_fields
        )

        # ..............................................................................................................
        # 2 - Build weak-form residual sides and optional ground-truth anchor
        # ..............................................................................................................
        zero = torch.zeros((), device=cleaning_mask.device, dtype=psi_fields.dtype)
        lhs_pred = torch.where(cleaning_mask, self._discrete_stiffness(psi_fields, r_matrix), zero)
        need_gt_reference = (self.gs_weights["lhs_gt"] > 0.0) or (self.gs_weights["rhs_gt"] > 0.0)

        lhs_gt = None
        if need_gt_reference or self._derives_j_tor or self.plot_check_type is not None:
            lhs_gt = torch.where(cleaning_mask, self._discrete_stiffness(psi_gt_fields, r_matrix), zero)

        if self._derives_j_tor:
            j_tor_fields = self._j_tor_from_psi_via_weak_operator(psi_gt_fields, r_matrix)
        rhs_pred = torch.where(cleaning_mask, mu0 * j_tor_fields, zero)

        # ..............................................................................................................
        # 3 - Optional diagnostics, then per-field eligibility and loss reductions
        # ..............................................................................................................
        self._run_plot_check(
            context=context,
            cleaning_mask=cleaning_mask,
            lhs_pred=lhs_pred,
            rhs_pred=rhs_pred,
            lhs_gt=lhs_gt,
            psi_fields=psi_fields,
            psi_gt_fields=psi_gt_fields,
            j_tor_fields=j_tor_fields,
            jtor_gt_fields=jtor_gt_fields,
            zero=zero,
        )

        no_gt_residual = lhs_pred - rhs_pred
        no_gt_per_field = self._field_norm(no_gt_residual, mask=cleaning_mask)
        valid = torch.isfinite(no_gt_per_field)
        batch_size, _, _, n_times = psi_bhwt.shape
        valid = valid & output_masks_to_field_mask(
            output_mask,
            self.required_output_keys,
            batch_size=batch_size,
            n_times=n_times,
            ref=ref,
        )

        lhs_gt_per_field = torch.zeros_like(no_gt_per_field)
        rhs_gt_per_field = torch.zeros_like(no_gt_per_field)
        gt_valid = torch.ones_like(valid)
        if need_gt_reference:
            gt_valid = cleaning_mask.flatten(start_dim=1).any(dim=1)
            lhs_gt_per_field = self._field_norm(lhs_pred - lhs_gt, mask=cleaning_mask)
            rhs_gt_per_field = self._field_norm(rhs_pred - lhs_gt, mask=cleaning_mask)

        if not bool(valid.any()):
            return ref.sum() * 0.0, logs

        # ..............................................................................................................
        # 4 - Aggregate enabled weak-form terms and report diagnostics
        # ..............................................................................................................
        loss = ref.sum() * 0.0
        if self.gs_weights["no_gt"] > 0.0:
            no_gt_loss = no_gt_per_field[valid].mean()
            loss += self.gs_weights["no_gt"] * no_gt_loss
            logs["weak_gs_no_gt"] = float(no_gt_loss.detach().cpu())
        if need_gt_reference:
            anchored_valid = valid & gt_valid.to(valid.device)
            if bool(anchored_valid.any()) and self.gs_weights["lhs_gt"] > 0.0:
                lhs_gt_loss = lhs_gt_per_field[anchored_valid].mean()
                loss += self.gs_weights["lhs_gt"] * lhs_gt_loss
                logs["weak_gs_lhs_gt"] = float(lhs_gt_loss.detach().cpu())
            if bool(anchored_valid.any()) and self.gs_weights["rhs_gt"] > 0.0:
                rhs_gt_loss = rhs_gt_per_field[anchored_valid].mean()
                loss += self.gs_weights["rhs_gt"] * rhs_gt_loss
                logs["weak_gs_rhs_gt"] = float(rhs_gt_loss.detach().cpu())
        logs["weak_gs_residual"] = float(loss.detach().cpu())
        return loss, logs


# ======================================================================================================================
if __name__ == "__main__":
    print("WeakFormGradShafranovLoss (residual, bounded).")
