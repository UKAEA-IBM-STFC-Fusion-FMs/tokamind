"""
Weak-form Grad-Shafranov loss in native (destandardized) space.

WeakFormGradShafranovLoss computes physics-informed loss by enforcing the weak-form Grad-Shafranov equation: the
discrete stiffness operator applied to predicted psi must match the RHS computed from toroidal current (j_tor). Loss is
computed in native (destandardized) signal space, with masking of missing (non-finite) ground-truth cells for sparse
measurements.

REMARK: This loss evaluates the discrete weak residual rather than solving the weak-form finite-element system.

Continuous weak formulation
---------------------------
Find ψ such that

    ∫Ω (1/R) ∇ψ · ∇v dΩ - μ₀ ∫Ω j_tor v dΩ = 0

for all admissible test functions v. This follows from the strong form `-Δ*ψ = μ₀ R j_tor` after dividing by R,
multiplying by v, and integrating by parts.

The boundary term from the integration by parts is not assembled. In the discrete operator, this amounts to a zero-flux
(natural) condition at the grid border: border cells only get fluxes through their interior edges, so their values do
not approximate the operator. The residual is evaluated only over the masked plasma region, away from the border, so
no boundary condition is imposed on the plasma itself.

Discrete weak residual
----------------------
The bilinear form is discretized as an edge-based stiffness operator `W`, applied matrix-free (no matrix is assembled):
each R- or Z-direction edge carries the flux `w · Δψ` between its two cells, with weights `w = 1 / (R_edge · h²)`,
where `R_edge` is the mean of R across the edge and `h` the grid spacing (`dr` or `dz`); see `lhs_calculation`. Per
unit cell area,

    W ψ ≈ -∇·((1/R) ∇ψ) = -Δ*ψ / R.

The source term uses a lumped (diagonal) mass approximation, so per unit cell area it reduces to `μ₀ j_tor`, yielding
the discrete residual

    r = W ψ − μ₀ j_tor.

This is the strong-form residual `-Δ*ψ − μ₀ R j_tor` divided by R, so, unlike the strong form, no explicit R factor
multiplies j_tor.

Key features
------------
- **Decoding & destandardization**: Predictions arrive in embedding space and are decoded to native space via
  per-signal TorchDecoder instances, then destandardized to physical units before GS calculations.
- **Sparse masking**: Ground-truth targets are made finite before the GS operator is applied, and their non-finite
  cells are excluded from the cleaning mask, so only valid measurements contribute. A field is skipped if it fails
  `output_mask`, has a non-finite predicted-RHS j_tor, or has no cell inside the cleaning mask. Non-finite values in
  predictions are not masked.
- **Flexible RHS sources** (`rhs_input.origin`, shared with the strong form):
  - `predicted_j_tor` (general case): both ψ and j_tor are model outputs. All three residual terms are available.
  - `derived_j_tor` (reduced case): ψ is the only required model output. The current is derived from ground-truth
    ψ using `rhs_input.calculation_method`:
    - `GS_operator`: through the same weak operator, `j_tor = max(0, M ⊙ W ψ_true / μ₀)`, with `M` the cleaning
      mask. Inside the mask (and where it is non-negative), the RHS is then ≈ `W ψ_true`, i.e., the anchor. So the
      `no_gt` residual `W ψ_pred - RHS` ≈ the `lhs_gt` residual `W ψ_pred - W ψ_true` (the two terms duplicate each
      other), and the `rhs_gt` residual is degenerate (≈ 0). The recommended configuration is
      {no_gt: 0.0, lhs_gt: 1.0, rhs_gt: 0.0}, giving the single-term loss: the `loss_metric` reduction of
      `W ψ_pred - W ψ_true`.
    - `parametric_approx`: through a smooth power-law current profile (see `j_tor_from_psi_via_parametric_approx`),
      with min-max normalization of the GS sides.

    With either method, the derived RHS is built from ground truth only, so the `rhs_gt` residual carries no gradient
    to the model and `rhs_gt` must be 0.
  - `predicted_profiles` (pprime, ffprime): accepted by validation but not implemented yet (raises
    `NotImplementedError` in `compute`).
- **Three loss terms**: Weighted combination of (1) LHS-RHS residual (no ground truth), (2) LHS vs ground-truth anchor,
  (3) RHS vs ground-truth anchor, allowing flexible supervision strategies. The anchor is the LHS of ground-truth psi
  (`W ψ_true`), which approximates the ground-truth RHS up to discretization error. Each term is reduced per field with
  `loss_metric` over the cleaning mask, the terms are combined with `grad_shafranov_weights`, and the loss is the mean
  over valid fields.
- **Gradient flow**: Gradients flow through decoders into model predictions; decoder parameters (frozen VAE weights,
  fixed indices) remain fixed.

Configuration
-------------
Expects one item of `train.loss.terms` shaped as follows (all keys but `type`, `weight`, and
`grad_shafranov_params_file` are optional; defaults are shown):

.. code-block:: yaml

    train:
      loss:
        terms:
          - type: weak_grad_shafranov
            weight: <float>
            grad_shafranov_params_file: <str or path>  # .npz asset with R/Z grids and limiter masks
            grad_shafranov_weights: {no_gt: 0.34, lhs_gt: 0.33, rhs_gt: 0.33}  # omitted keys count as 0
            rhs_input:
              origin: predicted_j_tor  # predicted_j_tor | derived_j_tor | predicted_profiles
              calculation_method: GS_operator  # GS_operator | parametric_approx (only used for derived_j_tor)
            loss_metric: mse  # mse | l2
            outputs:  # generic loss-term key; must keep every output required by `rhs_input.origin`
              include: [<output names>]  # or `exclude: [<output names>]`
            plot_check:  # visualize GS sides during training (disabled by default)
              type: save_png  # show_plots | save_png | save_pdf | save_svg | null
              probability: <float>  # fraction of training steps to plot, in [0, 1]

With `rhs_input.origin: derived_j_tor`, `rhs_gt` must be 0 (see `WeakFormGradShafranovLoss`).

Data requirements
-----------------
- `data.keep_output_native=True` in dataset config to provide ground-truth native targets.
- Output signals required by `rhs_input.origin`: "equilibrium-psi" always, plus "equilibrium-j_tor" for
  `predicted_j_tor`, or "equilibrium-dpressure_dpsi" and "equilibrium-f_df_dpsi" for `predicted_profiles`.
- Per-signal standardization statistics (`signal_stats`), passed to the constructor by the loss aggregator.
- LCFS data (`lcfs_r`, `lcfs_z`) is not provided yet (see TODOs), so the LCFS-based plasma mask is never used.

Implementation notes
--------------------
- All GS calculations operate on (F, n_r, n_z) fields where F = B*T (batch × forecast times). The stiffness operator
  is applied to all F fields of a batch at once, through vectorized edge-flux differences.
- Predicted/ground-truth psi and j_tor are transposed from model native shape (B, H, W, T) to
  match GS operator orientation.
- Both GS sides are computed in float32, so the residuals are formed in a single dtype.
- The plasma region (j_tor limiter mask) is taken from the first available source: time-varying LCFS masks (not
  provided yet), then ground-truth j_tor (cells where j_tor > 1e-6, available when j_tor is a required output), and
  finally the static limiter mask from the asset file.
- TODOs:
  - "predicted-profiles" pipeline is NOT ported to the field contract: it now raises NotImplementedError instead of
    running silently-wrong. It requires differentiable psi_N computation and torch.searchsorted interpolation
    (currently disabled).
  - Implement and check the provision of "lcfs_r" and "lcfs_z".

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from matplotlib.path import Path as MatplotlibPath
from typing import Hashable, Any, Literal, cast
import numpy as np
import scipy.sparse as sp

import torch
from torch import Tensor
import torch.nn.functional as torch_functional

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
    validate_params_file_cfg,
    validate_rhs_input_cfg,
    validate_loss_metric_cfg,
    validate_grad_shafranov_weights,
    validate_rhs_input_origin,
    validate_j_tor_calculation_method,
    validate_required_keys,
    validate_base_j_tor_limiter_mask,
)
from .plots import make_gs_plots
from mmt.data.embeddings.torch_decoder import TorchDecoder
from mmt.data.standardization import destandardize_torch
from mmt.train.losses.base import BaseLoss, LossComputeContext
from mmt.train.losses.constants import (
    GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR,
    GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR,
    GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES,
    LOSS_ACRONYM_MAP,
    GRAD_SHAFRANOV_WEIGHT_MAP,
)
from mmt.utils.paths import REPO_ROOT


# ----------------------------------------------------------------------------------------------------------------------
# Preliminaries

# Opt out of invariant checks
torch.sparse.check_sparse_tensor_invariants.disable()


# ======================================================================================================================
class WeakFormGradShafranovLoss(BaseLoss):
    """
    Physics-informed loss enforcing the weak-form Grad-Shafranov equilibrium equation in native, destandardized space.

    Computes residuals of the form: ||W ψ - μ₀·j_tor|| over tokamak equilibrium grids, where ψ (psi) and j_tor are
    decoded from model predictions, and `W` is the discrete stiffness operator of the weak form, applied matrix-free
    through 1/R-weighted edge fluxes (see `lhs_calculation`). `W ψ` approximates `-Δ*ψ / R`, so, unlike the strong
    form, no explicit R factor multiplies j_tor. Loss combines three weighted terms:
    (1) LHS–RHS residual (no ground truth),
    (2) LHS vs ground-truth anchor,
    (3) RHS vs ground-truth anchor,
    allowing flexible supervision strategies. The anchor is the LHS of ground-truth psi (`W ψ_true`), which
    approximates the ground-truth RHS up to discretization error.

    Predictions arrive in embedding space and are decoded to native space via per-signal `TorchDecoder` instances, then
    destandardized to physical units before GS operator application. Ground-truth native targets come from
    `batch['output_native']`, which requires `data.keep_output_native=True` in config.

    Sparse measurements: non-finite values in ground truth represent missing data (e.g., unmeasured plasma regions).
    Targets are made finite before the GS operator is applied, and their non-finite cells are excluded from the
    cleaning mask, so the loss is computed only over valid target cells. A field contributes only if it passes
    `output_mask`, has a finite predicted-RHS j_tor, and has at least one cell inside the cleaning mask. Non-finite
    values in predictions are neither replaced nor masked, so they propagate to the loss.

    Gradient flow: Gradients flow through decoders into model predictions; decoder parameters (frozen VAE weights,
    fixed indices) remain static.

    Parameters
    ----------
    decoders : dict[Hashable, TorchDecoder]
        Per-signal differentiable decoders, keyed by signal_id. Must not be empty and must include a decoder for every
        output required by `rhs_input_origin` (see `required_output_keys`): psi always, plus j_tor for
        `predicted_j_tor`, or p' and FF' for `predicted_profiles`.
    signal_stats : Mapping[str, Mapping[str, Any]]
        Per-signal statistics (mean, std) for destandardization. Keyed by signal name (str).
    output_name_to_id : Mapping[str, Hashable]
        Mapping from output signal names (e.g., "equilibrium-psi") to numeric signal IDs.
    grad_shafranov_params_file : str | Path
        Path to `.npz` asset file containing the R/Z grids and limiter masks (plus the sparse strong-form operator,
        which this loss does not use).
        Required. Absolute or repo-relative path.
    grad_shafranov_weights : dict[Hashable, float] | None
        Weights for the three loss terms: {"no_gt": w1, "lhs_gt": w2, "rhs_gt": w3}. `no_gt` scores `W ψ_pred`
        against `μ₀ · j_tor_pred`, `lhs_gt` scores `W ψ_pred` against the anchor `W ψ_true`, and `rhs_gt` scores
        `μ₀ · j_tor_pred` against the anchor. Each must lie in [0, 1] and they must sum to 1; any key left out counts
        as 0.
        Optional. Default: None (or an empty mapping), which results in {"no_gt": 0.34, "lhs_gt": 0.33, "rhs_gt": 0.33}.
    rhs_input_origin : str | None
        RHS computation mode, shared with the strong form: "predicted_j_tor", "derived_j_tor", or "predicted_profiles".
        `derived_j_tor` derives the current from ground-truth psi (see `j_tor_calculation_method`), so the `rhs_gt`
        residual carries no gradient to the model and `rhs_gt` must be 0, whatever the calculation method.
        `predicted_profiles` is accepted by validation but not implemented yet (raises `NotImplementedError` in
        `compute`).
        Optional. Default: "predicted_j_tor".
    j_tor_calculation_method : str | None
        Method used to derive j_tor from psi in `derived_j_tor` mode, shared with the strong form: "GS_operator"
        (through the weak operator W) or "parametric_approx". Ignored for other origins.
        Optional. Default: "GS_operator".
    loss_metric : Literal["l2", "mse"]
        Per-field reduction applied to each GS residual (`no_gt`, `lhs_gt`, `rhs_gt`) over the active cells of the
        cleaning mask, before the terms are combined with `grad_shafranov_weights`:
          - "mse": mean of squared residuals over active cells (i.e., sum(r**2) / n_active).
          - "l2": Euclidean norm of the residual over active cells (i.e., sqrt(sum(r**2))). Not normalized by the
            number of cells, so its magnitude grows with grid size and mask coverage.
        Fields with no active cells yield NaN under either metric.
        Optional. Default: "mse".
    output_weights : dict[Hashable, float] | None
        Deprecated. Per-output scalar weights (not currently used).
        Optional. Default: None.
    output_filter : set[Hashable] | None
        Output signal IDs selected for this loss term (from the term's generic `outputs` config). Only checked to
        contain every key in `required_output_keys`; it does not otherwise restrict which outputs `compute` uses.
        Optional. Default: None (no check).
    plot_check_cfg : Mapping[str, Any] | None
        Plotting configuration: {"type": <type>, "probability": float, "all_losses_weights": {...}}, with `type` in
        `ALLOWED_PLOT_CHECK_TYPES` ("show_plots", "save_png", "save_pdf", "save_svg", or None to disable), and
        `all_losses_weights` (added by the loss aggregator) mapping each loss-term type to its weight. When enabled,
        plots the GS sides during training on deterministically selected steps.
        Optional. Default: None (plotting disabled).

    Raises
    ------
    KeyError, TypeError, ValueError
        If the configuration, the required outputs, or the asset file is invalid; see `__init__` for the full list.

    Attributes
    ----------
    requires_native_target : bool
        Class attribute. Whether targets must be provided in native (decoded) space. Always True.
    requires_decode : bool
        Class attribute. Whether predictions must be decoded before computing the loss. Always True.
    requires_destandardize : bool
        Class attribute. Whether the caller must destandardize fields beforehand. Always False, since this loss
        destandardizes internally using `signal_stats`.
    rhs_input_origin : str
        Resolved RHS computation mode (one of `GRAD_SHAFRANOV_RHS_INPUT_ORIGINS_FULL`, shared with the strong form).
    j_tor_calculation_method : str
        Resolved toroidal-current calculation method (one of `GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS`), used in
        `derived_j_tor` mode.
    gs_weights : dict[str, float]
        Weights for the `no_gt`, `lhs_gt`, and `rhs_gt` residual terms: `grad_shafranov_weights` with missing keys
        set to 0, or `DEFAULT_GRAD_SHAFRANOV_WEIGHTS` if none (or an empty mapping) were given. Each weight lies in
        [0, 1] and they sum to 1; terms with zero weight are skipped.
    loss_metric : Literal["l2", "mse"]
        Per-field reduction form applied to each residual term (see `loss_metric` parameter).
    signal_stats : dict[str, dict[str, Any]]
        Per-signal standardization statistics (`mean`, `std`), keyed by signal name.
    required_output_names : list[str]
        Output signal names required by `rhs_input_origin` (e.g., psi and j_tor for `predicted_j_tor`).
    required_output_keys : set[Hashable]
        Signal IDs resolved from `required_output_names`. Each must have a decoder and, if set, be in
        `output_filter`.
    gs_op_coo : torch.Tensor
        Sparse COO strong-form Grad-Shafranov (Δ*) operator of shape `(n_r * n_z, n_r * n_z)`, loaded from the asset
        file. Loaded and moved to the runtime device, but not used by the weak-form loss, which applies `W` through
        `wR_unsqueezed` and `wZ_unsqueezed` instead.
    n_r, n_z : int
        Number of grid points along R and Z, loaded from the asset file in `__init__` (at least 2 each).
    dr, dz : float
        Uniform grid spacings along R and Z, derived from the R/Z grids in `__init__` (always positive).
    R_matrix, Z_matrix : torch.Tensor
        Major-radius and vertical-elevation grid coordinates shaped `(n_r, n_z)`. R varies along axis 0 and Z along
        axis 1.
    wR_unsqueezed, wZ_unsqueezed : torch.Tensor
        Edge weights `1 / (R_edge * dr**2)` and `1 / (R_edge * dz**2)` at the R- and Z-direction cell edges, shaped
        `(1, n_r - 1, n_z)` and `(1, n_r, n_z - 1)`, where `R_edge` is the mean of R across the edge. They define the
        discrete stiffness operator `W` applied in `lhs_calculation`. Kept in float32 on the runtime device.
    rz_points_for_lcfs_calculation : torch.Tensor
        Flattened grid points shaped `(n_r * n_z, 2)`, holding (R, Z) pairs, used to test which cells lie inside the
        last closed flux surface (LCFS).
    zero_lcfs_rz_mask : torch.Tensor
        All-False boolean mask shaped `(n_r, n_z)`, intended as the fallback when no valid LCFS can be determined.
        Built in `_load_gs_params`, but not currently used.
    base_j_tor_limiter_mask_rz : torch.Tensor
        Static boolean limiter mask shaped `(1, n_r, n_z)` from the asset file (scalar placeholder if absent). Used as
        the last-resort cleaning mask for j_tor.
    mast_limiter_mask_rz : torch.Tensor
        MAST limiter mask shaped `(1, n_r, n_z)` loaded from the asset file (float). Not currently used.

    Private attributes (internal use): `_decoders`, `_output_weights`, `_output_filter`, `_output_name_to_id`,
    `_name_by_key` (inverse of `_output_name_to_id`), `_psi_key`, `_j_tor_key`, `_pprime_key`, `_ffprime_key`
    (signal IDs of psi, j_tor, p', and FF'; only psi is mandatory), `_runtime_device` (device the cached tensors were
    last moved to), and plotting state derived from `plot_check_cfg`: `_plot_check_cfg`, `_all_losses_weights`,
    `_plot_check_type`, `_run_plotting`, `_plot_check_probability`, `_save_plots`, `_save_plot_format`,
    `_plain_loss_name`, `_fig_title`, `_fig_subtitle`.

    Methods
    -------
    Configuration validation (class methods):

    validate_common_terms(term_def, path)
        Validate config fields shared by all Grad-Shafranov losses (known keys, `grad_shafranov_params_file`,
        `rhs_input`, `loss_metric`, `plot_check`, `grad_shafranov_weights`). Raises on the first invalid field.
    validate_specific_terms(term_def, path)
        Validate config fields specific to the weak form. Currently a no-op placeholder.
    validate_term_cfg(term_def, path)
        Validate one `weak_grad_shafranov` config term: the shared checks, then the weak-form-specific ones.

    Main entry points:

    compute(preds, y_emb, y_native, output_mask, context)
        Compute the masked weak-form GS loss in native, destandardized space: decode and destandardize the
        predictions, build the LHS/RHS and ground-truth anchor, compute the weighted residual terms, and optionally
        emit diagnostic plots. NaN target samples are excluded. Returns the loss tensor and a dict of logged values.
    grad_shafranov_loss(gs_lhs_pred, gs_rhs_pred, gs_anchor, cleaning_mask=None, min_max_normalize_sides=False)
        Build the `no_gt`, `lhs_gt`, and `rhs_gt` residuals, reduce each one per field with `loss_metric` over the
        cleaning mask, and return their `gs_weights`-weighted sum, shaped `(F,)`.

    Grad-Shafranov equation sides:

    lhs_calculation(psi_fields_, cleaning_mask_=None)
        Apply the stiffness operator to psi fields through edge fluxes, returning `W ψ` shaped `(F, n_r, n_z)`,
        masked if required.
    rhs_calculation(j_tor_fields_, cleaning_mask_=None)
        Static. Compute the weak-form RHS `μ₀ · j_tor` shaped `(F, n_r, n_z)`, masked if required.
    j_tor_for_pred_rhs(j_tor_destdized_pred, psi_destdized_true, j_tor_limiter_mask)
        Select the toroidal current used for the predicted RHS according to `rhs_input_origin`: either the predicted
        j_tor or one derived from ground-truth psi.

    Toroidal-current derivation:

    calculate_j_tor_from_psi(psi_fields, j_tor_limiter_mask)
        Derive j_tor from psi, dispatching on `j_tor_calculation_method`. Returns `(j_tor_fields,
        perform_min_max_normalization)`.
    j_tor_from_psi_via_operator(psi_fields, j_tor_limiter_mask)
        Derive j_tor as `μ₀⁻¹ · W ψ` within the valid region, clamped to [0, ∞).
    j_tor_from_psi_via_parametric_approx(psi_fields, j_tor_limiter_mask, alpha_m, alpha_n, beta_m, beta_n, lamda, beta)
        Derive j_tor from a smooth power-law profile approximation over the plasma region inferred from psi.
    j_tor_from_pprime_ffprime_psi(pprime_native, ffprime_native, psi_native, psi_norm_native, mag_r_axis, mag_z_axis,
                                  xpt_r, xpt_z)
        Estimate j_tor from the p' and FF' profiles and psi, using EFIT magnetic-axis and X-point geometry.
    compute_psi_n_from_efit_geometry(psi_zr, mag_r_axis, mag_z_axis, xpt_r, xpt_z)
        Compute normalized flux `ψ_N` from the EFIT magnetic axis and the first valid X-point.
    nearest_grid_value(field_zr, r0, z0)
        Read a 2D field at the grid point nearest to `(r0, z0)`.

    Normalization and masking utilities (static methods unless noted):

    min_max_normalization(x, nan_mode=False)
        Scale an array or tensor to [0, 1], optionally ignoring NaNs.
    mean_std_normalization(x, mean_scaling=1.0, std_scaling=1.0)
        Normalize a tensor field by its scaled mean and standard deviation.
    compute_plasma_region(psi_n)
        Return a binary plasma-region mask from normalized psi (cells where `ψ_N <= 1`).

    Private helpers (internal use): `_load_gs_params` (load the grid geometry and edge weights from the asset file),
    `_check_device_and_dtype` (move cached tensors to the runtime device), `_required_output_names_for_rhs`
    (static; output names required by an RHS origin), `_gather_required_fields`, `_decode_fields_to_stdized_native`,
    `_get_gs_safe_destdized_fields` (destandardize fields and build the finite-cell mask), `_get_j_tor_limiter_mask`
    (select the plasma-region mask: LCFS, then target j_tor, then the static limiter mask), `_make_lcfs_mask`
    (boolean mask of grid points inside the LCFS polygon), `_min_max_normalize_per_field` (static),
    `_psi_n_from_limiter_mask` (static; normalized flux `ψ_N` over the limiter region, for the parametric approx.),
    and the plotting helpers `_run_plot_check`, `_build_plot_strings`, and `_gs_weights_latex`.

    """

    requires_native_target: bool = True
    requires_decode: bool = True
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod  # TODO: This must be abstract in a future generic parent class.
    def validate_specific_terms(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate the config fields specific to the Grad-Shafranov loss in this module.

        Called by `validate_term_cfg` after `validate_common_terms`, so `term_def` has already passed the checks shared
        by all Grad-Shafranov losses. The Grad-Shafranov loss in this module has no config fields of its own, so this
        method is a no-op placeholder for future specific checks.

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.

        Returns
        -------
        None

        Raises
        ------
        None
            No checks are implemented yet.

        Notes
        -----
        `validate_common_terms` rejects any key outside the generic loss keys and the shared Grad-Shafranov keys, and
        it runs first. A new loss-specific key must therefore also be added to its `allowed_specific_keys`, or
        it will be rejected before this method is reached.

        """

        # Implement validation for specific terms, if needed.

        return

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_common_terms(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate the config fields shared by all Grad-Shafranov losses.

        Checks that `term_def` contains only generic loss keys plus the Grad-Shafranov keys
        (`grad_shafranov_params_file`, `grad_shafranov_weights`, `rhs_input`, `loss_metric`, `plot_check`), then
        validates each one with the shared helpers in `helpers.py`. Omitted (or null) optional keys take their
        defaults: `rhs_input.origin="predicted_j_tor"`, `rhs_input.calculation_method="GS_operator"`,
        `loss_metric="mse"`, plotting disabled, and `DEFAULT_GRAD_SHAFRANOV_WEIGHTS` (also used for an empty
        `grad_shafranov_weights`). If `grad_shafranov_weights` names any weight, the ones it omits count as 0.
        Validation stops at the first invalid field.

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `term_def` contains unknown keys.
            If `grad_shafranov_params_file` is missing.
            If `rhs_input` or `grad_shafranov_weights` contains unknown keys.
        TypeError
            If `grad_shafranov_params_file` is not a str or Path.
            If `rhs_input`, `plot_check`, or `grad_shafranov_weights` is provided but is not a mapping.
            If a weight or `plot_check.probability` is not an int or float (bools and null are rejected).
        ValueError
            If `rhs_input.origin` is not in `GRAD_SHAFRANOV_RHS_INPUT_ORIGINS_FULL`.
            If `rhs_input.origin="derived_j_tor"` and `rhs_input.calculation_method` is not in
            `GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS`.
            If `loss_metric` is not "l2" or "mse".
            If `plot_check.probability` is outside [0, 1] (NaN included), or `plot_check.type` is not in
            `ALLOWED_PLOT_CHECK_TYPES`.
            If any weight is outside [0, 1], or the weights (missing keys counted as 0) do not sum to 1.
            If `rhs_input.origin="derived_j_tor"` and `rhs_gt > 0` (whatever `rhs_input.calculation_method` is).

        """

        cls._validate_known_term_keys(
            term_def=term_def,
            path=path,
            allowed_specific_keys={
                "grad_shafranov_params_file",
                "grad_shafranov_weights",
                "rhs_input",
                "loss_metric",
                "plot_check",
            },
        )

        _ = validate_params_file_cfg(params_file=term_def.get("grad_shafranov_params_file"), path=path)
        rhs_input_origin, j_tor_calculation_method = validate_rhs_input_cfg(
            rhs_input_cfg=term_def.get("rhs_input"),
            path=path,
        )
        _ = validate_loss_metric_cfg(loss_metric=term_def.get("loss_metric"), path=path)
        _ = validate_plot_check_cfg(plot_check=term_def.get("plot_check"), path=path)

        _ = validate_grad_shafranov_weights(
            grad_shafranov_weights=term_def.get("grad_shafranov_weights"),
            rhs_input_origin=rhs_input_origin,
            j_tor_calculation_method=j_tor_calculation_method,
            path=path,
        )

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_term_cfg(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate the config of one weak-form Grad-Shafranov loss term.

        Called by the config validator for each `train.loss.terms` item of type `weak_grad_shafranov`, so invalid
        configs fail before training starts. Runs the checks shared by all Grad-Shafranov losses
        (`validate_common_terms`), then the weak-form-specific checks (`validate_specific_terms`, currently none).

        Parameters
        ----------
        term_def : Mapping[str, Any]
            One configured loss term.
        path : str
            Human-readable config path used in error messages.

        Returns
        -------
        None

        Raises
        ------
        KeyError, TypeError, ValueError
            If a field is invalid; see `validate_common_terms` for the full list.

        """

        # ..............................................................................................................
        # Common validation across GS losses

        cls.validate_common_terms(term_def=term_def, path=path)

        # ..............................................................................................................
        # Specific validation for the GS loss in this module

        cls.validate_specific_terms(term_def=term_def, path=path)

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        decoders: dict[Hashable, TorchDecoder],
        signal_stats: Mapping[str, Mapping[str, Any]],
        output_name_to_id: Mapping[str, Hashable],
        grad_shafranov_params_file: str | Path,
        grad_shafranov_weights: dict[Hashable, float] | None = None,
        rhs_input_origin: str | None = None,
        j_tor_calculation_method: str | None = None,
        loss_metric: Literal["l2", "mse"] = "mse",
        output_weights: dict[Hashable, float] | None = None,
        output_filter: set[Hashable] | None = None,
        plot_check_cfg: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Validate the configuration, resolve the required outputs, and load the Grad-Shafranov grid and edge weights.

        Construction runs in four steps, and stops at the first failure:

          1. Configuration: checks that `decoders` is not empty, then resolves and validates `rhs_input_origin`,
             `j_tor_calculation_method`, `grad_shafranov_weights` (into `gs_weights`), `loss_metric`, and
             `plot_check_cfg` with the same helpers used by `validate_common_terms`. This way, direct instantiation
             is checked as strictly as a config passed through the config validator.
          2. Plotting: builds the figure title and subtitle from the resolved configuration.
          3. Outputs: resolves the psi signal ID (always required) and, if present in `output_name_to_id`, the j_tor,
             p', and FF' IDs. Then it builds `required_output_names` and `required_output_keys` for
             `rhs_input_origin`, and checks that `decoders` (and `output_filter`, when given) contain every required
             key.
          4. Assets: loads the R/Z grids and the limiter masks from `grad_shafranov_params_file`, and derives the
             edge weights `wR_unsqueezed` and `wZ_unsqueezed` that define the stiffness operator `W` (see
             `_load_gs_params`). If `rhs_input_origin` is not `predicted_j_tor`, it also checks that the static j_tor
             limiter mask is present.

        The loaded tensors stay on the CPU until the first call to `compute`, which moves them to the device of the
        predictions (see `_check_device_and_dtype`).

        Parameters
        ----------
        decoders : dict[Hashable, TorchDecoder]
            Per-signal differentiable decoders, keyed by signal ID.
        signal_stats : Mapping[str, Mapping[str, Any]]
            Per-signal standardization statistics (`mean`, `std`), keyed by signal name. Copied into plain dicts.
        output_name_to_id : Mapping[str, Hashable]
            Mapping from output signal names to signal IDs. Must contain "equilibrium-psi", plus any other output
            required by `rhs_input_origin`.
        grad_shafranov_params_file : str | Path
            Absolute or repo-relative path to the `.npz` GS asset file.
            Required. A None value (e.g., from a config that omits the key) raises a KeyError.
        grad_shafranov_weights : dict[Hashable, float] | None
            Weights for the `no_gt`, `lhs_gt`, and `rhs_gt` residuals.
            Optional. Default: None (`DEFAULT_GRAD_SHAFRANOV_WEIGHTS`).
        rhs_input_origin : str | None
            RHS source: "predicted_j_tor", "derived_j_tor", or "predicted_profiles".
            Optional. Default: None ("predicted_j_tor").
        j_tor_calculation_method : str | None
            Method used to derive j_tor in `derived_j_tor` mode: "GS_operator" or "parametric_approx".
            Optional. Default: None ("GS_operator").
        loss_metric : Literal["l2", "mse"]
            Per-field reduction applied to each residual.
            Optional. Default: "mse".
        output_weights : dict[Hashable, float] | None
            Deprecated. Stored but not used.
            Optional. Default: None.
        output_filter : set[Hashable] | None
            Output signal IDs selected for this loss term. Only checked to contain every required output key.
            Optional. Default: None (no check).
        plot_check_cfg : Mapping[str, Any] | None
            Plotting configuration (`type`, `probability`, and `all_losses_weights` from the loss aggregator).
            Optional. Default: None (plotting disabled).

        See the class docstring for full details on each parameter.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If `decoders` is empty.
            If `decoders`, or `output_filter` when given, is missing any key in `required_output_keys`.
            If `rhs_input_origin`, `j_tor_calculation_method`, `grad_shafranov_weights`, `loss_metric`, or
            `plot_check_cfg` is invalid (see `validate_common_terms`; e.g., `rhs_gt > 0` with `derived_j_tor`).
            If the asset grid has fewer than 2 points along R or Z, or a non-positive spacing (see `_load_gs_params`).
        KeyError
            If a required output name (including "equilibrium-psi") is not in `output_name_to_id`.
            If `grad_shafranov_params_file` is None, or a required key is missing from the asset file.
            If `rhs_input_origin` is not `predicted_j_tor` and the asset file has no `base_j_tor_lim_mask_rz`.
        TypeError
            If `grad_shafranov_params_file` is not a str or Path, or a weight or `plot_check_cfg` entry has the
            wrong type (see `validate_common_terms`).

        """

        if not decoders:
            raise ValueError(f"{self.__class__.__name__} requires at least one decoder.")

        self.rhs_input_origin: str = validate_rhs_input_origin(
            rhs_input_origin=rhs_input_origin,
            path=self.__class__.__name__,
        )

        self.j_tor_calculation_method: str = validate_j_tor_calculation_method(
            method=j_tor_calculation_method,
            rhs_input_origin=self.rhs_input_origin,
            path=self.__class__.__name__,
        )

        self._runtime_device: torch.device | None = None
        self._decoders = decoders
        self._output_weights = output_weights or {}
        self.gs_weights = validate_grad_shafranov_weights(
            grad_shafranov_weights=grad_shafranov_weights,
            rhs_input_origin=self.rhs_input_origin,
            j_tor_calculation_method=self.j_tor_calculation_method,
            path=self.__class__.__name__,
        )

        self._output_filter = set(output_filter) if output_filter is not None else None
        self._output_name_to_id = {str(name): sid for name, sid in output_name_to_id.items()}
        self.signal_stats = {str(name): dict(stats) for name, stats in signal_stats.items()}

        self.loss_metric = validate_loss_metric_cfg(loss_metric=loss_metric, path=self.__class__.__name__)

        self._plot_check_cfg = plot_check_cfg or {}
        (
            self._plot_check_type,
            self._plot_check_probability,
            self._run_plotting,
            self._save_plots,
            self._save_plot_format,
            self._all_losses_weights,
        ) = validate_plot_check_cfg(plot_check=self._plot_check_cfg, path=self.__class__.__name__)

        self._plain_loss_name = "Weak-form Grad-Shafranov loss"
        self._fig_title = ""
        self._fig_subtitle = ""
        self._build_plot_strings()

        # ..............................................................................................................
        # Set relevant output keys

        self._name_by_key = {key: name for name, key in self._output_name_to_id.items()}
        self._psi_key = resolve_output_key(  # -> Only required key (and it thus needs resolution), others are optional.
            output_name_to_id=self._output_name_to_id,
            name="equilibrium-psi",
            loss_name=self._plain_loss_name,
        )
        self._j_tor_key = self._output_name_to_id.get("equilibrium-j_tor")
        self._pprime_key = self._output_name_to_id.get("equilibrium-dpressure_dpsi")
        self._ffprime_key = self._output_name_to_id.get("equilibrium-f_df_dpsi")

        self.required_output_names = self._required_output_names_for_rhs(rhs_input_origin=self.rhs_input_origin)
        self.required_output_keys = set(
            resolve_output_key(
                output_name_to_id=self._output_name_to_id,
                name=name,
                loss_name=self._plain_loss_name,
            )
            for name in self.required_output_names
        )

        if self._output_filter is not None:
            validate_required_keys(
                input_name="output_filter",
                input_values=self._output_filter,
                required_keys=self.required_output_keys,
                path=self.__class__.__name__,
            )

        validate_required_keys(
            input_name="decoders",
            input_values=self._decoders,
            required_keys=self.required_output_keys,
            path=self.__class__.__name__,
        )

        # ..............................................................................................................
        # Load Grad-Shafranov parameters

        self.gs_op_coo: Tensor = torch.tensor(0)

        self.n_r = None
        self.dr = None
        self.R_matrix = None

        self.n_z = None
        self.dz = None
        self.Z_matrix = None

        self.wR_unsqueezed: Tensor = torch.tensor(0)
        self.wZ_unsqueezed: Tensor = torch.tensor(0)

        self.rz_points_for_lcfs_calculation: Tensor = torch.tensor(0)
        self.zero_lcfs_rz_mask: Tensor = torch.tensor(0)
        self.base_j_tor_limiter_mask_rz = torch.tensor(0)
        self.mast_limiter_mask_rz: Tensor = torch.tensor(0)

        grad_shafranov_params_file = validate_params_file_cfg(
            params_file=grad_shafranov_params_file,
            path=self.__class__.__name__,
        )
        self._load_gs_params(filename=grad_shafranov_params_file)

        validate_base_j_tor_limiter_mask(
            limiter_mask=self.base_j_tor_limiter_mask_rz,
            rhs_input_origin=self.rhs_input_origin,
            grad_shafranov_params_file=grad_shafranov_params_file,
            path=self.__class__.__name__,
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _gs_weights_latex(self) -> str:

        gs_weights_latex = ""
        for kk, vv in self.gs_weights.items():
            if vv > 0:
                gs_weights_latex += f"{GRAD_SHAFRANOV_WEIGHT_MAP[kk]}: {str(vv)}, "

        gs_weights_latex = gs_weights_latex[:-2]

        return gs_weights_latex

    # ------------------------------------------------------------------------------------------------------------------
    def _build_plot_strings(self) -> None:
        """Build the loss-configuration summary shown in diagnostic plots."""

        self._fig_title = f"{self._plain_loss_name} related plots"

        if self._plot_check_type is not None:
            self._fig_subtitle = r"Output: ($\psi^{pred}$"
            if self.rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
                self._fig_subtitle += r", $J^{pred}_\phi$)"
            else:
                self._fig_subtitle += rf"), $J^{{appr}}_\phi$ via {self.j_tor_calculation_method}"

            losses_weights_latex = ""
            for kk, vv in self._all_losses_weights.items():
                loss_acronym = "".join(i[0].upper() for i in kk.split("_"))
                if loss_acronym in LOSS_ACRONYM_MAP:
                    loss_acronym = LOSS_ACRONYM_MAP[loss_acronym]
                loss_acronym = "{" + loss_acronym + "}"
                losses_weights_latex += rf", $\omega_{loss_acronym}$: {str(vv)}"

                if kk.endswith("_grad_shafranov") and (vv > 0):
                    losses_weights_latex += f" (with {self._gs_weights_latex()})"

            self._fig_subtitle += losses_weights_latex

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _required_output_names_for_rhs(rhs_input_origin: str | None) -> list[str]:
        """
        Return output names required to build the RHS of the specified Grad-Shafranov equation arrangement.

        Parameters
        ----------
        rhs_input_origin : str | None
            Configured input for the RHS of the specified Grad-Shafranov equation arrangement.

        Returns
        -------
        list[str]
            Required output signal names.

        Notes
        -----
        - Signal "equilibrium-psi" must always be part of the GS loss.
        - Other relevant out keys include:
          - "equilibrium-psi"
          - "equilibrium-j_tor"
          - "equilibrium-dpressure_dpsi"
          - "equilibrium-f_df_dpsi"
          - "equilibrium-psi_norm"

        """

        if rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR:
            return ["equilibrium-psi"]
        if rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
            return ["equilibrium-psi", "equilibrium-j_tor"]
        if rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES:
            return ["equilibrium-psi", "equilibrium-dpressure_dpsi", "equilibrium-f_df_dpsi"]

        raise ValueError(f"Unsupported `rhs_input_origin={rhs_input_origin!r}`.")

    # ------------------------------------------------------------------------------------------------------------------
    def _load_gs_params(self, filename: str | Path) -> None:
        """
        Load the Grad-Shafranov asset file and populate the operator and grid-geometry attributes.

        Loads the sparse GS operator (`gs_op_coo`, from the stored COO arrays), the grid sizes `n_r`, `n_z`, the R/Z
        grids `R_matrix`, `Z_matrix`, the MAST limiter mask `mast_limiter_mask_rz` (required) and the j_tor limiter
        mask `base_j_tor_limiter_mask_rz` (optional; scalar placeholder if absent). From the grids it derives the
        R/Z spacings `dr`, `dz` (computed as `(max - min) / (n - 1)`, assuming a uniform grid), the LCFS helpers
        `rz_points_for_lcfs_calculation` and `zero_lcfs_rz_mask`, and the stiffness-form edge weights
        `wR_unsqueezed`, `wZ_unsqueezed`.

        Parameters
        ----------
        filename : str | Path
            Absolute path, or path relative to the repository root, to a `.npz` asset file.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If a required key is missing from the asset file.
        ValueError
            If `n_r` or `n_z` is smaller than 2 (grid spacings undefined), or if the derived spacing `dr` or `dz` is
            not positive (e.g. R/Z axes swapped relative to the expected orientation: R along axis 0, Z along axis 1).

        """

        with np.load(resolve_gs_asset_path(filename=filename, repo_root=REPO_ROOT)) as loaded_data:
            grid_assets = parse_gs_grid_assets(loaded=loaded_data)
            gs_op_coo_scipy = sp.coo_matrix(
                (loaded_data["GS_op_coo_data"], (loaded_data["GS_op_coo_row"], loaded_data["GS_op_coo_col"])),
                shape=loaded_data["GS_op_coo_shape"],
            )
            self.gs_op_coo = torch.sparse_coo_tensor(  # -> This is COO
                indices=torch.LongTensor(np.array(gs_op_coo_scipy.nonzero())),
                values=torch.as_tensor(gs_op_coo_scipy.data, dtype=torch.float32),
                size=torch.Size(gs_op_coo_scipy.shape),
            )
            self.mast_limiter_mask_rz = torch.tensor(loaded_data["MAST_lim_mask_rz"]).unsqueeze(0).float()

        self.n_r = grid_assets.n_r
        self.n_z = grid_assets.n_z
        self.R_matrix = grid_assets.r_matrix
        self.Z_matrix = grid_assets.z_matrix

        if self.n_r < 2 or self.n_z < 2:
            raise ValueError(
                f"GS grid must have at least 2 points along R and Z to define spacings; got n_r={self.n_r}, "
                f"n_z={self.n_z}."
            )

        # R varies down rows (axis 0), Z across cols (axis 1) — confirmed from the asset.
        r_axis = self.R_matrix[:, 0]  # column 0, R changes down rows
        z_axis = self.Z_matrix[0, :]  # row 0, Z changes across cols
        self.dr = float((r_axis.max() - r_axis.min()) / (self.n_r - 1))
        self.dz = float((z_axis.max() - z_axis.min()) / (self.n_z - 1))
        if self.dr <= 0 or self.dz <= 0:
            raise ValueError(f"Bad GS grid spacing dr={self.dr}, dz={self.dz}; check R/Z axis orientation.")

        self.rz_points_for_lcfs_calculation = torch.column_stack([self.R_matrix.ravel(), self.Z_matrix.ravel()])
        self.zero_lcfs_rz_mask = torch.zeros_like(input=self.R_matrix, dtype=torch.bool)
        self.base_j_tor_limiter_mask_rz = grid_assets.base_j_tor_limiter_mask_rz

        # R-direction edges
        R_edge_R = 0.5 * (self.R_matrix[1:, :] + self.R_matrix[:-1, :])
        wR = (1.0 / R_edge_R) / (self.dr * self.dr)
        self.wR_unsqueezed = wR.unsqueeze(0)

        # Z-direction edges
        R_edge_Z = 0.5 * (self.R_matrix[:, 1:] + self.R_matrix[:, :-1])
        wZ = (1.0 / R_edge_Z) / (self.dz * self.dz)
        self.wZ_unsqueezed = wZ.unsqueeze(0)

    # ------------------------------------------------------------------------------------------------------------------
    def _check_device_and_dtype(
        self,
        ref: Tensor,
    ) -> None:
        """
        Move the cached GS tensors used in the loss computation to the device of `ref`, casting them to float32.

        Runs only when `ref.device` differs from `self._runtime_device` (always on the first call), then updates
        `self._runtime_device`. Moved: `gs_op_coo` (float32, also coalesced), `R_matrix`, `wR_unsqueezed`,
        `wZ_unsqueezed` (float32), and `base_j_tor_limiter_mask_rz` (device only, dtype kept). Float32 is forced
        regardless of `ref.dtype` because CUDA sparse matrix multiplication does not support bfloat16.
        `Z_matrix`, `rz_points_for_lcfs_calculation`, `zero_lcfs_rz_mask` and `mast_limiter_mask_rz` are not moved.

        Parameters
        ----------
        ref : Tensor
            Runtime tensor whose device is matched. Its dtype is ignored.

        Returns
        -------
        None

        """

        _device = ref.device

        if self._runtime_device != _device:
            self.gs_op_coo = runtime_tensor(tensor=self.gs_op_coo, ref=ref, dtype=torch.float32).coalesce()
            self.R_matrix = runtime_tensor(tensor=self.R_matrix, ref=ref, dtype=torch.float32)
            self.wR_unsqueezed = runtime_tensor(tensor=self.wR_unsqueezed, ref=ref, dtype=torch.float32)
            self.wZ_unsqueezed = runtime_tensor(tensor=self.wZ_unsqueezed, ref=ref, dtype=torch.float32)
            self.base_j_tor_limiter_mask_rz = runtime_tensor(tensor=self.base_j_tor_limiter_mask_rz, ref=ref)
            self._runtime_device = _device

    # ------------------------------------------------------------------------------------------------------------------
    def lhs_calculation(  # TODO: This must be abstract in a future generic parent class.
        self,
        psi_fields_: Tensor | None,
        cleaning_mask_: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the weak-form Grad-Shafranov left-hand side `W ψ` for a stack of psi fields.

        `W` is the discrete stiffness operator associated with the weak-form bilinear form

           a(ψ, v) = ∫Ω (1/R) ∇ψ · ∇v dΩ.

        It is applied matrix-free, through edge-based flux differences: no operator matrix is assembled and no linear
        system is solved. For each R-direction edge between cells `i` and `i+1`, the flux is
        `wR · (ψ[i+1] - ψ[i])`, with `wR = 1 / (R_edge · dr²)` and `R_edge = (R[i] + R[i+1]) / 2` (the arithmetic
        mean of R across the edge, then inverted); Z-direction edges use `wZ = 1 / (R_edge · dz²)` likewise. Each
        flux is subtracted from the cell on its lower side and added to the cell on its upper side, so every cell gets

           (W ψ)[i, j] = -[wR₊ (ψ[i+1, j] - ψ[i, j]) - wR₋ (ψ[i, j] - ψ[i-1, j])]
                         -[wZ₊ (ψ[i, j+1] - ψ[i, j]) - wZ₋ (ψ[i, j] - ψ[i, j-1])],

        which approximates `-∇·((1/R) ∇ψ) = -Δ*ψ / R` per unit cell area. This is the weak-form counterpart of the
        strong-form `-Δ*ψ`, divided by R, so it is matched against the RHS `μ₀ · j_tor` (see `rhs_calculation`) with
        no explicit R factor.

        Cells on the grid border only get the fluxes through their interior edges (zero flux through the grid border),
        so their values do not approximate `-Δ*ψ / R`; in `compute`, the cleaning mask restricts the residual to the
        plasma region, away from the border.

        Computation is done in float32, matching the strong-form `lhs_calculation` and the weak-form `rhs_calculation`,
        so the residual `LHS - RHS` is formed in a single dtype. `wR_unsqueezed` and `wZ_unsqueezed` must already be
        float32 and on the same device as `psi_fields_` (see `_check_device_and_dtype`).

        Parameters
        ----------
        psi_fields_ : Tensor
            Poloidal-flux fields in physical units, shaped `(F, n_r, n_z)` (R along axis 1, Z along axis 2). Inputs
            whose last two dimensions do not match `(n_r, n_z)` fail at broadcasting (including a transposed
            `(F, n_z, n_r)` input, unless `n_r == n_z`).
            Required. Although it is typed as optional, None raises a ValueError.
        cleaning_mask_ : Tensor | None
            Boolean or numeric physical-domain mask broadcastable to `(F, n_r, n_z)`, multiplied into the result.
            Optional. Default: `None` (no masking).

        Returns
        -------
        Tensor
            Float32 `W ψ` fields shaped `(F, n_r, n_z)`, masked if `cleaning_mask_` is given. Differentiable with
            respect to `psi_fields_` (gradients are returned in its original dtype).

        Raises
        ------
        ValueError
            If `psi_fields_` is None.

        """

        if psi_fields_ is None:
            raise ValueError("Argument `psi_fields_` cannot be None")

        psi_fields_ = psi_fields_.to(torch.float32)
        gs_lhs = torch.zeros_like(input=psi_fields_)

        flux_R = self.wR_unsqueezed * (psi_fields_[:, 1:, :] - psi_fields_[:, :-1, :])
        gs_lhs[:, :-1, :] = gs_lhs[:, :-1, :] - flux_R
        gs_lhs[:, 1:, :] = gs_lhs[:, 1:, :] + flux_R

        flux_Z = self.wZ_unsqueezed * (psi_fields_[:, :, 1:] - psi_fields_[:, :, :-1])
        gs_lhs[:, :, :-1] = gs_lhs[:, :, :-1] - flux_Z
        gs_lhs[:, :, 1:] = gs_lhs[:, :, 1:] + flux_Z

        if cleaning_mask_ is None:
            return gs_lhs  # (F, n_r, n_z)
        else:
            return cleaning_mask_.to(dtype=gs_lhs.dtype) * gs_lhs  # (F, n_r, n_z)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def rhs_calculation(  # TODO: This must be abstract in a future generic parent class.
        j_tor_fields_: Tensor,
        cleaning_mask_: Tensor | None = None,
    ) -> Tensor:
        """
        Compute the weak-form Grad-Shafranov right-hand side `μ₀ · j_tor` for a stack of j_tor fields.

        The source term `μ₀ ∫Ω j_tor v dΩ` of the weak form is discretized with a lumped (diagonal) mass approximation,
        so it reduces to `μ₀ · j_tor` cell by cell; no mass matrix is assembled. It is matched against the LHS `W ψ`
        (see `lhs_calculation`), which approximates `-Δ*ψ / R` per unit cell area. That makes it the weak-form
        counterpart of the strong-form RHS `μ₀ · R · j_tor`, divided by R, so no explicit R factor appears here.

        Computation is done in float32, matching the dtype of `lhs_calculation`, so the residual `LHS - RHS` is formed
        in a single dtype. As a static method, it uses no cached grid tensors, so it runs on the device of
        `j_tor_fields_`.

        Parameters
        ----------
        j_tor_fields_ : Tensor
            Toroidal-current fields in physical units (A/m²), shaped `(F, n_r, n_z)` (R along axis 1, Z along axis 2).
            No shape check is done, so any shape works as long as `cleaning_mask_` broadcasts against it.
        cleaning_mask_ : Tensor | None
            Boolean or numeric physical-domain mask broadcastable to `(F, n_r, n_z)`, multiplied into the result.
            Optional. Default: `None` (no masking).

        Returns
        -------
        Tensor
            Float32 `μ₀ · j_tor` fields with the shape of `j_tor_fields_` (`(F, n_r, n_z)` in `compute`), masked if
            `cleaning_mask_` is given. Differentiable with respect to `j_tor_fields_` (gradients are returned in its
            original dtype).

        """

        # Perform point-wise mu0 * j_tor in float32, matching the float32 LHS from `lhs_calculation`.
        gs_rhs = mu0 * j_tor_fields_.to(torch.float32)  # (F, n_r, n_z)

        if cleaning_mask_ is None:
            return gs_rhs  # (F, n_r, n_z)
        else:
            return cleaning_mask_.to(dtype=gs_rhs.dtype) * gs_rhs  # (F, n_r, n_z)

    # ------------------------------------------------------------------------------------------------------------------
    def j_tor_for_pred_rhs(
        self,
        j_tor_destdized_pred: Tensor | None,
        psi_destdized_true: Tensor | None,
        j_tor_limiter_mask: Tensor,
    ) -> tuple[Tensor, bool]:
        """
        Select the toroidal current density j_tor used for the predicted GS right-hand side, according to
        `rhs_input_origin`.

          - "predicted_j_tor": returns `j_tor_destdized_pred` unchanged (same tensor, no copy or cast).
          - "derived_j_tor": derives j_tor from ground-truth psi via `calculate_j_tor_from_psi`, restricted to
            `j_tor_limiter_mask` and using `j_tor_calculation_method`.
          - "predicted_profiles": not implemented yet; always raises.

        Parameters
        ----------
        j_tor_destdized_pred : Tensor | None
            Predicted j_tor fields in physical units, shaped `(F, n_r, n_z)`. None when j_tor is not a required output.
            Used only for "predicted_j_tor".
        psi_destdized_true : Tensor | None
            Ground-truth psi fields in physical units, shaped `(F, n_r, n_z)`. Used only for "derived_j_tor".
        j_tor_limiter_mask : Tensor
            Region where j_tor is derived, broadcastable to `(F, n_r, n_z)`; cells outside it get zero current. In
            `compute`, this is the full sides cleaning mask (finite psi/j_tor targets and plasma region). Used only for
            "derived_j_tor".

        Returns
        -------
        tuple[Tensor, bool]
            `(j_tor_for_pred_rhs, requires_min_max_normalization)`:
            - j_tor_for_pred_rhs: j_tor fields for the predicted RHS, shaped `(F, n_r, n_z)`.
            - requires_min_max_normalization: whether the GS sides must be min-max normalized before computing the
              residuals. True only for "derived_j_tor" with `j_tor_calculation_method="parametric_approx"` (j_tor is
              then only meaningful up to a scale factor); False otherwise.

        Raises
        ------
        ValueError
            If "predicted_j_tor" is selected and `j_tor_destdized_pred` is None, or if "predicted_profiles" is selected
            and the p'/FF' outputs are not available.
        RuntimeError
            If "derived_j_tor" is selected and `psi_destdized_true` is None, or if `rhs_input_origin` is unsupported.
        NotImplementedError
            If "predicted_profiles" is selected (with its outputs available), since that path is disabled for now.

        """

        # Values of signals "equilibrium-j_tor" and "equilibrium-psi" correspond to matrices that must be transposed
        # before they are used in the Grad-Shafranov equation along with the Grad-Shafranov operator from FreeGSNKE.
        # `native_to_gs_fields` applies that transpose for every (sample, time).

        # ..............................................................................................................
        # Info for the predicted Right-Hand Side of the GS equation.
        # REMARK: Each branch to calculate j_tor for predicted RHS must yield j_tor as (F, n_r, n_z) fields.
        # ..............................................................................................................

        requires_min_max_normalization = False

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields from predicted j_tor

        if self.rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
            if j_tor_destdized_pred is None:
                raise ValueError(
                    f"`rhs_input_origin={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR!r}` requires model output "
                    "'equilibrium-j_tor'."
                )

            j_tor_for_pred_rhs = j_tor_destdized_pred

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields from derived j_tor

        elif self.rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR:
            if psi_destdized_true is None:
                raise RuntimeError("Grad-Shafranov loss needs native target 'equilibrium-psi' when j_tor is derived.")

            j_tor_for_pred_rhs, requires_min_max_normalization = self.calculate_j_tor_from_psi(
                psi_fields=psi_destdized_true,  # noqa - Proper check for None is done before this point.
                j_tor_limiter_mask=j_tor_limiter_mask,  # (F, n_r, n_z)
            )

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields from predicted profiles (pprime and ffprime)

        elif self.rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES:
            if (self._pprime_key is None) or (self._ffprime_key is None):
                raise ValueError(
                    f"`rhs_input_origin={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES!r}` requires model outputs "
                    "'equilibrium-dpressure_dpsi' and 'equilibrium-f_df_dpsi'."
                )

            # TODO(FIX):
            #  The predicted-profiles RHS path is not functional yet — see `j_tor_from_pprime_ffprime_psi` for the
            #  problem description and proposed fix. Disabled here so it cannot be silently used.
            raise NotImplementedError(
                f"`rhs_input_origin={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES!r}` is not implemented yet "
                "(see j_tor_from_pprime_ffprime_psi)."
            )

            # j_tor_for_pred_rhs = self.j_tor_from_pprime_ffprime_psi(...)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields cannot be obtained

        else:
            raise RuntimeError(f"Unsupported `rhs_input_origin={self.rhs_input_origin!r}`.")

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # If here, `j_tor_for_pred_rhs` has been properly specified.

        # ..............................................................................................................
        # Return calculated info
        # ..............................................................................................................

        return j_tor_for_pred_rhs, requires_min_max_normalization

    # ------------------------------------------------------------------------------------------------------------------
    def compute(  # NOSONAR - Ignore cognitive complexity
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: None | Mapping[Hashable, Tensor] = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute the masked weak-form Grad-Shafranov (GS) loss in native, destandardized space.

        Required predictions are decoded to native space; predictions and targets are destandardized and folded into
        `(F, n_r, n_z)` fields (`F = B * T`). The predicted LHS uses predicted psi; the predicted RHS uses the j_tor
        selected by `rhs_input_origin` (predicted j_tor, or j_tor derived from ground-truth psi); the anchor is the LHS
        of ground-truth psi. Residuals are restricted to the sides cleaning mask (finite psi/j_tor target cells and the
        j_tor limiter region). Only fields that pass `output_mask`, have finite predicted-RHS j_tor, and have at
        least one cell inside the cleaning mask contribute; the loss is the mean of their weighted per-field losses
        (see `grad_shafranov_loss`). Non-finite values in predictions are not masked.

        A zero loss that stays in the computation graph (`ref.sum() * 0.0`), with empty logs, is returned when the
        `weak_grad_shafranov` weight is not positive and plotting is off, or when no field is valid. If the weight is
        not positive but plotting is on, the GS sides are computed for the plots and the loss stays zero.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Prediction tensors in embedding space, keyed by signal ID, each shaped `(B, D)`. Must contain every key in
            `required_output_keys`; other keys are ignored.
        y_emb : Mapping[Hashable, Tensor]
            Unused.
        y_native : Mapping[Hashable, Tensor] | None
            Ground-truth tensors in native standardized space, keyed by signal ID; psi and j_tor are shaped
            `(B, H, W, T)`. Required (None raises); must contain every key in `required_output_keys`.
        output_mask : None | Mapping[Hashable, Tensor]
            Per-output boolean masks shaped `(B,)`, True for supervised samples; samples marked False in any required
            output are excluded. Missing entries leave that output unconstrained.
            Optional. Default: None (no sample is excluded).
        context : LossComputeContext | None
            Run metadata used for plotting. Must provide `run_dir` when plots are saved.
            Optional. Default: None.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            `(loss, logs)`: the scalar loss, and `{"weak_gs_loss": float}` when the `weak_grad_shafranov` weight is
            positive and at least one field is valid (otherwise empty).

        Raises
        ------
        RuntimeError
            If `preds` is empty or None, if `y_native` is None, if `rhs_input_origin="derived_j_tor"` and target psi is
            unavailable or `rhs_input_origin` is unsupported, or if plots are saved and `context` has no `run_dir`.
        ValueError
            If a required output is missing from `preds` or `y_native`, or if the outputs required by
            `rhs_input_origin` are unavailable.
        NotImplementedError
            If `rhs_input_origin="predicted_profiles"` is selected (path disabled).

        """

        # ..............................................................................................................
        # 0 - Preliminary checks
        # ..............................................................................................................

        if not preds:
            raise RuntimeError(f"{self.__class__.__name__} received empty predictions from the model.")

        if y_native is None:
            raise RuntimeError(f"{self.__class__.__name__} received predictions with empty native targets.")

        ref = next(iter(preds.values()))
        self._check_device_and_dtype(ref=ref)

        loss: Tensor = ref.sum() * 0.0  # Scalar
        logs: dict[Hashable, float] = {}

        # If the weight for the "weak_grad_shafranov" loss term is not greater than 0 and:
        # - Plotting is not needed, then return from here a default value 0 for the loss.
        # - Plotting is needed, calculation of all the required values for plotting is allowed, but calculations to
        #   update the default value 0 of the loss is later disallowed.
        if (not self._all_losses_weights.get("weak_grad_shafranov", 0) > 0) and (self._plot_check_type is None):
            return loss, logs  # At this point, loss is 0.

        # ..............................................................................................................
        # 1 - Gather required, decoded ground truth and predicted data in native space
        # ..............................................................................................................

        required_stdized_native_fields: dict[Hashable, dict[Hashable, Tensor]] = {
            "pred": self._decode_fields_to_stdized_native(
                encoded_fields=self._gather_required_fields(input_fields=preds),
            ),
            "true": self._gather_required_fields(input_fields=y_native),
        }

        # ..............................................................................................................
        # 2 - Destandardize fields for LHS and RHS
        # ..............................................................................................................

        # psi (LHS) related
        psi_fields_destdized, psi_no_nan_mask, bhwt_shape = self._get_gs_safe_destdized_fields(
            stdized_native_fields=required_stdized_native_fields,
            signal_key=self._psi_key,
        )

        # In this pipeline, `psi_fields_destdized` is always required to be non-None, and there is an early check for
        # it. Thus, the following cast is valid, and it is needed to avoid downstream issues.
        psi_fields_destdized = cast(dict[Hashable, Tensor], psi_fields_destdized)

        # j_tor (RHS) related
        j_tor_fields_destdized, j_tor_no_nan_mask, _ = self._get_gs_safe_destdized_fields(
            stdized_native_fields=required_stdized_native_fields,
            signal_key=self._j_tor_key,
        )

        n_fields = bhwt_shape[0] * bhwt_shape[3]  # F = B * T

        # ..............................................................................................................
        # 3 - Create composite cleaning mask
        # ..............................................................................................................

        # Limiter mask for j_tor
        j_tor_limiter_mask = self._get_j_tor_limiter_mask(
            lcfs_true={  # NOTE: Pipeline using this option is incomplete.
                "lcfs_r": required_stdized_native_fields["true"].get("lcfs_r"),
                "lcfs_z": required_stdized_native_fields["true"].get("lcfs_z"),
            },
            j_tor_true_destdized=j_tor_fields_destdized["true"],  # Not None only if j_tor is a required output.
            num_fields=n_fields,
        )

        # Consolidated cleaning mask
        sides_cleaning_mask = psi_no_nan_mask & j_tor_no_nan_mask & j_tor_limiter_mask

        # ..............................................................................................................
        # 4 - Get final info for predicted sides of the Grad-Shafranov equation
        # ..............................................................................................................

        j_tor_fields_for_pred_rhs, perform_min_max_normalization = self.j_tor_for_pred_rhs(
            j_tor_destdized_pred=j_tor_fields_destdized["pred"],
            psi_destdized_true=psi_fields_destdized["true"],
            j_tor_limiter_mask=sides_cleaning_mask,
        )

        # ..............................................................................................................
        # 5 - Calculate Grad-Shafranov sides to be used in the residual calculation
        # ..............................................................................................................

        # Drop field values whose j_tor contains NaNs (e.g., from ground-truth psi approximation where psi is all-NaN).
        # As psi is in general either fully NaN or fully valid, this would mean to drop whole fields.
        valid_fields = torch.isfinite(j_tor_fields_for_pred_rhs).flatten(start_dim=1).all(dim=1)  # (F,)
        valid_fields &= sides_cleaning_mask.flatten(start_dim=1).any(dim=1).to(valid_fields.device)  # Includes gt psi.

        # Required output mask follows the configured RHS policy: derived j_tor involves ground-truth psi only;
        # predicted j_tor involves ground-truth psi and ground-truth j_tor. The profiles path will use its declared
        # required outputs when implemented.
        batch_size, _, _, n_times = bhwt_shape
        valid_fields &= output_masks_to_field_mask(
            output_mask=output_mask,
            required_keys=self.required_output_keys,
            batch_size=batch_size,
            n_times=n_times,
            ref=ref,
        )

        # Check for valid fields, and return if no valid fields found
        if not bool(valid_fields.any()):
            # If no valid fields, we can return from here with a physics residual loss of the form ref.sum() * 0.0,
            # which is zero but stays in the computation graph, so backward() does not crash when native_sparse_mse is
            # the only active loss term.
            # REMARK: No reasons for plot checks (if active) since no valid fields are available.
            return loss, logs  # At this point, loss is 0.

        # Calculate GS anchor from ground-truth psi.
        # REMARK: For ground-truth data, LHS is equal to RHS, so either of them serves as anchor.
        gs_anchor = self.lhs_calculation(
            psi_fields_=psi_fields_destdized["true"],
            cleaning_mask_=sides_cleaning_mask,
        )

        # Calculate LHS from psi predictions.
        gs_lhs_pred = self.lhs_calculation(
            psi_fields_=psi_fields_destdized["pred"],
            cleaning_mask_=sides_cleaning_mask,
        )

        # Calculate RHS from j_tor predictions/derivations/approximations.
        gs_rhs_pred = self.rhs_calculation(
            j_tor_fields_=j_tor_fields_for_pred_rhs,
            cleaning_mask_=sides_cleaning_mask,
        )

        # ..............................................................................................................
        # 6 - Plot data (if specified)
        # ..............................................................................................................

        self._run_plot_check(
            context_info=context,
            cleaning_mask=sides_cleaning_mask,
            gs_lhs_pred=gs_lhs_pred,
            gs_rhs_pred=gs_rhs_pred,
            gs_lhs_ref=gs_anchor,
            psi_fields_for_pred_lhs=psi_fields_destdized["pred"],
            psi_fields_gt_destdized=psi_fields_destdized["true"],
            j_tor_fields_for_pred_rhs=j_tor_fields_for_pred_rhs,
            j_tor_fields_gt_destdized=j_tor_fields_destdized["true"],
        )

        # ..............................................................................................................
        # 7 - Calculate Grad-Shafranov residuals for every (sample, time) field, computed in one vectorized pass.
        # ..............................................................................................................

        # Only update the default loss value 0 if the weight for the "weak_grad_shafranov" loss term is > 0.
        if self._all_losses_weights.get("weak_grad_shafranov", 0) > 0:
            per_field_losses = self.grad_shafranov_loss(
                gs_lhs_pred=gs_lhs_pred,
                gs_rhs_pred=gs_rhs_pred,
                gs_anchor=gs_anchor,
                cleaning_mask=sides_cleaning_mask,
                min_max_normalize_sides=perform_min_max_normalization,
            )  # (F,)

            loss = per_field_losses[valid_fields].mean()  # Scalar
            logs["weak_gs_loss"] = float(loss.detach().cpu())

        # ..............................................................................................................
        # 8 - Return loss and logs
        # ..............................................................................................................

        return loss, logs

    # ------------------------------------------------------------------------------------------------------------------
    def _gather_required_fields(self, input_fields: Mapping[Hashable, Tensor]) -> dict[Hashable, Tensor]:
        """
        Select the fields required by the configured RHS and LHS, and cast them to float32.

        Parameters
        ----------
        input_fields : Mapping[Hashable, Tensor]
            Fields keyed by output signal ID, e.g., embedding-space predictions or native-space `y_native` targets.
            May contain additional, non-required fields, which are ignored.

        Returns
        -------
        dict[Hashable, Tensor]
            The fields whose keys are in `self.required_output_keys`, cast to float32 and keyed by signal ID.
            Tensors that are already float32 are returned as is, not copied.

        Raises
        ------
        ValueError
            If any key in `self.required_output_keys` is missing from `input_fields`.

        """

        missing_fields = self.required_output_keys - input_fields.keys()
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}.")

        required_fields: dict[Hashable, Tensor] = {}
        for f_key in self.required_output_keys:
            required_fields[f_key] = input_fields[f_key].to(torch.float32)

        return required_fields

    # ------------------------------------------------------------------------------------------------------------------
    def _decode_fields_to_stdized_native(self, encoded_fields: Mapping[Hashable, Tensor]) -> dict[Hashable, Tensor]:
        """
        Decode embedding-space fields to native standardized space using the per-signal differentiable decoders.

        Parameters
        ----------
        encoded_fields : Mapping[Hashable, Tensor]
            Embedding-space tensors (e.g., model predictions) keyed by output signal ID, each shaped `(B, D)`.

        Returns
        -------
        dict[Hashable, Tensor]
            Decoded float32 fields in native standardized space, keyed by signal ID, each shaped
            `(B, *native_shape)`. Gradients are preserved with respect to `encoded_fields`.

        Raises
        ------
        KeyError
            If a key in `encoded_fields` has no corresponding decoder in `self._decoders`.

        """
        stdized_native_fields: dict = {}
        for f_key, f_val in encoded_fields.items():
            # Decode fields: (B, D) → (B, *native_shape)
            decoder = self._decoders[f_key]
            stdized_native_fields[f_key] = decoder(
                f_val.to(torch.float32)  # Gradients flow through here
            ).to(torch.float32)

        return stdized_native_fields

    # ------------------------------------------------------------------------------------------------------------------
    def grad_shafranov_loss(
        self,
        gs_lhs_pred: Tensor,
        gs_rhs_pred: Tensor,
        gs_anchor: Tensor,
        cleaning_mask: Tensor | None = None,
        min_max_normalize_sides: bool = False,
    ) -> Tensor:
        """
        Compute the weighted Grad-Shafranov residual loss for each field.

        All inputs are stacks of 2D fields shaped `(F, n_r, n_z)`, with one field per (sample, time) pair. Three
        residuals are built from the predicted LHS, the predicted RHS, and the ground-truth anchor (which approximates
        the true RHS up to discretization error):

          - `no_gt`:  LHS_pred - RHS_pred   (physics consistency of the prediction, no ground truth needed)
          - `lhs_gt`: LHS_pred - anchor     (predicted psi checked against the true equilibrium)
          - `rhs_gt`: RHS_pred - anchor     (predicted current checked against the true equilibrium)

        Only residuals with a positive weight in `self.gs_weights` are computed. Each one is reduced per field with
        `masked_reduce` using `self.loss_metric` ("mse": mean of squares over active cells; "l2": Euclidean norm over
        active cells), and the results are combined as a weighted sum (the weights are validated to lie in [0, 1] and
        sum to 1). Optionally, each side is min-max normalized per field beforehand, over all cells of the field
        (including cells outside `cleaning_mask`); constant fields map to 0.

        Parameters
        ----------
        gs_lhs_pred : Tensor
            LHS of the Grad-Shafranov equation formed with predicted ψ data.
        gs_rhs_pred : Tensor
            RHS of the Grad-Shafranov equation formed with predicted j_tor data.
        gs_anchor : Tensor
            Ground-truth anchor for the Grad-Shafranov equation.
        cleaning_mask : Tensor | None
            Cleaning mask for both sides of the Grad-Shafranov equation.
            Optional. Default: None.
        min_max_normalize_sides : bool
            If True, min-max normalize the LHS, RHS, and anchor per field before computing the residuals.
            Optional. Default: False.

        Returns
        -------
        Tensor
            Weighted per-field GS loss shaped `(F,)`. Fields with no active cell in `cleaning_mask` are NaN, so the
            caller can drop them.

        Raises
        ------
        ValueError
            If `cleaning_mask` does not have the same number of cells per field as the inputs, or if
            `self.loss_metric` is not in `ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS`.

        """

        # TODO:
        #  - Check if the Rc term (equal to (R_max - R_min)/2) from the PINNs approach should be added.

        # ..............................................................................................................
        # Normalize sides, if required.

        if min_max_normalize_sides:
            gs_lhs_pred = self._min_max_normalize_per_field(x=gs_lhs_pred)
            gs_rhs_pred = self._min_max_normalize_per_field(x=gs_rhs_pred)
            gs_anchor = self._min_max_normalize_per_field(x=gs_anchor)

        # ..............................................................................................................
        # Calculate the base GS residuals.

        flat_gs_residuals = {}
        if self.gs_weights["no_gt"] > 0:
            flat_gs_residuals["no_gt"] = (gs_lhs_pred - gs_rhs_pred).flatten(start_dim=1)
        if self.gs_weights["lhs_gt"] > 0:
            flat_gs_residuals["lhs_gt"] = (gs_lhs_pred - gs_anchor).flatten(start_dim=1)
        if self.gs_weights["rhs_gt"] > 0:
            flat_gs_residuals["rhs_gt"] = (gs_rhs_pred - gs_anchor).flatten(start_dim=1)

        # ..............................................................................................................
        # Calculate and return the GS loss

        gs_loss = {
            res_idx: masked_reduce(residual=res_val, mask=cleaning_mask, kind=self.loss_metric)
            for res_idx, res_val in flat_gs_residuals.items()
        }

        total_gs_loss = 0 * gs_anchor.flatten(start_dim=1).mean(dim=1)
        for loss_idx, loss_val in gs_loss.items():
            total_gs_loss += self.gs_weights[loss_idx] * loss_val

        return total_gs_loss

    # ------------------------------------------------------------------------------------------------------------------
    def _get_gs_safe_destdized_fields(
        self,
        stdized_native_fields: Mapping[Hashable, Mapping[Hashable, Tensor]],
        signal_key: Hashable | None,
    ) -> tuple[dict[Hashable, Tensor | None], Tensor, torch.Size | list[int]]:
        """
        Convert standardized predicted and target native tensors into GS-safe, destandardized physical fields.

        Each `(B, H, W, T)` tensor is folded into `(F, n_r, n_z)` fields (`F = B * T`, with each `(H, W)` slice
        transposed to `(n_r, n_z)`; see `native_to_gs_fields`) and destandardized using `self.signal_stats`.
        Predictions are destandardized as they are (non-finite values are neither replaced nor masked; gradients are
        preserved). Target non-finite cells are set to zero in standardized space (so they hold the signal mean after
        destandardization), so that no NaN reaches the GS operator, and they are excluded by the returned mask.

        Parameters
        ----------
        stdized_native_fields : Mapping[Hashable, Mapping[Hashable, Tensor]]
            Standardized native tensors under the keys "pred" and "true", each a mapping from output signal ID to a
            tensor shaped `(B, H, W, T)`. In `compute`, these contain only the required outputs.
        signal_key : Hashable | None
            Signal ID of the field to prepare: `self._psi_key` or `self._j_tor_key`. May be None when j_tor is not a
            model output (i.e., `self._j_tor_key` is None).

        Returns
        -------
        tuple[dict[Hashable, Tensor | None], Tensor, torch.Size | list[int]]
            - Destandardized fields `{"pred": ..., "true": ...}`, each shaped `(F, n_r, n_z)`.
            - Boolean finite-cell mask of the target, shaped `(F, n_r, n_z)`.
            - Shape `(B, H, W, T)` of the predicted tensor.
            If `signal_key` is in neither "pred" nor "true" (e.g., j_tor is not a required output, as with
            `rhs_input_origin="derived_j_tor"`), returns `({"pred": None, "true": None}, torch.tensor(True),
            [0, 0, 0, 0])`, i.e., no fields, a no-op mask, and a placeholder shape.

        Raises
        ------
        ValueError
            If `signal_key` is neither `self._psi_key` nor `self._j_tor_key`.
        KeyError
            If `signal_key` is present in only one of "pred" and "true", or if its signal name has no entry in
            `self.signal_stats`.

        """

        # ..............................................................................................................
        # Fold (B, T) into a single field axis F = B * T, then (F, n_r, n_z) fields.
        # Decoded predictions have native shape (B, H, W, T); T is the forecast-horizon time axis.
        #
        # REMARK: For Grad-Shafranov related calculations, all data must be destandardized and in native space.
        # ..............................................................................................................

        if signal_key not in [self._psi_key, self._j_tor_key]:
            raise ValueError(
                "Invalid key {}. Valid options: [{}, {}].".format(signal_key, self._psi_key, self._j_tor_key)
            )

        # ..............................................................................................................
        # Return early if `signal_key` is not available (e.g., if `signal_key` for j_tor, but j_tor is not an output).

        if (signal_key not in stdized_native_fields["true"]) and (signal_key not in stdized_native_fields["pred"]):
            return {"pred": None, "true": None}, torch.tensor(True), [0, 0, 0, 0]

        # ..............................................................................................................
        # Proceed only if `signal_key` is available.

        signal_key_srt = "equilibrium-psi" if (signal_key == self._psi_key) else "equilibrium-j_tor"

        # Get GS-safe standardized signal fields
        gs_safe_stdized_fields = {
            "pred": native_to_gs_fields(
                x_bhwt=stdized_native_fields["pred"][signal_key]  # (B, H, W, T), standardized native
            ),  # (F, n_r, n_z)
            "true": native_to_gs_fields(
                x_bhwt=stdized_native_fields["true"][signal_key]  # (B, H, W, T), standardized native
            ),  # (F, n_r, n_z),
        }

        # Predictions are already safe. Targets are made finite before any GS operator is applied; the returned mask
        # still excludes their original non-finite cells from the physical residual domain.
        gs_safe_destdized_fields = {
            "pred": destandardize_torch(
                x=gs_safe_stdized_fields["pred"],
                mean=self.signal_stats[signal_key_srt]["mean"],
                std=self.signal_stats[signal_key_srt]["std"],
            ),
        }
        gs_safe_destdized_fields["true"], no_nan_mask = prepare_target_field(
            field=gs_safe_stdized_fields["true"],
            stats=self.signal_stats[signal_key_srt],
        )

        bhwt_shape = stdized_native_fields["pred"][signal_key].shape

        return gs_safe_destdized_fields, no_nan_mask, bhwt_shape

    # ------------------------------------------------------------------------------------------------------------------
    def _get_j_tor_limiter_mask(
        self,
        lcfs_true: Mapping[str, Tensor | None],  #  TODO: Check if these should be destandardized.
        j_tor_true_destdized: Tensor | None,
        num_fields: int,
    ) -> Tensor:
        """
        Select the plasma-region (limiter) mask that restricts the Grad-Shafranov residual domain.

        The mask is combined into the sides cleaning mask in `compute`, so it restricts all residuals (and the region
        where j_tor is derived). Three sources are tried in order:

          1. LCFS polygon: cells inside the polygon given by `lcfs_true` (via `_make_lcfs_mask`).
          2. Target j_tor: cells where the ground-truth current exceeds `1e-6` (`j_tor_true_destdized > 1e-6`). Target
             cells that were non-finite hold the signal mean (see `_get_gs_safe_destdized_fields`) and may be marked
             True; `compute` excludes them through the finite-cell mask.
          3. Static fallback: `self.base_j_tor_limiter_mask_rz` from the Grad-Shafranov asset, repeated per field.

        Parameters
        ----------
        lcfs_true : Mapping[str, Tensor | None]
            LCFS boundary coordinates in metres, as {"lcfs_r": lcfs_r_data, "lcfs_z": lcfs_z_data}, each shaped
            `(n_pts,)` or `(F, n_pts)`. Both keys must be present; option 1 is used only if both values are not None.
        j_tor_true_destdized : Tensor | None
            Ground-truth j_tor in physical units, shaped `(F, n_r, n_z)`. None when j_tor is not a required output
            (e.g., `rhs_input_origin="derived_j_tor"`).
        num_fields : int
            Number of fields `F = B * T`, used to repeat the static fallback mask.

        Returns
        -------
        Tensor
            Boolean (non-differentiable) plasma-region mask broadcastable to `(F, n_r, n_z)`:
              - option 1: shaped `(n_r, n_z)` (one contour shared by all fields) or `(F, n_r, n_z)`, on the device of
                the LCFS tensors;
              - option 2: shaped `(F, n_r, n_z)`, on the device of `j_tor_true_destdized`;
              - option 3: shaped `(F, n_r, n_z)`, on the runtime device (see `_check_device_and_dtype`).

        Raises
        ------
        KeyError
            If "lcfs_r" or "lcfs_z" is missing from `lcfs_true`.
        ValueError
            If option 1 is used and the LCFS tensors have different shapes or are neither 1D nor 2D.

        Notes
        -----
        - When j_tor is derived, it is constructed from psi's ground truth data. In that case, the derived j_tor cannot
          be used to define its domain or field eligibility since psi is also defined out of the plasma boundaries.
        - When j_tor is predicted, ground-truth data for j_tor can be used to define a validity mask for predicted
          j_tor values.

        """

        # ..............................................................................................................
        # Build a j_tor limiter mask
        # ..............................................................................................................

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # First option (the best possible): From time-varying lcfs_r and lcfs_z data, if available.
        if (lcfs_true["lcfs_r"] is not None) and (lcfs_true["lcfs_z"] is not None):
            j_tor_limiter_mask = self._make_lcfs_mask(
                lcfs_r=cast(Tensor, lcfs_true["lcfs_r"]),
                lcfs_z=cast(Tensor, lcfs_true["lcfs_z"]),
            )  # (n_r, n_z) or (F, n_r, n_z)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Second option: From time-varying j_tor ground truth data, if available.
        elif j_tor_true_destdized is not None:
            j_tor_limiter_mask = j_tor_true_destdized > 1e-6  # (F, n_r, n_z)
            # TODO: Adopt approach to soften the mask for better backpropagation performance.

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Last resort: From static reference data (i.e., from self.base_j_tor_limiter_mask_rz).
        else:
            j_tor_limiter_mask = self.base_j_tor_limiter_mask_rz.repeat(num_fields, 1, 1)  # (F, n_r, n_z)

        # ..............................................................................................................
        # Return calculated mask
        # ..............................................................................................................

        return j_tor_limiter_mask

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _min_max_normalize_per_field(x: Tensor) -> Tensor:
        """
        Min-max normalize each (sample, time) field independently to [0, 1], over all cells of the field.

        Min and max are taken over the last two axes of `x` (`(F, n_r, n_z)`), including cells outside any cleaning
        mask. Constant fields map to 0. Fields containing any NaN also map to 0, since their min/max are NaN.
        """

        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        x_range = x_max - x_min

        # Constant fields (e.g., all-zero j_tor) would otherwise give 0/0 = NaN. A safe denominator is also needed:
        # `torch.where` backpropagates through both branches, so 0/0 in the unselected branch would give NaN gradients.
        is_varying = x_range > 0
        safe_range = torch.where(is_varying, x_range, torch.ones_like(x_range))

        return torch.where(is_varying, (x - x_min) / safe_range, torch.zeros_like(x))

    # ------------------------------------------------------------------------------------------------------------------
    def _run_plot_check(
        self,
        *,
        context_info: LossComputeContext | None,
        cleaning_mask: Tensor,
        gs_lhs_pred: Tensor,
        gs_rhs_pred: Tensor,
        gs_lhs_ref: Tensor,
        psi_fields_for_pred_lhs: Tensor,
        psi_fields_gt_destdized: Tensor,
        j_tor_fields_for_pred_rhs: Tensor,
        j_tor_fields_gt_destdized: Tensor | None,
        plot_single_slice: bool = True,
    ) -> None:
        """
        Optionally emit weak-form diagnostic plots (GS sides, psi, and j_tor) for selected field slices.

        Nothing is plotted unless plotting is enabled in `plot_check_cfg` (a plot `type` and `probability > 0`)
        and `context_info` is provided. Whether this batch is plotted, and which slice, is decided deterministically
        from the batch metadata in `context_info` (see `select_diagnostic_plot_slice`), so plotting does not consume
        the global RNG. Plots are computed under `torch.no_grad()` and either saved under
        `<run_dir>/plots/training/stage_<idx>_<name>/` (for `save_*` types) or shown (for `show_plots`).


        The plotted slice is chosen among all `F` fields, so it may be one excluded from the loss (e.g., by
        `output_mask` or the field-validity checks in `compute`). GS sides are plotted as passed, i.e., without the
        min-max normalization that `grad_shafranov_loss` may apply. With `show_plots`, `plt.show()` may block
        training under interactive matplotlib backends.

        Parameters
        ----------
        context_info : LossComputeContext | None
            Run, stage, and batch metadata, used to select the plotted batch/slice and to build the save path. If None,
            nothing is plotted.
        cleaning_mask : Tensor
            Physical-domain mask shaped `(F, n_r, n_z)`, applied when deriving the reference right-hand side.
        gs_lhs_pred, gs_rhs_pred, gs_lhs_ref : Tensor
            Predicted LHS (`Wψ_pred`), predicted RHS (`μ₀ · j_tor`), and reference LHS (the ground-truth anchor
            `Wψ_true`), each shaped `(F, n_r, n_z)`.
        psi_fields_for_pred_lhs, psi_fields_gt_destdized : Tensor
            Predicted and target poloidal-flux fields in physical units, each shaped `(F, n_r, n_z)`.
        j_tor_fields_for_pred_rhs : Tensor
            Toroidal-current fields used for the predicted RHS, shaped `(F, n_r, n_z)`: either the predicted j_tor or
            one derived from ground-truth psi, depending on `rhs_input_origin`.
        j_tor_fields_gt_destdized : Tensor | None
            Target toroidal-current fields in physical units, shaped `(F, n_r, n_z)`. If None (j_tor is not a
            required output), NaN fields are plotted in its place, so the reference RHS and j_tor panels are empty, and
            the predicted current is labeled "approximated".
        plot_single_slice : bool
            If True, plot only the selected field slice; otherwise, plot every slice of a selected batch.
            Optional. Default: True.

        Returns
        -------
        None

        Raises
        ------
        RuntimeError
            If plots are saved and `context_info.run_dir` is not set.

        """

        if not self._run_plotting:
            return

        plot_index = select_diagnostic_plot_slice(
            context=context_info,
            probability=self._plot_check_probability,
            n_fields=psi_fields_for_pred_lhs.shape[0],
            diagnostic_name="weak_grad_shafranov",
        )
        if plot_index is None:
            return

        with torch.no_grad():
            # ..........................................................................................................
            # Preliminaries

            j_tor_case = "predicted"
            if j_tor_fields_gt_destdized is None:
                j_tor_case = "approximated"
                j_tor_fields_gt_destdized = torch.nan * psi_fields_gt_destdized

            # ..........................................................................................................
            # Reference RHS (from ground truth, if available)

            gs_rhs_ref = self.rhs_calculation(
                j_tor_fields_=j_tor_fields_gt_destdized,
                cleaning_mask_=cleaning_mask,
            )

            # ..........................................................................................................
            # Plot over selected indices

            indices_to_plot = [plot_index] if plot_single_slice else range(psi_fields_gt_destdized.shape[0])
            for ii in indices_to_plot:
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
                        "fig_title": self._fig_title,
                        "fig_subtitle": self._fig_subtitle,
                        "subplot_titles": [
                            r"$\mathrm{LHS}^{pred}: M{\odot}\left(W\psi^{pred}\right)$",
                            r"$\mathrm{LHS}^{true}: M{\odot}\left(W\psi^{true}\right)$",
                            r"$\mathrm{RHS}^{pred}: M{\odot}\left(\mu_{0}{\cdot}J^{"
                            + j_tor_case[:4]
                            + r"}_{\phi}\right)$",
                            r"$\mathrm{RHS}^{true}: \mu_{0}{\cdot}J^{true}_{\phi}$",
                            r"$\psi^{pred}$",
                            r"$\psi^{true}$",
                            r"$J^{" + j_tor_case[:4] + r"}_{\phi}$",
                            r"$J^{true}_{\phi}$",
                        ],
                        "gs_data": {
                            "lhs_pred_data": gs_lhs_pred[ii].detach(),
                            "lhs_ref_data": gs_lhs_ref[ii].detach(),
                            "rhs_pred_data": gs_rhs_pred[ii].detach(),
                            "rhs_ref_data": gs_rhs_ref[ii].detach(),
                        },
                        "signal_data": {
                            "psi_pred_data": psi_fields_for_pred_lhs[ii].detach(),
                            "psi_ref_data": psi_fields_gt_destdized[ii].detach(),
                            "j_tor_pred_data": j_tor_fields_for_pred_rhs[ii].detach(),
                            "j_tor_ref_data": j_tor_fields_gt_destdized[ii].detach(),
                            "j_tor_case": j_tor_case,
                        },
                    },
                    save_plots=self._save_plots,
                    save_path=(
                        training_plot_path(
                            context=context_info,
                            slice_index=ii,
                            plot_format=self._save_plot_format,
                        )
                        if self._save_plots
                        else None
                    ),
                )

    # ------------------------------------------------------------------------------------------------------------------
    def _make_lcfs_mask(
        self,
        lcfs_r: Tensor,
        lcfs_z: Tensor,
    ) -> Tensor:
        """
        Build a boolean mask of the grid points inside the last closed flux surface (LCFS), for one or more contours.

        Each LCFS contour is treated as a closed polygon in (R, Z), and every point of the rectangular equilibrium mesh
        is tested against it (via `matplotlib.path.Path.contains_points`). Contour points with a non-finite R or Z, or
        with `R <= 0` (e.g., padding in EFIT arrays), are discarded first.

        Parameters
        ----------
        lcfs_r, lcfs_z : Tensor
            R and Z coordinates (in m) of the LCFS contour(s), with equal shapes: `(n_pts,)` for one contour, or
            `(F, n_pts)` for one contour per field.
            Must be a NumPy-convertible dtype (e.g., float32; bfloat16 is not supported), with `n_pts >= 1` and
            `F >= 1`.

        Returns
        -------
        Tensor
            Boolean mask on the input's device, True for grid points inside the LCFS. Shaped `(n_r, n_z)` (the shape of
            `self.R_matrix`) for 1D inputs, or `(F, n_r, n_z)` for batched inputs. Contours with fewer than 3 valid
            points yield an all-False mask.

        Raises
        ------
        ValueError
            If `lcfs_r` and `lcfs_z` have different shapes, or are neither 1D nor 2D.

        Notes
        -----
        - Inputs are detached and processed on CPU via NumPy/Matplotlib, so the returned mask is not differentiable.
        - Contours are processed one at a time, so the cost grows linearly with `F`.
        - Grid points lying exactly on a contour may be classified either way.
        - Discarded points inside a contour are bridged by a straight segment between their valid neighbors.

        """

        if (lcfs_r.shape != lcfs_z.shape) or (lcfs_r.ndim not in (1, 2)):
            raise ValueError(
                f"[{self.__class__.__name__}] `lcfs_r` and `lcfs_z` must share a (n_pts,) or (F, n_pts) shape, got "
                f"{tuple(lcfs_r.shape)} and {tuple(lcfs_z.shape)}."
            )

        # Polygon tests run in NumPy/Matplotlib: detach and move to CPU, then return on the input device.
        lcfs_r_np = lcfs_r.detach().cpu().numpy().reshape(-1, lcfs_r.shape[-1])  # (F, n_pts)
        lcfs_z_np = lcfs_z.detach().cpu().numpy().reshape(-1, lcfs_z.shape[-1])  # (F, n_pts)
        rz_points = self.rz_points_for_lcfs_calculation.detach().cpu().numpy()  # (n_r * n_z, 2)
        grid_shape = tuple(self.R_matrix.shape)  # (n_r, n_z)

        masks = []
        for r_f, z_f in zip(lcfs_r_np, lcfs_z_np):
            valid = np.isfinite(r_f) & np.isfinite(z_f) & (r_f > 0)
            if valid.sum() < 3:
                masks.append(np.zeros(grid_shape, dtype=bool))
                continue
            lcfs_rz_path = MatplotlibPath(np.column_stack([r_f[valid], z_f[valid]]))
            masks.append(lcfs_rz_path.contains_points(points=rz_points).reshape(grid_shape))

        lcfs_mask = torch.from_numpy(np.stack(masks)).to(device=lcfs_r.device)  # (F, n_r, n_z), bool

        return lcfs_mask[0] if (lcfs_r.ndim == 1) else lcfs_mask

    # ------------------------------------------------------------------------------------------------------------------
    def calculate_j_tor_from_psi(
        self,
        psi_fields: Tensor,
        j_tor_limiter_mask: Tensor | None = None,
    ) -> tuple[Tensor, bool]:
        """
        Derive the toroidal current j_tor from ground-truth psi, using the configured `j_tor_calculation_method`.

        Used when `rhs_input_origin="derived_j_tor"` to build the predicted RHS from ground-truth psi. Dispatches on
        `self.j_tor_calculation_method`:

          - "GS_operator" (`GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR`): uses the corresponding GS operator to derive j_tor
            from psi (see `j_tor_from_psi_via_operator`). The result is in physical units, so no side normalization is
            needed.
          - "parametric_approx" (`GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX`): builds a smooth power-law current
            profile from the normalized flux `ψ_N` (0 at the magnetic axis, 1 on the plasma boundary; see
            `j_tor_from_psi_via_parametric_approx`). The result is only meaningful up to a scale factor, so the GS
            sides must be min-max normalized before computing the residuals.

        Parameters
        ----------
        psi_fields : Tensor
            Ground-truth psi fields in physical units, shaped `(F, n_r, n_z)`.
        j_tor_limiter_mask : Tensor | None
            Region where j_tor is derived, shaped `(F, n_r, n_z)` or broadcastable to it; cells outside it get zero
            current. In `compute`, this is the full sides cleaning mask (finite psi/j_tor targets and plasma region).
            Optional. Default: None. However, although it defaults to None, both methods require it: None raises a
            ValueError.

        Returns
        -------
        tuple[Tensor, bool]
            `(j_tor_fields, perform_min_max_normalization)`:
            - j_tor_fields: derived j_tor, shaped `(F, n_r, n_z)`. Built from ground-truth psi only, so it carries no
              gradient to the model, whichever method is used (hence `rhs_gt` must be 0). For "GS_operator", it
              reproduces the ground-truth RHS (up to the clipping of negative values), so the `rhs_gt` residual is
              also degenerate (≈ 0); for "parametric_approx", it generally differs from it. Gradients still reach the
              model through the predicted-psi LHS in the `no_gt` and `lhs_gt` residuals.
            - perform_min_max_normalization: whether the GS sides must be min-max normalized before computing the
              residuals (passed to `grad_shafranov_loss` as `min_max_normalize_sides`). True for "parametric_approx",
              False for "GS_operator".

        Raises
        ------
        ValueError
            If `j_tor_limiter_mask` is None.
        NotImplementedError
            If `self.j_tor_calculation_method` is not supported (cannot happen after `__init__` validation).

        """

        # ..............................................................................................................
        if self.j_tor_calculation_method == GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR:
            if j_tor_limiter_mask is None:
                raise ValueError(
                    f"A valid j_tor limiter mask (e.g., from the LCFS) is needed when j_tor is calculated through "
                    f"the {GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR} method."
                )

            return (
                self.j_tor_from_psi_via_operator(
                    psi_fields=psi_fields,
                    j_tor_limiter_mask=j_tor_limiter_mask.squeeze(0),
                ),
                False,  # -> Side normalization is not required.
            )

        # ..............................................................................................................
        elif self.j_tor_calculation_method == GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX:
            if j_tor_limiter_mask is None:
                raise ValueError(
                    f"A valid j_tor limiter mask (e.g., from the LCFS) is needed when j_tor is approximated through "
                    f"the {GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX} method."
                )
            return (
                self.j_tor_from_psi_via_parametric_approx(
                    psi_fields=psi_fields,
                    j_tor_limiter_mask=j_tor_limiter_mask.squeeze(0),
                ),
                True,  # -> Side normalization is required.
            )

        # ..............................................................................................................
        else:
            raise NotImplementedError

    # ------------------------------------------------------------------------------------------------------------------
    def j_tor_from_psi_via_operator(  # TODO: This must be abstract in a future generic parent class.
        self,
        psi_fields: Tensor,
        j_tor_limiter_mask: Tensor,
    ) -> Tensor:
        """
        Derive j_tor from psi by applying the GS operator and dividing by `μ₀`, i.e.,
        `j_tor = max(0, M ⊙ (-Wψ) / (μ₀))`.

        The sparse GS operator is applied to psi (via `lhs_calculation`), and the result is divided cell by cell by
        `μ₀`; no linear system is solved. The mask `M` is applied first, so cells outside it are zero, and negative
        values (e.g., discretization noise near the boundary) are then clipped, assuming a non-negative plasma current.
        Used when `rhs_input_origin="derived_j_tor"` and `j_tor_calculation_method="GS_operator"`.

        Parameters
        ----------
        psi_fields : Tensor
            Ground-truth psi fields in physical units, shaped `(F, n_r, n_z)`.
        j_tor_limiter_mask : Tensor
            Region where j_tor is derived, shaped `(F, n_r, n_z)` or broadcastable to it (boolean or numeric). In
            `compute`, this is the full sides cleaning mask (finite targets and plasma region).

        Returns
        -------
        Tensor
            Derived j_tor fields in physical units, shaped `(F, n_r, n_z)`, zero outside the mask and clamped to
            [0, ∞). Differentiable with respect to `psi_fields` (zero gradient where clipped).

        Notes
        -----
        In `compute`, this method receives ground-truth psi only, so the derived current density carries no gradient
        to the model (as with any `j_tor_calculation_method`; see `calculate_j_tor_from_psi`). It acts as a fixed
        target for the predicted-psi LHS in the `no_gt` residual, and it can also be used for diagnostics (e.g., to
        check the integrity of ground-truth data).

        """

        numerator = self.lhs_calculation(psi_fields_=psi_fields)
        denominator = mu0

        j_tor = torch.clamp(
            input=j_tor_limiter_mask.to(dtype=numerator.dtype) * (numerator / denominator),
            min=0.0,
        )

        return j_tor  # (F, n_r, n_z)

    # ------------------------------------------------------------------------------------------------------------------
    def j_tor_from_psi_via_parametric_approx(
        self,
        psi_fields: Tensor,
        j_tor_limiter_mask: Tensor,
        alpha_m: float = 2.0,
        alpha_n: float = 2.0,
        beta_m: float = 1.0,
        beta_n: float = 1.0,
        lamda: float = 1e-1,
        beta: float = 0.5,
    ) -> Tensor:
        """
        Derive an approximate j_tor from psi using a parametric (power-law) current profile.

        psi is first converted to a normalized flux `ψ_N` (0 at the magnetic axis, 1 on the plasma boundary; see
        `_psi_n_from_limiter_mask`). The current is then modeled as a blend of the two terms of
        `j_tor = R · p' + FF' / (μ₀ R)`:

            j_p = (1 - ψ_N^alpha_m)^alpha_n · R        (pressure term)
            j_f = (1 - ψ_N^beta_m)^beta_n / R          (poloidal-current, FF', term)
            j_tor = M ⊙ lamda · [beta · j_p + (1 - beta) · j_f] ⊙ P

        where `M` is `j_tor_limiter_mask` and `P` is the plasma region `ψ_N <= 1` (see
        `compute_plasma_region`). Both terms fall from their maximum at the magnetic axis to zero on the boundary.
        `ψ_N` is clamped to [0, 1] inside the power laws, so fractional exponents cannot produce NaN. Used when
        `j_tor_calculation_method="parametric_approx"`.

        Parameters
        ----------
        psi_fields : Tensor
            Ground-truth psi fields in physical units, shaped `(F, n_r, n_z)`. The magnetic axis is assumed to be a
            maximum of psi.
        j_tor_limiter_mask : Tensor
            Region where j_tor is derived, shaped `(F, n_r, n_z)` or broadcastable to it. Used both to locate the
            magnetic axis and plasma boundary, and to mask the output.
        alpha_m, alpha_n : float
            Exponents of the pressure term `j_p`.
            Optional. Default: 2.0, 2.0.
        beta_m, beta_n : float
            Exponents of the poloidal-current term `j_f`.
            Optional. Default: 1.0, 1.0.
        lamda : float
            Overall scale factor.
            Optional. Default: 1e-1.
        beta : float
            Blend weight of the pressure term (`1 - beta` weights the poloidal-current term).
            Optional. Default: 0.5.

        Returns
        -------
        Tensor
            Approximate, non-negative j_tor fields shaped `(F, n_r, n_z)` (dtype from `R_matrix`). Zero outside
            the mask, outside the plasma region (`ψ_N > 1`), and for fields without a peak of psi inside the mask
            (see `_psi_n_from_limiter_mask`). Invariant to the scale of psi, and only meaningful up to a scale factor.

        Notes
        -----
        - This is a smooth approximation, so it may miss sharp profile features that the operator-based method
          (`j_tor_from_psi_via_operator`) captures.
        - If j_tor obtained via this method is used to build the GS equation, then min-max normalization of the GS
          sides is required before the calculation of the GS residuals.
        - The plasma boundary is estimated as for a limited plasma (highest psi on the mask edge). For a diverted
          plasma, the true LCFS is set by the X-point and lies inside the limiter, so this estimate is cruder.

        """

        mask = j_tor_limiter_mask.to(device=psi_fields.device, dtype=torch.bool).expand_as(psi_fields)  # (F, n_r, n_z)

        # psi_n: (F, n_r, n_z); valid_fields: (F, 1, 1)
        psi_n, valid_fields = self._psi_n_from_limiter_mask(psi_fields=psi_fields, mask=mask)

        plasma_region = self.compute_plasma_region(psi_n=psi_n)  # (F, n_r, n_z)

        # Keep the power-law bases in [0, 1], so that fractional exponents cannot produce NaN.
        psi_n_clamped = psi_n.clamp(min=0.0, max=1.0)

        jp = torch.pow((1 - torch.pow(input=psi_n_clamped, exponent=alpha_m)), alpha_n) * self.R_matrix
        jf = torch.pow((1 - torch.pow(input=psi_n_clamped, exponent=beta_m)), beta_n) / self.R_matrix

        j_tor = lamda * (jp * beta + jf * (1 - beta))

        return mask * j_tor * plasma_region * valid_fields

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _psi_n_from_limiter_mask(psi_fields: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """
        Compute the normalized poloidal flux `ψ_N = (ψ_axis - ψ) / (ψ_axis - ψ_boundary)` per field.

        `ψ_axis` is the maximum of psi inside `mask` (the magnetic axis). `ψ_boundary` is the highest psi on the
        mask edge (mask cells with at least one neighbor outside the mask; the grid border counts as outside), i.e.,
        the flux surface of a limited plasma that touches the limiter. Hence, `ψ_N = 0` at the axis and `ψ_N = 1` on
        the boundary.

        Parameters
        ----------
        psi_fields : Tensor
            Poloidal-flux fields shaped `(F, n_r, n_z)`. The magnetic axis is assumed to be a maximum of psi.
        mask : Tensor
            Boolean region mask shaped `(F, n_r, n_z)`.

        Returns
        -------
        tuple[Tensor, Tensor]
            `(psi_n, valid_fields)`:
            - psi_n: normalized flux shaped `(F, n_r, n_z)`, with the dtype of `psi_fields`. Set to 1 on invalid
              fields.
            - valid_fields: boolean tensor shaped `(F, 1, 1)`. False when `ψ_axis <= ψ_boundary`, e.g., for flat
              fields, fields whose maximum lies on the mask edge, or fields with an empty mask.

        Notes
        -----
        - Remaining limitation: sign of psi. The method assumes the magnetic axis is a maximum of psi. If MAST data
          uses the opposite sign, every field would get zero current. That's worth checking on a real shot before using
          parametric_approx.

        """

        lowest = torch.finfo(psi_fields.dtype).min

        # Magnetic axis: maximum of psi inside the mask.
        psi_axis = torch.where(mask, psi_fields, lowest).amax(dim=(-2, -1), keepdim=True)  # (F, 1, 1)

        # Mask edge: cells inside the mask with at least one neighbor outside it (the grid border counts as outside).
        outside = torch_functional.pad((~mask).float().unsqueeze(1), pad=(1, 1, 1, 1), value=1.0)  # (F, 1, ., .)
        edge = mask & torch_functional.max_pool2d(outside, kernel_size=3, stride=1).squeeze(1).bool()  # (F, n_r, n_z)

        # Boundary of a limited plasma: the highest psi contour that touches the mask edge.
        psi_boundary = torch.where(edge, psi_fields, lowest).amax(dim=(-2, -1), keepdim=True)  # (F, 1, 1)

        span = psi_axis - psi_boundary
        valid_fields = span > 0  # (F, 1, 1)
        psi_n = torch.where(valid_fields, (psi_axis - psi_fields) / torch.where(valid_fields, span, 1.0), 1.0)

        return psi_n, valid_fields

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def min_max_normalization(x, nan_mode=False):
        """Scale a NumPy array or tensor to the interval from zero to one.

        Parameters
        ----------
        x : array-like
            Values to normalize.
        nan_mode : bool
            If true, use NumPy NaN-aware extrema.

        Returns
        -------
        array-like
            `(x - min(x)) / (max(x) - min(x))` in the input representation.
        """

        min_ = np.nanmin(x) if nan_mode else x.min()
        max_ = np.nanmax(x) if nan_mode else x.max()

        return (x - min_) / (max_ - min_)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mean_std_normalization(x: Tensor, mean_scaling: float = 1.0, std_scaling: float = 1.0) -> Tensor:
        """
        Normalize a tensor field by its (scaled) mean and standard deviation, i.e.,
        `(x - mean_scaling · mean(x)) / (std_scaling · std(x))`.

        A single mean and standard deviation are computed over all elements of `x`. The standard deviation is the
        unbiased one (`torch.std`, divisor `N - 1`). Both are treated as constants (detached), so gradients flow
        only through the explicit `x` terms.

        Parameters
        ----------
        x : Tensor
            Tensor field to normalize, of any shape (e.g., one `(n_r, n_z)` field).
        mean_scaling, std_scaling : float
            Multipliers applied to the field mean and standard deviation before normalization.
            Optional. Default: 1.0, 1.0.

        Returns
        -------
        Tensor
            Normalized tensor with the same shape and dtype as `x`.

        Raises
        ------
        ValueError
            If the scaled standard deviation is zero (e.g., a constant field).

        """

        mean_x = mean_scaling * torch.mean(input=x).detach()
        std_x = std_scaling * torch.std(input=x).detach()

        if std_x == 0:
            raise ValueError("[mean_std_normalization] Standard deviation is zero; cannot normalize a constant field.")

        return (x - mean_x) / std_x

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_plasma_region(psi_n: Tensor) -> Tensor:
        """
        Return the binary plasma-region mask from normalized poloidal flux, i.e., cells where `ψ_N <= 1`.

        Parameters
        ----------
        psi_n : Tensor
            Normalized poloidal flux (`ψ_N`: 0 at the magnetic axis, 1 on the plasma boundary), e.g., from
            `_psi_n_from_limiter_mask`.

        Returns
        -------
        Tensor
            float32 mask with the same shape as `psi_n`: 1 where `psi_n <= 1`, else 0 (NaN and +inf give 0). Carries
            no gradient.

        """

        return psi_n.le(1).float()

    # ------------------------------------------------------------------------------------------------------------------
    def j_tor_from_pprime_ffprime_psi(
        self,
        pprime_native: Tensor,
        ffprime_native: Tensor,
        psi_native: Tensor,
        psi_norm_native: Tensor,
        mag_r_axis,
        mag_z_axis,
        xpt_r,
        xpt_z,
    ) -> Tensor:
        """Estimate toroidal current from pressure/current profiles and poloidal flux.

        Parameters
        ----------
        pprime_native, ffprime_native : Tensor
            Profile samples for pressure and `FF'` derivatives.
        psi_native, psi_norm_native : Tensor
            Poloidal flux and its profile-normalization coordinate.
        mag_r_axis, mag_z_axis, xpt_r, xpt_z : array-like
            EFIT magnetic-axis and X-point geometry used to construct `psi_N`.

        Returns
        -------
        Tensor
            Estimated toroidal-current field.

        Notes
        -----
        This legacy path is currently disabled in `compute`. It mixes NumPy and Torch operations, is not
        differentiable, and does not satisfy the vectorized `(F, n_r, n_z)` field contract.
        """

        # TODO(FIX): This predicted-profiles -> j_tor path is NOT functional and is currently disabled in compute().
        #
        # PROBLEM:
        #   - It mixes NumPy ops (np.interp, np.clip, np.nanmin) with torch tensors that require grad: this breaks
        #     autograd (the graph is detached) and will error for multi-dimensional / GPU tensors.
        #   - It needs EFIT geometry (magnetic axis, X-points) to normalize psi_N, but compute() has no such
        #     metadata and passes mag_r_axis/mag_z_axis/xpt_r/xpt_z = None, so compute_psi_n_from_efit_geometry
        #     cannot run.
        #   - It operates on a single (H, W) field and ignores the (F, n_r, n_z) multi-time field contract used by
        #     the rest of the loss.
        #
        # PROPOSED SOLUTION:
        #   - Compute psi_N differentiably from psi itself (magnetic axis = arg-extremum of psi; boundary = LCFS /
        #     limiter contact) so no EFIT metadata is required; or plumb EFIT axis/boundary through the batch.
        #   - Replace np.interp with a differentiable torch interpolation (e.g. torch.searchsorted + linear blend)
        #     of pprime/ffprime over psi_N so gradients flow to the predictions.
        #   - Vectorize over the F = B*T fields and return j_tor as (F, n_r, n_z), matching the other RHS sources
        #     (see j_tor_from_psi_via_operator).

        psi_n_efit_zr, _, _ = self.compute_psi_n_from_efit_geometry(
            psi_zr=psi_native,
            mag_r_axis=mag_r_axis,
            mag_z_axis=mag_z_axis,
            xpt_r=xpt_r,
            xpt_z=xpt_z,
        )

        psi_n_eval = np.clip(
            a=psi_n_efit_zr,
            a_min=float(np.nanmin(psi_norm_native)),
            a_max=float(np.nanmax(psi_norm_native)),
        )

        pprime_zr = np.interp(
            x=psi_n_eval,
            xp=psi_native,
            fp=pprime_native,
        )

        ffprime_zr = np.interp(
            x=psi_n_eval,
            xp=psi_native,
            fp=ffprime_native,
        )

        # TODO: Check next expression (is it correct adding .T?)
        j_tor_zr = self.R_matrix.T * pprime_zr + ffprime_zr / (mu0 * self.R_matrix.T)

        return j_tor_zr

    # ------------------------------------------------------------------------------------------------------------------
    def compute_psi_n_from_efit_geometry(  # TODO: Check this method after fixing j_tor_from_pprime_ffprime_psi method.
        self,
        psi_zr,
        mag_r_axis,
        mag_z_axis,
        xpt_r,
        xpt_z,
    ):
        """
        Compute psi_N using EFIT magnetic axis and first valid X-point.

        This is the simple version used for the first loss check. For training, we may use EFIT-provided axis/boundary
        metadata directly as batch metadata.

        """

        psi_axis = self.nearest_grid_value(field_zr=psi_zr, r0=mag_r_axis, z0=mag_z_axis)

        valid_xpt = np.isfinite(xpt_r) & np.isfinite(xpt_z) & (xpt_r > 0)
        if not valid_xpt.any():
            raise RuntimeError("No valid X-point found for psi_N normalization.")

        psi_sep = self.nearest_grid_value(field_zr=psi_zr, r0=float(xpt_r[valid_xpt][0]), z0=float(xpt_z[valid_xpt][0]))
        if abs(psi_sep - psi_axis) < 1e-12:
            raise RuntimeError("Degenerate psi_N normalization: psi_sep == psi_axis.")

        return (psi_zr - psi_axis) / (psi_sep - psi_axis), psi_axis, psi_sep

    # ------------------------------------------------------------------------------------------------------------------
    def nearest_grid_value(self, field_zr, r0, z0):
        """Read a 2D field at the nearest R/Z grid point."""

        ir = int(np.argmin(np.abs(self.R_matrix - r0)))
        iz = int(np.argmin(np.abs(self.Z_matrix - z0)))

        return float(field_zr[iz, ir])


# ----------------------------------------------------------------------------------------------------------------------

# ======================================================================================================================
if __name__ == "__main__":
    print("WeakFormGradShafranovLoss (residual, bounded).")
