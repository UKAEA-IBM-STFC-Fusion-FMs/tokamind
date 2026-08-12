"""
Grad-Shafranov residual loss in native space.

GradShafranovResidualLoss computes physics-informed loss by enforcing the Grad-Shafranov equation: the sparse
diffemputed in native (destandardized) signal space with NaN-masking for sparse measurements.  # FIXME: Check wording.

Key features
------------
- **Decoding & destandardization**: Predictions arrive in embedding space and are decoded to native space via
  per-signal TorchDecoder instances, then destandardized to physical units before GS calculations.
- **Sparse masking**: NaN positions in ground-truth targets are excluded from loss computation, only valid measurements
  contribute. Per-field validity checks skip entirely-NaN fields.
- **Flexible RHS sources**: Three modes for computing j_tor on the RHS:
  - Derived from ground-truth psi (via sparse operator or parametric approximation)
  - Predicted j_tor directly from model outputs
  - Derived from predicted profiles (pprime, ffprime) — currently unimplemented
- **Three loss terms**: Weighted combination of (1) LHS-RHS residual (no ground truth), (2) LHS vs ground-truth LHS,
  (3) RHS vs ground-truth RHS, allowing flexible supervision strategies.
- **Gradient flow**: Gradients flow through decoders into model predictions; decoder parameters (frozen VAE weights,
  fixed indices) remain fixed.

Configuration
-------------rential operator applied to predicted psi must match the RHS computed from toroidal current (j_tor). Loss is
co
Expects config with:

.code-block:: python

    train.loss_aggregator.losses[*]:
      name: "grad_shafranov"
      weight: <float>
      grad_shafranov_params_file: <path>      # .npz asset with GS operator, R/Z grids, limiter masks
      rhs_input: "predicted_j_tor" | "derived_j_tor" | "predicted_profiles"
      j_tor_calculation_method: "via_gs_operator" | "via_parametric_approx"  # for derived_j_tor
      grad_shafranov_weights: {no_gt: 0.34, lhs_gt: 0.33, rhs_gt: 0.33}
      output_filter: <set of output signal_ids> | null  # optional: restrict to subset of outputs
      plot_check_cfg:  # optional: visualize GS residuals during training
        type: "save_plots" | "show_plots" | null
        probability: <float>  # fraction of batches to plot

Data requirements
-----------------
- `data.keep_output_native=True` in dataset config to provide ground-truth native targets.
- Output signals "equilibrium-psi" (always required) and others depend on ``rhs_input`` mode.
- Per-window metadata in batch (signal stats for destandardization, optional LCFS/geometry data).

Implementation notes
--------------------
- All GS calculations operate on (F, n_r, n_z) fields where F = B*T (batch × forecast times),
  vectorized in a single sparse matmul per epoch.
- Predicted/ground-truth psi and j_tor are transposed from model native shape (B, H, W, T) to
  match GS operator orientation.
- Limiter masks filter valid regions; LCFS masks (if available) provide time-varying physics
  constraints; static base masks are fallback.
- TODO:
  - 'predicted-profiles' pipeline is NOT ported to the field contract: it now raises NotImplementedError instead of
    running silently-wrong. It requires differentiable psi_N computation and torch.searchsorted interpolation (currently
    disabled).

"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from matplotlib.path import Path as MatplotlibPath
from typing import Hashable, Any, Literal
import numpy as np
import scipy.sparse as sp

import torch
from torch import Tensor
import torch.nn.functional as torch_functional
from torchvision import transforms

from mmt.data.embeddings.torch_decoder import TorchDecoder
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
from .plots import make_gs_plots
from mmt.train.losses.constants import (
    GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR,
    GRAD_SHAFRANOV_J_TOR_VIA_PARAMETRIC_APPROX,
    GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS,
    GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGINS,
    GRAD_SHAFRANOV_RHS_KEYS,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR,
    GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES,
)
from mmt.data.standardization import destandardize_torch
from mmt.utils.paths import REPO_ROOT


# ----------------------------------------------------------------------------------------------------------------------
# Preliminaries

# Opt out of invariant checks
torch.sparse.check_sparse_tensor_invariants.disable()

# Vacuum permeability `mu0` is imported from the local ``helpers`` module (shared with the weak-form loss).
DEFAULT_GRAD_SHAFRANOV_WEIGHTS: dict[str, float] = {"no_gt": 0.34, "lhs_gt": 0.33, "rhs_gt": 0.33}


# ======================================================================================================================
class GradShafranovResidualLoss(BaseLoss):
    """
    Physics-informed loss enforcing the Grad-Shafranov equilibrium equation in native, destandardized space.

    TODO: Check (compiled by Claude)

    Computes residuals of the form: ||Δ*ψ - μ₀·R·j_tor|| over tokamak equilibrium grids, where ψ (psi)
    and j_tor are decoded from model predictions. Loss combines three weighted terms:
    (1) LHS–RHS residual (no ground truth),
    (2) LHS vs ground-truth LHS,
    (3) RHS vs ground-truth RHS,
    allowing flexible supervision strategies.

    Predictions arrive in embedding space and are decoded to native space via per-signal `TorchDecoder` instances, then
    destandardized to physical units before GS operator application. Ground-truth native targets come from
    `batch['output_native']`, which requires `data.keep_output_native=True` in config.

    Sparse measurements: NaN values in ground truth represent missing data (e.g., unmeasured plasma regions). Loss is
    computed only over valid (non-NaN) positions; entirely-NaN fields are skipped. Per-field validity checks prevent
    NaN propagation through gradients.

    Gradient flow: Gradients flow through decoders into model predictions; decoder parameters (frozen VAE weights,
    fixed indices) remain static.

    Parameters
    ----------
    decoders : dict[Hashable, TorchDecoder]
        Per-signal differentiable decoders, keyed by signal_id. Must include "equilibrium-psi" at minimum.
    signal_stats : Mapping[str, Mapping[str, Any]]
        Per-signal statistics (mean, std) for destandardization. Keyed by signal name (str).
    output_name_to_id : Mapping[str, Hashable]
        Mapping from output signal names (e.g., "equilibrium-psi") to numeric signal IDs.
    grad_shafranov_params_file : str | Path | None
        Path to `.npz` asset file containing sparse GS operator, R/Z grids, and limiter masks.
        Required. Absolute or repo-relative path.
    grad_shafranov_weights : dict[Hashable, float] | None
        Weights for the three loss terms: {"no_gt": w1, "lhs_gt": w2, "rhs_gt": w3}.
        Optional. Default: {"no_gt": 0.34, "lhs_gt": 0.33, "rhs_gt": 0.33}.
    rhs_input : str | None
        RHS computation mode: "predicted_j_tor", "derived_j_tor", or "predicted_profiles".
        Optional. Default: "predicted_j_tor".
    j_tor_calculation_method : str | None
        For derived_j_tor mode: "via_gs_operator" or "via_parametric_approx".
        Optional. Default: "via_gs_operator".
    output_weights : dict[Hashable, float] | None
        Deprecated. Per-output scalar weights (not currently used).  # TODO: Check this.
    output_filter : set[Hashable] | None
        Optional: restrict loss to subset of output signal IDs. Must include all required RHS outputs.
    plot_check_cfg : Mapping[str, Any] | None
        Optional plotting config: {"type": "save_plots"|"show_plots", "probability": float, ...}.
        When set, visualizes LHS/RHS residuals during training on deterministically selected batches.

    Attributes
    ----------
    gs_op_coo : torch.sparse_coo_tensor | None
        Sparse Grad-Shafranov operator (loaded from asset file).
    R_matrix, Z_matrix : torch.Tensor | None
        (n_r, n_z) equilibrium grid coordinates.
    n_r, n_z : int | None
        Grid dimensions.

    Methods
    -------
    TODO

    """

    requires_native_target: bool = True
    requires_decode: bool = True
    requires_destandardize: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    @classmethod
    def validate_term_cfg(cls, term_def: Mapping[str, Any], path: str) -> None:
        """
        Validate config fields owned by the strong Grad-Shafranov residual loss.

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
            If a required field is missing or an unknown nested key is provided.
        TypeError
            If a field has the wrong type.
        ValueError
            If a field has an unsupported value.

        """

        cls._validate_known_term_keys(
            term_def=term_def,
            path=path,
            allowed_specific_keys={
                "grad_shafranov_params_file",
                "grad_shafranov_weights",
                "rhs_input",
                "loss_type",
                "plot_check",
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

        unknown_rhs_keys = sorted(str(key) for key in rhs_input.keys() if key not in GRAD_SHAFRANOV_RHS_KEYS)
        if unknown_rhs_keys:
            raise KeyError(
                f"Unknown {path}.rhs_input keys: {unknown_rhs_keys}. "
                f"Supported keys are {sorted(GRAD_SHAFRANOV_RHS_KEYS)}."
            )

        rhs_input_origin = str(
            rhs_input.get(GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY, GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR)
        )
        if rhs_input_origin not in GRAD_SHAFRANOV_RHS_INPUT_ORIGINS:
            raise ValueError(
                f"{path}.rhs_input.origin={rhs_input_origin!r} is unsupported. "
                f"Supported: {sorted(GRAD_SHAFRANOV_RHS_INPUT_ORIGINS)}."
            )

        rhs_input_calculation_method = str(
            rhs_input.get(GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY, GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR)
        )
        if rhs_input_calculation_method not in GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS:
            raise ValueError(
                f"{path}.rhs_input.calculation_method={rhs_input_calculation_method!r} is unsupported. "
                f"Supported: {sorted(GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS)}."
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
            allowed_keys=set(DEFAULT_GRAD_SHAFRANOV_WEIGHTS),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        decoders: dict[Hashable, TorchDecoder],
        signal_stats: Mapping[str, Mapping[str, Any]],
        output_name_to_id: Mapping[str, Hashable],
        grad_shafranov_params_file: str | Path | None,
        grad_shafranov_weights: dict[Hashable, float] | None = None,
        rhs_input: str | None = None,
        j_tor_calculation_method: str | None = None,
        loss_type: Literal["l2", "mse"] = "mse",
        output_weights: dict[Hashable, float] | None = None,
        output_filter: set[Hashable] | None = None,
        plot_check_cfg: Mapping[str, Any] = None,
    ) -> None:

        if not decoders:
            raise ValueError("GradShafranovResidualLoss requires at least one decoder.")

        if grad_shafranov_params_file is None:
            raise ValueError("GradShafranovResidualLoss requires grad_shafranov_params_file.")

        if not isinstance(grad_shafranov_params_file, (str, Path)):
            raise TypeError(
                f"`grad_shafranov_params_file` must be a str or Path, got {type(grad_shafranov_params_file).__name__}."
            )

        rhs_input = rhs_input or GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR
        if rhs_input not in GRAD_SHAFRANOV_RHS_INPUT_ORIGINS:
            raise ValueError(
                f"Unsupported `rhs_input={rhs_input!r}`. Supported: {sorted(GRAD_SHAFRANOV_RHS_INPUT_ORIGINS)}."
            )

        j_tor_calculation_method = j_tor_calculation_method or GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR
        if j_tor_calculation_method not in GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS:
            raise ValueError(
                f"Unsupported `j_tor_calculation_method={j_tor_calculation_method!r}`. "
                f"Supported: {sorted(GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS)}."
            )

        self._runtime_device: torch.device | None = None
        self._decoders = decoders
        self._output_weights = output_weights or {}
        self.gs_weights = {**DEFAULT_GRAD_SHAFRANOV_WEIGHTS, **(grad_shafranov_weights or {})}
        for gs_weight in self.gs_weights.values():
            if not 0 <= gs_weight <= 1:
                raise ValueError(
                    "[GradShafranovResidualLoss] all weights in `grad_shafranov_weights` must be between 0 and 1."
                )

        self.rhs_input = rhs_input
        self.j_tor_calculation_method = j_tor_calculation_method
        self._output_filter = set(output_filter) if (output_filter is not None) else None
        self._output_name_to_id = {str(name): sid for name, sid in output_name_to_id.items()}
        self.signal_stats = {str(name): dict(stats) for name, stats in signal_stats.items()}

        if loss_type not in ("l2", "mse"):
            raise ValueError(
                f"[GradShafranovResidualLoss] Invalid `loss_type`: must be in ['l2', 'mse'], got '{loss_type}'."
            )
        self.loss_type = loss_type

        self._plot_check_cfg = plot_check_cfg or {}
        self._all_losses_weights = self._plot_check_cfg.get("all_losses_weights", {"NA": "NA"})
        self._plot_check_type = self._plot_check_cfg.get("type", None)  # Options: "show_plots", "save_plots", None.
        self._plot_check_probability = float(self._plot_check_cfg.get("probability", 0.0))

        self._use_case_suffix = ""
        self._use_case_suffix_latex = ""
        self._build_plot_strings()

        # ..............................................................................................................
        # Set relevant output keys

        self.required_output_names = self._required_output_names_for_rhs(rhs_input=rhs_input)
        self.required_output_keys = [
            resolve_output_key(self._output_name_to_id, name, loss_name="Grad-Shafranov loss")
            for name in self.required_output_names
        ]
        self._name_by_key = {key: name for name, key in self._output_name_to_id.items()}
        self._psi_key = resolve_output_key(self._output_name_to_id, "equilibrium-psi", loss_name="Grad-Shafranov loss")
        self._j_tor_key = self._output_name_to_id.get("equilibrium-j_tor")
        self._pprime_key = self._output_name_to_id.get("equilibrium-dpressure_dpsi")
        self._ffprime_key = self._output_name_to_id.get("equilibrium-f_df_dpsi")

        if self._output_filter is not None:
            missing_filter_names = [
                name
                for name, key in zip(self.required_output_names, self.required_output_keys, strict=True)
                if key not in self._output_filter
            ]
            if missing_filter_names:
                raise ValueError(
                    "Grad-Shafranov loss output filter must include all outputs required by `rhs_input.origin="
                    f"{rhs_input!r}: {missing_filter_names}`."
                )

        # ..............................................................................................................
        # Load Grad-Shafranov parameters

        self.gs_op_coo: Tensor | None = None

        self.n_r = None
        self.R_matrix: Tensor | None = None

        self.n_z = None
        self.Z_matrix: Tensor | None = None

        self.rz_points_for_lcfs_calculation: Tensor | None = None
        self.zero_lcfs_rz_mask: Tensor | None = None

        self.base_j_tor_limiter_mask_rz: Tensor | None = None
        self.mast_limiter_mask_rz: Tensor | None = None

        self._load_gs_params(filename=grad_shafranov_params_file)

    # ------------------------------------------------------------------------------------------------------------------
    def _build_plot_strings(self) -> None:
        """Build the loss-configuration summary shown in diagnostic plots."""

        if self._plot_check_type is not None:
            self._use_case_suffix = "j_tor "
            self._use_case_suffix_latex = r"Output: ($\psi^{pred}$"
            if self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
                self._use_case_suffix += "pred,"
                self._use_case_suffix_latex += r", $j^{pred}_\phi$)"
            else:
                self._use_case_suffix += f"via {self.j_tor_calculation_method}, "
                self._use_case_suffix_latex += rf"), $j^{{approx}}_\phi$ via {self.j_tor_calculation_method}"

            losses_weights = ""
            losses_weights_latex = ""
            for kk, vv in self._all_losses_weights.items():
                losses_weights += f" {''.join(i[0].upper() for i in kk.split('_'))}_{str(vv)}"
                losses_weights_latex += f", w_{''.join(i[0].upper() for i in kk.split('_'))}: {str(vv)}"

            gs_weights = ""
            gs_weights_latex = ""
            if self._all_losses_weights["grad_shafranov_residual"] > 0:
                gs_weights += ", GSR["
                gs_weights_latex += ", GSR["

                for kk, vv in self.gs_weights.items():
                    if vv > 0:
                        gs_weights += f"{kk[0]}_{str(vv)} "
                        gs_weights_latex += f"{kk}: {str(vv)}, "

                gs_weights = gs_weights[:-1] + "]"
                gs_weights_latex = gs_weights_latex[:-2] + "]"

            self._use_case_suffix += losses_weights + gs_weights
            self._use_case_suffix_latex += losses_weights_latex + gs_weights_latex

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _required_output_names_for_rhs(rhs_input: str) -> list[str]:
        """
        Return output names required to build the RHS of the specified Grad-Shafranov equation arrangement.

        Parameters
        ----------
        rhs_input : str
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

        if rhs_input == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR:
            return ["equilibrium-psi"]
        if rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
            return ["equilibrium-psi", "equilibrium-j_tor"]
        if rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES:
            return ["equilibrium-psi", "equilibrium-dpressure_dpsi", "equilibrium-f_df_dpsi"]

        raise ValueError(f"Unsupported `rhs_input={rhs_input!r}`.")

    # ------------------------------------------------------------------------------------------------------------------
    def _load_gs_params(
        self,
        filename: str | Path,
    ) -> None:
        """
        Load sparse Grad-Shafranov operator assets from disk.

        Parameters
        ----------
        filename : str | Path
            Absolute path or repository-relative path to a `.npz` asset file.

        Returns
        -------
        None

        """

        with np.load(resolve_gs_asset_path(filename, REPO_ROOT)) as loaded_data:
            grid_assets = parse_gs_grid_assets(loaded_data)
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

        self.rz_points_for_lcfs_calculation = torch.column_stack(
            [self.R_matrix.ravel(), self.Z_matrix.ravel()]  # noqa - Ignore missing attribute warning
        )
        self.zero_lcfs_rz_mask = torch.zeros_like(self.R_matrix, dtype=torch.bool)  # noqa - Ignore missing attribute

        self.base_j_tor_limiter_mask_rz = grid_assets.base_j_tor_limiter_mask_rz

    # ------------------------------------------------------------------------------------------------------------------
    def rhs_calculation(
        self,
        j_tor_fields_,
        cleaning_mask_=None,
    ):
        """
        Compute the strong-form Grad-Shafranov right-hand side `mu0 * R * j_tor`.

        Parameters
        ----------
        j_tor_fields_ : Tensor
            Toroidal-current fields with shape `(F, n_r, n_z)`.
        cleaning_mask_ : Tensor
            Boolean or numeric physical-domain mask broadcastable to `j_tor_fields_`.
            Optional. Default: `None`.

        Returns
        -------
        Tensor
            Masked (if required) right-hand-side fields with shape `(F, n_r, n_z)`.

        """

        # Perform point-wise mu0 * R * j_tor in float32. CUDA sparse GS ops do not support bf16.
        gs_rhs = mu0 * self.R_matrix * j_tor_fields_.to(torch.float32)  # (F, n_r, n_z)

        if cleaning_mask_ is None:
            return gs_rhs  # (F, n_r, n_z)
        else:
            return cleaning_mask_.to(dtype=gs_rhs.dtype) * gs_rhs  # (F, n_r, n_z)

    # ------------------------------------------------------------------------------------------------------------------
    def lhs_calculation(
        self,
        psi_fields_,
        cleaning_mask_=None,
    ):
        """
        Apply the sparse Grad-Shafranov operator to a stack of psi fields.

        Parameters
        ----------
        psi_fields_ : Tensor
            Poloidal-flux fields with shape `(F, n_r, n_z)`.
        cleaning_mask_ : Tensor
            Boolean or numeric physical-domain mask broadcastable to `psi_fields_`.

        Returns
        -------
        Tensor
            Masked (if required) `-Delta*psi` fields with shape `(F, n_r, n_z)`.

        """

        # Apply the sparse GS operator to every flattened psi field in a single matmul.
        # Per field this equals (-gs_op @ psi.ravel()).reshape(n_r, n_z) from a per-sample iteration.

        n_fields_ = psi_fields_.shape[0]
        psi_flat = psi_fields_.to(torch.float32).reshape(
            n_fields_, self.n_r * self.n_z
        )  # (F, N), row-major matches the per-field ravel
        gs_lhs = (
            -torch.sparse.mm(  # -> mm: Matrix multiplication
                self.gs_op_coo, psi_flat.t()
            )
        ).t()
        gs_lhs = gs_lhs.reshape(n_fields_, self.n_r, self.n_z)  # (F, n_r, n_z)

        if cleaning_mask_ is None:
            return gs_lhs  # (F, n_r, n_z)
        else:
            return cleaning_mask_.to(dtype=gs_lhs.dtype) * gs_lhs  # (F, n_r, n_z)

    # ------------------------------------------------------------------------------------------------------------------
    def _check_device_and_dtype(
        self,
        ref: Tensor,
    ):
        """Move cached Grad-Shafranov tensors to the runtime device.

        Parameters
        ----------
        ref : Tensor
            Runtime tensor whose device should be matched. Grad-Shafranov operator tensors stay float32 because
            CUDA sparse matrix multiplication does not support bfloat16.

        Returns
        -------
        None
        """

        _device = ref.device

        if self._runtime_device != _device:
            if self.gs_op_coo is not None:
                self.gs_op_coo = runtime_tensor(self.gs_op_coo, ref=ref, dtype=torch.float32).coalesce()

            if self.R_matrix is not None:
                self.R_matrix = runtime_tensor(self.R_matrix, ref=ref, dtype=torch.float32)

            if self.base_j_tor_limiter_mask_rz is not None:
                self.base_j_tor_limiter_mask_rz = runtime_tensor(self.base_j_tor_limiter_mask_rz, ref=ref)

            self._runtime_device = _device

    # ------------------------------------------------------------------------------------------------------------------
    def _make_lcfs_mask(
        self,
        lcfs_r: Tensor,
        lcfs_z: Tensor,
    ) -> Tensor:
        """
        Build a boolean mask of points inside LCFS.

        The LCFS contour is treated as a polygon in (R, Z) and tested over the rectangular equilibrium mesh.
        """

        valid_l = torch.isfinite(input=lcfs_r) & torch.isfinite(input=lcfs_z) & (lcfs_r > 0)
        if valid_l.sum() < 3:
            if self.zero_lcfs_rz_mask is not None:
                return self.zero_lcfs_rz_mask
            else:
                raise RuntimeError("Grad-Shafranov matrices R or Z were not properly loaded.")

        lcfs_rz_path = MatplotlibPath(np.column_stack([lcfs_r[valid_l], lcfs_z[valid_l]]))

        if self.rz_points_for_lcfs_calculation is not None:
            return torch.tensor(
                lcfs_rz_path.contains_points(points=self.rz_points_for_lcfs_calculation).reshape(self.R_matrix.shape)  # noqa - Ignore missing attribute warning
            )
        else:
            raise RuntimeError("Grad-Shafranov matrices R or Z were not properly loaded.")

    # ------------------------------------------------------------------------------------------------------------------
    def run_plot_checks(
        self,
        gs_lhs_pred: Tensor,
        gs_rhs_pred: Tensor,
        gs_lhs_ref: Tensor,
        psi_fields_for_pred_lhs: Tensor,
        psi_fields_gt_destdized: Tensor,
        j_tor_fields_for_pred_rhs: Tensor,
        j_tor_fields_gt_destdized: Tensor | None,
        cleaning_mask: Tensor,
        context_info: LossComputeContext | None,
        plot_single_slice: bool = True,
    ):
        """Optionally emit strong-form diagnostic plots for selected field slices.

        Parameters
        ----------
        gs_lhs_pred, gs_rhs_pred, gs_lhs_ref : Tensor
            Predicted and reference Grad-Shafranov sides, each shaped ``(F, n_r, n_z)``.
        psi_fields_for_pred_lhs, psi_fields_gt_destdized : Tensor
            Predicted and target poloidal-flux fields.
        j_tor_fields_for_pred_rhs : Tensor
            Toroidal-current fields used for the predicted right-hand side.
        j_tor_fields_gt_destdized : Tensor | None
            Target toroidal-current fields, if available.
        cleaning_mask : Tensor
            Physical-domain mask applied when deriving the reference right-hand side.
        context_info : LossComputeContext | None
            Run, stage, and batch metadata used when saving plots.
        plot_single_slice : bool
            If true, plot the deterministically selected field slice; otherwise plot every slice from a selected batch.

        Returns
        -------
        None
        """

        plot_index = select_diagnostic_plot_slice(
            context=context_info,
            probability=self._plot_check_probability,
            n_fields=psi_fields_gt_destdized.shape[0],
            diagnostic_name="grad_shafranov_residual",
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

            gs_rhs_ref = torch.nan * psi_fields_gt_destdized
            if j_tor_fields_gt_destdized is not None:
                gs_rhs_ref = self.rhs_calculation(
                    j_tor_fields_=j_tor_fields_gt_destdized,
                    cleaning_mask_=cleaning_mask,
                )

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
                        "title_sufix": self._use_case_suffix_latex,
                        "gs_data": {
                            "lhs_pred_data": gs_lhs_pred[ii, :, :].detach(),
                            "lhs_ref_data": gs_lhs_ref[ii, :, :].detach(),
                            "rhs_pred_data": gs_rhs_pred[ii, :, :].detach(),
                            "rhs_ref_data": gs_rhs_ref[ii, :, :].detach(),
                        },
                        "signal_data": {
                            "psi_pred_data": psi_fields_for_pred_lhs[ii, :, :].detach(),
                            "psi_ref_data": psi_fields_gt_destdized[ii, :, :].detach(),
                            "j_tor_pred_data": j_tor_fields_for_pred_rhs[ii, :, :].detach(),
                            "j_tor_ref_data": j_tor_fields_gt_destdized[ii, :, :].detach(),
                            "j_tor_case": j_tor_case,
                        },
                    },
                    save_plots=(self._plot_check_type == "save_plots"),
                    save_path=(
                        training_plot_path(context_info, slice_index=ii)
                        if self._plot_check_type == "save_plots"
                        else None
                    ),
                )

    # ------------------------------------------------------------------------------------------------------------------
    def get_destdized_fields_for_gs_eq(
        self,
        stdized_native_data,
        key: Literal["psi", "j_tor"],
    ):
        """Convert predicted and target native tensors into safe physical GS fields.

        Parameters
        ----------
        stdized_native_data : Mapping[str, Mapping[Hashable, Tensor]]
            ``pred`` and ``true`` native standardized tensors keyed by output signal ID.
        key : {"psi", "j_tor"}
            Equilibrium field to prepare.

        Returns
        -------
        tuple[dict[str, Tensor], Tensor, torch.Size]
            Destandardized prediction/target fields shaped ``(F, n_r, n_z)``, the target finite-cell mask, and the
            original predicted ``(B, H, W, T)`` shape.

        Raises
        ------
        ValueError
            If ``key`` is not a supported Grad-Shafranov field.
        """

        # ..............................................................................................................
        # Fold (B, T) into a single field axis F = B * T, then (F, n_r, n_z) fields.
        # Decoded predictions have native shape (B, H, W, T); T is the forecast-horizon time axis.
        #
        # REMARK: For Grad-Shafranov related calculations, all data must be destandardized and in native space.
        # ..............................................................................................................

        if key not in ["psi", "j_tor"]:
            raise ValueError(f"Invalid key {key}. Valid options: ['psi', 'j_tor'].")

        # TODO: Try to unify the following two lines
        signal_key = self._psi_key if key == "psi" else self._j_tor_key
        signal_key_srt = "equilibrium-psi" if key == "psi" else "equilibrium-j_tor"

        # Get standardized signal fields
        signal_fields_stdized = {
            "pred": native_to_gs_fields(
                x_bhwt=stdized_native_data["pred"][signal_key]  # (B, H, W, T), standardized native
            ),  # (F, n_r, n_z)
            "true": native_to_gs_fields(
                x_bhwt=stdized_native_data["true"][signal_key]  # (B, H, W, T), standardized native
            ),  # (F, n_r, n_z),
        }

        # Predictions are already safe. Targets are made finite before any GS operator is applied; the returned mask
        # still excludes their original non-finite cells from the physical residual domain.
        signal_fields_destdized = {
            "pred": destandardize_torch(
                x=signal_fields_stdized["pred"],
                mean=self.signal_stats[signal_key_srt]["mean"],
                std=self.signal_stats[signal_key_srt]["std"],
            ),
        }
        signal_fields_destdized["true"], no_nan_mask = prepare_target_field(
            signal_fields_stdized["true"], self.signal_stats[signal_key_srt]
        )

        bhwt_shape = stdized_native_data["pred"][signal_key].shape

        return signal_fields_destdized, no_nan_mask, bhwt_shape

    # ------------------------------------------------------------------------------------------------------------------
    def _get_j_tor_limiter_mask(
        self,
        lcfs_true,  # {"lcfs_r": lcfs_r_data, "lcfs_rz": lcfs_z_data} # TODO: Check if this data should be destandardized
        j_tor_true_destdized,
        num_fields,
    ):
        """Select the plasma-region mask used by the strong-form current terms.

        The method prefers a time-varying LCFS polygon, then derives a mask from target ``j_tor``, and finally falls
        back to the static limiter mask stored in the Grad-Shafranov asset.

        Parameters
        ----------
        lcfs_true : Mapping[str, Tensor | None]
            Optional LCFS R/Z coordinate arrays.
        j_tor_true_destdized : Tensor | None
            Optional physical target current fields shaped ``(F, n_r, n_z)``.
        num_fields : int
            Number of flattened batch/time fields required by the static fallback.

        Returns
        -------
        Tensor
            Limiter/plasma mask with shape ``(F, n_r, n_z)`` or a broadcastable static mask.
        """

        # ..............................................................................................................
        # Build a j_tor limiter mask
        # ..............................................................................................................

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # First option (best possible): From time-varying lcfs_r and lcfs_z data, if available.
        # TODO: Implement provision of "lcfs_r" and "lcfs_z".
        if (lcfs_true["lcfs_r"] is not None) and (lcfs_true["lcfs_z"] is not None):
            j_tor_limiter_mask = self._make_lcfs_mask(
                lcfs_r=lcfs_true["lcfs_r"],  # TODO: Check if this data should be destandardized
                lcfs_z=lcfs_true["lcfs_z"],  # TODO: Check if this data should be destandardized
            )  # REMARK: This should be of size (F, n_r, n_z)

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Second option: From time-varying j_tor ground truth data, if available.
        elif j_tor_true_destdized is not None:
            j_tor_limiter_mask = j_tor_true_destdized > 1e-6  # (F, n_r, n_z)
            # TODO: Adopt later Tobia's approach to soften the mask for better backpropagation performance.

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # Third option: From static reference data (i.e., from self.base_j_tor_limiter_mask_rz).
        else:
            if self.base_j_tor_limiter_mask_rz is None:
                raise RuntimeError(
                    "Base j_tor limiter mask not found at last resource for the creation of the `j_tor_limiter_mask`."
                )

            j_tor_limiter_mask = self.base_j_tor_limiter_mask_rz.repeat(num_fields, 1, 1)  # (F, n_r, n_z)

        # ..............................................................................................................
        # Return calculated mask
        # ..............................................................................................................

        return j_tor_limiter_mask

    # ------------------------------------------------------------------------------------------------------------------
    def info_for_pred_gs_sides(
        self,
        j_tor_destdized,
        psi_destdized,
        j_tor_limiter_mask,
    ):
        """Choose the predicted GS right-hand-side current according to the configured RHS policy.

        Parameters
        ----------
        j_tor_destdized : Mapping[str, Tensor | None]
            Predicted and target physical current fields.
        psi_destdized : Mapping[str, Tensor | None]
            Predicted and target physical poloidal-flux fields.
        j_tor_limiter_mask : Tensor
            Plasma-region mask used when deriving current from target psi.

        Returns
        -------
        tuple[Tensor, Tensor, bool]
            Current fields for the predicted RHS, predicted psi fields for the LHS, and whether downstream min-max
            normalization is required.

        Raises
        ------
        ValueError
            If the selected RHS policy requires an unavailable output.
        NotImplementedError
            If the disabled predicted-profiles RHS policy is selected.
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

        if self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
            if j_tor_destdized["pred"] is None:
                raise ValueError(
                    f"`rhs_input={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR!r}` requires model output "
                    "'equilibrium-j_tor'."
                )

            j_tor_for_pred_rhs = j_tor_destdized["pred"]

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields from derived j_tor

        elif self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR:
            if psi_destdized["true"] is None:
                raise RuntimeError("Grad-Shafranov loss needs native target 'equilibrium-psi' when j_tor is derived.")

            j_tor_for_pred_rhs, requires_min_max_normalization = self.calculate_j_tor_from_psi(
                psi_fields=psi_destdized["true"],
                j_tor_limiter_mask=j_tor_limiter_mask,  # (F, n_r, n_z)
            )

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # j_tor fields from predicted profiles (pprime and ffprime)

        elif self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES:
            if (self._pprime_key is None) or (self._ffprime_key is None):
                raise ValueError(
                    f"`rhs_input={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES!r}` requires model outputs "
                    "'equilibrium-dpressure_dpsi' and 'equilibrium-f_df_dpsi'."
                )

            # TODO(FIX):
            #  The predicted-profiles RHS path is not functional yet — see `j_tor_from_pprime_ffprime_psi` for the
            #  problem description and proposed fix. Disabled here so it cannot be silently used.
            raise NotImplementedError(
                f"`rhs_input.origin={GRAD_SHAFRANOV_RHS_FROM_PREDICTED_PROFILES!r}` is not implemented yet "
                "(see j_tor_from_pprime_ffprime_psi)."
            )

            # j_tor_for_pred_rhs = SOMETHING_TO_BE_PORTED

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

        else:
            raise RuntimeError(f"Unsupported `rhs_input={self.rhs_input!r}`.")

        # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        # If here, j_tor_for_pred_rhs has been properly specified.

        # ..............................................................................................................
        # Info for the predicted Right-Hand Side of the GS equation.
        # ..............................................................................................................

        psi_for_pred_lhs = psi_destdized["pred"]

        # ..............................................................................................................
        # Return calculated info
        # ..............................................................................................................

        return j_tor_for_pred_rhs, psi_for_pred_lhs, requires_min_max_normalization

    # ------------------------------------------------------------------------------------------------------------------
    def compute(  # NOSONAR - Ignore cognitive complexity
        self,
        preds: Mapping[Hashable, Tensor],
        y_emb: Mapping[Hashable, Tensor],
        y_native: Mapping[Hashable, Tensor] | None,
        output_mask: None | Mapping[Hashable, Tensor] = None,
        pred_dist: Mapping[Hashable, Mapping[str, Any]] | None = None,
        context: LossComputeContext | None = None,
    ) -> tuple[Tensor, dict[Hashable, float]]:
        """
        Compute masked strong Grad-Shafranov (GS) loss in native, destandardized space. For this, predictions are
        required to be decoded and destandardized. If `j_tor` is not part of native targets, it may be approximated by
        using other native targets (i.e., `psi`).

        NaN samples in `y_native` are excluded from the GS loss — only valid measurements contribute.

        Parameters
        ----------
        preds : Mapping[Hashable, Tensor]
            Prediction tensors in embedding space, keyed by signal_id. Shape: `(B, D)`.
        y_emb : Mapping[Hashable, Tensor]
            Unused. May be None.
        y_native : Mapping[Hashable, Tensor] | None
            Ground-truth tensors in native standardized space, keyed by signal_id. Shape: `(B, *native_shape)`.
            Must be non-None when any supervised output is present.
        output_mask : None | Mapping[Hashable, Tensor]
            If specified, boolean mask tensors of shape `(B,)`, True for supervised samples.
            Optional. Default: None
        pred_dist : Mapping[Hashable, Mapping[str, Any]] | None
            Unused; accepted for `BaseLoss.compute` signature compatibility. May be None.
        context : LossComputeContext | None
            Optional metadata context for logging or plotting. May be None.

        Returns
        -------
        tuple[Tensor, dict[Hashable, float]]
            `(loss_scalar, per_output_logs)`

        Raises
        ------
        RuntimeError
            If `preds` is None.
            If `y_native` is None when a supervised output is encountered.
            If the Grad-Shafranov operator or the Grad-Shafranov R matrix were not loaded.

        """

        # ..............................................................................................................
        # 0 - Preliminary checks
        # ..............................................................................................................

        if not preds:
            raise RuntimeError("GradShafranovResidualLoss received empty predictions from the model.")

        if not y_native:
            raise RuntimeError("GradShafranovResidualLoss received predictions with empy native targets.")

        if self.gs_op_coo is None:
            raise RuntimeError("Grad-Shafranov operator was not loaded.")

        if self.R_matrix is None:
            raise RuntimeError("Grad-Shafranov R matrix was not loaded.")

        ref = next(iter(preds.values()))
        self._check_device_and_dtype(ref=ref)

        loss: Tensor = torch.tensor(0).to(self._runtime_device)  # Scalar
        logs: dict[Hashable, float] = {}

        # If the weight for the "grad_shafranov_residual" loss term is not greater than 0 and:
        # - Plotting is not needed, then return from here a default value 0 for the loss.
        # - Plotting is needed, calculation of all the required values for plotting is allowed, but calculations to
        #   update the default value 0 of the loss is later disallowed.
        if (not self._all_losses_weights.get("grad_shafranov_residual", 0) > 0) and (self._plot_check_type is None):
            return loss, logs  # TODO: Populate logs

        # ..............................................................................................................
        # 1 - Gather relevant ground truth and predicted data in native space
        # ..............................................................................................................

        # REMARK: Only ground truth for "equilibrium-psi" would be required in case predictions for "equilibrium-j_tor"
        # are not available, so that predicted j_tor data can be approximated form "equilibrium-psi"'s ground truth.

        stdized_native_data: dict[Hashable, dict[Hashable, Any]] = {"pred": {}, "true": {}}
        for out_key in self.required_output_keys:
            if (self._output_filter is not None) and (out_key not in self._output_filter):
                continue
            if (out_key not in self._decoders) or (out_key not in preds) or (out_key not in y_native):
                continue

            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            # Predicted data
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

            y_pred = preds[out_key]

            # Decode predictions: (B, D) → (B, *native_shape)
            decoder = self._decoders[out_key]
            pred_native = decoder(y_pred.to(torch.float32))  # Gradients flow through here

            stdized_native_data["pred"][out_key] = pred_native.to(torch.float32)

            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
            # Ground truth data
            # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .

            stdized_native_data["true"][out_key] = y_native[out_key].to(torch.float32)

        if self._psi_key not in stdized_native_data["pred"]:
            raise RuntimeError("Grad-Shafranov loss requires decoded predictions for 'equilibrium-psi'.")

        # ..............................................................................................................
        # 2 - Destandardize fields for LHS and RHS
        # ..............................................................................................................

        # psi (LHS) related
        psi_fields_destdized, psi_no_nan_mask, bhwt_shape = self.get_destdized_fields_for_gs_eq(
            stdized_native_data=stdized_native_data,
            key="psi",
        )
        n_fields = bhwt_shape[0] * bhwt_shape[3]  # F = B * T

        # j_tor (RHS) related
        j_tor_fields_destdized = {"pred": None, "true": None}
        j_tor_no_nan_mask = torch.ones_like(psi_no_nan_mask)
        if (self._j_tor_key is not None) and (self._j_tor_key in y_native):
            j_tor_fields_destdized, j_tor_no_nan_mask, _ = self.get_destdized_fields_for_gs_eq(
                stdized_native_data=stdized_native_data,
                key="j_tor",
            )

        # The derived-current RHS is constructed from psi ground truth, so current labels must not define its domain
        # or field eligibility. Predicted-current RHS legitimately requires current target validity.
        limiter_j_tor_target = (
            j_tor_fields_destdized["true"] if self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR else None
        )
        j_tor_limiter_mask = self._get_j_tor_limiter_mask(
            lcfs_true={
                "lcfs_r": stdized_native_data["true"].get("lcfs_r"),  # TODO: Should this be destandardized?
                "lcfs_z": stdized_native_data["true"].get("lcfs_z"),  # TODO: Should this be destandardized?
            },
            j_tor_true_destdized=limiter_j_tor_target,
            num_fields=n_fields,
        )
        if self.rhs_input == GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR:
            j_tor_limiter_mask = j_tor_limiter_mask & j_tor_no_nan_mask

        # Consolidated cleaning mask
        sides_cleaning_mask = psi_no_nan_mask & j_tor_limiter_mask

        # ..............................................................................................................
        # 3 - Get final info for predicted sides of the Grad-Shafranov equation
        # ..............................................................................................................

        j_tor_fields_for_pred_rhs, psi_fields_for_pred_lhs, perform_min_max_normalization = self.info_for_pred_gs_sides(
            j_tor_destdized=j_tor_fields_destdized,
            psi_destdized=psi_fields_destdized,
            j_tor_limiter_mask=j_tor_limiter_mask,
        )

        # ..............................................................................................................
        # 4 - Calculate Grad-Shafranov sides to be used in the residual calculation
        # ..............................................................................................................

        # Drop fields whose j_tor contains NaNs (e.g., from ground-truth psi approximation where psi is all-NaN).
        # As psi is in general either fully NaN or fully valid, so this would mean to drop whole fields.
        valid_fields = torch.isfinite(j_tor_fields_for_pred_rhs).flatten(start_dim=1).all(dim=1)  # (F,)
        valid_fields = valid_fields & sides_cleaning_mask.flatten(start_dim=1).any(dim=1).to(valid_fields.device)

        # Required output masks follow the configured RHS policy: derived current needs psi only; predicted current
        # requires psi and j_tor. The profiles path will use its declared required outputs when implemented.
        batch_size, _, _, n_times = bhwt_shape
        valid_fields = valid_fields & output_masks_to_field_mask(
            output_mask,
            self.required_output_keys,
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

            return ref.sum() * 0.0, logs

        # Calculate GS anchor from ground-truth psi: -A @ psi_gt
        # REMARK: For ground-truth data, LHS is equal to RHS, so either of them serves as anchor.
        gs_anchor = self.lhs_calculation(
            psi_fields_=psi_fields_destdized["true"],
            cleaning_mask_=sides_cleaning_mask,
        )

        # Calculate LSH from psi predictions: -A @ psi_pred
        gs_lhs_pred = self.lhs_calculation(
            psi_fields_=psi_fields_for_pred_lhs,
            cleaning_mask_=sides_cleaning_mask,
        )

        # Calculate LHS from j_tor predictions (or approximations): mu0 * R * j_tor_pred_or_approx
        gs_rhs_pred = self.rhs_calculation(
            j_tor_fields_=j_tor_fields_for_pred_rhs,
            cleaning_mask_=sides_cleaning_mask,
        )

        # ..............................................................................................................
        # 5 - Plot data (if specified)
        # ..............................................................................................................

        if self._plot_check_type is not None:
            self.run_plot_checks(
                gs_lhs_pred=gs_lhs_pred,
                gs_rhs_pred=gs_rhs_pred,
                gs_lhs_ref=gs_anchor,
                psi_fields_for_pred_lhs=psi_fields_for_pred_lhs,
                psi_fields_gt_destdized=psi_fields_destdized["true"],
                j_tor_fields_for_pred_rhs=j_tor_fields_for_pred_rhs,
                j_tor_fields_gt_destdized=j_tor_fields_destdized["true"],
                cleaning_mask=sides_cleaning_mask,
                context_info=context,
            )

        # ..............................................................................................................
        # 6 - Calculate Grad-Shafranov residuals for every (sample, time) field, computed in one vectorized pass.
        # ..............................................................................................................

        # Only update the default loss value 0 if the weight for the "grad_shafranov_residual" loss term is > 0.
        if self._all_losses_weights.get("grad_shafranov_residual", 0) > 0:
            per_field_losses = self.grad_shafranov_loss(
                gs_lhs_pred=gs_lhs_pred,
                gs_rhs_pred=gs_rhs_pred,
                gs_anchor=gs_anchor,
                cleaning_mask=sides_cleaning_mask,
                min_max_normalize_sides=perform_min_max_normalization,
            )  # (F,)

            loss = per_field_losses[valid_fields].mean()  # Scalar

        # ..............................................................................................................
        # 7 - Return loss and logs
        # ..............................................................................................................

        return loss, logs  # TODO: Populate logs.

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _min_max_normalize_per_field(x: Tensor) -> Tensor:
        """Min-max normalize each (sample, time) field independently. `x`: (F, n_r, n_z)."""

        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)

        return (x - x_min) / (x_max - x_min)

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
        gs_lhs, gs_rhs: (F, n_r, n_z) — one field per (sample, time).

        Parameters
        ----------
        gs_lhs_pred : Tensor
            LHS of the Grad-Shafranov equation without ground-truth anchor.
        gs_rhs_pred : Tensor
            RHS of the Grad-Shafranov equation without ground-truth anchor.
        gs_anchor : Tensor
            Ground-truth anchor for the Grad-Shafranov equation.
        min_max_normalize_sides : bool
            Optional. Default: False.
        cleaning_mask : Tensor | None
            Cleaning mask for both sides of the Grad-Shafranov equation.
            Optional. Default: None.

        Returns
        -------
        Tensor
            Per-field residual norms of size (F,).

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
        # Calculate the base GS residual

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
            kk: masked_reduce(residual=vv, mask=cleaning_mask, kind=self.loss_type)
            for kk, vv in flat_gs_residuals.items()
        }

        total_gs_loss = 0 * gs_anchor.flatten(start_dim=1).mean(dim=1)
        for kk, vv in gs_loss.items():
            total_gs_loss += self.gs_weights[kk] * vv

        return total_gs_loss

    # ------------------------------------------------------------------------------------------------------------------
    def calculate_j_tor_from_psi(
        self,
        psi_fields: Tensor,
        j_tor_limiter_mask: Tensor | None = None,
    ) -> tuple[Tensor, bool]:
        """
        Derive toroidal current j_tor from psi using configured `j_tor_calculation_method`.

        Dispatches to either sparse-operator inversion or parametric approximation based on the specified value for
        `self.j_tor_calculation_method`. Used when `rhs_input="derived_j_tor" to compute RHS from ground-truth psi.

        Parameters
        ----------
        psi_fields : torch.Tensor
            Ground-truth psi fields in physical units. Shape: (F, n_r, n_z).
        j_tor_limiter_mask : torch.Tensor | None
            Valid region mask (e.g., from LCFS or limiter). Shape: (F, n_r, n_z) or None.
            Required unless using parametric approximation without masking.
            Optional. Default: None.

        Returns
        -------
        tuple[torch.Tensor, bool]
            (j_tor_fields, perform_min_max_normalization):
            - j_tor_fields: Derived j_tor. Shape: (F, n_r, n_z).
            - perform_min_max_normalization: True if parametric approx (recommends min-max norm), False if via operator
              (no normalization recommended).

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
    def j_tor_from_psi_via_operator(
        self,
        psi_fields: Tensor,
        j_tor_limiter_mask: Tensor,
    ) -> Tensor:
        """
        Derive j_tor from psi and the GS operator as j_tor = (μ₀ R)⁻¹ · (-GS_op @ psi).

        Applies the sparse differential operator to ground-truth psi and solves for j_tor in valid regions,
        clamped to [0, ∞). Used when `rhs_input="derived_j_tor"` and `j_tor_calculation_method="via_gs_operator"`.

        Parameters
        ----------
        psi_fields : torch.Tensor
            Ground-truth psi fields in physical units. Shape: (F, n_r, n_z).
        j_tor_limiter_mask : torch.Tensor
            Boolean mask of valid regions (plasma interior). Shape: (F, n_r, n_z).

        Returns
        -------
        torch.Tensor
            Derived j_tor fields, clamped to [0, ∞). Shape: (F, n_r, n_z).

        """

        n_fields_ = psi_fields.shape[0]
        denominator = mu0 * self.R_matrix  # (n_r, n_z)

        # Apply the sparse operator to every flattened psi field in one matmul (was a per-sample loop over [:, :, 0]).
        psi_flat = psi_fields.to(torch.float32).reshape(n_fields_, self.n_r * self.n_z)  # (F, N)
        numerator = (
            -torch.sparse.mm(  # -> mm: Matrix multiplication
                self.gs_op_coo,
                psi_flat.t(),
            )
        ).t()  # (F, N)
        numerator = numerator.reshape(n_fields_, self.n_r, self.n_z)  # (F, n_r, n_z)

        j_tor = torch.clamp(
            input=j_tor_limiter_mask.to(dtype=numerator.dtype) * (numerator / denominator),
            min=0,
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
        Derive j_tor from psi using parametric profile approximation.

        Applies a smooth power-law approximation to model poloidal and toroidal current contributions independently,
        then blends them. Normalized psi is computed per-sample (mean-std normalization), and plasma region is inferred
        as points where 1 - psi ≥ 0. Used when `j_tor_calculation_method="via_parametric_approx"`.

        Parameters
        ----------
        psi_fields : torch.Tensor
            Ground-truth ψ fields in physical units. Shape: (F, n_r, n_z).
        j_tor_limiter_mask : torch.Tensor
            Mask of valid regions. Shape: (F, n_r, n_z).
        alpha_m, alpha_n : float
            Exponents for poloidal current power law.
            Optional. Default: 2.0, 2.0.
        beta_m, beta_n : float
            Exponents for toroidal current power law.
            Optional. Default: 1.0, 1.0.
        lamda : float
            Scaling coefficient for total current.
            Optional. Default: 1e-1.
        beta : float
            Blend factor: j_tor ∝ lamda · [β · j_pol + (1-β) · j_tor].
            Optional. Default: 0.5.

        Returns
        -------
        torch.Tensor
            Parametric j_tor fields (F, n_r, n_z), masked to valid region.

        Notes
        -----
        - This is a smoothed approximation; may miss sharp profile features captured by direct inversion.
        - If j_tor obtained via this method is used to build the GS equation, then min-max normalization of the GS
          sides is required before the calculation of the GS residuals.

        """

        psi_fields_masked = torch.clamp(input=psi_fields, min=0) * j_tor_limiter_mask

        ms_normalized_psi = []  # -> For mean-std (ms) normalized psi data
        for psi_sample in psi_fields_masked:
            # REMARK: Normalization must be done per-sample (cannot be vectorized).
            ms_normalized_psi_sample = self.mean_std_normalization(
                x=psi_sample,
            )

            ms_normalized_psi.append(ms_normalized_psi_sample)

        ms_normalized_psi = torch.stack(ms_normalized_psi)  # (F, n_r, n_z)

        plasma_region = self.compute_plasma_region(ms_normalized_psi=ms_normalized_psi)

        jp = torch.pow((1 - torch.pow(input=ms_normalized_psi, exponent=alpha_m)), alpha_n) * self.R_matrix
        jf = torch.pow((1 - torch.pow(input=ms_normalized_psi, exponent=beta_m)), beta_n) / self.R_matrix

        j_tor = lamda * (jp * beta + jf * (1 - beta))

        return (j_tor_limiter_mask * j_tor) * plasma_region

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def min_max_normalisation(x, nan_mode=False):
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
            ``(x - min(x)) / (max(x) - min(x))`` in the input representation.
        """

        min_ = np.nanmin(x) if nan_mode else x.min()
        max_ = np.nanmax(x) if nan_mode else x.max()

        return (x - min_) / (max_ - min_)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def mean_std_normalization(x, mean_scaling=1, std_scaling=1):
        """Normalize one tensor field using its scaled mean and standard deviation.

        Parameters
        ----------
        x : Tensor
            Tensor field to normalize.
        mean_scaling, std_scaling : float
            Multipliers applied to the field mean and standard deviation before normalization.

        Returns
        -------
        Tensor
            Mean/std-normalized tensor with the same shape as ``x``.
        """

        mean_x = mean_scaling * torch.mean(input=x)
        std_x = std_scaling * torch.std(input=x)

        normalizer = transforms.Normalize(
            mean=(mean_x,),
            std=(std_x,),
        )
        normalised_x = normalizer(x.unsqueeze(0))

        return normalised_x.squeeze()

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def compute_plasma_region(ms_normalized_psi):
        """Return the current binary plasma-region mask from normalized poloidal flux.

        Parameters
        ----------
        ms_normalized_psi : Tensor
            Mean/std-normalized poloidal-flux field.

        Returns
        -------
        Tensor
            Floating-point binary mask with the same shape as the input.
        """

        return torch_functional.relu(1 - ms_normalized_psi).ge(0).float()

    # -------------------------------------------------------------------------------------------------------------------
    def nearest_grid_value(self, field_zr, r0, z0):
        """Read a 2D field at the nearest R/Z grid point."""

        ir = int(np.argmin(np.abs(self.R_matrix - r0)))
        iz = int(np.argmin(np.abs(self.Z_matrix - z0)))

        return float(field_zr[iz, ir])

    # -------------------------------------------------------------------------------------------------------------------
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
            Profile samples for pressure and ``F F'`` derivatives.
        psi_native, psi_norm_native : Tensor
            Poloidal flux and its profile-normalization coordinate.
        mag_r_axis, mag_z_axis, xpt_r, xpt_z : array-like
            EFIT magnetic-axis and X-point geometry used to construct ``psi_N``.

        Returns
        -------
        Tensor
            Estimated toroidal-current field.

        Notes
        -----
        This legacy path is currently disabled in ``compute``. It mixes NumPy and Torch operations, is not
        differentiable, and does not satisfy the vectorized ``(F, n_r, n_z)`` field contract.
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
