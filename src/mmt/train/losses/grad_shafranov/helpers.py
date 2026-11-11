"""
Shared helpers for the Grad-Shafranov residual losses (strong and weak formulations).

Both :class:`~mmt.train.losses.grad_shafranov.strong.StrongFormGradShafranovLoss` and
:class:`~mmt.train.losses.grad_shafranov.weak.WeakFormGradShafranovLoss` fold decoded native
predictions into `(F, n_r, n_z)` fields, prepare safe physical target fields, reduce a per-field
residual to a per-field norm, and translate batch-level output masks to field masks. Those mechanics are
independent of whether the residual is strong or weak, so they live here as pure functions with a single
source of truth.

What is intentionally *not* shared: formulation-specific mask composition (the j_tor limiter/plasma mask
and which outputs define a physically usable sample). Each loss composes those masks from the common
finite-target and field-mask primitives according to its RHS policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Hashable

import numpy as np
import torch
from torch import Tensor

from mmt.data.standardization import destandardize_torch
from mmt.train.losses.base import LossComputeContext
from mmt.train.losses.constants import (
    ALLOWED_PLOT_FORMATS_TYPE,
    ALLOWED_PLOT_FORMATS,
    ALLOWED_PLOT_CHECK_TYPES,
    ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS,
    DEFAULT_GRAD_SHAFRANOV_LOSS_METRIC,
    GRAD_SHAFRANOV_RHS_KEYS,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY,
    GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR,
    GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY,
    GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR,
    GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS,
    GRAD_SHAFRANOV_RHS_INPUT_ORIGINS_FULL,
    DEFAULT_GRAD_SHAFRANOV_WEIGHTS,
    GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR,
    ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS_TYPE,
)

# Vacuum permeability (single source of truth for both GS losses).
mu0 = 1.2566370614359173e-06


# ======================================================================================================================
@dataclass(frozen=True)
class GSGridAssets:
    """Grid data common to the strong and weak Grad-Shafranov loss assets."""

    n_r: int
    n_z: int
    r_matrix: Tensor
    z_matrix: Tensor
    base_j_tor_limiter_mask_rz: Tensor


# ----------------------------------------------------------------------------------------------------------------------
def native_to_gs_fields(x_bhwt: Tensor) -> Tensor:
    """
    Fold a decoded native tensor `(B, H, W, T)` into Grad-Shafranov operator fields `(B * T, n_r, n_z)`.

    For every `(sample, time)` the native `(H, W)` slice is transposed to `(n_r, n_z)` — the
    orientation the sparse GS operator and limiter masks expect. This generalizes a per-sample
    `field[:, :, 0].T` to all forecast times; the field index is `f = b * T + t`.

    Parameters
    ----------
    x_bhwt : Tensor
        Decoded native tensor of shape `(B, H, W, T)` where `T` is the forecast-horizon time axis.

    Returns
    -------
    Tensor
        Fields of shape `(F, n_r, n_z)` with `F = B * T`.

    """

    b, h, w, t = x_bhwt.shape

    # (B, H, W, T) -> (B, T, W, H) -> (B * T, W, H) == (F, n_r, n_z)
    return x_bhwt.permute(0, 3, 2, 1).reshape(b * t, w, h)


# ----------------------------------------------------------------------------------------------------------------------
def prepare_target_field(field: Tensor, stats: Mapping[str, Any]) -> tuple[Tensor, Tensor]:
    """
    Make a standardized target safe for a GS operator and destandardize it to physical native units.

    Non-finite cells are replaced with zero *before* any operator is applied. The returned finite mask must
    still be combined with the formulation-specific physical-domain mask before reducing a residual. This
    avoids `0 * NaN` propagation without allowing invalid cells to contribute to the loss.

    Parameters
    ----------
    field : Tensor
        Standardized native target field.
    stats : Mapping[str, Any]
        Per-signal statistics containing `"mean"` and `"std"`.

    Returns
    -------
    tuple[Tensor, Tensor]
        `(safe_destandardized_field, finite_mask)`. Both have the shape of `field`; the mask is boolean.

    """

    finite_mask = torch.isfinite(field)
    safe_standardized = torch.where(finite_mask, field, torch.zeros_like(field))
    safe_field = destandardize_torch(x=safe_standardized, mean=stats["mean"], std=stats["std"])

    return safe_field, finite_mask


# ----------------------------------------------------------------------------------------------------------------------
def output_masks_to_field_mask(
    output_mask: Mapping[Hashable, Tensor] | None,
    required_keys: tuple[Hashable, ...] | list[Hashable] | set[Hashable],
    *,
    batch_size: int,
    n_times: int,
    ref: Tensor,
) -> Tensor:
    """Fold required per-sample output masks `(B,)` into a per-field mask `(B*T,)`.

    A missing mask entry preserves the existing loss-system convention and leaves that output unconstrained.
    Callers select `required_keys` according to their RHS policy: derived-current strong mode requires only
    psi, while predicted-current strong mode and weak form require both psi and j_tor.
    """

    field_mask = torch.ones(batch_size * n_times, device=ref.device, dtype=torch.bool)
    if output_mask is None:
        return field_mask

    for key in required_keys:
        sample_mask = output_mask.get(key)
        if sample_mask is None:
            continue
        if sample_mask.numel() != batch_size:
            raise ValueError(
                f"output_mask[{key!r}] has {sample_mask.numel()} values, expected one per batch item ({batch_size})."
            )

        field_mask &= (
            sample_mask.to(device=ref.device, dtype=torch.bool)
            .reshape(batch_size, 1)
            .expand(batch_size, n_times)
            .reshape(-1)
        )

    return field_mask


# ----------------------------------------------------------------------------------------------------------------------
def resolve_output_key(output_name_to_id: Mapping[str, Hashable], name: str, *, loss_name: str) -> Hashable:
    """Resolve one required output name with a consistent error message."""

    if name not in output_name_to_id:
        raise KeyError(f"{loss_name} requires output {name!r}, not in available outputs {sorted(output_name_to_id)!r}.")

    return output_name_to_id[name]


# ----------------------------------------------------------------------------------------------------------------------
def resolve_gs_asset_path(filename: str | Path, repo_root: Path) -> Path:
    """Resolve an absolute or repository-relative Grad-Shafranov asset path."""

    path = Path(filename)

    return path if path.is_absolute() else (repo_root / path).resolve()


# ----------------------------------------------------------------------------------------------------------------------
def parse_gs_grid_assets(loaded: Mapping[str, Any], *, tensor_dtype: torch.dtype | None = None) -> GSGridAssets:
    """Extract the grid portion shared by strong and weak Grad-Shafranov `.npz` assets."""

    tensor_kwargs: dict[str, torch.dtype] = {} if tensor_dtype is None else {"dtype": tensor_dtype}
    r_matrix = torch.tensor(np.asarray(loaded["R_matrix"]), **tensor_kwargs)
    z_matrix = torch.tensor(np.asarray(loaded["Z_matrix"]), **tensor_kwargs)
    limiter = torch.tensor(0)
    if "base_j_tor_lim_mask_rz" in loaded:
        limiter = torch.tensor(np.asarray(loaded["base_j_tor_lim_mask_rz"])).unsqueeze(0).bool()

    return GSGridAssets(
        n_r=int(loaded["n_r"]),
        n_z=int(loaded["n_z"]),
        r_matrix=r_matrix,
        z_matrix=z_matrix,
        base_j_tor_limiter_mask_rz=limiter,
    )


# ----------------------------------------------------------------------------------------------------------------------
def runtime_tensor(tensor: Tensor, *, ref: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """Move a stored GS tensor to the runtime device, optionally matching a reference dtype."""

    if dtype is None:
        return tensor.to(device=ref.device)

    return tensor.to(device=ref.device, dtype=dtype)


# ----------------------------------------------------------------------------------------------------------------------
def validate_rhs_input_cfg(rhs_input_cfg: Any, path: str) -> tuple[str, str]:
    """
    Validate the rhs_input fragment common to strong and weak GS loss configuration, and return parsed values if all
    the checks pass. A null `rhs_input` is treated as omitted (all defaults).
    """

    if rhs_input_cfg is None:
        rhs_input_cfg = {}

    if not isinstance(rhs_input_cfg, Mapping):
        raise TypeError(f"{path}.rhs_input must be a mapping when provided.")

    unknown_rhs_input_keys = sorted(str(key) for key in rhs_input_cfg.keys() if (key not in GRAD_SHAFRANOV_RHS_KEYS))
    if unknown_rhs_input_keys:
        raise KeyError(
            f"Unknown {path}.rhs_input keys: {unknown_rhs_input_keys}. Supported: {sorted(GRAD_SHAFRANOV_RHS_KEYS)}."
        )

    rhs_input_origin = validate_rhs_input_origin(
        rhs_input_origin=rhs_input_cfg.get(GRAD_SHAFRANOV_RHS_INPUT_ORIGIN_KEY),
        path=path,
    )

    j_tor_calculation_method = validate_j_tor_calculation_method(
        method=rhs_input_cfg.get(GRAD_SHAFRANOV_RHS_INPUT_CALCULATION_METHOD_KEY),
        rhs_input_origin=rhs_input_origin,
        path=path,
    )

    return rhs_input_origin, j_tor_calculation_method


# ----------------------------------------------------------------------------------------------------------------------
def validate_rhs_input_origin(rhs_input_origin: Any, path: str) -> str:
    """Validate RHS input origin, and return parsed values if all the checks pass."""

    rhs_input_origin: str = rhs_input_origin or GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR
    if rhs_input_origin not in GRAD_SHAFRANOV_RHS_INPUT_ORIGINS_FULL:
        raise ValueError(
            f"{path}.rhs_input.origin={rhs_input_origin!r} is unsupported. "
            f"Supported: {sorted(GRAD_SHAFRANOV_RHS_INPUT_ORIGINS_FULL)}."
        )

    return rhs_input_origin


# ----------------------------------------------------------------------------------------------------------------------
def validate_j_tor_calculation_method(method: Any, rhs_input_origin: str, path: str) -> str:
    """
    Validate j_tor calculation method, and return parsed values if all the checks pass. A missing value (None)
    defaults to `GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR` ("GS_operator").
    """

    derived_j_tor = rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR
    method: str = method or GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR
    if derived_j_tor and (method not in GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS):
        raise ValueError(
            f"[{path}] Unsupported `j_tor_calculation_method={method!r}`. "
            f"Supported: {sorted(GRAD_SHAFRANOV_J_TOR_CALCULATION_METHODS)}."
        )

    return method


# ----------------------------------------------------------------------------------------------------------------------
def validate_params_file_cfg(params_file: Any, path: str) -> str | Path:
    """
    Validate the parameters file fragment common to strong and weak GS loss configuration, and return parsed values if
    all the checks pass.
    """

    if not params_file:
        raise KeyError(f"{path}.grad_shafranov_params_file is required.")

    if not isinstance(params_file, (str, Path)):
        raise TypeError(f"{path}.grad_shafranov_params_file must be str or Path, got {type(params_file).__name__}.")

    return params_file


# ----------------------------------------------------------------------------------------------------------------------
def validate_grad_shafranov_weights(
    grad_shafranov_weights: Any,
    rhs_input_origin: str,
    j_tor_calculation_method: str,
    path: str,
) -> dict:
    """
    Validate the grad_shafranov_weights fragment common to strong and weak GS loss configuration, and return parsed
    values if all the checks pass. If weights are given, any missing key counts as 0. If no weights are given (None
    or an empty mapping), `DEFAULT_GRAD_SHAFRANOV_WEIGHTS` is used. If `rhs_input_origin="derived_j_tor"`, `rhs_gt`
    must be 0 (whatever `j_tor_calculation_method` is), since that residual carries no gradient to the model.
    """

    required_keys = list(DEFAULT_GRAD_SHAFRANOV_WEIGHTS.keys())
    is_a_mapping = isinstance(grad_shafranov_weights, Mapping)

    if is_a_mapping and grad_shafranov_weights:
        unknown_weight_keys = sorted(str(key) for key in grad_shafranov_weights.keys() if str(key) not in required_keys)
        if unknown_weight_keys:
            raise KeyError(
                f"Unknown {path}.grad_shafranov_weights keys: {unknown_weight_keys}. "
                f"Required keys are {sorted(required_keys)}."
            )

    else:
        if (grad_shafranov_weights is not None) and not is_a_mapping:
            raise TypeError(f"{path}.grad_shafranov_weights must be a mapping when provided.")

        # None or an empty mapping: use the defaults.
        grad_shafranov_weights = DEFAULT_GRAD_SHAFRANOV_WEIGHTS

    # Missing keys count as 0, so a partial mapping (e.g., {lhs_gt: 1.0}) selects only the terms it names.
    gs_weights = {**{key: 0.0 for key in required_keys}, **grad_shafranov_weights}
    for gs_weight_key, gs_weight_val in gs_weights.items():
        if isinstance(gs_weight_val, bool) or not isinstance(gs_weight_val, (float, int)):
            raise TypeError(
                f"{path}.grad_shafranov_weights.{gs_weight_key} must be an int or float (bools are not accepted)."
            )

        if not 0 <= gs_weight_val <= 1:
            raise ValueError(f"[{path}] All weights in `grad_shafranov_weights` must be between 0 and 1.")

    # Check and enforce convex combination
    if abs(sum(gs_weights.values()) - 1) > 1e-5:
        raise ValueError(f"[{path}].grad_shafranov_weights {required_keys} must sum to 1.")

    # With a derived j_tor, the predicted RHS is built from ground-truth psi only, so the `rhs_gt` residual
    # (RHS_pred - anchor) carries no gradient to the model, whatever the calculation method. With "GS_operator", it is
    # also degenerate (≈ 0), since the derived RHS reproduces the ground-truth RHS.
    if (rhs_input_origin == GRAD_SHAFRANOV_RHS_FROM_DERIVED_J_TOR) and (gs_weights["rhs_gt"] > 0):
        degeneracy_note = (
            " With `rhs_input.calculation_method='GS_operator'`, it is also degenerate (~0), since the derived RHS "
            "reproduces the ground-truth RHS."
            if j_tor_calculation_method == GRAD_SHAFRANOV_J_TOR_VIA_GS_OPERATOR
            else ""
        )
        raise ValueError(
            f"[{path}] If `rhs_input.origin='derived_j_tor'`, the predicted RHS is built from ground-truth psi only, "
            f"so the `rhs_gt` residual carries no gradient to the model.{degeneracy_note} Set rhs_gt: 0."
        )

    return gs_weights


# ----------------------------------------------------------------------------------------------------------------------
def validate_loss_metric_cfg(loss_metric: Any, path: str) -> ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS_TYPE:
    """
    Validate the loss-metric fragment common to strong and weak GS loss configuration, and return parsed values if all
    the checks pass. A missing value (None) defaults to `DEFAULT_GRAD_SHAFRANOV_LOSS_METRIC` ("mse").
    """

    if loss_metric is None:
        loss_metric = DEFAULT_GRAD_SHAFRANOV_LOSS_METRIC

    if loss_metric not in ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS:
        raise ValueError(f"{path}.loss_metric must be in {ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS}, got '{loss_metric}'.")

    return loss_metric


# ----------------------------------------------------------------------------------------------------------------------
def validate_plot_check_cfg(plot_check: Any, path: str) -> Any:
    """
    Validate the plot-check fragment common to strong and weak GS loss configuration, and return parsed values
    if all the checks pass.
    """

    if plot_check is None:
        return None

    if not isinstance(plot_check, Mapping):
        raise TypeError(f"{path}.plot_check must be a mapping when provided.")

    probability = plot_check.get("probability", 0.0)
    if (probability is None) or isinstance(probability, bool) or not isinstance(probability, (float, int)):
        raise TypeError(
            f"{path}.plot_check.probability must be an int or float (bools are not accepted), "
            f"got {type(probability).__name__}."
        )

    probability = float(probability)
    if not (0.0 <= probability <= 1.0):  # Also rejects NaN.
        raise ValueError(f"{path}.plot_check.probability must be between 0.0 and 1.0, got {probability}.")

    plot_check_type = plot_check.get("type")
    if plot_check_type not in ALLOWED_PLOT_CHECK_TYPES:
        raise ValueError(f"{path}.plot_check.type must be in {ALLOWED_PLOT_CHECK_TYPES}.")

    plot_check_type = plot_check_type or "none_none"
    run_plotting = not plot_check_type.startswith("none_none")
    save_plots = plot_check_type.startswith("save_")
    save_plot_format: ALLOWED_PLOT_FORMATS_TYPE = plot_check_type.split("_")[-1] if save_plots else "png"

    all_losses_weights = plot_check.get(
        "all_losses_weights",
        {"NA": "NA"},  # Default value in case "all_losses_weights" field was not added in aggregator.py
    )

    return plot_check_type, probability, run_plotting, save_plots, save_plot_format, all_losses_weights


# ----------------------------------------------------------------------------------------------------------------------
def validate_base_j_tor_limiter_mask(
    limiter_mask: Tensor,
    rhs_input_origin: Any,
    grad_shafranov_params_file,
    path: str,
) -> None:
    """Validate the specified base_j_tor_limiter_mask parameter."""

    # The static limiter mask is the only plasma-region source when target j_tor is unavailable (and LCFS data is
    # not provided), so it is required for every RHS origin except `predicted_j_tor`.
    if (rhs_input_origin != GRAD_SHAFRANOV_RHS_FROM_PREDICTED_J_TOR) and (limiter_mask.ndim == 0):
        raise KeyError(
            f"[{path}] `rhs_input.origin={rhs_input_origin!r}` requires a static "
            f"limiter mask, but `base_j_tor_lim_mask_rz` is missing from {grad_shafranov_params_file!r}."
        )


# ----------------------------------------------------------------------------------------------------------------------
def validate_required_keys(input_name: str, input_values: Any, required_keys, path: str) -> None:
    """Validate the required keys against the specified input."""

    missing_keys = [key for key in required_keys if (key not in input_values)]
    if missing_keys:
        raise ValueError(
            f"[{path}] {input_name} must include all required keys by the specified configuration. "
            f"Missing keys: {missing_keys}."
        )


# ----------------------------------------------------------------------------------------------------------------------
def select_diagnostic_plot_slice(
    *,
    context: LossComputeContext | None,
    probability: float,
    n_fields: int,
    diagnostic_name: str,
) -> int | None:
    """Deterministically select one field slice for an optional diagnostic plot.

    Plotting must not advance Python's process-global RNG: the data collation path also uses that stream for
    stochastic masking, so doing so makes plot-enabled training differ from the same run without plots. This helper
    derives both the plot decision and selected slice from immutable batch metadata instead. The same context always
    yields the same result, independent of call order or worker timing.

    Returns `None` when the batch is not selected, plotting is disabled by its probability, or no runtime context is
    available. A missing context is intentionally treated as no plot: diagnostic side effects must never be required
    for the scalar loss calculation.
    """

    if probability <= 0.0 or n_fields <= 0 or context is None:
        return None

    run_id = Path(context.run_dir).name if context.run_dir else "unknown_run"
    key = "|".join(
        (
            run_id,
            diagnostic_name,
            "train" if context.train else "validation",
            str(context.stage_index),
            str(context.stage_name),
            str(context.epoch_global),
            str(context.epoch_in_stage),
            str(context.global_step),
            str(context.batch_index),
            str(context.term_index),
        )
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=16).digest()
    draw = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    if draw >= probability:
        return None

    return int.from_bytes(digest[8:], byteorder="big") % n_fields


# ----------------------------------------------------------------------------------------------------------------------
def training_plot_path(
    context: LossComputeContext | None,
    *,
    slice_index: int,
    plot_format: ALLOWED_PLOT_FORMATS_TYPE = "png",
) -> Path:
    """Build a reproducible diagnostic-plot path inside the current training run.

    The path is intentionally loss-agnostic. The configured term `type` distinguishes plot-producing terms that run
    on the same batch without exposing a Python implementation name in the output layout.
    """

    if (context is None) or (not context.run_dir):
        raise RuntimeError("Saving diagnostic plots requires LossComputeContext.run_dir.")

    if plot_format not in ALLOWED_PLOT_FORMATS:
        raise ValueError(
            f"Saving diagnostic plots requires a valid `plot_format` in {ALLOWED_PLOT_FORMATS}, got {plot_format}]."
        )

    stage_index = "unknown" if context.stage_index is None else f"{context.stage_index:02d}"
    raw_stage_name = str(context.stage_name or "unnamed")
    stage_name = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_stage_name).strip("._") or "unnamed"
    plot_dir = Path(context.run_dir) / "plots" / "training" / f"stage_{stage_index}_{stage_name}"

    phase = "train" if context.train else "validation"
    epoch = "unknown" if context.epoch_global is None else f"{context.epoch_global:03d}"
    step = "unknown" if context.global_step is None else f"{context.global_step:06d}"
    batch = "unknown" if context.batch_index is None else f"{context.batch_index:05d}"
    raw_term_type = str((context.term_config or {}).get("type") or context.term_name or "unknown")
    term_type = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_term_type).strip("._") or "unknown"

    matching_term_indices = [
        index for index, term in enumerate(context.stage_loss_terms) if str(term.get("type")) == raw_term_type
    ]
    if (len(matching_term_indices) > 1) and (context.term_index in matching_term_indices):
        occurrence = matching_term_indices.index(context.term_index) + 1  # noqa - Proper check for None done above.
        term_type = f"{term_type}_{occurrence:02d}"

    filename = f"{phase}_epoch_{epoch}_step_{step}_batch_{batch}_slice_{slice_index:03d}_{term_type}.{plot_format}"

    return plot_dir / filename


# ----------------------------------------------------------------------------------------------------------------------
def masked_reduce(residual: Tensor, mask: Tensor | None, kind: ALLOWED_GRAD_SHAFRANOV_LOSS_METRICS_TYPE) -> Tensor:
    """
    Reduce a stack of residual fields to a per-field metric.

    This reproduces the reduction forms used by both GS losses with an explicit mask contract:

    - `kind="l2"`: the Euclidean norm over active cells.
    - `kind="mse"` with `mask is None`: the plain mean of squares over all cells.
    - `kind="mse"` with a mask: the mean of squares over the active (masked) cells only, i.e.
      `sum(residual**2) / active_cells`, with fields having zero active cells set to NaN so they can be
      dropped by the caller's validity check.

    The helper applies the mask itself rather than relying on callers to zero inactive cells beforehand. The L2
    branch keeps `vector_norm` semantics and is deliberately not redefined as RMSE here.

    Parameters
    ----------
    residual : Tensor
        Residual fields shaped `(F, n_r, n_z)` or already flattened as `(F, N)`.
    mask : Tensor | None
        Optional boolean/other mask with the same number of cells per field as `residual`.
    kind : Literal["l2", "mse"]
        Reduction form.

    Returns
    -------
    Tensor
        Per-field norms shaped `(F,)`.

    Raises
    ------
    ValueError
        If `kind` is not `"l2"` or `"mse"`.

    """

    flat = residual.flatten(start_dim=1)
    if mask is None:
        if kind == "l2":
            return torch.linalg.vector_norm(flat, dim=1)
        if kind == "mse":
            return flat.square().mean(dim=1)
        raise ValueError(f"[masked_reduce] Invalid `kind`: must be in ['l2', 'mse'], got {kind!r}.")

    flat_mask = mask.flatten(start_dim=1).to(device=flat.device, dtype=torch.bool)
    if flat_mask.shape != flat.shape:
        raise ValueError(f"mask shape {tuple(mask.shape)} is incompatible with residual shape {tuple(residual.shape)}.")

    active_cells = flat_mask.sum(dim=1).to(dtype=flat.dtype)
    active_residual = torch.where(flat_mask, flat, torch.zeros_like(flat))

    if kind == "l2":
        loss = torch.linalg.vector_norm(active_residual, dim=1)
    elif kind == "mse":
        loss = active_residual.square().sum(dim=1) / active_cells.clamp_min(1)
    else:
        raise ValueError(f"[masked_reduce] Invalid `kind`: must be in ['l2', 'mse'], got {kind!r}.")

    return torch.where(active_cells > 0, loss, torch.full_like(loss, torch.nan))


# ----------------------------------------------------------------------------------------------------------------------
def is_tensor_zero(x):
    return torch.is_tensor(x) and x.ndim == 0 and x.item() == 0


# ----------------------------------------------------------------------------------------------------------------------
