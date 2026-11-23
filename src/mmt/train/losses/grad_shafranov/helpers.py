"""
Shared helpers for the Grad-Shafranov residual losses (strong and weak formulations).

Both :class:`~mmt.train.losses.grad_shafranov.GradShafranovResidualLoss` and
:class:`~mmt.train.losses.grad_shafranov.WeakFormGradShafranovLoss` fold decoded native
predictions into ``(F, n_r, n_z)`` fields, prepare safe physical target fields, reduce a per-field
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
from typing import Any, Hashable, Literal

import numpy as np
import torch
from torch import Tensor

from mmt.data.standardization import destandardize_torch
from mmt.train.losses.base import LossComputeContext


# Vacuum permeability (single source of truth for both GS losses).
mu0 = 1.2566370614359173e-06


@dataclass(frozen=True)
class GSGridAssets:
    """Grid data common to the strong and weak Grad-Shafranov loss assets."""

    n_r: int
    n_z: int
    r_matrix: Tensor
    z_matrix: Tensor
    base_j_tor_limiter_mask_rz: Tensor | None


# ----------------------------------------------------------------------------------------------------------------------
def native_to_gs_fields(x_bhwt: Tensor) -> Tensor:
    """
    Fold a decoded native tensor ``(B, H, W, T)`` into Grad-Shafranov operator fields ``(B * T, n_r, n_z)``.

    For every ``(sample, time)`` the native ``(H, W)`` slice is transposed to ``(n_r, n_z)`` — the
    orientation the sparse GS operator and limiter masks expect. This generalizes a per-sample
    ``field[:, :, 0].T`` to all forecast times; the field index is ``f = b * T + t``.

    Parameters
    ----------
    x_bhwt : Tensor
        Decoded native tensor of shape ``(B, H, W, T)`` where ``T`` is the forecast-horizon time axis.

    Returns
    -------
    Tensor
        Fields of shape ``(F, n_r, n_z)`` with ``F = B * T``.

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
    avoids ``0 * NaN`` propagation without allowing invalid cells to contribute to the loss.

    Parameters
    ----------
    field : Tensor
        Standardized native target field.
    stats : Mapping[str, Any]
        Per-signal statistics containing ``"mean"`` and ``"std"``.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(safe_destandardized_field, finite_mask)``. Both have the shape of ``field``; the mask is boolean.

    """

    finite_mask = torch.isfinite(field)
    safe_standardized = torch.where(finite_mask, field, torch.zeros_like(field))
    safe_field = destandardize_torch(x=safe_standardized, mean=stats["mean"], std=stats["std"])
    return safe_field, finite_mask


# ----------------------------------------------------------------------------------------------------------------------
def output_masks_to_field_mask(
    output_mask: Mapping[Hashable, Tensor] | None,
    required_keys: tuple[Hashable, ...] | list[Hashable],
    *,
    batch_size: int,
    n_times: int,
    ref: Tensor,
) -> Tensor:
    """Fold required per-sample output masks ``(B,)`` into a per-field mask ``(B*T,)``.

    A missing mask entry preserves the existing loss-system convention and leaves that output unconstrained.
    Callers select ``required_keys`` according to their RHS policy: derived-current strong mode requires only
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
    """Extract the grid portion shared by strong and weak Grad-Shafranov ``.npz`` assets."""

    tensor_kwargs: dict[str, torch.dtype] = {} if tensor_dtype is None else {"dtype": tensor_dtype}
    r_matrix = torch.tensor(np.asarray(loaded["R_matrix"]), **tensor_kwargs)
    z_matrix = torch.tensor(np.asarray(loaded["Z_matrix"]), **tensor_kwargs)
    limiter = None
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
def validate_plot_check_cfg(plot_check: Any, path: str) -> None:
    """Validate the plot-check fragment common to strong and weak GS loss configuration."""

    if plot_check is None:
        return
    if not isinstance(plot_check, Mapping):
        raise TypeError(f"{path}.plot_check must be a mapping when provided.")
    probability = plot_check.get("probability")
    if probability is not None and (isinstance(probability, bool) or not isinstance(probability, (float, int))):
        raise TypeError(f"{path}.plot_check.probability must be a number.")
    if plot_check.get("type") not in {None, "show_plots", "save_plots"}:
        raise ValueError(f"{path}.plot_check.type must be 'show_plots', 'save_plots', or null.")


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

    Returns ``None`` when the batch is not selected, plotting is disabled by its probability, or no runtime context is
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
def training_plot_path(context: LossComputeContext | None, *, slice_index: int) -> Path:
    """Build a reproducible diagnostic-plot path inside the current training run.

    The path is intentionally loss-agnostic. The configured term ``type`` distinguishes plot-producing terms that run
    on the same batch without exposing a Python implementation name in the output layout.
    """

    if context is None or not context.run_dir:
        raise RuntimeError("Saving diagnostic plots requires LossComputeContext.run_dir.")

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
    if len(matching_term_indices) > 1 and context.term_index in matching_term_indices:
        occurrence = matching_term_indices.index(context.term_index) + 1
        term_type = f"{term_type}_{occurrence:02d}"

    filename = f"{phase}_epoch_{epoch}_step_{step}_batch_{batch}_slice_{slice_index:03d}_{term_type}.png"
    return plot_dir / filename


# ----------------------------------------------------------------------------------------------------------------------
def masked_reduce(residual: Tensor, mask: Tensor | None, kind: Literal["l2", "mse"]) -> Tensor:
    """
    Reduce a stack of residual fields to a per-field norm.

    This reproduces the reduction forms used by both GS losses with an explicit mask contract:

    - ``kind="l2"``: the Euclidean norm over active cells.
    - ``kind="mse"`` with ``mask is None``: the plain mean of squares over all cells.
    - ``kind="mse"`` with a mask: the mean of squares over the active (masked) cells only, i.e.
      ``sum(residual**2) / active_cells``, with fields having zero active cells set to NaN so they can be
      dropped by the caller's validity check.

    The helper applies the mask itself rather than relying on callers to zero inactive cells beforehand. The L2
    branch keeps ``vector_norm`` semantics and is deliberately not redefined as RMSE here.

    Parameters
    ----------
    residual : Tensor
        Residual fields shaped ``(F, n_r, n_z)`` or already flattened as ``(F, N)``.
    mask : Tensor | None
        Optional boolean/other mask with the same number of cells per field as ``residual``.
    kind : Literal["l2", "mse"]
        Reduction form.

    Returns
    -------
    Tensor
        Per-field norms shaped ``(F,)``.

    Raises
    ------
    ValueError
        If ``kind`` is not ``"l2"`` or ``"mse"``.

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
