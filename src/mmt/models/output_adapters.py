"""
Output adapters for MMT.

Each output signal has a small adapter network that maps from the modality
latent space (G_mod) to the target output embedding dimension (K_t).

Adapters are lightweight (linear or a tiny MLP with one hidden layer) and are
keyed by stable canonical keys ("output:<name>") to ensure predictable
checkpoint loading and warm-start across tasks.
"""

from __future__ import annotations

from typing import Any, Hashable
from collections.abc import Iterable, Mapping, Sequence

import torch
import torch.nn as nn


OutputAdapterItem = tuple[Hashable, str, torch.Tensor]


# ----------------------------------------------------------------------------------------------------------------------
def _activation_layer(activation: str) -> nn.Module:
    """Build the requested nonlinearity for a small output adapter."""

    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported output-adapter activation={activation!r}.")


# ======================================================================================================================
class OutputAdapter(nn.Module):
    """
    Per-output adapter: group latent (G_mod) -> output embedding (K_t).
    Optionally with a tiny hidden layer.

    Attributes
    ----------
    net = nn.Sequential
        ModalityHead's network.
    out_dim : int
        Output dimension.

    Methods
    -------
    forward(h)
        OutputAdapter's forward function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0, activation: str = "relu"):
        """

        Initialize class attributes.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Dimension of hidden layer.
        activation : str
            Activation applied after the optional hidden projection. Supported
            values are ``"relu"`` and ``"gelu"``.
            Optional. Default: ``"relu"``.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                _activation_layer(activation),
                nn.Linear(in_features=hidden_dim, out_features=out_dim),
            )
        else:
            self.net = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.out_dim = int(out_dim)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        OutputAdapter's forward function.

        Parameters
        ----------
        h : torch.Tensor
            Input for the network.

        Returns
        -------
        torch.Tensor
            Forward pass over specified `h`.

        """

        return self.net(h)  # (B, K_t)


# ----------------------------------------------------------------------------------------------------------------------
def apply_output_adapters(
    *,
    output_adapters: nn.ModuleDict,
    items: Iterable[OutputAdapterItem],
    output_adapter_type: str = "deterministic",
) -> dict[str, Any]:
    """Apply per-output adapters and return predictions keyed by output signal ID."""

    if output_adapter_type != "deterministic":
        raise ValueError(f"Unsupported output_adapter_type={output_adapter_type!r}. Expected 'deterministic'.")
    preds: dict[Hashable, torch.Tensor] = {}
    for out_key, adapter_key, h_out in items:
        adapter = output_adapters[adapter_key]
        if not isinstance(adapter, OutputAdapter):
            raise TypeError(f"Expected OutputAdapter for adapter_key={adapter_key!r}.")
        preds[out_key] = adapter(h_out)
    return {"pred": preds}


# ----------------------------------------------------------------------------------------------------------------------
def apply_output_residual(
    *,
    adapter_output: dict[str, Any],
    baseline_emb: Mapping[Hashable, torch.Tensor] | None,
    residual_output_ids: set[Hashable],
) -> dict[str, Any]:
    """
    Add output-space persistence baselines to predicted corrections.

    Parameters
    ----------
    adapter_output : dict[str, Any]
        Result from :func:`apply_output_adapters`.
    baseline_emb : Mapping[Hashable, torch.Tensor] | None
        Batched baseline embeddings keyed by output signal ID.
    residual_output_ids : set[Hashable]
        Output IDs whose adapters represent residual corrections.

    Returns
    -------
    dict[str, Any]
        Adapter output with absolute predictions.

    Raises
    ------
    KeyError
        If residual outputs are enabled but a required baseline is missing.
    ValueError
        If a baseline and prediction have different shapes.
    """

    if not residual_output_ids:
        return adapter_output
    if not isinstance(baseline_emb, Mapping):
        raise KeyError("Residual outputs require batch['output_baseline_emb'].")

    preds = adapter_output.get("pred") or {}
    for output_id in residual_output_ids:
        if output_id not in preds:
            continue
        if output_id not in baseline_emb:
            raise KeyError(f"Missing residual baseline for output signal ID {output_id!r}.")

        pred = preds[output_id]
        baseline = baseline_emb[output_id].to(device=pred.device, dtype=pred.dtype)
        if baseline.shape != pred.shape:
            raise ValueError(
                f"Residual baseline shape {tuple(baseline.shape)} != prediction shape {tuple(pred.shape)} "
                f"for output signal ID {output_id!r}."
            )

        absolute = pred + baseline
        preds[output_id] = absolute

    adapter_output["pred"] = preds
    return adapter_output


# ----------------------------------------------------------------------------------------------------------------------
def zero_initialize_output_corrections(*, output_adapters: nn.ModuleDict, adapter_keys: Iterable[str]) -> None:
    """
    Zero the final prediction layer for residual output adapters.

    Parameters
    ----------
    output_adapters : nn.ModuleDict
        Output adapter modules keyed by canonical output key.
    adapter_keys : Iterable[str]
        Adapter keys whose predictions represent residual corrections.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If an adapter has no supported prediction network.
    """

    for adapter_key in adapter_keys:
        adapter = output_adapters[adapter_key]
        net = getattr(adapter, "net", None)
        last: nn.Linear | None = None
        if isinstance(net, nn.Linear):
            last = net
        elif isinstance(net, nn.Sequential) and isinstance(net[-1], nn.Linear):
            last = net[-1]
        if last is None:
            raise TypeError(f"Output adapter {adapter_key!r} has no supported prediction layer.")

        with torch.no_grad():
            last.weight.zero_()
            if last.bias is not None:
                last.bias.zero_()


# ----------------------------------------------------------------------------------------------------------------------
def resolve_output_adapter_hiddens(  # NOSONAR - Ignore cognitive complexity
    *,
    output_specs: Sequence[Any],
    d_model: int,
    hidden_dim_cfg: Mapping[str, Any] | None,
) -> dict[str, int]:
    """
    Resolve per-output adapter hidden dims from config.

    Validation is done in the config validator. Manual overrides always win.

    Parameters
    ----------
    output_specs : Sequence[Any]
        List of output specifications used for resolution of adapter hidden dims.
    d_model : int
        The number of expected features in the input.
    hidden_dim_cfg : Mapping[str, Any]
        Mapping of adapter hidden dims.
        Optional. Default: None.

    Returns
    -------
    dict[str, int]
        Dictionary with resolved per-output adapter hidden dims.

    """

    # ..................................................................................................................
    def _to_hidden_dim(v: str | int):
        """Resolve a hidden dim value: return `d_model` if the value is the string "d_model", else cast to int."""
        return int(d_model) if (v == "d_model") else int(v)

    # ..................................................................................................................

    cfg = dict(hidden_dim_cfg or {})

    default_hidden_dim = int(cfg.get("default", 0) or 0)
    bucketed = cfg.get("bucketed") or {}
    bucket_enable = bool(bucketed.get("enable", False))
    rules = bucketed.get("rules") or []
    manual = {str(k): v for k, v in (cfg.get("manual") or {}).items()}

    out: dict[str, int] = {}
    for spec in output_specs:
        name = str(getattr(spec, "name"))
        out_dim = int(getattr(spec, "embedding_dim"))
        hidden_dim = default_hidden_dim

        if bucket_enable:
            for r in rules:
                max_out = r.get("max_out_dim")
                if (max_out is None) or (out_dim <= int(max_out)):
                    hidden_dim = _to_hidden_dim(v=r.get("hidden", default_hidden_dim))
                    break

        if name in manual:
            hidden_dim = _to_hidden_dim(v=manual[name])

        out[name] = int(hidden_dim)

    return out
