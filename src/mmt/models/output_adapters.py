"""
Output adapters for MMT.

Each output signal has a small adapter network that maps from the modality
latent space (G_mod) to the target output embedding dimension (K_t).

Adapters are lightweight (linear or a tiny MLP with one hidden layer) and are
keyed by stable canonical keys ("output:<name>") to ensure predictable
checkpoint loading and warm-start across tasks.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn


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
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 0):
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

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        if hidden_dim and hidden_dim > 0:
            self.net = nn.Sequential(
                nn.Linear(in_features=in_dim, out_features=hidden_dim),
                nn.ReLU(),
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
                    hidden_dim = _to_hidden_dim(v=r.get("hidden_dim", default_hidden_dim))
                    break

        if name in manual:
            hidden_dim = _to_hidden_dim(v=manual[name])

        out[name] = int(hidden_dim)

    return out
