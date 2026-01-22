"""
Output adapters for MMT.

Each output signal has a small adapter network that maps from the modality
latent space (G_mod) to the target output embedding dimension (K_t).

Adapters are lightweight (linear or a tiny MLP with one hidden layer) and are
keyed by stable canonical keys ("output:<name>") to ensure predictable
checkpoint loading and warm-start across tasks.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn


class OutputAdapter(nn.Module):
    """
    Per-output adapter: group latent (G_mod) -> output embedding (K_t).
    Optionally with a tiny hidden layer.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )
        else:
            self.net = nn.Linear(in_dim, out_dim)
        self.out_dim = int(out_dim)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)  # (B, K_t)


def resolve_output_adapter_hiddens(
    *,
    output_specs: Sequence[Any],
    d_model: int,
    hidden_dim_cfg: Mapping[str, Any] | None,
) -> Dict[str, int]:
    """Resolve per-output adapter hidden dims from config.

    Validation is done in the config validator.
    Manual overrides always win.
    """
    cfg = dict(hidden_dim_cfg or {})

    default_hidden = int(cfg.get("default", 0) or 0)
    bucketed = cfg.get("bucketed") or {}
    bucket_enable = bool(bucketed.get("enable", False))
    rules = bucketed.get("rules") or []
    manual = {str(k): v for k, v in (cfg.get("manual") or {}).items()}

    def _to_hidden(v):
        return int(d_model) if v == "d_model" else int(v)

    out: Dict[str, int] = {}
    for spec in output_specs:
        name = str(getattr(spec, "name"))
        out_dim = int(getattr(spec, "embedding_dim"))
        hidden = default_hidden

        if bucket_enable:
            for r in rules:
                max_out = r.get("max_out_dim")
                if max_out is None or out_dim <= int(max_out):
                    hidden = _to_hidden(r.get("hidden", default_hidden))
                    break

        if name in manual:
            hidden = _to_hidden(manual[name])

        out[name] = int(hidden)
    return out
