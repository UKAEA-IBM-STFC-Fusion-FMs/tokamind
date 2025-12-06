from __future__ import annotations

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
