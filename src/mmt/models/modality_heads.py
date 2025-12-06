from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ModalityHead(nn.Module):
    """
    Shared per-modality MLP: maps CLS (d_model) -> group latent (G_mod).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, layers: int = 2):
        super().__init__()
        blocks: List[nn.Module] = []
        if hidden and hidden > 0:
            blocks.append(nn.Linear(in_dim, hidden))
            blocks.append(nn.ReLU())
            for _ in range(max(0, layers - 1)):
                blocks.append(nn.Linear(hidden, hidden))
                blocks.append(nn.ReLU())
            blocks.append(nn.Linear(hidden, out_dim))
        else:
            blocks.append(nn.Linear(in_dim, out_dim))
        self.net = nn.Sequential(*blocks)
        self.out_dim = int(out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)  # (B, G_mod)
