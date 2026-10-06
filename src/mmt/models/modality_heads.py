"""
Modality heads for MMT.

A modality head is a small MLP that maps the pooled transformer representation (CLS token, size d_model) into a
modality-specific latent space (G_mod).

These heads provide a shared representation per modality (e.g., timeseries, profile, video), which is then consumed by
per-output adapters to produce signal-specific predictions.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ======================================================================================================================
class ModalityHead(nn.Module):
    """
    Shared per-modality MLP: maps CLS (d_model) -> group latent (G_mod).

    Attributes
    ----------
    net = nn.Sequential
        ModalityHead's network.
    out_dim : int
        Output dimension.

    Methods
    -------
    forward(z)
        ModalityHead's forward function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 128, layers: int = 2):
        """
        Initialize class parameters.

        Parameters
        ----------
        in_dim : int
            Input dimension.
        out_dim : int
            Output dimension.
        hidden_dim : int
            Dimension of hidden layers.
        layers : int
            Number of hidden layers.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes.

        """

        super().__init__()
        blocks: list[nn.Module] = []
        if hidden_dim and (hidden_dim > 0):
            blocks.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
            blocks.append(nn.ReLU())
            for _ in range(max(0, layers - 1)):
                blocks.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                blocks.append(nn.ReLU())
            blocks.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))
        else:
            blocks.append(nn.Linear(in_features=in_dim, out_features=out_dim))

        self.net = nn.Sequential(*blocks)
        self.out_dim = int(out_dim)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        ModalityHead's forward function.

        Parameters
        ----------
        z : torch.Tensor
            Input for the network.

        Returns
        -------
        torch.Tensor
            Forward pass over provided input.

        """

        return self.net(z)  # (B, G_mod)
