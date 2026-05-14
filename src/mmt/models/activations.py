"""
Custom activation functions for MMT models.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ======================================================================================================================
class WaveletActivation(nn.Module):
    """
    PINNsFormer's Wavelet activation function.

    Source: PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks
    https://arxiv.org/abs/2307.11833

    Attributes
    ----------
    w1 : nn.Parameter
        Learnable weight for the sine component.
    w2 : nn.Parameter
        Learnable weight for the cosine component.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

    # ------------------------------------------------------------------------------------------------------------------
