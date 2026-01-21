"""
Transformer backbone for MMT.

A thin wrapper around PyTorch's nn.TransformerEncoder (batch_first=True).
Kept as its own module to support clear checkpointing, freezing, and warm-start
behaviour independent from the TokenEncoder and task-specific heads.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional


class Backbone(nn.Module):
    """
    Thin wrapper around nn.TransformerEncoder.

    Keeps the Backbone as its own module so we can easily freeze / save it.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        n_layers: int,
        dropout: float,
        activation: str = "relu",
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation=activation,  # explicit, default "relu"
        )
        # NOTE:
        # PyTorch's nested tensor API is still marked as prototype and may emit
        # warnings when TransformerEncoder internally constructs nested tensors.
        # Disabling nested tensors keeps behaviour stable and avoids the warning.
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # Backwards compatibility for older PyTorch versions.
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
