"""
Transformer backbone for MMT.

A thin wrapper around PyTorch's nn.TransformerEncoder (batch_first=True).
Kept as its own module to support clear checkpointing, freezing, and warm-start behaviour independent of the
TokenEncoder and task-specific heads.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ======================================================================================================================
class Backbone(nn.Module):
    """
    Thin wrapper around nn.TransformerEncoder.

    Keeps the Backbone as its own module so we can easily freeze / save it.

    Attributes
    ----------
    encoder : nn.TransformerEncoder
        Built instance of TransformerEncoder class.

    Methods
    -------
    forward(x, src_key_padding_mask)
        Backbone's forward function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_ff: int,
        n_layers: int,
        dropout: float,
        activation: str = "relu",
    ) -> None:
        """
        Initialize class parameters.

        Parameters
        ----------
        d_model : int
            The number of expected features in the input, to be passed to the `nn.TransformerEncoderLayer` constructor.
        n_heads : int
            Number of heads.
        dim_ff : int
            The dimension of the feedforward network model, to be passed to the `nn.TransformerEncoderLayer`
            constructor.
        n_layers : int
            Number of layers.
        dropout : float
            The dropout value, to be passed to the `nn.TransformerEncoderLayer` constructor.
        activation : str
            The activation function, to be passed to the `nn.TransformerEncoderLayer` constructor.
            Optional. Default: "relu".

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        """

        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation=activation,  # Explicit. Default "relu".
        )

        # NOTE:
        # PyTorch's nested tensor API is still marked as prototype and may emit warnings when TransformerEncoder
        # internally constructs nested tensors. Disabling nested tensors keeps behaviour stable and avoids the warning.
        try:
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
        except TypeError:
            # Backwards compatibility for older PyTorch versions.
            self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, x: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Backbone's forward function.

        Parameters
        ----------
        x : torch.Tensor
            The sequence to the encoder.
        src_key_padding_mask : Optional[torch.Tensor]
            The mask for the `x` keys per batch.
            Optional. Default: None.

        Returns
        -------
        torch.Tensor
            Forward pass over specified `x` and `src_key_padding_mask`.

        """

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    # ------------------------------------------------------------------------------------------------------------------
