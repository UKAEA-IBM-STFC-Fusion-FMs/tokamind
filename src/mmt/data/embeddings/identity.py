"""
Identity Codec
--------------

This module provides a pass-through codec for the MMT embedding pipeline, together with
its differentiable torch decoder.

The module provides:
    • ``IdentityCodec``          — numpy encoder (offline use: flattens the input chunk)
    • ``IdentityTorchDecoder``   — differentiable nn.Module decoder (training losses + eval)

The identity codec performs no feature transform beyond flattening (and optional dtype cast).
It is useful for:
    - Bypassing compression for specific signals.
    - Debugging the end-to-end token/embedding pipeline.
    - Keeping signals in native space while still using the same embedding interfaces.

Usage
-----
    # Offline: encode (numpy)
    codec = IdentityCodec()
    z = codec.encode(x)                              # x: np.ndarray, z: flat (D,)

    # Online: decode (torch, differentiable)
    decoder = IdentityTorchDecoder(original_shape=x.shape)
    x_hat = decoder(z_tensor)                        # z_tensor: (B, D)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from .torch_decoder import TorchDecoder


# ======================================================================================================================
# Encoder
# ======================================================================================================================


# ======================================================================================================================
@dataclass
class IdentityCodec:
    """
    Identity encoder for signal chunks.

    Flattens the input array and optionally casts its dtype. No transform is applied.

    Parameters
    ----------
    out_dtype : np.dtype
        Dtype of the encoded vector. Default: float32.
    requires_finite_input : bool
        Whether the codec requires finite (non-NaN) inputs. ``True`` for IdentityCodec: NaN values
        would propagate into ``output_emb`` and break embedding-space losses such as ``embed_mse``.

    """

    out_dtype: type = np.float32
    requires_finite_input: bool = True

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a signal chunk by flattening it.

        Parameters
        ----------
        x : np.ndarray
            Input array of any shape, e.g., (T,), (C, T), (H, W, T).

        Returns
        -------
        z : np.ndarray
            Flattened representation of shape (D,), dtype = out_dtype.

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa - Ignore unreachable code warning

        return x.astype(self.out_dtype, copy=False).reshape(-1)


# ======================================================================================================================
# Differentiable decoder
# ======================================================================================================================


# ======================================================================================================================
class IdentityTorchDecoder(TorchDecoder):
    """
    Differentiable identity decoder for training losses and eval.

    Reshapes the flat embedding vector back to the original native shape.
    This is a no-op beyond a reshape and requires no parameters or buffers.

    Parameters
    ----------
    original_shape : tuple[int, ...]
        Native signal shape (without batch dimension), e.g. ``(T,)``, ``(C, T)``,
        ``(H, W, T)``.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, original_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._native_shape: tuple[int, ...] = tuple(original_shape)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: Tensor) -> Tensor:
        """
        Reshape a batch of flat embedding vectors to native standardized space.

        Parameters
        ----------
        z : Tensor
            Flat embedding vectors of shape (B, D).

        Returns
        -------
        Tensor
            Native standardized output of shape (B, *native_shape).
            Gradients are preserved with respect to ``z``.

        """

        return z.view(z.shape[0], *self._native_shape)
