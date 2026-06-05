"""
Base classes for the MMT embedding pipeline.

This module defines abstract interfaces shared across all codec types, keeping them decoupled from any specific
embedding implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


# ======================================================================================================================
class TorchDecoder(nn.Module, ABC):
    """
    Abstract base class for differentiable embedding decoders.

    All decoders accept a batch of embedding vectors and return the corresponding native-standardized tensors,
    preserving gradients for use in training losses.

    Subclasses implement :meth:`forward` with the signature::

        forward(z: Tensor) -> Tensor
            z:      (B, D)             — batch of embedding vectors
            return: (B, *native_shape) — native standardized output

    Implementation constraints
    --------------------------
    - No :func:`torch.no_grad`, no ``.detach()``, no ``.cpu()``, no NumPy inside :meth:`forward`. These are eval-only
      concerns handled by the caller.
    - Fixed constants (basis matrices, scatter indices) must be stored via :meth:`~torch.nn.Module.register_buffer`
      with ``persistent=False``, so they move to the correct device with the module but are never saved to checkpoints.
      They are recomputed from the codec at construction time.
    - Learned parameters (e.g., a VAE decoder network) must have ``requires_grad_(False)`` so gradients flow *through*
      the decoder to the embedding prediction ``z``, but do not update the decoder weights.

    Eval usage
    ----------
    Callers that only need a NumPy output (e.g., eval/decode.py) should wrap the forward pass in ``torch.no_grad()``
    and detach afterwards::

        with torch.no_grad():
            x_hat = decoder(z_tensor)          # (B, *native_shape) tensor
        x_hat_np = x_hat.cpu().numpy()         # hand off to xarray / file output
    """

    @abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        """
        Decode a batch of embedding vectors to native standardized space.

        Parameters
        ----------
        z : Tensor
            Embedding vectors of shape (B, D).

        Returns
        -------
        Tensor
            Native standardized output of shape (B, *native_shape). Gradients are preserved with respect to ``z``.
        """
