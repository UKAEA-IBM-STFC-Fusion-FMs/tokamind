# src/mmt/data/embeddings/identity_codec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class IdentityCodec:
    """
    Identity "encoder" / "decoder" for signal chunks.

    This codec does not perform any transform in feature space: it simply
    flattens the input array and optionally changes dtype.

    It is useful when you want to:
      - bypass DCT3D or any compression,
      - or handle certain signals in native space, while still going through
        the same embedding / caching machinery.

    Parameters
    ----------
    out_dtype : np.dtype, optional
        Dtype of the encoded vector. Calculations are trivial here, so
        out_dtype is both the internal and output dtype. Default is float32.
    """

    out_dtype: np.dtype = np.float32

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a signal chunk by flattening it.

        Parameters
        ----------
        x : np.ndarray
            Input array of any shape, e.g. (T,), (C, T), (H, W, T), ...

        Returns
        -------
        z : np.ndarray
            Flattened representation of shape (D,), dtype = out_dtype.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        return x.astype(self.out_dtype, copy=False).reshape(-1)

    def decode(self, z: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decode a flattened vector back to the original shape.

        Parameters
        ----------
        z : np.ndarray
            Encoded representation of shape (D,)
        original_shape : tuple of int
            Shape of the original input array, used for reshape.

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed signal with shape == original_shape.
        """
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)

        if z.ndim != 1:
            raise ValueError(f"Expected z of shape (D,), got {z.shape}. ")

        return z.astype(self.out_dtype, copy=False).reshape(original_shape)
