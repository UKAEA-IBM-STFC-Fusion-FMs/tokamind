"""
DCT3D Codec
-----------

This module implements a lightweight 3D Discrete Cosine Transform (DCT) codec used for compressing
multi-dimensional time-dependent signals, together with its differentiable torch decoder.

The module provides:
    • ``DCT3DCodec``          — numpy encoder (offline use: embedding generation, index tuning)
    • ``DCT3DTorchDecoder``   — differentiable nn.Module decoder (training losses + eval)

Encoder design
--------------
    • **Orthonormal transform** (energy-preserving)
      MSE in coefficient space equals MSE in native space, up to a (K/N) scale factor.
    • **Fixed-size latent representation**
      Truncation to (keep_h, keep_w, keep_t) or ranked coefficient selection produces a
      stable embedding dimension.
    • **Shape-robust interface**
      Inputs of shape (T,), (C, T), or (H, W, T) are all handled internally.

Decoder design
--------------
The torch decoder mirrors the numpy codec: it scatters the predicted coefficients into
the full DCT tensor, then applies an orthonormal inverse DCT along the three native
axes using ``torch.fft``. This avoids materialising a huge dense ``(D, N)`` IDCT basis
matrix while preserving gradients for native-space losses.

Usage
-----
    # Offline: encode (numpy)
    codec = DCT3DCodec(keep_h=8, keep_w=8, keep_t=16)
    z = codec.encode(x)                              # x: np.ndarray

    # Online: decode (torch, differentiable)
    decoder = DCT3DTorchDecoder(codec, original_shape=x.shape)
    decoder.to(device)
    x_hat = decoder(z_tensor)                        # z_tensor: (B, D)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.fftpack import dct, idct
from torch import Tensor

import torch.nn.functional as F

from .torch_decoder import TorchDecoder


# ======================================================================================================================
# Private math helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _dct3(x: np.ndarray) -> np.ndarray:
    """3D DCT (type-II, orthonormal) over the last 3 axes of `x`."""
    y = dct(x, type=2, axis=-1, norm="ortho")
    y = dct(y, type=2, axis=-2, norm="ortho")
    y = dct(y, type=2, axis=-3, norm="ortho")
    return y


# ----------------------------------------------------------------------------------------------------------------------
def _idct3(x: np.ndarray) -> np.ndarray:
    """3D inverse DCT (type-II, orthonormal) over the last 3 axes of `x`."""
    y = idct(x, type=2, axis=-1, norm="ortho")
    y = idct(y, type=2, axis=-2, norm="ortho")
    y = idct(y, type=2, axis=-3, norm="ortho")
    return y


# ----------------------------------------------------------------------------------------------------------------------
def _to_3d_view(x: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """
    Convert a 1D / 2D / 3D array into a (H, W, T) view and return also the original shape.

    Conventions:
      - (T,)      -> (1, 1, T)
      - (C, T)    -> (C, 1, T)
      - (H, W, T) -> (H, W, T)

    Parameters
    ----------
    x : np.ndarray
        Input array to be turned into (H, W, T) view.

    Returns
    -------
    tuple[np.ndarray, tuple[int, ...]]
        3D view of input ``x`` array, along with the original shape.

    Raises
    ------
    ValueError
        If ``x`` is not a 1D/2D/3D array.

    """

    if x.ndim == 1:
        H, W, T = 1, 1, x.shape[0]
        x3 = x.reshape(H, W, T)
    elif x.ndim == 2:
        H, W, T = x.shape[0], 1, x.shape[1]
        x3 = x.reshape(H, W, T)
    elif x.ndim == 3:
        x3 = x
    else:
        raise ValueError(f"DCT3DCodec only supports 1D/2D/3D inputs, got shape={x.shape}.")

    return x3, x.shape


# ======================================================================================================================
# Torch inverse DCT helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _shape_to_3d(original_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """Return the canonical ``(H, W, T)`` view for a native signal shape."""

    if len(original_shape) == 1:
        return 1, 1, int(original_shape[0])
    if len(original_shape) == 2:
        return int(original_shape[0]), 1, int(original_shape[1])
    if len(original_shape) == 3:
        return int(original_shape[0]), int(original_shape[1]), int(original_shape[2])
    raise ValueError(f"Unsupported original_shape={original_shape!r}: expected 1D, 2D, or 3D native shape.")


# ----------------------------------------------------------------------------------------------------------------------
def _torch_idct_ortho_last(x: Tensor) -> Tensor:
    """Inverse DCT-II with ``norm='ortho'`` along the last dimension, matching ``scipy.fftpack.idct``."""

    original_shape = x.shape
    n = int(original_shape[-1])
    if n == 1:
        return x.clone()

    original_dtype = x.dtype
    x_work = x.contiguous().reshape(-1, n)
    if x_work.dtype not in (torch.float32, torch.float64):
        x_work = x_work.float()

    x_v = (x_work / 2.0).clone()
    x_v[:, 0] *= (n**0.5) * 2.0
    x_v[:, 1:] *= ((n / 2.0) ** 0.5) * 2.0

    # irfft only reads the first m = n//2+1 complex values; build only those.
    m = n // 2 + 1
    k = torch.arange(m, dtype=x_v.dtype, device=x_v.device).reshape(1, m) * np.pi / (2.0 * n)
    w_r = torch.cos(k)
    w_i = torch.sin(k)

    v_t_r = x_v[:, :m]
    v_t_i = torch.cat([x_v[:, :1].new_zeros((x_v.shape[0], 1)), -x_v.flip(dims=[1])[:, : m - 1]], dim=1)
    v_r = v_t_r * w_r - v_t_i * w_i
    v_i = v_t_r * w_i + v_t_i * w_r

    v = torch.fft.irfft(torch.complex(v_r, v_i), n=n, dim=1)
    out = v.new_empty(v.shape)
    out[:, ::2] = v[:, : n - (n // 2)]
    out[:, 1::2] = v.flip(dims=[1])[:, : n // 2]

    return out.reshape(original_shape).to(dtype=original_dtype)


# ----------------------------------------------------------------------------------------------------------------------
def _torch_idct_ortho_dim(x: Tensor, dim: int) -> Tensor:
    """Inverse DCT-II with ``norm='ortho'`` along an arbitrary dimension."""

    dim = dim % x.ndim
    if dim == x.ndim - 1:
        return _torch_idct_ortho_last(x)
    return _torch_idct_ortho_last(torch.movedim(x, dim, -1)).movedim(-1, dim)


# ----------------------------------------------------------------------------------------------------------------------
def _torch_idct3(x: Tensor) -> Tensor:
    """3D inverse DCT-II with ``norm='ortho'`` over the last three axes."""

    y = _torch_idct_ortho_dim(x, dim=-1)
    y = _torch_idct_ortho_dim(y, dim=-2)
    y = _torch_idct_ortho_dim(y, dim=-3)
    return y


# ======================================================================================================================
# Encoder
# ======================================================================================================================


# ======================================================================================================================
@dataclass
class DCT3DCodec:
    """
    3D DCT-based encoder for time-dependent signals.

    This codec supports the three canonical signal shapes:
      - timeseries: (T,)
      - profile:    (C, T)
      - video/map:  (H, W, T)

    Internally, all inputs are viewed as (H, W, T), a 3D DCT is applied, and coefficients are
    selected using one of two modes:

    **Spatial mode** (default):
      Keeps the top-left-front (keep_h, keep_w, keep_t) block of DCT coefficients.

    **Rank mode**:
      Keeps the top-K coefficients by explained variance (energy), regardless of spatial
      position. Requires ``coeff_indices``.

    Parameters
    ----------
    keep_h : int
        Number of DCT coefficients to keep along the "H" dimension (spatial mode).
    keep_w : int
        Number of DCT coefficients to keep along the "W" dimension (spatial mode).
    keep_t : int
        Number of DCT coefficients to keep along the "T" (time) dimension (spatial mode).
    dtype : np.dtype
        Data type for encoded coefficients (default: float32).
    selection_mode : str
        Coefficient selection strategy: ``"spatial"`` or ``"rank"`` (default: ``"spatial"``).
    coeff_indices : np.ndarray | None
        1D array of coefficient indices for rank mode. Required if ``selection_mode="rank"``.
    coeff_shape : tuple[int, int, int] | None
        Expected (H, W, T) shape for validation in rank mode (optional).
    requires_finite_input : bool
        Whether the codec requires finite (non-NaN) inputs. Always ``True`` for DCT3D: the
        DCT transform is undefined for NaN values.

    Notes
    -----
    - **Spatial mode**: actual coefficients kept = min(keep_h, H) * min(keep_w, W) * min(keep_t, T)
    - **Rank mode**: coefficients kept = ``len(coeff_indices)``
    - The encoder returns a (D,) array.

    """

    keep_h: int
    keep_w: int
    keep_t: int
    dtype: type = np.float32
    selection_mode: str = "spatial"
    coeff_indices: np.ndarray | None = None
    coeff_shape: tuple[int, int, int] | None = None
    requires_finite_input: bool = True

    # ------------------------------------------------------------------------------------------------------------------
    def __post_init__(self):
        """
        Validate codec parameters.

        Raises
        ------
        ValueError
            If ``self.selection_mode`` not in ``["spatial", "rank"]``.
            If ``self.coeff_indices`` is None when ``selection_mode="rank"``.
            If ``self.coeff_indices`` is not a 1D array when ``selection_mode="rank"``.
            If ``self.coeff_indices`` is empty when ``selection_mode='rank'``.
            If ``self.coeff_indices`` contains negative integers when ``selection_mode='rank'``.

        """

        if self.selection_mode not in ["spatial", "rank"]:
            raise ValueError(f"`selection_mode` must be 'spatial' or 'rank', got {self.selection_mode!r}.")

        if self.selection_mode == "rank":
            if self.coeff_indices is None:
                raise ValueError("`coeff_indices` required when `selection_mode='rank'`.")

            coeff_indices = np.asarray(self.coeff_indices, dtype=np.int32)
            self.coeff_indices = coeff_indices

            if coeff_indices.ndim != 1:
                raise ValueError(
                    f"`coeff_indices` must be 1D array when `selection_mode='rank'`, got shape {coeff_indices.shape}."
                )

            if len(coeff_indices) == 0:
                raise ValueError("`coeff_indices` cannot be empty when `selection_mode='rank'`.")

            if np.any(coeff_indices < 0):
                raise ValueError("`coeff_indices` must contain non-negative integers when `selection_mode='rank'`.")

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def keep_shape(self) -> tuple[int, int, int]:
        """Requested (keep_h, keep_w, keep_t). Actual kept dims depend on input."""
        return self.keep_h, self.keep_w, self.keep_t

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single signal chunk.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (T,), (C, T), or (H, W, T).

        Returns
        -------
        z : np.ndarray
            Encoded representation of shape (D,), dtype = self.dtype.
            D = h_eff * w_eff * t_eff (spatial mode) or len(coeff_indices) (rank mode).

        Raises
        ------
        ValueError
            If input shape mismatches ``self.coeff_shape`` in rank mode.
            If ``self.coeff_indices`` contains out-of-bounds indices.

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa - Ignore unreachable code warning

        x = x.astype(self.dtype, copy=False)
        x3, _orig_shape = _to_3d_view(x)
        H, W, T = x3.shape  # NOSONAR # noqa - Ignore lowercase warning

        X = _dct3(x3)  # NOSONAR # noqa - Ignore lowercase warning

        if self.selection_mode == "rank":
            if self.coeff_shape is not None:
                expected_H, expected_W, expected_T = self.coeff_shape  # NOSONAR # noqa - Ignore lowercase warning
                if (H, W, T) != (expected_H, expected_W, expected_T):
                    raise ValueError(
                        f"Input shape mismatch: expected coeff_shape={self.coeff_shape}, "
                        f"got (H,W,T)={H, W, T} from input shape {x.shape}."
                    )

            X_flat = X.reshape(-1)  # NOSONAR # noqa - Ignore lowercase warning
            max_idx = H * W * T
            if np.any(self.coeff_indices >= max_idx):
                raise ValueError(f"`coeff_indices` contains out-of-bounds indices (max={max_idx - 1}).")

            z = X_flat[self.coeff_indices].astype(self.dtype, copy=False)

        else:
            h_eff = min(self.keep_h, H)
            w_eff = min(self.keep_w, W)
            t_eff = min(self.keep_t, T)
            X_crop = X[:h_eff, :w_eff, :t_eff]  # NOSONAR # noqa - Ignore lowercase warning
            z = X_crop.reshape(-1).astype(self.dtype, copy=False)

        return z


# ======================================================================================================================
# Differentiable decoder
# ======================================================================================================================


# ======================================================================================================================
class DCT3DTorchDecoder(TorchDecoder):
    """
    Differentiable DCT3D decoder for training losses and eval.

    The decoder mirrors :meth:`DCT3DCodec.decode` in torch: predicted coefficients are scattered into a full DCT
    tensor and decoded with separable orthonormal inverse DCT operations. This avoids the previous dense ``(D, N)``
    basis matrix, which is prohibitively expensive for large sparse/ranked outputs such as ``(18, 1, 5000)``.

    Parameters
    ----------
    codec : DCT3DCodec
        Fully initialised encoder instance. Its ``selection_mode``, ``coeff_indices`` and ``keep_h/w/t`` determine how
        the coefficient vector is scattered back into DCT space.
    original_shape : tuple[int, ...]
        Native signal shape (without batch dimension), e.g. ``(T,)``, ``(C, T)``,
        ``(H, W, T)``. Must be consistent with the shape used during offline encoding.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, codec: DCT3DCodec, original_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._native_shape: tuple[int, ...] = tuple(original_shape)
        self._full_shape: tuple[int, int, int] = _shape_to_3d(self._native_shape)
        self._selection_mode = codec.selection_mode

        h_full, w_full, t_full = self._full_shape

        if codec.selection_mode == "rank":
            if codec.coeff_shape is not None and tuple(codec.coeff_shape) != self._full_shape:
                raise ValueError(
                    f"Shape mismatch: expected coeff_shape={codec.coeff_shape}, got {self._full_shape} from "
                    f"original_shape={original_shape!r}."
                )
            coeff_indices = torch.as_tensor(np.asarray(codec.coeff_indices, dtype=np.int64), dtype=torch.long)
            self.register_buffer("coeff_indices", coeff_indices, persistent=False)
            self._encoded_dim = int(coeff_indices.numel())
            self._spatial_shape: tuple[int, int, int] | None = None
        else:
            h_eff = min(int(codec.keep_h), h_full)
            w_eff = min(int(codec.keep_w), w_full)
            t_eff = min(int(codec.keep_t), t_full)
            self._spatial_shape = (h_eff, w_eff, t_eff)
            self._encoded_dim = h_eff * w_eff * t_eff

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: Tensor) -> Tensor:
        """
        Decode a batch of DCT coefficient vectors to native standardized space.

        Parameters
        ----------
        z : Tensor
            Coefficient vectors of shape ``(B, D)``.

        Returns
        -------
        Tensor
            Native standardized output of shape ``(B, *native_shape)``.
            Gradients are preserved with respect to ``z``.

        """

        if z.ndim != 2:
            raise ValueError(f"Expected z of shape (B, D), got {tuple(z.shape)}.")
        if int(z.shape[1]) != self._encoded_dim:
            raise ValueError(f"Expected z.shape[1]={self._encoded_dim}, got {int(z.shape[1])}.")

        z = z.float()
        batch_size = int(z.shape[0])
        h_full, w_full, t_full = self._full_shape

        if self._selection_mode == "rank":
            x_flat = z.new_zeros((batch_size, h_full * w_full * t_full))
            x_flat.scatter_(dim=1, index=self.coeff_indices.view(1, -1).expand(batch_size, -1), src=z)  # type: ignore[attr-defined]
            x_full = x_flat.view(batch_size, h_full, w_full, t_full)
        else:
            if self._spatial_shape is None:
                raise RuntimeError("Spatial DCT decoder is missing _spatial_shape.")
            h_eff, w_eff, t_eff = self._spatial_shape
            x_full = F.pad(
                z.view(batch_size, h_eff, w_eff, t_eff),
                (0, t_full - t_eff, 0, w_full - w_eff, 0, h_full - h_eff),
            )

        x_native = _torch_idct3(x_full)
        return x_native.view(batch_size, *self._native_shape)
