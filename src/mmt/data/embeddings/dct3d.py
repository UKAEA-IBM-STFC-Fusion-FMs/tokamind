"""
DCT3D Codec
-----------

This module implements a lightweight 3D Discrete Cosine Transform (DCT) codec used for compressing
multi-dimensional time-dependent signals, together with its differentiable torch decoder.

The module provides:
    • ``DCT3DCodec``          — numpy encoder (offline use: embedding generation, index tuning)
    • ``DCT3DTorchDecoder``   — differentiable nn.Module decoder (training losses + eval)
    • ``_build_dct3d_basis``  — precomputes the (D, N) IDCT basis matrix from a codec instance

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
The inverse DCT is a linear operation: z → IDCT(scatter(z)) = z @ basis, where basis is
the (D, N) matrix of IDCT basis vectors. This matrix is:

    • Precomputed once at ``DCT3DTorchDecoder`` construction time using the existing
      numpy DCT helpers, so the torch decoder is always numerically consistent with
      the offline encoder.
    • Stored as a ``register_buffer`` with ``persistent=False``: it moves to device
      with the module but is never written to checkpoints and is recomputed from the
      codec (and its ``coeff_indices`` loaded from ``runs/<run_id>/embeddings/``) each run.

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
# Basis matrix computation
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _build_dct3d_basis(codec: DCT3DCodec, original_shape: tuple[int, ...]) -> np.ndarray:
    """
    Precompute the (D, N) IDCT basis matrix for a given codec and native signal shape.

    Each row ``basis[i]`` is the native-space signal produced by placing a single unit
    coefficient at position ``i`` in the coefficient vector and applying the inverse DCT.
    The decode operation then reduces to a single matmul: ``x_flat = z @ basis``.

    Parameters
    ----------
    codec : DCT3DCodec
        Fully initialised encoder instance. Its ``selection_mode``, ``coeff_indices``
        (rank mode), and ``keep_h/w/t`` (spatial mode) determine which rows to build.
    original_shape : tuple[int, ...]
        Native signal shape (without batch dimension), e.g. ``(T,)``, ``(C, T)``,
        ``(H, W, T)``. Must be consistent with the shape used during offline encoding.

    Returns
    -------
    np.ndarray
        Basis matrix of shape (D, N) and dtype float32, where
        D = number of coefficients and N = product of original_shape.

    Raises
    ------
    ValueError
        If ``len(original_shape)`` is not in [1, 2, 3].

    """

    if len(original_shape) == 1:
        H_full, W_full, T_full = 1, 1, original_shape[0]
    elif len(original_shape) == 2:
        H_full, W_full, T_full = original_shape[0], 1, original_shape[1]
    elif len(original_shape) == 3:
        H_full, W_full, T_full = original_shape
    else:
        raise ValueError(f"Unsupported original_shape={original_shape!r}: expected 1D, 2D, or 3D native shape.")

    N = H_full * W_full * T_full

    if codec.selection_mode == "rank":
        D = len(codec.coeff_indices)  # type: ignore[arg-type]
        basis = np.zeros((D, N), dtype=np.float32)
        for i, flat_idx in enumerate(codec.coeff_indices):  # type: ignore[union-attr]
            X_full = np.zeros((H_full, W_full, T_full), dtype=np.float32)
            X_full.reshape(-1)[flat_idx] = 1.0
            basis[i] = _idct3(X_full).reshape(-1)

    else:  # spatial
        h_eff = min(codec.keep_h, H_full)
        w_eff = min(codec.keep_w, W_full)
        t_eff = min(codec.keep_t, T_full)
        D = h_eff * w_eff * t_eff
        basis = np.zeros((D, N), dtype=np.float32)
        idx = 0
        for hi in range(h_eff):
            for wi in range(w_eff):
                for ti in range(t_eff):
                    X_full = np.zeros((H_full, W_full, T_full), dtype=np.float32)
                    X_full[hi, wi, ti] = 1.0
                    basis[idx] = _idct3(X_full).reshape(-1)
                    idx += 1

    return basis


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

    The inverse DCT is a linear operation, so the full decode is expressed as a single
    matrix multiplication: ``x_flat = z @ basis``, where ``basis`` is the (D, N) matrix
    of IDCT basis vectors precomputed from the codec.

    The basis is stored as a ``register_buffer`` with ``persistent=False``:

      - Moves to the correct device/dtype with the module.
      - Never saved to checkpoints — recomputed from the codec at construction time.
      - The codec carries the ``coeff_indices`` loaded from ``runs/<run_id>/embeddings/``,
        so the decoder is always numerically consistent with the offline encoder.

    Parameters
    ----------
    codec : DCT3DCodec
        Fully initialised encoder instance (used only at construction to build the basis).
    original_shape : tuple[int, ...]
        Native signal shape (without batch dimension), e.g. ``(T,)``, ``(C, T)``,
        ``(H, W, T)``. Must be consistent with the shape used during offline encoding.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, codec: DCT3DCodec, original_shape: tuple[int, ...]) -> None:
        super().__init__()
        self._native_shape: tuple[int, ...] = tuple(original_shape)
        basis = _build_dct3d_basis(codec=codec, original_shape=original_shape)
        # persistent=False: recomputed from codec at init, never written to checkpoints
        self.register_buffer("basis", torch.from_numpy(basis), persistent=False)

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: Tensor) -> Tensor:
        """
        Decode a batch of DCT coefficient vectors to native standardized space.

        Parameters
        ----------
        z : Tensor
            Coefficient vectors of shape (B, D).

        Returns
        -------
        Tensor
            Native standardized output of shape (B, *native_shape).
            Gradients are preserved with respect to ``z``.

        """

        x_flat = z.float() @ self.basis  # (B, N)
        return x_flat.view(z.shape[0], *self._native_shape)
