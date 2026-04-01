"""
DCT3D Codec
-----------

This module implements a lightweight 3D Discrete Cosine Transform (DCT) codec used for compressing and reconstructing
multi-dimensional time-dependent signals.

The codec provides:
    • A 3D orthonormal DCT transform (type-II) applied along (H, W, T)
    • Optional coefficient truncation via (keep_h, keep_w, keep_t)
    • Flattened coefficient vectors for storage or model input
    • Exact inverse transform with zero-padding of truncated coefficients
    • Automatic handling of 1D / 2D / 3D signals by reshaping to (H, W, T)

Design goals
------------
    • **Orthonormal transform** (energy-preserving)
      Ensures that MSE in coefficient space matches MSE in native space.

    • **Fixed-size latent representation**
      Truncation to (keep_h, keep_w, keep_t) produces a stable, task-controlled embedding dimension.

    • **Non-destructive decode**
      Missing coefficients are padded with zeros before IDCT, providing a consistent reconstruction even when inputs
      have different native shapes.

    • **Shape-robust interface**
      Inputs of shape (T), (H, T), or (H, W, T) are all supported and internally normalized to a 3D view.

Usage
-----
    codec = DCT3DCodec(keep_h=8, keep_w=8, keep_t=16)
    z = codec.encode(x)                 # x: np.ndarray
    x_hat = codec.decode(z, x.shape)

The module also includes a small demo/test suite (isolated inside private functions) that can be executed directly:
    $ python dct3d_codec.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy.fftpack import dct, idct


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
    Convert a 1D / 2D / 3D array into a (H, W, T) view and return also the original shape (for reconstruction).

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
        3D view of input `x` array, along with the original shape.

    Raises
    ------
    ValueError
        If `x` is not a 1D/2D/3D array.

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


# ----------------------------------------------------------------------------------------------------------------------
def _from_3d_view(x3: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
    """
    Restore original shape from a (H, W, T) array using the original_shape convention of `_to_3d_view`.

    Parameters
    ----------
    x3 : np.ndarray
        Input (H, W, T) array to be restored to original shape.
    original_shape : tuple[int, ...]
        Original shape of the input `x3` array.

    Returns
    -------
    np.ndarray
        Restored array from 3D (H, W, T) view.

    Raises
    ------
    ValueError
        If shape `len(original_shape)` is not in [1, 2, 3].

    """

    if len(original_shape) == 1:
        # (1, 1, T) -> (T,)
        return x3.reshape(original_shape)
    if len(original_shape) == 2:
        # (C, 1, T) -> (C, T)
        return x3.reshape(original_shape)
    if len(original_shape) == 3:
        # (H, W, T) -> (H, W, T)
        return x3.reshape(original_shape)

    raise ValueError(f"Unsupported `original_shape={original_shape!r}` in from_3d_view.")


# ======================================================================================================================
@dataclass
class DCT3DCodec:
    """
    3D DCT-based encoder/decoder for time-dependent signals.

    This codec supports the three canonical signal shapes:
      - timeseries: (T,)
      - profile:    (C, T)
      - video/map:  (H, W, T)

    Internally, all inputs are viewed as (H, W, T), a 3D DCT is applied, and coefficients are selected using one of two
    modes:

    **Spatial mode** (default):
      Keeps the top-left-front (keep_h, keep_w, keep_t) block of DCT coefficients (low-frequency components).

    **Rank mode**:
      Keeps the top-K coefficients by explained variance (energy), regardless of spatial position. Requires
      coeff_indices parameter.

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
        Coefficient selection strategy: "spatial" or "rank" (default: "spatial").
    coeff_indices : np.ndarray | None
        1D array of coefficient indices for rank mode. Required if selection_mode="rank".
    coeff_shape : tuple[int, int, int] | None
        Expected (H, W, T) shape for validation in rank mode (optional).

    Notes
    -----
    - **Spatial mode**: The actual number of coefficients kept is:
          D = min(keep_h, H) * min(keep_w, W) * min(keep_t, T)

    - **Rank mode**: The number of coefficients is:
          D = len(coeff_indices)

    - The encoder returns a (D,) array.
    - The decoder requires the original input shape and reconstructs via zero-padding in DCT space + inverse DCT.

    """

    keep_h: int
    keep_w: int
    keep_t: int
    dtype: type = np.float32
    selection_mode: str = "spatial"
    coeff_indices: Optional[np.ndarray] = None
    coeff_shape: Optional[tuple[int, int, int]] = None

    # ------------------------------------------------------------------------------------------------------------------
    def __post_init__(self):
        """
        Validate codec parameters.

        Raises
        ------
        ValueError
            If `self.selection_mode` not in ["spatial", "rank"].
            If `self.coeff_indices` is None when `selection_mode="rank"`.
            If `self.coeff_indices` is not a 1D array when `selection_mode="rank"`.
            If `self.coeff_indices` is  empty when `selection_mode='rank'`.
            If `self.coeff_indices` contains negative integers when `selection_mode='rank'`.

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

            # Validate indices are non-negative
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
            If input shape mismatch between `self.coeff_shape` and `x.shape`.
            If `self.coeff_indices` contains out-of-bounds indices (calculated from 3D view of `x`).

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa - Ignore unreachable code warning

        x = x.astype(self.dtype, copy=False)
        x3, _orig_shape = _to_3d_view(x)
        H, W, T = x3.shape

        # Apply 3D DCT
        X = _dct3(x3)

        if self.selection_mode == "rank":
            # Rank mode: select coefficients by variance-ordered indices
            if self.coeff_shape is not None:
                expected_H, expected_W, expected_T = self.coeff_shape  # NOSONAR # noqa - Ignore lowercase warning
                if (H, W, T) != (expected_H, expected_W, expected_T):
                    raise ValueError(
                        f"Input shape mismatch: expected coeff_shape={self.coeff_shape}, "
                        f"got (H,W,T)={H, W, T} from input shape {x.shape}."
                    )

            X_flat = X.reshape(-1)  # NOSONAR # noqa - Ignore lowercase warning

            # Validate indices are within bounds
            max_idx = H * W * T
            if np.any(self.coeff_indices >= max_idx):
                raise ValueError(f"`coeff_indices` contains out-of-bounds indices (max={max_idx - 1}).")

            z = X_flat[self.coeff_indices].astype(self.dtype, copy=False)

        else:
            # Spatial mode: keep top-left-front block
            h_eff = min(self.keep_h, H)
            w_eff = min(self.keep_w, W)
            t_eff = min(self.keep_t, T)

            X_crop = X[:h_eff, :w_eff, :t_eff]  # NOSONAR # noqa - Ignore lowercase warning
            z = X_crop.reshape(-1).astype(self.dtype, copy=False)

        return z

    # ------------------------------------------------------------------------------------------------------------------
    def decode(  # NOSONAR - Ignore cognitive complexity
        self, z: np.ndarray, original_shape: tuple[int, ...]
    ) -> np.ndarray:
        """
        Decode a latent code back to the original signal shape.

        Parameters
        ----------
        z : np.ndarray
            Encoded representation of shape (D,).
        original_shape : tuple[int, ...]
            Original input shape used in encode:
                (T,), (C, T), or (H, W, T).

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed signal with shape == original_shape.

        Raises
        ------
        ValueError
            If `z.ndim` is not 1.
            If `len(original_shape)` is not in [1, 2, 3].
            If shape mismatch between `self.coeff_shape` and (H,W,T) from `original_shape`.
            If `z.size` is different than the expected dimension when `self.selection_mode` is "rank".
            If `z.size` is different than the expected dimension when `self.selection_mode` is "spatial".
        """

        if not isinstance(z, np.ndarray):
            z = np.asarray(z)  # noqa - Ignore unreachable code warning

        z = z.astype(self.dtype, copy=False)

        if z.ndim != 1:
            raise ValueError(f"Expected z of shape (D,), got {z.shape}.")

        # Recover the 3D full shape from original_shape. Match conventions of _to_3d_view.
        H_full, W_full, T_full = (  # NOSONAR # noqa - Ignore lowercase warning
            None,
            None,
            None,
        )
        if len(original_shape) == 1:
            H_full, W_full, T_full = 1, 1, original_shape[0]
        elif len(original_shape) == 2:
            H_full, W_full, T_full = original_shape[0], 1, original_shape[1]
        elif len(original_shape) == 3:
            H_full, W_full, T_full = original_shape
        else:
            raise ValueError(f"Unsupported `original_shape={original_shape!r}`.")

        # Build full DCT tensor with zeros
        X_full = np.zeros(shape=(H_full, W_full, T_full), dtype=self.dtype)  # NOSONAR # noqa - Ignore lowercase warning

        if self.selection_mode == "rank":
            # Rank mode: scatter coefficients to variance-ordered positions
            if self.coeff_shape is not None:
                expected_H, expected_W, expected_T = self.coeff_shape  # NOSONAR # noqa - Ignore lowercase warning
                if (H_full, W_full, T_full) != (expected_H, expected_W, expected_T):
                    raise ValueError(
                        f"Shape mismatch: `expected coeff_shape={self.coeff_shape}`, "
                        f"got (H,W,T)={H_full, W_full, T_full} from `original_shape={original_shape}`."
                    )

            expected_dim = len(self.coeff_indices)  # type: ignore[arg-type]  # Guaranteed non-None by __post_init__
            if z.size != expected_dim:
                raise ValueError(
                    f"Encoded vector `z` has size {z.size}, but expected {expected_dim} (len(coeff_indices)) for "
                    f"original_shape={original_shape!r}."
                )

            X_flat = X_full.reshape(-1)  # NOSONAR # noqa - Ignore lowercase warning
            X_flat[self.coeff_indices] = z
            X_full = X_flat.reshape(H_full, W_full, T_full)

        else:  # -> I.e., self.selection_mode is "spatial".
            # Spatial mode: place in top-left-front block
            h_eff = min(self.keep_h, H_full)
            w_eff = min(self.keep_w, W_full)
            t_eff = min(self.keep_t, T_full)

            expected_dim = h_eff * w_eff * t_eff
            if z.size != expected_dim:
                raise ValueError(
                    f"Encoded vector `z` has size {z.size}, but expected {expected_dim} for "
                    f"original_shape={original_shape!r} with keep=(h={self.keep_h}, w={self.keep_w}, t={self.keep_t})."
                )

            X_crop = z.reshape(h_eff, w_eff, t_eff)  # NOSONAR # noqa - Ignore lowercase warning
            X_full[:h_eff, :w_eff, :t_eff] = X_crop

        x3_hat = _idct3(x=X_full)
        x_hat = _from_3d_view(x3=x3_hat, original_shape=original_shape)

        return x_hat.astype(self.dtype, copy=False)


# ======================================================================================================================
# Demo / tests
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _demo_roundtrip(codec: DCT3DCodec, x: np.ndarray, desc: str) -> None:
    """Single encode/decode test."""

    print(f"\n[TEST] {desc}  shape={x.shape}  keep={codec.keep_shape}")
    z = codec.encode(x)
    print(f"  encoded: D={z.size}, shape={z.shape}")
    x_hat = codec.decode(z, original_shape=x.shape)
    print(f"  decoded: shape={x_hat.shape}")

    diff = x_hat - x
    print(f"  max |x-x_hat| = {np.max(np.abs(diff)):.3e}")
    print(f"  MSE           = {np.mean(diff**2):.3e}")


# ----------------------------------------------------------------------------------------------------------------------
def _demo_timeseries():
    T = 128
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    x_sine = np.sin(2 * t).astype(np.float32)

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=1, keep_w=1, keep_t=keep_t)
        _demo_roundtrip(codec=codec, x=x_sine, desc=f"Timeseries sine, keep_t={keep_t}")


# ----------------------------------------------------------------------------------------------------------------------
def _demo_timeseries_noisy():
    T = 128
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    rng = np.random.default_rng()  # NOSONAR - Ignore request for seed
    noise = (0.3 * rng.standard_normal(T)).astype(np.float32)
    x = (np.sin(2 * t) + noise).astype(np.float32)

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=1, keep_w=1, keep_t=keep_t)
        _demo_roundtrip(codec=codec, x=x, desc=f"Timeseries noisy, keep_t={keep_t}")


# ----------------------------------------------------------------------------------------------------------------------
def _demo_profile():
    C = 5
    T = 128
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    x = np.stack([np.sin((1 + c) * t + 0.3 * c) for c in range(C)], axis=0).astype(np.float32)

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=C, keep_w=1, keep_t=keep_t)
        _demo_roundtrip(codec=codec, x=x, desc=f"Profile (C={C}), keep_t={keep_t}")


# ----------------------------------------------------------------------------------------------------------------------
def _demo_video():

    # ..................................................................................................................
    def gaussian(mux, muy, sigma=0.3):
        """2D Gaussian function."""
        return np.exp(-((X - mux) ** 2 + (Y - muy) ** 2) / (2 * sigma**2))

    # ..................................................................................................................

    H, W, T = 16, 16, 32
    yv = np.linspace(-1, 1, H)
    xv = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(yv, xv, indexing="ij")

    frames = []
    for k in range(T):
        cx = -0.5 + k * (1.0 / (T - 1))
        cy = 0.0
        frames.append(gaussian(mux=cx, muy=cy))
    x_video = np.stack(frames, axis=-1).astype(np.float32)

    configs = [(4, 4, 4), (8, 8, 8), (16, 16, 8), (16, 16, 16)]

    for keep_h, keep_w, keep_t in configs:
        codec = DCT3DCodec(keep_h=keep_h, keep_w=keep_w, keep_t=keep_t)
        label = f"Video Gaussian, keep(h={keep_h}, w={keep_w}, t={keep_t})"
        _demo_roundtrip(codec=codec, x=x_video, desc=label)


# ----------------------------------------------------------------------------------------------------------------------
def _run_all_demos():
    np.random.seed(0)
    _demo_timeseries()
    _demo_timeseries_noisy()
    _demo_profile()
    _demo_video()


# ======================================================================================================================
if __name__ == "__main__":
    _run_all_demos()
