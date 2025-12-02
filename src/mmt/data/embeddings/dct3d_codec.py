# src/mmt/data/embeddings/dct3d_codec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from scipy.fftpack import dct, idct


def _dct3(x: np.ndarray) -> np.ndarray:
    """3D DCT (type-II, orthonormal) over the last 3 axes of `x`."""
    y = dct(x, type=2, axis=-1, norm="ortho")
    y = dct(y, type=2, axis=-2, norm="ortho")
    y = dct(y, type=2, axis=-3, norm="ortho")
    return y


def _idct3(x: np.ndarray) -> np.ndarray:
    """3D inverse DCT (type-II, orthonormal) over the last 3 axes of `x`."""
    y = idct(x, type=2, axis=-1, norm="ortho")
    y = idct(y, type=2, axis=-2, norm="ortho")
    y = idct(y, type=2, axis=-3, norm="ortho")
    return y


def _to_3d_view(x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """
    Convert a 1D / 2D / 3D array into a (H, W, T) view and return
    also the original shape (for reconstruction).

    Conventions:
      - (T,)      -> (1, 1, T)
      - (C, T)    -> (C, 1, T)
      - (H, W, T) -> (H, W, T)
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
        raise ValueError(
            f"DCT3DCodec only supports 1D/2D/3D inputs, got shape={x.shape}"
        )

    return x3, x.shape


def _from_3d_view(x3: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Restore original shape from a (H, W, T) array using the original_shape
    convention of `_to_3d_view`.
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

    raise ValueError(f"Unsupported original_shape={original_shape!r} in from_3d_view")


@dataclass
class DCT3DCodec:
    """
    3D DCT-based encoder/decoder for time-dependent signals.

    This codec supports the three canonical signal shapes:
      - timeseries: (T,)
      - profile:    (C, T)
      - video/map:  (H, W, T)

    Internally, all inputs are viewed as (H, W, T), a 3D DCT is applied,
    and only the top-left-front (keep_h, keep_w, keep_t) coefficients are
    kept and flattened.

    Parameters
    ----------
    keep_h : int
        Number of DCT coefficients to keep along the "H" dimension.
    keep_w : int
        Number of DCT coefficients to keep along the "W" dimension.
    keep_t : int
        Number of DCT coefficients to keep along the "T" (time) dimension.

    Notes
    -----
    - The actual number of coefficients kept in each dimension is:

          h_eff = min(keep_h, H)
          w_eff = min(keep_w, W)
          t_eff = min(keep_t, T)

      where (H, W, T) is the 3D shape of the input view.

    - The encoder returns a (D,) array, where:

          D = h_eff * w_eff * t_eff

    - The decoder requires the original input shape and reconstructs an
      array of the same shape via zero-padding in DCT space + inverse DCT.
    """

    keep_h: int
    keep_w: int
    keep_t: int
    dtype: np.dtype = np.float32

    @property
    def keep_shape(self) -> Tuple[int, int, int]:
        """Requested (keep_h, keep_w, keep_t). Actual kept dims depend on input."""
        return self.keep_h, self.keep_w, self.keep_t

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
            Encoded representation of shape (1, D), dtype = self.dtype.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        x = x.astype(self.dtype, copy=False)
        x3, _orig_shape = _to_3d_view(x)
        H, W, T = x3.shape

        # Apply 3D DCT
        X = _dct3(x3)

        # Effective keep dims (cannot exceed actual dims)
        h_eff = min(self.keep_h, H)
        w_eff = min(self.keep_w, W)
        t_eff = min(self.keep_t, T)

        X_crop = X[:h_eff, :w_eff, :t_eff]

        z = X_crop.reshape(-1).astype(self.dtype, copy=False)  # shape (D,)

        return z

    def decode(self, z: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """
        Decode a latent code back to the original signal shape.

        Parameters
        ----------
        z : np.ndarray
            Encoded representation of shape or (D,).
        original_shape : tuple of int
            Original input shape used in encode:
                (T,), (C, T), or (H, W, T).

        Returns
        -------
        x_hat : np.ndarray
            Reconstructed signal with shape == original_shape.
        """
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)

        z = z.astype(self.dtype, copy=False)

        if z.ndim != 1:
            raise ValueError(f"Expected z of shape (D,), got {z.shape}. ")

        # Recover the 3D full shape from original_shape
        # match conventions of _to_3d_view
        if len(original_shape) == 1:
            H_full, W_full, T_full = 1, 1, original_shape[0]
        elif len(original_shape) == 2:
            H_full, W_full, T_full = original_shape[0], 1, original_shape[1]
        elif len(original_shape) == 3:
            H_full, W_full, T_full = original_shape
        else:
            raise ValueError(f"Unsupported original_shape={original_shape!r}")

        h_eff = min(self.keep_h, H_full)
        w_eff = min(self.keep_w, W_full)
        t_eff = min(self.keep_t, T_full)

        expected_dim = h_eff * w_eff * t_eff
        if z.size != expected_dim:
            raise ValueError(
                f"Encoded vector has size {z.size}, but expected "
                f"{expected_dim} for original_shape={original_shape!r} "
                f"with keep=(h={self.keep_h}, w={self.keep_w}, t={self.keep_t})"
            )

        X_crop = z.reshape(h_eff, w_eff, t_eff)

        # Build full DCT tensor with zeros outside the kept region
        X_full = np.zeros((H_full, W_full, T_full), dtype=self.dtype)
        X_full[:h_eff, :w_eff, :t_eff] = X_crop

        x3_hat = _idct3(X_full)
        x_hat = _from_3d_view(x3_hat, original_shape)

        return x_hat.astype(self.dtype, copy=False)


if __name__ == "__main__":
    """
    simple test of the codec
    """

    np.random.seed(0)

    def roundtrip(codec: DCT3DCodec, x: np.ndarray, desc: str) -> None:
        print(f"\n[TEST] {desc}  shape={x.shape}  keep={codec.keep_shape}")
        z = codec.encode(x)
        print(f"  encoded shape: {z.shape} (D={z.size})")
        x_hat = codec.decode(z, original_shape=x.shape)
        print(f"  decoded shape: {x_hat.shape}")

        diff = x_hat - x
        max_err = float(np.max(np.abs(diff)))
        mse = float(np.mean(diff**2))
        rel_energy = float(np.linalg.norm(x_hat) / (np.linalg.norm(x) + 1e-12))

        print(f"  max |x - x_hat| = {max_err:.3e}")
        print(f"  MSE              = {mse:.3e}")
        print(f"  ||x_hat|| / ||x||= {rel_energy:.3f}")

    # -------------------------------------------------------------
    # 1) Timeseries: smooth sine wave (low frequency)
    # -------------------------------------------------------------
    T = 128
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    x_sine = np.sin(2 * t).astype(np.float32)  # smooth, low-freq

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=1, keep_w=1, keep_t=keep_t)
        roundtrip(codec, x_sine, f"timeseries (low-freq sine), keep_t={keep_t}")

    # -------------------------------------------------------------
    # 2) Timeseries: sine + noise (higher freq content)
    # -------------------------------------------------------------
    noise = 0.3 * np.random.randn(T).astype(np.float32)
    x_noisy = (np.sin(2 * t) + noise).astype(np.float32)

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=1, keep_w=1, keep_t=keep_t)
        roundtrip(codec, x_noisy, f"timeseries (sine + noise), keep_t={keep_t}")

    # -------------------------------------------------------------
    # 3) Profile: multi-channel sinusoids (C, T)
    # -------------------------------------------------------------
    C = 5
    T_prof = 128
    t_prof = np.linspace(0, 2 * np.pi, T_prof, endpoint=False)

    # ogni canale ha una frequenza e fase diversa
    x_profile = []
    for c in range(C):
        freq = 1 + c  # 1x, 2x, 3x, ...
        phase = 0.3 * c
        x_c = np.sin(freq * t_prof + phase)
        x_profile.append(x_c)
    x_profile = np.stack(x_profile, axis=0).astype(np.float32)  # (C, T)

    for keep_t in [4, 8, 16, 32, 64]:
        codec = DCT3DCodec(keep_h=C, keep_w=1, keep_t=keep_t)
        roundtrip(codec, x_profile, f"profile (C={C}, T) multi-sine, keep_t={keep_t}")

    # -------------------------------------------------------------
    # 4) "Video": Gaussian blob moving slowly in time (H, W, T)
    # -------------------------------------------------------------
    H, W, T_vid = 16, 16, 32
    y = np.linspace(-1, 1, H)
    x = np.linspace(-1, 1, W)
    Y, X = np.meshgrid(y, x, indexing="ij")

    def gaussian(cx, cy, sigma=0.3):
        return np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sigma**2))

    frames = []
    for k in range(T_vid):
        # blob moves slowly along x
        cx = -0.5 + k * (1.0 / (T_vid - 1))
        cy = 0.0
        frame = gaussian(cx, cy)
        frames.append(frame)
    x_video = np.stack(frames, axis=-1).astype(np.float32)  # (H, W, T)

    for keep_h, keep_w, keep_t in [(4, 4, 4), (8, 8, 8), (16, 16, 8), (16, 16, 16)]:
        codec = DCT3DCodec(keep_h=keep_h, keep_w=keep_w, keep_t=keep_t)
        roundtrip(
            codec,
            x_video,
            f"video Gaussian blob, keep=(h={keep_h},w={keep_w},t={keep_t})",
        )
