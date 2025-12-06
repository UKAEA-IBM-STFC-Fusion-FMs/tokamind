# src/mmt/data/embeddings/codec_utils.py
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .dct3d_codec import DCT3DCodec
from .identity_codec import IdentityCodec


def _infer_hw_from_values_shape(values_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Map values_shape (excluding time) to (H, W) used by DCT3D.

    Conventions (must match those used in DCT3DCodec._to_3d_view):
      - () or (1,)     -> (1, 1)          (scalar / single-channel timeseries)
      - (C,) with C>1  -> (C, 1)          (profile: C channels)
      - (H, W)         -> (H, W)          (2D map / video frame)
    """
    if len(values_shape) == 0:
        return 1, 1
    if len(values_shape) == 1:
        c = int(values_shape[0])
        return (1, 1) if c == 1 else (c, 1)
    if len(values_shape) == 2:
        return int(values_shape[0]), int(values_shape[1])
    raise ValueError(f"Unsupported values_shape={values_shape!r} for H/W inference")


def compute_embedding_dim_for_encoder(
    *,
    encoder_name: str,
    encoder_kwargs: Mapping[str, Any],
    values_shape: Tuple[int, ...],
    dt: float,
    chunk_length_sec: float,
) -> int:
    """
    Compute the encoded dimension for a single chunk of a signal, given:

      • encoder_name / encoder_kwargs
      • per-sample values_shape (excluding time)
      • dt (sampling period, seconds)
      • chunk_length_sec (chunk length, seconds)

    This should mirror the behaviour of the corresponding codec's encode()
    logic, but without actually running the transform.
    """
    # Number of time samples per chunk
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    n_samples = max(1, int(round(chunk_length_sec / dt)))

    values_shape = tuple(values_shape or ())

    if encoder_name == "identity":
        # Identity: chunk shape = (n_samples, *values_shape), flattened.
        spatial_dim = 1
        for s in values_shape:
            spatial_dim *= int(s)
        return n_samples * spatial_dim

    if encoder_name == "dct3d":
        # DCT3D: work on a (H, W, T) view. H/W depend on values_shape, T on n_samples.
        H, W = _infer_hw_from_values_shape(values_shape)
        T = n_samples

        keep_h = int(encoder_kwargs.get("keep_h", H))
        keep_w = int(encoder_kwargs.get("keep_w", W))
        keep_t = int(encoder_kwargs.get("keep_t", T))

        # Effective kept sizes cannot exceed actual dims
        h_eff = min(keep_h, H)
        w_eff = min(keep_w, W)
        t_eff = min(keep_t, T)

        return h_eff * w_eff * t_eff

    raise ValueError(f"Unknown encoder_name={encoder_name!r} in compute_embedding_dim")


def build_codecs(signal_specs) -> Dict[int, Any]:
    """
    Build one encoder codec instance per signal.

    Parameters
    ----------
    signal_specs : SignalSpecRegistry or any object exposing `.specs`
        Each spec must have fields: encoder_name, encoder_kwargs, signal_id.

    Returns
    -------
    codecs : dict
        Mapping ``signal_id -> codec`` where each codec exposes
        an ``encode(x: np.ndarray) -> np.ndarray(D,)`` method.
    """
    codecs: Dict[int, Any] = {}
    for spec in signal_specs.specs:
        if spec.encoder_name == "dct3d":
            codecs[spec.signal_id] = DCT3DCodec(**spec.encoder_kwargs)
        elif spec.encoder_name == "identity":
            codecs[spec.signal_id] = IdentityCodec()
        else:
            raise ValueError(
                f"Unknown encoder_name={spec.encoder_name!r} for signal {spec.name!r}"
            )
    return codecs
