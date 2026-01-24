"""
Codec utilities for the embedding pipeline.

This module provides:
- small shape helpers (e.g., inferring (H, W) from non-time value shapes),
- a lightweight embedding-dimension estimator for a given encoder + signal shape,
- a factory to build per-signal codec instances from the SignalSpec registry.

The utilities here are intentionally simple and mirror the behaviour of the
corresponding codec implementations
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .dct3d_codec import DCT3DCodec
from .identity_codec import IdentityCodec
from .vae_codec import VAECodec, read_vae_model_meta


def infer_hw_from_values_shape(values_shape: Tuple[int, ...]) -> Tuple[int, int]:
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


def _prod(shape: Tuple[int, ...]) -> int:
    out = 1
    for s in shape:
        out *= int(s)
    return int(out)


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
    n_samples = max(1, int(round(float(chunk_length_sec) / float(dt))))

    values_shape = tuple(values_shape or ())

    if encoder_name == "identity":
        # Identity: chunk shape = (n_samples, *values_shape), flattened.
        spatial_dim = _prod(values_shape)
        return int(n_samples * spatial_dim)

    if encoder_name == "dct3d":
        # DCT3D: work on a (H, W, T) view. H/W depend on values_shape, T on n_samples.
        H, W = infer_hw_from_values_shape(values_shape)
        T = n_samples

        keep_h = int(encoder_kwargs.get("keep_h", H))
        keep_w = int(encoder_kwargs.get("keep_w", W))
        keep_t = int(encoder_kwargs.get("keep_t", T))

        # Effective kept sizes cannot exceed actual dims
        h_eff = min(keep_h, H)
        w_eff = min(keep_w, W)
        t_eff = min(keep_t, T)

        return int(h_eff * w_eff * t_eff)

    if encoder_name == "vae":
        # VAE: embedding dim is the model latent dim (read from the model folder).
        # We also validate that the expected (channels, T) matches the signal.
        if "model_dir" not in encoder_kwargs:
            raise KeyError(
                "encoder_kwargs.model_dir is required when encoder_name='vae'."
            )

        meta = read_vae_model_meta(str(encoder_kwargs["model_dir"]))
        latent_dim = int(meta.latent_dim)
        in_channels = int(meta.in_channels)
        seq_len = int(meta.seq_len)

        # values_shape excludes time; derive expected channels from (H,W).
        H, W = infer_hw_from_values_shape(values_shape)
        expected_channels = H * W

        if expected_channels != in_channels:
            raise ValueError(
                "VAE model expects a different number of channels than the signal provides: "
                f"signal_channels={expected_channels}, model_channels={in_channels}. "
                f"(values_shape={values_shape}, model_dir={meta.model_dir})"
            )

        if n_samples != seq_len:
            raise ValueError(
                "VAE model expects a different time length than the chunk length produces: "
                f"signal_T={n_samples} (from chunk_length_sec/dt), model_T={seq_len}. "
                f"(dt={dt}, chunk_length_sec={chunk_length_sec}, model_dir={meta.model_dir})"
            )

        return latent_dim

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
        elif spec.encoder_name == "vae":
            # Config files are deep-merged; when switching encoder_name from dct3d→vae,
            # DCT3D-specific kwargs (keep_h/keep_w/keep_t) may remain in encoder_kwargs.
            # Filter to the kwargs accepted by VAECodec.
            kw = dict(spec.encoder_kwargs or {})
            allowed = ("model_dir", "device", "use_mu", "model_entrypoint")
            vae_kw = {k: kw[k] for k in allowed if k in kw}
            if "model_dir" not in vae_kw:
                raise KeyError(
                    f"Missing required encoder_kwargs.model_dir for VAE signal {spec.name!r}."
                )
            codecs[spec.signal_id] = VAECodec(**vae_kw)
        else:
            raise ValueError(
                f"Unknown encoder_name={spec.encoder_name!r} for signal {spec.name!r}"
            )
    return codecs
