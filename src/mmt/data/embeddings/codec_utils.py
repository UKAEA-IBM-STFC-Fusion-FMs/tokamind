"""
Codec utilities for the embedding pipeline.

This module provides:
- small shape helpers (e.g., inferring (H, W) from non-time value shapes),
- a lightweight embedding-dimension estimator for a given encoder + signal shape,
- a factory to build per-signal codec instances from the SignalSpec registry.

The utilities here are intentionally simple and mirror the behaviour of the
corresponding codec implementations

VAE support assumes the refactored VAE_fairmast package (`vae_pipeline`) and
the trained VAE artifacts under: vae_pipeline/data/trained_vaes/<MODEL_DIR>.

"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from .dct3d_codec import DCT3DCodec
from .identity_codec import IdentityCodec
from .vae_codec import VAECodec, read_vae_model_meta


def infer_hw_from_values_shape(values_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Map values_shape (excluding time) to (H, W).

    Conventions:
      - () or (1,)     -> (1, 1)
      - (C,) with C>1  -> (C, 1)
      - (H, W)         -> (H, W)
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
    Compute the encoded dimension for a single chunk of a signal.

    Must mirror the codec behaviour (without executing the transform).
    """
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    n_samples = max(1, int(round(float(chunk_length_sec) / float(dt))))

    values_shape = tuple(values_shape or ())

    if encoder_name == "identity":
        spatial_dim = _prod(values_shape)
        return int(n_samples * spatial_dim)

    if encoder_name == "dct3d":
        H, W = infer_hw_from_values_shape(values_shape)
        T = n_samples

        keep_h = int(encoder_kwargs.get("keep_h", H))
        keep_w = int(encoder_kwargs.get("keep_w", W))
        keep_t = int(encoder_kwargs.get("keep_t", T))

        h_eff = min(keep_h, H)
        w_eff = min(keep_w, W)
        t_eff = min(keep_t, T)

        return int(h_eff * w_eff * t_eff)

    if encoder_name == "vae":
        if "model_dir" not in encoder_kwargs:
            raise KeyError("encoder_kwargs.model_dir is required when encoder_name='vae'.")

        meta = read_vae_model_meta(str(encoder_kwargs["model_dir"]))
        latent_dim = int(meta["latent_dim"])
        in_channels = int(meta["in_channels"])
        seq_len = int(meta["seq_len"])

        H, W = infer_hw_from_values_shape(values_shape)
        expected_channels = H * W

        if expected_channels != in_channels:
            raise ValueError(
                "VAE in_channels mismatch: "
                f"signal values_shape={values_shape} -> C={expected_channels}, "
                f"but model expects in_channels={in_channels} (model_dir={meta['model_dir']})."
            )

        if int(n_samples) != int(seq_len):
            raise ValueError(
                "VAE seq_len mismatch: "
                f"chunk_length_sec={chunk_length_sec} with dt={dt} -> T={n_samples}, "
                f"but model expects T={seq_len} (model_dir={meta['model_dir']})."
            )

        return int(latent_dim)

    raise ValueError(f"Unknown encoder_name={encoder_name!r} in compute_embedding_dim_for_encoder")


def build_codecs(signal_specs) -> Dict[int, Any]:
    """
    Build one codec instance per signal.

    Returns: mapping signal_id -> codec.
    """
    codecs: Dict[int, Any] = {}
    for spec in signal_specs.specs:
        if spec.encoder_name == "dct3d":
            codecs[spec.signal_id] = DCT3DCodec(**spec.encoder_kwargs)
        elif spec.encoder_name == "identity":
            codecs[spec.signal_id] = IdentityCodec()
        elif spec.encoder_name == "vae":
            # Config dicts are deep-merged; when switching encoder_name from dct3d→vae,
            # DCT3D-specific kwargs (keep_h/keep_w/keep_t) may remain in encoder_kwargs.
            kw = dict(spec.encoder_kwargs or {})
            allowed = ("model_dir", "device", "use_mu")
            vae_kw = {k: kw[k] for k in allowed if k in kw}
            if "model_dir" not in vae_kw:
                raise KeyError(f"Missing required encoder_kwargs.model_dir for VAE signal {spec.name!r}.")
            codecs[spec.signal_id] = VAECodec(**vae_kw)
        else:
            raise ValueError(
                f"Unknown encoder_name={spec.encoder_name!r} for signal {spec.name!r}"
            )
    return codecs
