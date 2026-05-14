"""
Codec utilities for the embedding pipeline.

This module provides:
- small shape helpers (e.g., inferring (H, W) from non-time value shapes),
- a lightweight embedding-dimension estimator for a given encoder + signal shape,
- a factory to build per-signal codec instances from the SignalSpec registry,
- a factory to build per-signal differentiable TorchDecoder instances for training and eval.

VAE support assumes the refactored VAE_fairmast package (`vae_pipeline`) and the trained VAE artifacts
under: vae_pipeline/data/VAEs/<MODEL_DIR>.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TYPE_CHECKING
from collections.abc import Mapping
import numpy as np

from .torch_decoder import TorchDecoder
from .dct3d import DCT3DCodec
from .identity import IdentityCodec
from .vae import VAECodec, read_vae_model_meta

if TYPE_CHECKING:
    from ..signal_spec import SignalSpecRegistry

__all__ = [
    "TorchDecoder",
    "build_torch_decoder",
    "build_codecs",
    "build_decoders",
    "compute_embedding_dim_for_encoder",
    "load_coeff_indices",
    "infer_hw_from_values_shape",
]


# ----------------------------------------------------------------------------------------------------------------------
def infer_hw_from_values_shape(values_shape: tuple[int, ...]) -> tuple[int, int]:
    """
    Map values_shape (excluding time) to (H, W).

    Conventions:
      - ( ) or (1, )     -> (1, 1)
      - (C, ) with C>1  -> (C, 1)
      - (H, W)         -> (H, W)

    Parameters
    ----------
    values_shape : tuple[int, ...]
        Values shape for H/W inference.

    Returns
    -------
    tuple[int, int]
        Inferred H/W values from input values shape.

    Raises
    ------
    ValueError
        If unsupported `values_shape.

    """

    if len(values_shape) == 0:
        return 1, 1
    if len(values_shape) == 1:
        c = int(values_shape[0])
        return (1, 1) if c == 1 else (c, 1)
    if len(values_shape) == 2:
        return int(values_shape[0]), int(values_shape[1])

    raise ValueError(f"Unsupported `values_shape={values_shape!r}` for H/W inference.")


# ----------------------------------------------------------------------------------------------------------------------
def _prod(shape: tuple[int, ...]) -> int:
    """Component-wise product of items in input shape."""
    out = 1
    for s in shape:
        out *= int(s)
    return int(out)


# ----------------------------------------------------------------------------------------------------------------------
def load_coeff_indices(config_dir: Path, rel_path: str) -> np.ndarray:
    """
    Load coefficient indices from a .npy file for rank mode DCT3D.

    The path is resolved relative to `config_dir`, which for trained models is `runs/<run_id>/embeddings/`.

    Parameters
    ----------
    config_dir : Path
        Base directory containing the indices (e.g., runs/<run_id>/embeddings/).
    rel_path : str
        Relative path to the .npy file.
        Example: "dct3d_indices/output_signal.npy"

    Returns
    -------
    np.ndarray
        1D array of coefficient indices (dtype: int32).

    Raises
    ------
    FileNotFoundError
        If resulting coefficient indices file from `config_dir` and `rel_path` is not found.
    ValueError
        If indices loaded from the coefficient indices file is not a 1D array.

    """

    full_path = config_dir / rel_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Coefficient indices file not found: {full_path}\n"
            f"Each training run saves its own indices to runs/<run_id>/embeddings/.\n"
            f"Original path: {rel_path}"
        )

    indices = np.load(full_path)
    if indices.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {indices.shape} from {full_path}.")

    return indices.astype(np.int32)


# ----------------------------------------------------------------------------------------------------------------------
def compute_embedding_dim_for_encoder(  # NOSONAR - Ignore cognitive complexity
    *,
    encoder_name: Literal["identity", "dct3d", "vae"],
    encoder_kwargs: Mapping[str, Any],
    values_shape: tuple[int, ...],
    dt: float,
    chunk_length_sec: float,
) -> int:
    """
    Compute the encoded dimension for a single chunk of a signal.

    Must mirror the codec behaviour (without executing the transform).

    Parameters
    ----------
    encoder_name : Literal["identity", "dct3d", "vae"]
        Encoder name in ["identity", "dct3d", "vae"].
    encoder_kwargs : Mapping[str, Any]
        Mapping (dict) of encoder keyword arguments.
    values_shape : tuple[int, ...]
        Values shape tuple.
    dt : float
        Sampling interval in seconds.
    chunk_length_sec : float
        Chunk length in seconds.

    Returns
    -------
    int
        Embedding dimension.

    Raises
    ------
    ValueError
        If `encoder_name` not in ["identity", "dct3d", "vae"].
        If `dt` not greater than 0.
        If `encoder_kwargs["num_coeffs"]` is not an int for DCT3D rank mode.
        If VAE `model_type` is unsupported for VAE encoder (`encoder_name="vae"`).
        If VAE `linear` model does not get input_shape [C,T].
        If VAE `linear` model does not satisfy input_mode in ["channels", "time"].
        If VAE `linear` channel mismatch is detected.
        If VAE `linear` time mismatch is detected.
        If VAE `conv1d` model does not get input_shape [C,T],
        If VAE `conv1d` model does not satisfy input_mode="channels".
        If VAE `conv1d` channel mismatch is detected.
        If VAE `conv1d` time mismatch is detected.
        If VAE `conv2d` model does not get input_shape [H,W,T]
        If VAE `conv2d` model does not satisfy input_mode="time".
        If VAE `conv2d` channel mismatch is detected.
        If VAE `conv2d` time mismatch is detected.
        If VAE VAE `model_type`not in ["linear", "conv1d", "conv2d"].

    """

    if encoder_name not in ["identity", "dct3d", "vae"]:
        raise ValueError(f"Unknown encoder_name={encoder_name!r} in compute_embedding_dim_for_encoder.")

    if dt <= 0:
        raise ValueError(f"`dt` must be > 0, got {dt}.")

    n_samples = max(1, int(round(float(chunk_length_sec) / float(dt))))
    values_shape = tuple(values_shape or ())

    # ..................................................................................................................
    # Embedding computation based on encoder name
    # ..................................................................................................................

    # Identity encoder
    if encoder_name == "identity":
        spatial_dim = _prod(shape=values_shape)
        return int(n_samples * spatial_dim)

    # DCT3D encoder
    elif encoder_name == "dct3d":
        selection_mode = encoder_kwargs.get("selection_mode", "spatial")

        if selection_mode == "rank":
            # Rank mode: dimension is number of selected coefficients
            if "num_coeffs" not in encoder_kwargs:
                raise KeyError("`encoder_kwargs['num_coeffs']` is required for DCT3D rank mode.")
            num_coeffs = encoder_kwargs["num_coeffs"]
            if (not isinstance(num_coeffs, int)) or isinstance(num_coeffs, bool):
                raise ValueError("`encoder_kwargs['num_coeffs']` must be an int for DCT3D rank mode.")

            return num_coeffs

        else:  # -> I.e., selection_mode is "spatial"
            # Spatial mode: dimension is keep_h * keep_w * keep_t (clamped)
            H, W = infer_hw_from_values_shape(values_shape=values_shape)  # noqa - Ignore lowercase warning
            T = n_samples  # noqa - Ignore lowercase warning

            keep_h = int(encoder_kwargs.get("keep_h", H))
            keep_w = int(encoder_kwargs.get("keep_w", W))
            keep_t = int(encoder_kwargs.get("keep_t", T))

            h_eff = min(keep_h, H)
            w_eff = min(keep_w, W)
            t_eff = min(keep_t, T)

            return int(h_eff * w_eff * t_eff)

    # VAE encoder
    else:  # -> I.e,  encoder_name="vae":
        if "model_dir" not in encoder_kwargs:
            raise KeyError("`encoder_kwargs['model_dir']` is required when VAE encoder (`encoder_name='vae')`.")

        meta = read_vae_model_meta(model_dir=str(encoder_kwargs["model_dir"]))
        latent_dim = int(meta["latent_dim"])
        model_type = str(meta["model_type"])
        input_shape = tuple(int(v) for v in meta["input_shape"])
        input_mode = str(meta["input_mode"])

        # Signal-level canonical dimensions:
        # - values_shape carries non-time dimensions
        # - n_samples is the chunk time dimension
        h_sig, w_sig = infer_hw_from_values_shape(values_shape=values_shape)  # noqa - Ignore lowercase warning
        c_sig = int(h_sig * w_sig)
        t_sig = int(n_samples)

        if model_type == "linear":
            if len(input_shape) != 2:
                raise ValueError(
                    f"VAE `linear` model expects input_shape=[C,T], got {list(input_shape)} "
                    f"(model_dir={meta['model_dir']})."
                )

            if input_mode not in {"channels", "time"}:
                raise ValueError(
                    f"VAE `linear` model expects input_mode in ['channels','time'], got {input_mode!r} "
                    f"(model_dir={meta['model_dir']})."
                )

            c_model, t_model = int(input_shape[0]), int(input_shape[1])
            if c_sig != c_model:
                raise ValueError(
                    "VAE `linear` channel mismatch: "
                    f"signal values_shape={values_shape} -> C={c_sig}, "
                    f"but model expects C={c_model} (model_dir={meta['model_dir']})."
                )

            if t_sig != t_model:
                raise ValueError(
                    "VAE `linear` time mismatch: "
                    f"chunk_length_sec={chunk_length_sec} with dt={dt} -> T={t_sig}, "
                    f"but model expects T={t_model} (model_dir={meta['model_dir']})."
                )

        elif model_type == "conv1d":
            if len(input_shape) != 2:
                raise ValueError(
                    f"VAE `conv1d` model expects input_shape=[C,T], got {list(input_shape)} "
                    f"(model_dir={meta['model_dir']})."
                )

            if input_mode != "channels":
                raise ValueError(
                    f"VAE `conv1d` model requires input_mode='channels', got {input_mode!r} "
                    f"(model_dir={meta['model_dir']})."
                )

            c_model, t_model = int(input_shape[0]), int(input_shape[1])
            if c_sig != c_model:
                raise ValueError(
                    "VAE `conv1d` channel mismatch: "
                    f"signal values_shape={values_shape} -> C={c_sig}, "
                    f"but model expects C={c_model} (model_dir={meta['model_dir']})."
                )

            if t_sig != t_model:
                raise ValueError(
                    "VAE `conv1d` time mismatch: "
                    f"chunk_length_sec={chunk_length_sec} with dt={dt} -> T={t_sig}, "
                    f"but model expects T={t_model} (model_dir={meta['model_dir']})."
                )

        elif model_type == "conv2d":
            if len(input_shape) != 3:
                raise ValueError(
                    f"VAE `conv2d` model expects input_shape=[H,W,T], got {list(input_shape)} "
                    f"(model_dir={meta['model_dir']})."
                )

            if input_mode != "time":
                raise ValueError(
                    f"VAE `conv2d` model requires input_mode='time', got {input_mode!r} "
                    f"(model_dir={meta['model_dir']})."
                )

            h_model, w_model, t_model = (
                int(input_shape[0]),
                int(input_shape[1]),
                int(input_shape[2]),
            )
            if (h_sig, w_sig) != (h_model, w_model):
                raise ValueError(
                    "VAE `conv2d` spatial mismatch: "
                    f"signal values_shape={values_shape} -> (H,W)=({h_sig},{w_sig}), "
                    f"but model expects (H,W)=({h_model},{w_model}) (model_dir={meta['model_dir']})."
                )

            if t_sig != t_model:
                raise ValueError(
                    "VAE `conv2d` time mismatch: "
                    f"chunk_length_sec={chunk_length_sec} with dt={dt} -> T={t_sig}, "
                    f"but model expects T={t_model} (model_dir={meta['model_dir']})."
                )

        else:
            raise ValueError(
                f"Unsupported VAE model_type={model_type!r} in metadata for model_dir={meta['model_dir']}."
            )

        return int(latent_dim)


# ----------------------------------------------------------------------------------------------------------------------
def build_codecs(  # NOSONAR - Ignore cognitive complexity
    signal_specs: SignalSpecRegistry, config_dir: Path | None = None
) -> dict[int, Any]:
    """
    Build one codec instance per signal.

    Parameters
    ----------
    signal_specs : SignalSpecRegistry
        Registry of signal specifications.
    config_dir : Path | None
        Base directory for loading coefficient indices (DCT3D rank mode only).
        Only required if any signal uses selection_mode="rank".
        Typically: runs/<run_id>/embeddings/
        For spatial mode or other encoders, this parameter is not needed.
        Optional. Default: None.

    Returns
    -------
    dict[int, Any]
        Mapping signal_id -> codec instance.

    Raises
    ------
    ValueError
        If `config_dir` is not available for DCT3D rank mode.
        If `encoder_name` attribute of an item in `signal_specs.specs` is not in ["dct3d", "identity", "vae"].
    KeyError
        If `encoder_kwargs["coeff_indices_path"]` is not available for DCT3D rank mode.
        If `encoder_kwargs["model_dir"]` is not available for a given VAE signal.

    """

    codecs: dict[int, Any] = {}
    for spec in signal_specs.specs:
        if spec.encoder_name == "dct3d":
            kw = dict(spec.encoder_kwargs or {})
            selection_mode = kw.get("selection_mode", "spatial")

            if selection_mode == "rank":
                # Rank mode: load coefficient indices from .npy file
                if config_dir is None:
                    raise ValueError(
                        f"`config_dir` required for DCT3D rank mode (signal {spec.role}:{spec.name}). "
                        f"Pass config_dir=Path(run_dir) / 'embeddings' to build_codecs()."
                    )

                coeff_indices_path = kw.get("coeff_indices_path")
                if coeff_indices_path is None:
                    raise KeyError(
                        f"`encoder_kwargs['coeff_indices_path']` required for DCT3D rank mode "
                        f"(signal {spec.role}:{spec.name})."
                    )

                # Load indices
                coeff_indices = load_coeff_indices(config_dir=config_dir, rel_path=str(coeff_indices_path))

                # Build codec with loaded indices
                codec_kw = {
                    "keep_h": kw.get("keep_h", 1),  # Dummy values for rank mode
                    "keep_w": kw.get("keep_w", 1),
                    "keep_t": kw.get("keep_t", 1),
                    "selection_mode": "rank",
                    "coeff_indices": coeff_indices,
                    "coeff_shape": tuple(kw["coeff_shape"]) if "coeff_shape" in kw else None,
                }
                codecs[spec.signal_id] = DCT3DCodec(**codec_kw)
            else:
                # Spatial mode: use keep_h/w/t directly (no config_dir needed)
                codecs[spec.signal_id] = DCT3DCodec(**kw)

        elif spec.encoder_name == "identity":
            codecs[spec.signal_id] = IdentityCodec()

        elif spec.encoder_name == "vae":
            # Config dicts are deep-merged; when switching encoder_name from dct3d→vae,
            # DCT3D-specific kwargs (keep_h/keep_w/keep_t) may remain in encoder_kwargs.
            kw = dict(spec.encoder_kwargs or {})
            allowed = ("model_dir", "device", "use_mu")
            vae_kw = {k: kw[k] for k in allowed if k in kw}
            if "model_dir" not in vae_kw:
                raise KeyError(f"Missing required `encoder_kwargs['model_dir']` for VAE signal {spec.name!r}.")

            codecs[spec.signal_id] = VAECodec(**vae_kw)

        else:
            raise ValueError(f"Unknown encoder_name={spec.encoder_name!r} for signal {spec.name!r}.")

    return codecs


# ----------------------------------------------------------------------------------------------------------------------
def build_torch_decoder(
    codec: Any,
    original_shape: tuple[int, ...],
) -> TorchDecoder:
    """
    Build a differentiable :class:`~mmt.data.embeddings.base.TorchDecoder` for the given codec.

    The decoder is an ``nn.Module`` — call ``.to(device)`` on it (or on a parent module that
    contains it) to move any registered buffers (e.g., the DCT3D basis matrix) to the correct
    device before training.

    Parameters
    ----------
    codec : DCT3DCodec | IdentityCodec | VAECodec
        Fully initialised encoder instance.
    original_shape : tuple[int, ...]
        Native signal shape (without batch dimension), e.g. ``(T, )``, ``(C, T)``,
        ``(H, W, T)``. Used by DCT3D and Identity decoders; ignored for VAE (shape
        is inferred from the model's ``mmt_info.json``).

    Returns
    -------
    TorchDecoder
        Decoder instance ready for use in training losses and eval.

    Raises
    ------
    NotImplementedError
        If no TorchDecoder is registered for the given codec type.

    """

    # Lazy imports avoid circular dependencies at module load time
    from .dct3d import DCT3DTorchDecoder
    from .identity import IdentityTorchDecoder
    from .vae import VAETorchDecoder

    if isinstance(codec, DCT3DCodec):
        return DCT3DTorchDecoder(codec=codec, original_shape=original_shape)

    if isinstance(codec, IdentityCodec):
        return IdentityTorchDecoder(original_shape=original_shape)

    if isinstance(codec, VAECodec):
        return VAETorchDecoder(vae_codec=codec)

    raise NotImplementedError(
        f"No TorchDecoder registered for codec type {type(codec).__name__!r}. "
        "Implement a TorchDecoder subclass and register it in build_torch_decoder()."
    )


# ----------------------------------------------------------------------------------------------------------------------
def build_decoders(
    registry: "SignalSpecRegistry",
    codecs: dict[int, Any],
    role: str = "output",
) -> dict[int, TorchDecoder]:
    """
    Build a TorchDecoder for every signal of ``role`` in the registry.

    The ``original_shape`` passed to each decoder is ``spec.values_shape + (n_samples,)``,
    which matches the shape of the collated batch tensor (``output_native[signal_id]``).
    This is ``(T,)`` for scalar timeseries, ``(C, T)`` for profiles, and ``(H, W, T)`` for
    video — not the canonical ``(H, W, T)`` form stored in ``spec.native_shape``.

    Parameters
    ----------
    registry : SignalSpecRegistry
        Registry produced by ``build_signal_specs()``.
    codecs : dict[int, Any]
        Per-signal codec instances keyed by ``signal_id``, as returned by ``build_codecs()``.
    role : str
        Which role to build decoders for.  Defaults to ``"output"``.

    Returns
    -------
    dict[int, TorchDecoder]
        Mapping of ``signal_id → TorchDecoder`` for all specs of the given role.

    Raises
    ------
    KeyError
        If a codec is missing for a spec in the registry.
    NotImplementedError
        If no TorchDecoder is registered for a codec type (propagated from
        :func:`build_torch_decoder`).

    """

    decoders: dict[int, TorchDecoder] = {}
    for spec in registry.specs_for_role(role):
        if spec.signal_id not in codecs:
            raise KeyError(
                f"No codec found for signal_id={spec.signal_id} (name={spec.name!r}, role={spec.role!r}). "
                "Ensure build_codecs() was called for the same registry."
            )
        n_samples = spec.native_shape[2]
        original_shape = spec.values_shape + (n_samples,)
        decoders[spec.signal_id] = build_torch_decoder(
            codec=codecs[spec.signal_id],
            original_shape=original_shape,
        )
    return decoders
