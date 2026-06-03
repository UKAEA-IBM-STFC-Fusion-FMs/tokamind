"""
VAE Codec
---------

This module provides a VAE-backed codec for the MMT embedding pipeline, together with
its differentiable torch decoder.

The module provides:
    • ``VAECodec``          — numpy encoder (offline use: wraps a pretrained beta_VAE)
    • ``VAETorchDecoder``   — differentiable nn.Module decoder (training losses + eval)

Strict expectations (current VAE_fairmast refactor)
---------------------------------------------------
- VAE_fairmast is installed and exposes the python package ``vae_pipeline``.
- Trained VAEs live under:
    <VAE_fairmast>/src/vae_pipeline/data/New_VAEs/<MODEL_DIR>/
  containing:
    - exactly one ``config_*.json``
    - exactly one ``best_*.pt``

MMT YAML usage (per-signal)
---------------------------
    encoder_name: vae
    encoder_kwargs:
      model_dir: conv1d_vae_config_coil_current   # folder name under New_VAEs/
      device: cuda:0                              # optional, empty/None -> cpu
      use_mu: true                                # optional, default true

Notes
-----
- The encoder casts inputs to float32 to match model weights.
- ``VAETorchDecoder`` freezes all VAE decoder parameters (``requires_grad_(False)``),
  so gradients flow *through* the decoder to the embedding prediction ``z`` but do not
  update the pretrained weights.
- If VAE_fairmast changes its API, update ``_import_vae_pipeline()``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json

import numpy as np
import torch
from torch import Tensor

from .torch_decoder import TorchDecoder


# ======================================================================================================================
# Lazy imports
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _import_vae_pipeline() -> tuple[Any, Any, Any]:
    """Import VAE-related packages, raising ImportError if packages not found."""

    try:
        import vae_pipeline.configs.config_setup as config_setup_mod
        from vae_pipeline.configs.config_setup import get_settings
        from vae_pipeline.models.vae_model import beta_VAE
    except ImportError as e:
        raise ImportError(
            "Failed to import required symbols from VAE_fairmast package 'vae_pipeline'.\n"
            "This likely means either VAE_fairmast is not installed in this environment,\n"
            "or the repo refactored its Python API.\n\n"
            "Expected imports:\n"
            "  - import vae_pipeline.configs.config_setup as config_setup_mod\n"
            "  - from vae_pipeline.configs.config_setup import get_settings\n"
            "  - from vae_pipeline.models.vae_model import beta_VAE\n\n"
            "Install VAE_fairmast in editable mode (recommended):\n"
            "  pip install -e /path/to/VAE_fairmast\n\n"
            "If VAE_fairmast changed its API, update `_import_vae_pipeline()` in:\n"
            "  src/mmt/data/embeddings/vae.py\n\n"
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    return config_setup_mod, get_settings, beta_VAE


# ======================================================================================================================
# Config + path helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _read_json(path: Path) -> dict[str, Any]:
    """Read JSON data from target path."""
    with path.open(mode="r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------------------------------------
def resolve_vae_model_dir(model_dir: str | Path) -> Path:
    """
    Resolve model_dir to an on-disk directory.

    Accepted:
      1) A filesystem path to a directory (absolute or relative) that exists.
      2) A folder name under: <vae_pipeline package>/data/New_VAEs/<model_dir>

    Parameters
    ----------
    model_dir : str | Path
        Input VAE model directory to be resolved.

    Returns
    -------
    Path
        Resolved VAE model directory.

    Raises
    ------
    RuntimeError
        If ``vae_pipeline.configs.config_setup`` cannot be located on disk.
    FileNotFoundError
        If the VAE model directory cannot be resolved.

    """

    p = Path(model_dir).expanduser()
    if p.is_dir():
        return p.resolve()

    config_setup_mod, _, _ = _import_vae_pipeline()
    cs_file_obj: object = getattr(config_setup_mod, "__file__", None)
    if not isinstance(cs_file_obj, (str, os.PathLike)):
        raise RuntimeError(
            "Could not locate vae_pipeline.configs.config_setup on disk (missing __file__). "
            "Your installation looks like a namespace package without source files. "
            "Install VAE_fairmast in editable mode."
        )

    vae_pkg_dir = Path(cs_file_obj).resolve().parent.parent
    trained_root = (vae_pkg_dir / "data" / "New_VAEs").resolve()
    cand = trained_root / p

    if cand.is_dir():
        return cand

    raise FileNotFoundError(
        "Could not resolve VAE model directory.\n"
        f"Provided model_dir: {str(model_dir)!r}\n"
        "Tried:\n"
        f"  - as a filesystem directory: {p.resolve()}\n"
        f"  - as a trained VAE folder:   {cand}\n\n"
        "Expected trained VAEs under: <VAE_fairmast>/src/vae_pipeline/data/New_VAEs/<MODEL_DIR>."
    )


# ----------------------------------------------------------------------------------------------------------------------
def read_vae_model_meta(  # NOSONAR - Ignore cognitive complexity
    model_dir: str | Path,
) -> dict[str, Any]:
    """
    Read minimal metadata needed by MMT from a trained VAE folder.

    STRICT:
    - Each model folder MUST contain:
        - ``mmt_info.json``
        - exactly one ``config_*.json``
    - ``mmt_info.json`` MUST contain:
        - model_type (str)        — one of ``{"linear", "conv1d", "conv2d"}``
        - latent_dim (int)
        - input_shape (list[int]) — [C, T] for linear/conv1d, [H, W, T] for conv2d
        - input_mode (str)        — ``{"channels", "time"}`` with model-specific constraints
        - checkpoint (str)        — filename or glob pattern relative to the model folder

    Parameters
    ----------
    model_dir : str | Path
        Target VAE model directory.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys: model_dir, config_path, checkpoint_path, model_type,
        latent_dim, input_shape, input_mode.

    Raises
    ------
    FileNotFoundError
        If ``mmt_info.json`` or the checkpoint file is not found.
    RuntimeError
        If ``config_*.json`` or checkpoint pattern does not match exactly 1 file.
    KeyError
        If ``mmt_info.json`` is missing a required key.
    ValueError
        If ``mmt_info.json`` has an invalid ``input_shape`` for the declared ``model_type``.

    """

    md = resolve_vae_model_dir(model_dir=model_dir)

    info_path = md / "mmt_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing required `mmt_info.json` in VAE model directory: {md}\n"
            "Create it with keys: model_type, latent_dim, input_shape, input_mode, checkpoint."
        )
    info = _read_json(path=info_path)

    for k in ["model_type", "latent_dim", "input_shape", "input_mode", "checkpoint"]:
        if k not in info:
            raise KeyError(f"`mmt_info.json` missing required key {k!r}: {info_path}.")

    model_type = str(info["model_type"])
    latent_dim = int(info["latent_dim"])
    input_shape = tuple(int(v) for v in info["input_shape"])
    input_mode = str(info["input_mode"])
    checkpoint_spec = str(info["checkpoint"])

    if model_type in {"linear", "conv1d"}:
        if len(input_shape) != 2:
            raise ValueError(
                f"`mmt_info.json` has invalid input_shape for model_type={model_type!r}: "
                f"expected length 2 [C,T], got {list(input_shape)} ({info_path})."
            )
    elif model_type == "conv2d":
        if len(input_shape) != 3:
            raise ValueError(
                "`mmt_info.json` has invalid input_shape for model_type='conv2d': "
                f"expected length 3 [H,W,T], got {list(input_shape)} ({info_path})."
            )
    else:
        raise ValueError(
            f"`mmt_info.json` has unsupported model_type={model_type!r} in {info_path}. "
            "Expected one of {'linear', 'conv1d', 'conv2d'}."
        )

    cfg_files = sorted(md.glob("config_*.json"))
    if len(cfg_files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 `config_*.json` in {md}, found {len(cfg_files)}: "
            + ", ".join([p.name for p in cfg_files])
        )
    config_path = cfg_files[0]

    has_glob = any(ch in checkpoint_spec for ch in ("*", "?", "["))
    if has_glob:
        matches = sorted(md.glob(checkpoint_spec))
        if len(matches) != 1:
            raise RuntimeError(
                f"Checkpoint pattern {checkpoint_spec!r} in {info_path} must match exactly 1 file in {md}, "
                f"matched {len(matches)}: " + ", ".join([p.name for p in matches])
            )
        checkpoint_path = matches[0]
    else:
        checkpoint_path = md / checkpoint_spec
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint file {checkpoint_spec!r} from {info_path} not found in {md}.")

    return {
        "model_dir": md,
        "config_path": config_path,
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "latent_dim": latent_dim,
        "input_shape": input_shape,
        "input_mode": input_mode,
    }


# ======================================================================================================================
# Encoder
# ======================================================================================================================


# ======================================================================================================================
class VAECodec:
    """
    Encoder wrapping a pretrained VAE_fairmast beta_VAE model.

    Provides a single ``encode(x) -> (latent_dim,)`` method for offline embedding
    generation. The underlying torch model is available as ``self.model`` for use
    by :class:`VAETorchDecoder`.

    Parameters
    ----------
    model_dir : str
        Path to the trained VAE directory (or folder name under ``New_VAEs/``).
    device : str | None
        Device for the model (default: cpu).
    use_mu : bool
        If True, use the encoder mean as the latent code; else sample from (mu, logvar).

    Attributes
    ----------
    meta : dict
        Metadata loaded from ``mmt_info.json``.
    model_type : str
        One of ``{"linear", "conv1d", "conv2d"}``.
    latent_dim : int
        Latent space dimension.
    input_shape : tuple[int, ...]
        Expected signal shape: (C, T) for linear/conv1d, (H, W, T) for conv2d.
    input_mode : str
        One of ``{"channels", "time"}``.
    expected_channels : int
        Expected number of channels derived from ``input_shape``.
    expected_time_steps : int
        Expected number of time steps derived from ``input_shape``.
    device : torch.device
        Device the model is loaded on.
    use_mu : bool
        Whether to use the encoder mean as the latent code.
    model : nn.Module
        The loaded beta_VAE model (eval mode, on ``self.device``).
    requires_finite_input : bool
        Whether the codec requires finite (non-NaN) inputs. ``True`` for the current VAE
        implementation. Set to ``False`` on a future NaN-aware VAE subclass.

    """

    requires_finite_input: bool = False

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, *, model_dir: str, device: str | None = None, use_mu: bool = True) -> None:
        """
        Load and initialize the VAE model.

        Parameters
        ----------
        model_dir : str
            Target model directory.
        device : str | None
            Target device. None or empty string defaults to cpu.
        use_mu : bool
            If True, use the encoder mean as latent code; else sample.

        Raises
        ------
        RuntimeError
            If the checkpoint format is unsupported.

        """

        self.meta = read_vae_model_meta(model_dir=model_dir)
        self.model_type = str(self.meta["model_type"])
        self.latent_dim = int(self.meta["latent_dim"])
        self.input_shape = tuple(int(v) for v in self.meta["input_shape"])
        self.input_mode = str(self.meta["input_mode"])

        self._validate_model_layout_contract()

        if self.model_type in {"linear", "conv1d"}:
            self.expected_channels = int(self.input_shape[0])
            self.expected_time_steps = int(self.input_shape[1])
        elif self.model_type == "conv2d":
            h, w, t = self.input_shape
            self.expected_channels = int(h * w)
            self.expected_time_steps = int(t)
        else:
            raise RuntimeError(
                f"[VAECodec] Unsupported model_type={self.model_type!r} in metadata for {self.meta['model_dir']}."
            )

        dev: str = device if (device is not None and str(device).strip() != "") else "cpu"
        self.device: torch.device = torch.device(dev)
        self.use_mu = bool(use_mu)

        _, get_settings, beta_VAE = _import_vae_pipeline()  # NOSONAR # noqa - Ignore lowercase warning

        settings = get_settings(str(self.meta["config_path"]))
        self.model = beta_VAE(settings)

        ckpt = torch.load(self.meta["checkpoint_path"], map_location=self.device)
        if (not isinstance(ckpt, dict)) or ("model_state_dict" not in ckpt):
            raise RuntimeError(
                f"[VAECodec] Unsupported checkpoint format at {self.meta['checkpoint_path']}. "
                "Expected a dict containing key 'model_state_dict'."
            )

        sd = ckpt["model_state_dict"]
        prefix = "module."
        len_prefix = len(prefix)
        if any(k.startswith(prefix) for k in sd.keys()):
            sd = {k[len_prefix:] if k.startswith(prefix) else k: v for k, v in sd.items()}

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    def _validate_model_layout_contract(  # NOSONAR - Ignore cognitive complexity
        self,
    ) -> None:
        """
        Validate the strict model/input contract defined by ``mmt_info.json``.

        Raises
        ------
        ValueError
            On any shape or mode mismatch.

        """

        if any(int(d) <= 0 for d in self.input_shape):
            raise ValueError(
                f"[VAECodec] All input_shape dimensions must be > 0, got {list(self.input_shape)} "
                f"(model_dir={self.meta['model_dir']})."
            )

        if self.model_type == "linear":
            if len(self.input_shape) != 2:
                raise ValueError(
                    f"[VAECodec] `linear` model requires input_shape=[C,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode not in {"channels", "time"}:
                raise ValueError(
                    f"[VAECodec] `linear` model requires input_mode in ['channels','time'], got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        if self.model_type == "conv1d":
            if len(self.input_shape) != 2:
                raise ValueError(
                    f"`conv1d` model requires input_shape=[C,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode != "channels":
                raise ValueError(
                    f"[VAECodec] `conv1d` model requires input_mode='channels', got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        if self.model_type == "conv2d":
            if len(self.input_shape) != 3:
                raise ValueError(
                    f"`conv2d` model requires input_shape=[H,W,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode != "time":
                raise ValueError(
                    f"`conv2d` model requires input_mode='time', got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        raise ValueError(
            f"[VAECodec] Unsupported model_type={self.model_type!r}; expected one of {{'linear', 'conv1d', 'conv2d'}}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _to_channel_time(x: np.ndarray) -> np.ndarray:
        """
        Reshape input array into (C, T) canonical form.

        Parameters
        ----------
        x : np.ndarray
            Input array with ``x.ndim`` in [1, 2, 3].

        Returns
        -------
        np.ndarray
            Array in (C, T) form.

        Raises
        ------
        ValueError
            If ``x.ndim`` not in [1, 2, 3].

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa - Ignore unreachable code warning

        if x.ndim == 1:
            return x.reshape(1, x.shape[0])
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            h, w, t = x.shape
            return x.reshape(h * w, t)

        raise ValueError(f"[VAECodec] VAECodec supports `x.ndim` in [1,2,3], got shape={x.shape}.")

    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_encode_array(  # NOSONAR - Ignore cognitive complexity
        self, x: np.ndarray
    ) -> np.ndarray:
        """
        Convert an input sample to the shape expected by the underlying VAE encoder.

        Returns a model-ready array without batch dimension:
          - linear (input_mode="time")      -> (C, T)
          - linear (input_mode="channels")  -> (T, C)
          - conv1d                          -> (C, T)
          - conv2d                          -> (T, H, W)

        Parameters
        ----------
        x : np.ndarray
            Input sample to be encoded.

        Returns
        -------
        np.ndarray
            Model-ready sample (without batch dimension).

        Raises
        ------
        ValueError
            On shape or mode mismatch.

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa - Ignore unreachable code warning

        if self.model_type == "conv2d":
            h_exp, w_exp, t_exp = (int(v) for v in self.input_shape)
            hw_exp = int(h_exp * w_exp)

            if x.ndim == 3:
                h_in, w_in, t_in = (int(v) for v in x.shape)
                if t_in != t_exp:
                    raise ValueError(
                        f"[VAECodec] `conv2d` length mismatch for model {self.meta['model_dir'].name!r}: "
                        f"expected T={t_exp}, got T={t_in} (input shape {tuple(x.shape)})."
                    )
                if (h_in * w_in) != hw_exp:
                    raise ValueError(
                        f"[VAECodec] `conv2d` spatial mismatch for model {self.meta['model_dir'].name!r}: "
                        f"expected H*W={hw_exp}, got H*W={h_in * w_in} (input shape {tuple(x.shape)})."
                    )
                x_hwt = x if (h_in, w_in) == (h_exp, w_exp) else x.reshape(h_exp, w_exp, t_exp)
            elif x.ndim == 2:
                if x.shape != (hw_exp, t_exp):
                    raise ValueError(
                        f"[VAECodec] `conv2d` expects (H*W,T)=({hw_exp},{t_exp}) or (H,W,T)=({h_exp},{w_exp},{t_exp}), "
                        f"got shape {tuple(x.shape)}."
                    )
                x_hwt = x.reshape(h_exp, w_exp, t_exp)
            elif x.ndim == 1 and hw_exp == 1 and x.shape[0] == t_exp:
                x_hwt = x.reshape(h_exp, w_exp, t_exp)
            else:
                raise ValueError(
                    f"[VAECodec] `conv2d` supports input shapes (H,W,T), (H*W,T), or (T,) when H=W=1. "
                    f"Expected [H,W,T]={list(self.input_shape)}, got shape {tuple(x.shape)}."
                )

            x_thw = np.transpose(x_hwt, (2, 0, 1))  # (T, H, W)
            return np.asarray(x_thw, dtype=np.float32)

        x_ct = self._to_channel_time(x=x)
        c_exp, t_exp = int(self.input_shape[0]), int(self.input_shape[1])

        if x_ct.shape != (c_exp, t_exp):
            raise ValueError(
                f"[VAECodec] Input shape mismatch for model {self.meta['model_dir'].name!r}: "
                f"expected (C,T)=({c_exp},{t_exp}), got (C,T)={tuple(x_ct.shape)} "
                f"(input shape {tuple(x.shape)})."
            )

        if self.model_type == "conv1d":
            return np.asarray(x_ct, dtype=np.float32)

        if self.model_type == "linear":
            if self.input_mode == "time":
                x_linear = x_ct
            elif self.input_mode == "channels":
                x_linear = x_ct.T
            else:
                raise ValueError(
                    f"[VAECodec] Unexpected input_mode={self.input_mode!r} for `linear` model "
                    f"{self.meta['model_dir'].name!r}."
                )
            return np.asarray(x_linear, dtype=np.float32)

        raise ValueError(
            f"[VAECodec] Unsupported model_type={self.model_type!r} in _prepare_encode_array for "
            f"model {self.meta['model_dir'].name!r}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a single signal sample to a latent vector.

        Parameters
        ----------
        x : np.ndarray
            Input array to be encoded.

        Returns
        -------
        np.ndarray
            Latent vector of shape (latent_dim,), dtype float32.

        Raises
        ------
        RuntimeError
            If the encoder produces a shape other than (latent_dim,).

        """

        x_model = self._prepare_encode_array(x=np.asarray(x))

        x_t = torch.from_numpy(np.asarray(x_model, dtype=np.float32)).unsqueeze(0)
        x_t = x_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mu, logvar = self.model.encode(x_t)
            z_t = mu if self.use_mu else self.model.reparameterize(mu, logvar)

        z = np.asarray(z_t.detach().cpu().numpy(), dtype=np.float32)
        if z.shape[-1] != self.latent_dim:
            raise RuntimeError(
                f"[VAECodec] encode produced latent shape {tuple(z_t.shape)}; expected latent_dim={self.latent_dim}."
            )

        leading_dims = tuple(int(d) for d in z.shape[:-1])
        if any(d != 1 for d in leading_dims):
            raise RuntimeError(
                f"[VAECodec] encode produced multiple latent vectors with shape {tuple(z_t.shape)}; "
                "expected exactly one latent vector per sample."
            )

        return z.reshape(self.latent_dim)


# ======================================================================================================================
# Differentiable decoder
# ======================================================================================================================


# ======================================================================================================================
class VAETorchDecoder(TorchDecoder):
    """
    Differentiable VAE decoder for training losses and eval.

    Wraps the pretrained ``beta_VAE.decode()`` network from ``VAECodec.model`` with all
    parameters frozen (``requires_grad_(False)``). Gradients flow *through* the decoder
    to the embedding prediction ``z`` but do not update the pretrained weights.

    Output shape is ``(B, *original_shape)`` where ``original_shape`` is the per-sample shape
    of ``output_native`` in the collated batch (e.g. ``(T,)`` for scalar timeseries, ``(C, T)``
    for profiles). The raw decoder output from the VAE model is reshaped to this contract after
    decoding, so singleton spatial dims present in the VAE's internal ``input_shape`` (e.g.
    ``(1, T)``) are correctly squeezed away.

    Parameters
    ----------
    vae_codec : VAECodec
        Fully initialized encoder instance. Its ``model``, ``model_type``,
        ``input_mode``, and ``input_shape`` are used to configure the decoder.
    original_shape : tuple[int, ...] | None
        Expected per-sample output shape (without batch dimension). When provided, the
        decoder's output is reshaped to ``(B, *original_shape)`` after decoding. When
        ``None``, the raw shape from ``_reshape_to_native`` is returned as-is.

    Notes
    -----
    Some beta_VAE variants return an extra singleton dimension from ``decode()``.
    The ``forward`` method handles this transparently via a try/except, matching
    the behavior of the original encoder.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, vae_codec: VAECodec, original_shape: tuple[int, ...] | None = None) -> None:
        super().__init__()

        # Freeze: gradients flow through ops, not into pretrained weights
        self._vae_model = vae_codec.model
        for p in self._vae_model.parameters():
            p.requires_grad_(False)

        self._model_type: str = vae_codec.model_type
        self._input_mode: str = vae_codec.input_mode
        self._input_shape: tuple[int, ...] = tuple(vae_codec.input_shape)
        self._native_shape: tuple[int, ...] = tuple(vae_codec.input_shape)
        self._output_shape: tuple[int, ...] | None = tuple(original_shape) if original_shape is not None else None

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, z: Tensor) -> Tensor:
        """
        Decode a batch of latent vectors to native standardized space.

        Parameters
        ----------
        z : Tensor
            Latent vectors of shape (B, D).

        Returns
        -------
        Tensor
            Native standardized output of shape (B, *native_shape).
            Gradients are preserved with respect to ``z``.

        """

        B = z.shape[0]

        try:
            x_hat = self._vae_model.decode(z)
        except Exception:  # noqa - Some VAE variants expect (B, 1, D)
            x_hat = self._vae_model.decode(z.unsqueeze(1))

        x_hat = self._reshape_to_native(x_hat, B)

        if self._output_shape is not None:
            x_hat = x_hat.reshape(B, *self._output_shape)

        return x_hat

    # ------------------------------------------------------------------------------------------------------------------
    def _reshape_to_native(  # NOSONAR - Ignore cognitive complexity
        self,
        x_hat: Tensor,
        B: int,  # noqa - Ignore lowercase warning
    ) -> Tensor:
        """
        Convert the raw model output to (B, *native_shape).

        Parameters
        ----------
        x_hat : Tensor
            Raw decoder output (shape depends on model_type and VAE architecture).
        B : int
            Batch size.

        Returns
        -------
        Tensor
            Output of shape (B, *native_shape).

        Raises
        ------
        ValueError
            If ``self._model_type`` is not in ``{"linear", "conv1d", "conv2d"}``.

        """

        if self._model_type == "conv1d":
            # Model outputs (B, C, T); strip spurious leading dim if model returned (C, T)
            if x_hat.dim() == 2:
                x_hat = x_hat.unsqueeze(0)
            return x_hat  # (B, C, T) == (B, *native_shape)

        if self._model_type == "linear":
            # Collapse any extra dims to (B, F)
            if x_hat.dim() > 2:
                x_hat = x_hat.view(B, -1)
            c_exp, t_exp = self._input_shape
            if self._input_mode == "time":
                # Model saw (B, C, T) → output is (B, C, T)
                return x_hat.view(B, c_exp, t_exp)
            else:  # channels: model saw (B, T, C) → transpose back to (B, C, T)
                return x_hat.view(B, t_exp, c_exp).permute(0, 2, 1).contiguous()

        if self._model_type == "conv2d":
            # Strip spurious leading dim if model returned (T, H, W)
            if x_hat.dim() == 3:
                x_hat = x_hat.unsqueeze(0)
            # x_hat: (B, T, H, W) → permute to (B, H, W, T) = native_shape
            return x_hat.permute(0, 2, 3, 1).contiguous()

        raise ValueError(
            f"[VAETorchDecoder] Unsupported model_type={self._model_type!r}; "
            "expected one of {'linear', 'conv1d', 'conv2d'}."
        )
