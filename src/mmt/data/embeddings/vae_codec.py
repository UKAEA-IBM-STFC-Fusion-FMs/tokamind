"""
mmt.data.embeddings.vae_codec

VAE-backed codec for the MMT embedding pipeline using VAE_fairmast.

Strict expectations (current VAE_fairmast refactor)
---------------------------------------------------
- VAE_fairmast is installed and exposes the python package `vae_pipeline`.
- Trained VAEs live under:
    <VAE_fairmast>/src/vae_pipeline/data/VAEs/<MODEL_DIR>/
  containing:
    - exactly one config_*.json
    - exactly one best_*.pt

MMT YAML usage (per-signal)
---------------------------
    encoder_name: vae
    encoder_kwargs:
      model_dir: conv1d_vae_config_coil_current   # folder name under VAEs/
      device: cuda:0                              # optional, empty/None -> cpu
      use_mu: true                                # optional, default true

Notes
-----
- This codec casts inputs to float32 to match model weights (avoids "Input type (double) and bias type (float)" errors).
- This codec is intentionally strict about the directory layout and files. If VAE_fairmast changes, update the imports
  inside `_import_vae_pipeline()`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import json
import numpy as np

import torch


# ======================================================================================================================
# Lazy imports in a "benchmark_imports.py style" (but wrapped in a function so that importing MMT does not require
# VAE_fairmast unless encoder_name='vae' is used).
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _import_vae_pipeline():
    try:
        # Import the module object too (for locating the package on disk)
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
            "  src/mmt/data/embeddings/vae_codec.py\n\n"
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
      2) A folder name under: <vae_pipeline package>/data/VAEs/<model_dir>

    This is strict: no other search paths.

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
        If `vae_pipeline.configs.config_setup` cannot be located using the provided model path `model_dir`.
    FileNotFoundError
        If VAE model directory cannot be resolved.

    """

    p = Path(model_dir).expanduser()
    if p.is_dir():
        return p.resolve()

    # Anchor to a real module file (works even if `vae_pipeline` is a namespace package)
    config_setup_mod, _, _ = _import_vae_pipeline()
    cs_file_obj: object = getattr(config_setup_mod, "__file__", None)
    if not isinstance(cs_file_obj, (str, os.PathLike)):
        raise RuntimeError(
            "Could not locate vae_pipeline.configs.config_setup on disk (missing __file__). "
            "Your installation looks like a namespace package without source files. "
            "Install VAE_fairmast in editable mode."
        )

    # .../vae_pipeline/configs/config_setup.py -> .../vae_pipeline
    vae_pkg_dir = Path(cs_file_obj).resolve().parent.parent
    trained_root = (vae_pkg_dir / "data" / "VAEs").resolve()
    cand = trained_root / p

    if cand.is_dir():
        return cand

    raise FileNotFoundError(
        "Could not resolve VAE model directory.\n"
        f"Provided model_dir: {str(model_dir)!r}\n"
        "Tried:\n"
        f"  - as a filesystem directory: {p.resolve()}\n"
        f"  - as a trained VAE folder:   {cand}\n\n"
        "Expected trained VAEs under: <VAE_fairmast>/src/vae_pipeline/data/VAEs/<MODEL_DIR>."
    )


# ----------------------------------------------------------------------------------------------------------------------
def read_vae_model_meta(model_dir: str | Path) -> dict[str, Any]:
    """
    Read minimal metadata needed by MMT from a trained VAE folder.

    STRICT:
    - Each model folder MUST contain:
        - mmt_info.json
        - exactly one config_*.json (used to build VAE settings)
    - mmt_info.json MUST contain:
        - model_type (str)   # one of {"linear", "conv1d", "conv2d"}
        - latent_dim (int)
        - input_shape (list[int])  # [C, T] for linear/conv1d, [H, W, T] for conv2d
        - input_mode (str)   # {"channels", "time"} with model-specific constraints
        - checkpoint (str)   # filename or glob pattern relative to the model folder

    Parameters
    ----------
    model_dir : str | Path
        Target VAE model directory.

    Returns
    -------
    dict[str, Any]
        Dictionary with model metadata:
            - model_dir (Path)
            - config_path (Path)
            - checkpoint_path (Path)
            - model_type (str)
            - latent_dim (int)
            - input_shape (tuple[int, ...])
            - input_mode (str)

    Raises
    ------
    FileNotFoundError
        If required mmt_info.json file not found in VAE model directory `model_dir`.
        If checkpoint pattern does not match 1 file in VAE model directory `model_dir`.
        If checkpoint file not found in VAE model directory `model_dir`.
    KeyError
        If mmt_info.json file in VAE model directory `model_dir` does not have a required key in "latent_dim",
        "model_type", "input_shape", "input_mode", "checkpoint"].

    """

    md = resolve_vae_model_dir(model_dir=model_dir)

    info_path = md / "mmt_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing required mmt_info.json in VAE model directory: {md}\n"
            "Create it with keys: model_type, latent_dim, input_shape, input_mode, checkpoint."
        )
    info = _read_json(path=info_path)

    for k in ["model_type", "latent_dim", "input_shape", "input_mode", "checkpoint"]:
        if k not in info:
            raise KeyError(f"mmt_info.json missing required key {k!r}: {info_path}.")

    model_type = str(info["model_type"])
    latent_dim = int(info["latent_dim"])
    input_shape = tuple(int(v) for v in info["input_shape"])
    input_mode = str(info["input_mode"])
    checkpoint_spec = str(info["checkpoint"])

    # Validate basic shape contract at metadata read time.
    if model_type in {"linear", "conv1d"}:
        if len(input_shape) != 2:
            raise ValueError(
                f"mmt_info.json has invalid input_shape for model_type={model_type!r}: "
                f"expected length 2 [C,T], got {list(input_shape)} ({info_path})."
            )
    elif model_type == "conv2d":
        if len(input_shape) != 3:
            raise ValueError(
                "mmt_info.json has invalid input_shape for model_type='conv2d': "
                f"expected length 3 [H,W,T], got {list(input_shape)} ({info_path})."
            )
    else:
        raise ValueError(
            f"mmt_info.json has unsupported model_type={model_type!r} in {info_path}. "
            "Expected one of {'linear', 'conv1d', 'conv2d'}."
        )

    cfg_files = sorted(md.glob("config_*.json"))
    if len(cfg_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 config_*.json in {md}, found {len(cfg_files)}: "
            + ", ".join([p.name for p in cfg_files])
        )
    config_path = cfg_files[0]

    # checkpoint_spec can be an exact filename or a glob (e.g., "best_*.pt")
    has_glob = any(ch in checkpoint_spec for ch in ("*", "?", "["))
    if has_glob:
        matches = sorted(md.glob(checkpoint_spec))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Checkpoint pattern {checkpoint_spec!r} in {info_path} must match exactly 1 file in {md}, "
                f"matched {len(matches)}: " + ", ".join([p.name for p in matches])
            )
        checkpoint_path = matches[0]
    else:
        checkpoint_path = md / checkpoint_spec
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint file {checkpoint_spec!r} from {info_path} not found in {md}."
            )

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
# Codec implementation
# ======================================================================================================================


# ======================================================================================================================
class VAECodec:
    """
    Wrap a pretrained VAE_fairmast beta_VAE model with a codec interface.

    encode(x) -> (latent_dim,) float32
    decode(z, original_shape) -> float32 array of original_shape

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self, *, model_dir: str, device: str | None = None, use_mu: bool = True
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        model_dir : str
            Target model directory.
        device : str | None
            Target device.
            Optional. Default: None.
        use_mu : bool
            If True, use the encoder mean (mu) as latent code; else sample from (mu, logvar).
            Optional. Default: True.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        RuntimeError
            If unsupported checkpoint format for the loaded model checkpoint.

        """

        self.meta = read_vae_model_meta(model_dir=model_dir)
        self.model_type = str(self.meta["model_type"])
        self.latent_dim = int(self.meta["latent_dim"])
        self.input_shape = tuple(int(v) for v in self.meta["input_shape"])
        self.input_mode = str(self.meta["input_mode"])

        self._validate_model_layout_contract()

        # Canonical expected dimensions are derived from input_shape.
        if self.model_type in {"linear", "conv1d"}:
            self.expected_channels = int(self.input_shape[0])
            self.expected_time_steps = int(self.input_shape[1])
        elif self.model_type == "conv2d":
            h, w, t = self.input_shape
            self.expected_channels = int(h * w)
            self.expected_time_steps = int(t)
        else:
            # read_vae_model_meta() already validates model_type; keep this defensive guard local.
            raise RuntimeError(
                f"Unsupported model_type={self.model_type!r} in metadata for {self.meta['model_dir']}."
            )

        dev = device if (device is not None and str(device).strip() != "") else "cpu"
        self.device = torch.device(str(dev))
        self.use_mu = bool(use_mu)

        # Lazy import VAE_fairmast symbols
        _, get_settings, beta_VAE = _import_vae_pipeline()

        settings = get_settings(str(self.meta["config_path"]))
        self.model = beta_VAE(settings)

        ckpt = torch.load(self.meta["checkpoint_path"], map_location=self.device)
        if (not isinstance(ckpt, dict)) or ("model_state_dict" not in ckpt):
            raise RuntimeError(
                f"Unsupported checkpoint format at {self.meta['checkpoint_path']}. "
                "Expected a dict containing key 'model_state_dict'."
            )

        sd = ckpt["model_state_dict"]

        # Handle DataParallel prefix
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {
                k[len("module.") :] if k.startswith("module.") else k: v
                for k, v in sd.items()
            }

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def _validate_model_layout_contract(self) -> None:
        """
        Validate the strict model/input contract defined by mmt_info.json.

        This codec enforces a single convention:
        - linear  : input_shape=[C,T], input_mode in {"channels", "time"}
        - conv1d  : input_shape=[C,T], input_mode == "channels"
        - conv2d  : input_shape=[H,W,T], input_mode == "time" (time mapped to Conv2d channels)

        Raises
        ------
        ValueError
            If model_type/input_shape/input_mode combination does not follow the strict contract.
            If any input_shape dimension is not strictly positive.

        """

        if any(int(d) <= 0 for d in self.input_shape):
            raise ValueError(
                f"All input_shape dimensions must be > 0, got {list(self.input_shape)} "
                f"(model_dir={self.meta['model_dir']})."
            )

        if self.model_type == "linear":
            if len(self.input_shape) != 2:
                raise ValueError(
                    f"linear model requires input_shape=[C,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode not in {"channels", "time"}:
                raise ValueError(
                    f"linear model requires input_mode in {{'channels','time'}}, got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        if self.model_type == "conv1d":
            if len(self.input_shape) != 2:
                raise ValueError(
                    f"conv1d model requires input_shape=[C,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode != "channels":
                raise ValueError(
                    f"conv1d model requires input_mode='channels', got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        if self.model_type == "conv2d":
            if len(self.input_shape) != 3:
                raise ValueError(
                    f"conv2d model requires input_shape=[H,W,T], got {list(self.input_shape)} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            if self.input_mode != "time":
                raise ValueError(
                    f"conv2d model requires input_mode='time', got {self.input_mode!r} "
                    f"(model_dir={self.meta['model_dir']})."
                )
            return

        raise ValueError(
            f"Unsupported model_type={self.model_type!r}; expected one of "
            "{'linear', 'conv1d', 'conv2d'}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _to_channel_time(x: np.ndarray) -> np.ndarray:
        """
        Reshape input array into (channel, time) form.

        Parameters
        ----------
        x : np.ndarray
            Input array with expected `x.ndim` in [1, 2, 3], reshaped into (channel, time) form.

        Returns
        -------
        np.ndarray
            Reshaped input array in (channel, time) form.

        Raises
        ------
        ValueError
            If `x.ndim` not in in [1,2,3]."

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa (omit unreachable code warning)

        if x.ndim == 1:
            x_ct = x.reshape(1, x.shape[0])  # (1, T)
        elif x.ndim == 2:
            x_ct = x  # (C, T)
        elif x.ndim == 3:
            h, w, t = x.shape
            x_ct = x.reshape(h * w, t)  # (H*W, T)
        else:
            raise ValueError(
                f"VAECodec supports `x.ndim` in [1,2,3], got shape={x.shape}."
            )

        return x_ct

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _reshape_back(x_ct: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        """
        Reshape a (channel, time) array to its original shape.

        Parameters
        ----------
        x_ct : np.ndarray
            Input array in (channel, time) form to be reshaped to its original form.
        original_shape : tuple[int, ...]
            Original shape of input array `x_ct`.

        Returns
        -------
        np.ndarray
            Reshaped input array to original shape.

        Raises
        ------
        ValueError
            If `x_ct`cannot be reshaped to `original_shape`.
            If reconstruction shape `x_ct.shape` is incompatible with `original_shape`.
            If `len(original_shape)` not in [1, 2, 3].

        """

        if len(original_shape) == 1:
            if x_ct.shape[0] != 1:
                raise ValueError(
                    f"Cannot reshape reconstruction {x_ct.shape} to original_shape={original_shape}: expected 1 "
                    f"channel."
                )
            return x_ct.reshape(original_shape)

        if len(original_shape) == 2:
            return x_ct.reshape(original_shape)

        if len(original_shape) == 3:
            h, w, t = original_shape
            if x_ct.shape != (h * w, t):
                raise ValueError(
                    f"Reconstruction shape {x_ct.shape} incompatible with original_shape={original_shape}."
                )
            return x_ct.reshape(original_shape)

        raise ValueError(f"Unsupported original_shape={original_shape}.")

    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_encode_array(self, x: np.ndarray) -> np.ndarray:
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
            If input sample shape is incompatible with model_type/input_shape/input_mode.
        RuntimeError
            If model_type/input_mode is internally inconsistent.

        """

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)  # noqa (omit unreachable code warning)

        # conv2d expects [H,W,T] in metadata and [T,H,W] at model input (time as channels).
        if self.model_type == "conv2d":
            h_exp, w_exp, t_exp = (int(v) for v in self.input_shape)
            hw_exp = int(h_exp * w_exp)

            if x.ndim == 3:
                h_in, w_in, t_in = (int(v) for v in x.shape)
                if t_in != t_exp:
                    raise ValueError(
                        f"VAECodec conv2d length mismatch for model {self.meta['model_dir'].name!r}: "
                        f"expected T={t_exp}, got T={t_in} (input shape {tuple(x.shape)})."
                    )
                if (h_in * w_in) != hw_exp:
                    raise ValueError(
                        f"VAECodec conv2d spatial mismatch for model {self.meta['model_dir'].name!r}: "
                        f"expected H*W={hw_exp}, got H*W={h_in * w_in} (input shape {tuple(x.shape)})."
                    )
                x_hwt = (
                    x
                    if (h_in, w_in) == (h_exp, w_exp)
                    else x.reshape(h_exp, w_exp, t_exp)
                )
            elif x.ndim == 2:
                if x.shape != (hw_exp, t_exp):
                    raise ValueError(
                        f"VAECodec conv2d expects (H*W,T)=({hw_exp},{t_exp}) or (H,W,T)=({h_exp},{w_exp},{t_exp}), "
                        f"got shape {tuple(x.shape)}."
                    )
                x_hwt = x.reshape(h_exp, w_exp, t_exp)
            elif x.ndim == 1 and hw_exp == 1 and x.shape[0] == t_exp:
                x_hwt = x.reshape(h_exp, w_exp, t_exp)
            else:
                raise ValueError(
                    f"VAECodec conv2d supports input shapes (H,W,T), (H*W,T), or (T,) when H=W=1. "
                    f"Expected [H,W,T]={list(self.input_shape)}, got shape {tuple(x.shape)}."
                )

            x_thw = np.transpose(x_hwt, (2, 0, 1))  # (T,H,W)
            return np.asarray(x_thw, dtype=np.float32)

        # linear and conv1d use canonical (C,T) from the generic helper.
        x_ct = self._to_channel_time(x=x)
        c_exp, t_exp = int(self.input_shape[0]), int(self.input_shape[1])

        if x_ct.shape != (c_exp, t_exp):
            raise ValueError(
                f"VAECodec input shape mismatch for model {self.meta['model_dir'].name!r}: "
                f"expected (C,T)=({c_exp},{t_exp}), got (C,T)={tuple(x_ct.shape)} "
                f"(input shape {tuple(x.shape)})."
            )

        if self.model_type == "conv1d":
            return np.asarray(x_ct, dtype=np.float32)

        if self.model_type == "linear":
            if self.input_mode == "time":
                x_linear = x_ct  # first linear sees T
            elif self.input_mode == "channels":
                x_linear = x_ct.T  # first linear sees C
            else:
                raise RuntimeError(
                    f"Unexpected input_mode={self.input_mode!r} for linear model {self.meta['model_dir'].name!r}."
                )
            return np.asarray(x_linear, dtype=np.float32)

        raise RuntimeError(
            f"Unsupported model_type={self.model_type!r} in _prepare_encode_array for "
            f"model {self.meta['model_dir'].name!r}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _decode_tensor_to_channel_time(self, x_hat: torch.Tensor) -> np.ndarray:
        """
        Convert model decoder output to canonical (C, T) form.

        Parameters
        ----------
        x_hat : torch.Tensor
            Raw decoder output tensor.

        Returns
        -------
        np.ndarray
            Decoded sample in canonical (C, T) form.

        Raises
        ------
        TypeError
            If decoder output is not a torch.Tensor.
        RuntimeError
            If decoder output shape is incompatible with model_type/input_shape/input_mode.

        """

        if not isinstance(x_hat, torch.Tensor):
            raise TypeError(
                f"VAE decode returned {type(x_hat).__name__}, expected torch.Tensor."
            )

        x_np = x_hat.detach().cpu().numpy().astype(np.float32, copy=False)

        if self.model_type == "conv1d":
            if x_np.ndim == 3:
                if x_np.shape[0] != 1:
                    raise RuntimeError(
                        f"VAE conv1d decode returned batch size {x_np.shape[0]}, expected 1."
                    )
                x_ct = x_np[0]
            elif x_np.ndim == 2:
                x_ct = x_np
            else:
                raise RuntimeError(
                    f"VAE conv1d decode returned shape {tuple(x_np.shape)}; expected (B,C,T) or (C,T)."
                )
            return np.asarray(x_ct, dtype=np.float32)

        if self.model_type == "linear":
            if x_np.ndim == 3:
                if x_np.shape[0] != 1:
                    raise RuntimeError(
                        f"VAE linear decode returned batch size {x_np.shape[0]}, expected 1."
                    )
                x_model = x_np[0]
            elif x_np.ndim == 2:
                x_model = x_np
            elif x_np.ndim == 1:
                x_model = x_np.reshape(1, -1)
            else:
                raise RuntimeError(
                    f"VAE linear decode returned shape {tuple(x_np.shape)}; expected 1D, (B,F), or (B,*,F)."
                )

            c_exp, t_exp = int(self.input_shape[0]), int(self.input_shape[1])
            expected_linear_shape = (
                (c_exp, t_exp) if self.input_mode == "time" else (t_exp, c_exp)
            )

            if tuple(x_model.shape) != tuple(expected_linear_shape):
                raise RuntimeError(
                    f"VAE linear decode produced shape {tuple(x_model.shape)} but expected "
                    f"{tuple(expected_linear_shape)} for input_mode={self.input_mode!r}."
                )

            x_ct = x_model if self.input_mode == "time" else x_model.T
            return np.asarray(x_ct, dtype=np.float32)

        if self.model_type == "conv2d":
            if x_np.ndim == 4:
                if x_np.shape[0] != 1:
                    raise RuntimeError(
                        f"VAE conv2d decode returned batch size {x_np.shape[0]}, expected 1."
                    )
                x_thw = x_np[0]
            elif x_np.ndim == 3:
                x_thw = x_np
            else:
                raise RuntimeError(
                    f"VAE conv2d decode returned shape {tuple(x_np.shape)}; expected (B,T,H,W) or (T,H,W)."
                )

            h_exp, w_exp, t_exp = (int(v) for v in self.input_shape)
            expected_thw = (t_exp, h_exp, w_exp)
            if tuple(x_thw.shape) != expected_thw:
                raise RuntimeError(
                    f"VAE conv2d decode produced shape {tuple(x_thw.shape)} but expected {expected_thw}."
                )

            x_hwt = np.transpose(x_thw, (1, 2, 0))  # (H,W,T)
            x_ct = x_hwt.reshape(h_exp * w_exp, t_exp)
            return np.asarray(x_ct, dtype=np.float32)

        raise RuntimeError(
            f"Unsupported model_type={self.model_type!r} in _decode_tensor_to_channel_time for "
            f"model {self.meta['model_dir'].name!r}."
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Public encode method for the VAECodec class.

        Parameters
        ----------
        x : np.ndarray
            Input array to be encoded.

        Returns
        -------
        np.ndarray
            Encoded input array.

        Raises
        ------
        ValueError
            If VAECodec channel mismatch between `self.expected_channels` and input shape `x.shape`.
            If VAECodec length mismatch between `self.expected_time_steps` and `x.shape`.
        RuntimeError
            If VAECodec produced a latent shape different than `self.latent_dim`.
            If VAECodec produced more than one latent vector for a single sample.

        """

        x_model = self._prepare_encode_array(x=np.asarray(x))

        # IMPORTANT: cast to float32 to match model weights/bias dtype
        x_t = torch.from_numpy(np.asarray(x_model, dtype=np.float32)).unsqueeze(0)
        x_t = x_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mu, logvar = self.model.encode(x_t)
            z_t = mu if self.use_mu else self.model.reparameterize(mu, logvar)

        # beta_VAE implementations may return extra singleton leading dimensions.
        # Accept only a single latent vector of size latent_dim.
        z = np.asarray(z_t.detach().cpu().numpy(), dtype=np.float32)
        if z.shape[-1] != self.latent_dim:
            raise RuntimeError(
                f"VAECodec produced latent shape {tuple(z_t.shape)}; "
                f"expected latent_dim={self.latent_dim}."
            )
        leading_dims = tuple(int(d) for d in z.shape[:-1])
        if any(d != 1 for d in leading_dims):
            raise RuntimeError(
                f"VAECodec produced multiple latent vectors with shape {tuple(z_t.shape)}; "
                "expected exactly one latent vector per sample."
            )
        z = z.reshape(self.latent_dim)

        return z

    # ------------------------------------------------------------------------------------------------------------------
    def decode(self, z: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        """
        Public decode method for the VAECodec class.

        Parameters
        ----------
        z : np.ndarray
            Input array to be decoded.
        original_shape : tuple[int, ...]
            Original shape of the input array `z` to be decoded.

        Returns
        -------
        np.ndarray
            Decoded input array.

        Raises
        ------
        ValueError
            If `z.shape` is different than `(self.latent_dim,)`.
        TypeError
            If decoded array not of type torch.Tensor.
        RuntimeError
            If the shape of the resulting decoded array does not match
            `(self.expected_channels, self.expected_time_steps)`.

        """

        if not isinstance(z, np.ndarray):
            z = np.asarray(z)  # noqa (omit unreachable code warning)

        if (z.ndim != 1) or (z.shape[0] != self.latent_dim):
            raise ValueError(
                f"VAECodec.decode expects `z.shape` ({self.latent_dim},), got {tuple(z.shape)}."
            )

        # IMPORTANT: latent must be float32 too
        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).unsqueeze(0)
        z_t = z_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            try:
                x_hat = self.model.decode(z_t)
            except Exception:  # noqa (omit broad exception warning)
                # Some VAE_fairmast variants expect an extra singleton dim: (B, 1, D)
                x_hat = self.model.decode(z_t.unsqueeze(1))

        x_ct_np = self._decode_tensor_to_channel_time(x_hat=x_hat)

        if x_ct_np.shape != (self.expected_channels, self.expected_time_steps):
            raise RuntimeError(
                "VAE decode produced shape "
                f"{tuple(x_ct_np.shape)} but expected "
                f"({self.expected_channels}, {self.expected_time_steps})."
            )

        return self._reshape_back(
            x_ct=x_ct_np, original_shape=tuple(original_shape)
        ).astype(np.float32, copy=False)
