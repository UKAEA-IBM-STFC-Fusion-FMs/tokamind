"""mmt.data.embeddings.vae_codec

Simplified VAE-backed codec for the MMT embedding pipeline.

Design goals
------------
- **Require** the VAE_fairmast codebase to be installed/importable.
- Use the reference implementation provided by VAE_fairmast:
    - SETTINGS are built with: `get_settings(config_path)`
    - Model class: `beta_VAE(SETTINGS)`
    - Weights: checkpoint dict with key `model_state_dict`
- Keep the integration **strict and predictable**:
    - If expected files/keys are missing -> raise a clear error.
    - Only the model-directory resolver supports multiple options (absolute path or
      derived from the installed VAE_fairmast package layout).

If VAE_fairmast changes its package/module layout, update ONLY the constants:
  - VAE_CONFIG_SETUP_MODULE
  - VAE_MODEL_MODULE
  - VAE_MODEL_CLASS

Expected VAE_fairmast layout (installed editable or with package data present)
---------------------------------------------------------------------------
  VAE_fairmast/
    src/pipelines/
      configs/config_setup.py
      models/vae_model.py
      data/<model_dir>/
        config_*.json
        best_*.pt

Configuration schema support (strict, but matches your examples)
--------------------------------------------------------------
The per-model config_*.json must contain at least:

  - beta-vae.latent_dim

  - time_settings.targeted_time_stamps_per_window
    (also supports the older typo: tergeted_time_stamp_per_window)

  - Either of the following encoder schemas:

    A) encoder_specs.conv1d_in_channels          (older)
    B) encoder.layers[0].params.in_channels      (newer)

Usage from MMT (YAML)
---------------------
  equilibrium-beta_normal:
    encoder_name: vae
    encoder_kwargs:
      model_dir: conv1d_vae_config_beta_normal
      device: cpu   # optional

Notes
-----
- Encoding uses deterministic `mu` from `beta_VAE.encode()`.
- Input arrays are assumed to be shaped (*values_shape, T). We flatten spatial dims
  into channels -> (C, T). Supported input ndim: 1, 2, 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json

import numpy as np
import torch


# -------------------------------------------------------------------------
# If VAE_fairmast moves modules, update only these strings.
# -------------------------------------------------------------------------
VAE_CONFIG_SETUP_MODULE = "pipelines.configs.config_setup"
VAE_MODEL_MODULE = "pipelines.models.vae_model"
VAE_MODEL_CLASS = "beta_VAE"


@dataclass(frozen=True)
class VAEModelMeta:
    model_dir: Path
    config_path: Path
    checkpoint_path: Path
    latent_dim: int
    in_channels: int
    seq_len: int


def _import_vae_fairmast_symbol(module_path: str, symbol: str) -> Any:
    """Import `symbol` from `module_path`, raising a clear 'install package' error."""
    try:
        mod = __import__(module_path, fromlist=[symbol])
        return getattr(mod, symbol)
    except Exception as e:
        raise ImportError(
            "VAE codec requires the VAE_fairmast package to be installed and importable.\n"
            "Install it (editable recommended) and ensure its repo includes the 'src/pipelines' tree.\n"
            f"Failed importing: {module_path}:{symbol}\n"
            f"Original error: {e}"
        ) from e


def resolve_vae_model_dir(model_dir: str | Path) -> Path:
    """Resolve a VAE model directory.

    - If `model_dir` is an existing absolute/relative directory path, use it.
    - Otherwise treat it as a folder name under the installed VAE_fairmast
      'src/pipelines/data' directory (derived from config_setup.py location).
    """
    p = Path(model_dir).expanduser()

    # If user gave a usable directory path (absolute OR relative), accept it.
    if p.is_dir():
        return p.resolve()

    # Otherwise, interpret it as a folder name under VAE_fairmast/src/pipelines/data
    get_settings = _import_vae_fairmast_symbol(VAE_CONFIG_SETUP_MODULE, "get_settings")
    # The file path of the module is our anchor to locate src/pipelines/
    mod = __import__(VAE_CONFIG_SETUP_MODULE, fromlist=["__file__"])
    mod_file = Path(mod.__file__).resolve()

    # .../src/pipelines/configs/config_setup.py -> .../src/pipelines
    pipelines_root = mod_file.parent.parent  # configs -> pipelines
    data_root = pipelines_root / "data"
    cand = data_root / p

    if cand.is_dir():
        return cand.resolve()

    raise FileNotFoundError(
        "Could not resolve VAE model directory.\n"
        f"Requested model_dir={str(model_dir)!r}.\n"
        "Provide either:\n"
        "  - a valid directory path to the model folder, OR\n"
        "  - a folder name located under VAE_fairmast/src/pipelines/data/\n"
        f"Tried: {cand}"
    )


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _require_one(glob_results: list[Path], what: str, where: Path) -> Path:
    if len(glob_results) == 0:
        raise FileNotFoundError(f"Missing {what} in {where}")
    if len(glob_results) > 1:
        names = ", ".join(p.name for p in glob_results)
        raise FileExistsError(f"Expected exactly one {what} in {where}, found: {names}")
    return glob_results[0]


def read_vae_model_meta(model_dir: str | Path) -> VAEModelMeta:
    """Read VAE model metadata from a VAE_fairmast model directory."""
    md = resolve_vae_model_dir(model_dir)

    config_path = _require_one(sorted(md.glob("config_*.json")), "config_*.json", md)
    cfg = _read_json(config_path)

    # latent dim (required)
    try:
        latent_dim = int(cfg["beta-vae"]["latent_dim"])
    except Exception as e:
        raise KeyError(f"Missing required beta-vae.latent_dim in {config_path}") from e

    # seq len (required; support both spellings)
    try:
        ts = cfg["time_settings"].get("targeted_time_stamps_per_window")
        if ts is None:
            ts = cfg["time_settings"].get("tergeted_time_stamp_per_window")
        if ts is None:
            raise KeyError("targeted_time_stamps_per_window")
        seq_len = int(ts)
    except Exception as e:
        raise KeyError(
            "Missing required time_settings.(targeted_time_stamps_per_window|tergeted_time_stamp_per_window) "
            f"in {config_path}"
        ) from e

    # in_channels (required; support both schemas)
    in_channels: Optional[int] = None

    if "encoder_specs" in cfg:
        # older schema
        try:
            in_channels = int(cfg["encoder_specs"]["conv1d_in_channels"])
        except Exception as e:
            raise KeyError(
                f"Missing encoder_specs.conv1d_in_channels in {config_path}"
            ) from e
    elif "encoder" in cfg:
        # newer schema (your coil_current example)
        try:
            layers = cfg["encoder"]["layers"]
            first = layers[0]
            in_channels = int(first["params"]["in_channels"])
        except Exception as e:
            raise KeyError(
                f"Missing encoder.layers[0].params.in_channels in {config_path}"
            ) from e
    else:
        raise KeyError(
            "Missing expected encoder schema in VAE config. Expected either 'encoder_specs' "
            "or 'encoder'.\n"
            f"Config: {config_path}"
        )

    # checkpoint (strict: one best_*.pt)
    ckpt_path = _require_one(sorted(md.glob("best_*.pt")), "best_*.pt", md)

    return VAEModelMeta(
        model_dir=md,
        config_path=config_path,
        checkpoint_path=ckpt_path,
        latent_dim=latent_dim,
        in_channels=in_channels,
        seq_len=seq_len,
    )


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


class VAECodec:
    """Codec wrapper around a pretrained VAE_fairmast beta_VAE model."""

    def __init__(self, *, model_dir: str, device: str = "cpu") -> None:
        self.meta = read_vae_model_meta(model_dir)
        self.latent_dim = int(self.meta.latent_dim)
        self.in_channels = int(self.meta.in_channels)
        self.seq_len = int(self.meta.seq_len)

        # Lazy import VAE_fairmast model + settings builder (only when VAE is used)
        get_settings = _import_vae_fairmast_symbol(
            VAE_CONFIG_SETUP_MODULE, "get_settings"
        )
        beta_VAE = _import_vae_fairmast_symbol(VAE_MODEL_MODULE, VAE_MODEL_CLASS)

        self.device = torch.device(str(device))
        settings = get_settings(str(self.meta.config_path))

        model = beta_VAE(settings)

        ckpt = torch.load(self.meta.checkpoint_path, map_location=self.device)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise RuntimeError(
                "Unsupported checkpoint format. Expected a dict with key 'model_state_dict'.\n"
                f"checkpoint={self.meta.checkpoint_path}"
            )

        state = _strip_module_prefix(ckpt["model_state_dict"])
        model.load_state_dict(state, strict=True)

        model.to(self.device)
        model.eval()
        self.model = model

    # -------------------------
    # shape helpers
    # -------------------------
    def _to_ct(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        """Normalise input to (C, T)."""
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)

        orig_shape = tuple(x.shape)

        if x.ndim == 1:
            # (T,) -> (1, T)
            x_ct = x.reshape(1, x.shape[0])
        elif x.ndim == 2:
            # (C, T)
            x_ct = x
        elif x.ndim == 3:
            # (H, W, T) -> (H*W, T)
            h, w, t = x.shape
            x_ct = x.reshape(h * w, t)
        else:
            raise ValueError(
                f"VAECodec supports inputs with ndim in {{1,2,3}}, got {x.ndim} shape={x.shape}"
            )

        return x_ct, orig_shape

    def _reshape_back(
        self, x_ct: np.ndarray, original_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Reshape (C,T) reconstruction back to original_shape."""
        if len(original_shape) == 1:
            if x_ct.shape[0] != 1:
                raise ValueError(
                    f"Cannot reshape recon {x_ct.shape} to original_shape={original_shape}: expected 1 channel"
                )
            return x_ct.reshape(original_shape)

        if len(original_shape) == 2:
            return x_ct.reshape(original_shape)

        if len(original_shape) == 3:
            h, w, t = original_shape
            if x_ct.shape != (h * w, t):
                raise ValueError(
                    f"Recon shape {x_ct.shape} incompatible with original_shape={original_shape}"
                )
            return x_ct.reshape(original_shape)

        raise ValueError(f"Unsupported original_shape={original_shape}")

    # -------------------------
    # API
    # -------------------------
    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode a signal chunk into a deterministic latent vector (mu)."""
        x_ct, _orig_shape = self._to_ct(np.asarray(x))
        c, t = x_ct.shape

        if c != self.in_channels:
            raise ValueError(
                f"VAE input channel mismatch: expected C={self.in_channels}, got C={c} (shape={np.asarray(x).shape})"
            )
        if t != self.seq_len:
            raise ValueError(
                f"VAE input length mismatch: expected T={self.seq_len}, got T={t} (shape={np.asarray(x).shape})"
            )
        if not np.isfinite(x_ct).all():
            raise ValueError("VAECodec received non-finite values (nan/inf) in input.")

        x_t = (
            torch.from_numpy(x_ct.astype(np.float32, copy=False))
            .unsqueeze(0)
            .to(self.device)
        )  # (1,C,T)

        with torch.no_grad():
            mu, _logvar = self.model.encode(x_t)

        z = mu.squeeze(0).detach().cpu().float().numpy()
        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise RuntimeError(
                f"VAE produced latent shape {z.shape}; expected ({self.latent_dim},)"
            )
        return z.astype(np.float32, copy=False)

    def decode(self, z: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        """Decode a latent vector back to the original signal shape."""
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)

        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise ValueError(
                f"VAECodec.decode expects z shape ({self.latent_dim},), got {z.shape}"
            )

        z_t = (
            torch.from_numpy(z.astype(np.float32, copy=False))
            .unsqueeze(0)
            .to(self.device)
        )  # (1,D)

        with torch.no_grad():
            x_hat = self.model.decode(z_t)

        if not isinstance(x_hat, torch.Tensor):
            raise RuntimeError(
                f"VAE decode returned {type(x_hat).__name__}, expected torch.Tensor"
            )

        if x_hat.ndim != 3:
            raise RuntimeError(
                f"VAE decode returned tensor with shape {tuple(x_hat.shape)}; expected (B,C,T)"
            )

        x_ct = x_hat.squeeze(0).detach().cpu().float().numpy()  # (C,T)

        if x_ct.shape != (self.in_channels, self.seq_len):
            raise RuntimeError(
                f"VAE decode produced shape {x_ct.shape} but model expects ({self.in_channels},{self.seq_len})"
            )

        return self._reshape_back(x_ct, tuple(original_shape)).astype(
            np.float32, copy=False
        )
