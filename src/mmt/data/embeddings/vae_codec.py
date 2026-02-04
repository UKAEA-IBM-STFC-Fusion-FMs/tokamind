"""
mmt.data.embeddings.vae_codec

VAE-backed codec for the MMT embedding pipeline using VAE_fairmast.

Strict expectations (current VAE_fairmast refactor)
---------------------------------------------------
- VAE_fairmast is installed and exposes the python package `vae_pipeline`.
- Trained VAEs live under:
    <VAE_fairmast>/src/vae_pipeline/data/trained_VAEs/<MODEL_DIR>/
  containing:
    - exactly one config_*.json
    - exactly one best_*.pt

MMT YAML usage (per-signal)
---------------------------
    encoder_name: vae
    encoder_kwargs:
      model_dir: conv1d_vae_config_coil_current   # folder name under trained_VAEs/
      device: cuda:0                              # optional, empty/None -> cpu
      use_mu: true                                # optional, default true

Notes
-----
- This codec casts inputs to float32 to match model weights (avoids
  "Input type (double) and bias type (float)" errors).
- This codec is intentionally strict about the directory layout and files.
  If VAE_fairmast changes, update the imports inside `_import_vae_pipeline()`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json
import numpy as np
import torch


# --------------------------------------------------------------------------------------
# Lazy imports in a "benchmark_imports.py style" (but wrapped in a function so that
# importing MMT does not require VAE_fairmast unless encoder_name='vae' is used).
# --------------------------------------------------------------------------------------

def _import_vae_pipeline():
    try:
        # Import the module object too (for locating the package on disk)
        import vae_pipeline.configs.config_setup as config_setup_mod
        from vae_pipeline.configs.config_setup import get_settings
        from vae_pipeline.models.vae_model import beta_VAE
    except Exception as e:
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


# --------------------------------------------------------------------------------------
# Config + path helpers
# --------------------------------------------------------------------------------------

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_vae_model_dir(model_dir: str | Path) -> Path:
    """
    Resolve model_dir to an on-disk directory.

    Accepted:
      1) A filesystem path to a directory (absolute or relative) that exists.
      2) A folder name under: <vae_pipeline package>/data/trained_VAEs/<model_dir>

    This is strict: no other search paths.
    """
    p = Path(model_dir).expanduser()
    if p.is_dir():
        return p.resolve()

    # Anchor to a real module file (works even if `vae_pipeline` is a namespace package)
    config_setup_mod, _, _ = _import_vae_pipeline()
    cs_file = getattr(config_setup_mod, "__file__", None)
    if not cs_file:
        raise RuntimeError(
            "Could not locate vae_pipeline.configs.config_setup on disk (missing __file__). "
            "Your installation looks like a namespace package without source files. "
            "Install VAE_fairmast in editable mode."
        )

    # .../vae_pipeline/configs/config_setup.py -> .../vae_pipeline
    vae_pkg_dir = Path(cs_file).resolve().parent.parent
    trained_root = (vae_pkg_dir / "data" / "trained_VAEs").resolve()
    cand = trained_root / p

    if cand.is_dir():
        return cand

    raise FileNotFoundError(
        "Could not resolve VAE model directory.\n"
        f"Provided model_dir: {str(model_dir)!r}\n"
        "Tried:\n"
        f"  - as a filesystem directory: {p.resolve()}\n"
        f"  - as a trained VAE folder:   {cand}\n\n"
        "Expected trained VAEs under: <VAE_fairmast>/src/vae_pipeline/data/trained_VAEs/<MODEL_DIR>."
    )


def read_vae_model_meta(model_dir: str | Path) -> Dict[str, Any]:
    """
    Read minimal metadata needed by MMT from a trained VAE folder.

    STRICT (no backwards compatibility):
    - Each model folder MUST contain:
        - mmt_info.json
        - exactly one config_*.json (used to build VAE settings)
    - mmt_info.json MUST contain:
        - latent_dim (int)
        - in_channels (int)
        - seq_len (int)
        - checkpoint (str)  # filename or glob pattern relative to the model folder

    Returns a dict:
      - model_dir (Path)
      - config_path (Path)
      - checkpoint_path (Path)
      - latent_dim (int)
      - in_channels (int)
      - seq_len (int)
    """
    md = resolve_vae_model_dir(model_dir)

    info_path = md / "mmt_info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Missing required mmt_info.json in VAE model directory: {md}\n"
            "Create it with keys: latent_dim, in_channels, seq_len, checkpoint."
        )
    info = _read_json(info_path)

    for k in ("latent_dim", "in_channels", "seq_len", "checkpoint"):
        if k not in info:
            raise KeyError(f"mmt_info.json missing required key {k!r}: {info_path}")

    latent_dim = int(info["latent_dim"])
    in_channels = int(info["in_channels"])
    seq_len = int(info["seq_len"])
    checkpoint_spec = str(info["checkpoint"])

    cfg_files = sorted(md.glob("config_*.json"))
    if len(cfg_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly 1 config_*.json in {md}, found {len(cfg_files)}: "
            + ", ".join([p.name for p in cfg_files])
        )
    config_path = cfg_files[0]

    # checkpoint_spec can be an exact filename or a glob (e.g. "best_*.pt")
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
        "latent_dim": latent_dim,
        "in_channels": in_channels,
        "seq_len": seq_len,
    }
# --------------------------------------------------------------------------------------
# Codec implementation
# --------------------------------------------------------------------------------------

class VAECodec:
    """
    Wrap a pretrained VAE_fairmast beta_VAE model with a codec interface.

    encode(x) -> (latent_dim,) float32
    decode(z, original_shape) -> float32 array of original_shape
    """

    def __init__(
        self,
        *,
        model_dir: str,
        device: Optional[str] = None,
        use_mu: bool = True,
    ) -> None:
        self.meta = read_vae_model_meta(model_dir)
        self.latent_dim = int(self.meta["latent_dim"])
        self.in_channels = int(self.meta["in_channels"])
        self.seq_len = int(self.meta["seq_len"])

        dev = device if (device is not None and str(device).strip() != "") else "cpu"
        self.device = torch.device(str(dev))
        self.use_mu = bool(use_mu)

        # Lazy import VAE_fairmast symbols
        _, get_settings, beta_VAE = _import_vae_pipeline()

        settings = get_settings(str(self.meta["config_path"]))
        self.model = beta_VAE(settings)

        ckpt = torch.load(self.meta["checkpoint_path"], map_location=self.device)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise RuntimeError(
                f"Unsupported checkpoint format at {self.meta['checkpoint_path']}. "
                "Expected a dict containing key 'model_state_dict'."
            )

        sd = ckpt["model_state_dict"]
        # handle DataParallel prefix
        if any(k.startswith("module.") for k in sd.keys()):
            sd = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in sd.items()}

        self.model.load_state_dict(sd, strict=True)
        self.model.to(self.device)
        self.model.eval()

    # -------------------------
    # shape helpers
    # -------------------------

    def _to_ct(self, x: np.ndarray) -> Tuple[np.ndarray, Tuple[int, ...]]:
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        orig_shape = tuple(x.shape)

        if x.ndim == 1:
            x_ct = x.reshape(1, x.shape[0])          # (1, T)
        elif x.ndim == 2:
            x_ct = x                                # (C, T)
        elif x.ndim == 3:
            h, w, t = x.shape
            x_ct = x.reshape(h * w, t)               # (H*W, T)
        else:
            raise ValueError(f"VAECodec supports x.ndim in {{1,2,3}}, got shape={x.shape}")

        return x_ct, orig_shape

    def _reshape_back(self, x_ct: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        if len(original_shape) == 1:
            if x_ct.shape[0] != 1:
                raise ValueError(
                    f"Cannot reshape reconstruction {x_ct.shape} to original_shape={original_shape}: expected 1 channel."
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
        raise ValueError(f"Unsupported original_shape={original_shape}")

    # -------------------------
    # public API
    # -------------------------

    def encode(self, x: np.ndarray) -> np.ndarray:
        x_ct, _orig_shape = self._to_ct(np.asarray(x))
        c, t = x_ct.shape

        if c != self.in_channels:
            raise ValueError(
                f"VAECodec channel mismatch for model {self.meta['model_dir'].name!r}: "
                f"expected C={self.in_channels}, got C={c} (input shape {tuple(np.asarray(x).shape)})."
            )
        if t != self.seq_len:
            raise ValueError(
                f"VAECodec length mismatch for model {self.meta['model_dir'].name!r}: "
                f"expected T={self.seq_len}, got T={t} (input shape {tuple(np.asarray(x).shape)})."
            )

        # IMPORTANT: cast to float32 to match model weights/bias dtype
        x_t = torch.from_numpy(np.asarray(x_ct, dtype=np.float32)).unsqueeze(0)  # (1,C,T)
        x_t = x_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            mu, logvar = self.model.encode(x_t)
            z_t = mu if self.use_mu else self.model.reparameterize(mu, logvar)

        z = z_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise RuntimeError(f"VAECodec produced latent shape {z.shape}; expected ({self.latent_dim},).")
        return z

    def decode(self, z: np.ndarray, original_shape: Tuple[int, ...]) -> np.ndarray:
        if not isinstance(z, np.ndarray):
            z = np.asarray(z)

        if z.ndim != 1 or z.shape[0] != self.latent_dim:
            raise ValueError(f"VAECodec.decode expects z shape ({self.latent_dim},), got {tuple(z.shape)}")

        # IMPORTANT: latent must be float32 too
        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).unsqueeze(0)
        z_t = z_t.to(device=self.device, dtype=torch.float32)

        with torch.no_grad():
            x_hat = self.model.decode(z_t)

        if not isinstance(x_hat, torch.Tensor):
            raise RuntimeError(f"VAE decode returned {type(x_hat).__name__}, expected torch.Tensor")

        if x_hat.ndim == 3:
            x_ct = x_hat.squeeze(0)  # (C, T)
        elif x_hat.ndim == 2:
            x_ct = x_hat
        else:
            raise RuntimeError(f"VAE decode returned tensor with shape {tuple(x_hat.shape)}; expected (B,C,T) or (C,T).")

        x_ct_np = x_ct.detach().cpu().numpy().astype(np.float32, copy=False)

        if x_ct_np.shape != (self.in_channels, self.seq_len):
            raise RuntimeError(
                f"VAE decode produced shape {tuple(x_ct_np.shape)} but expected ({self.in_channels}, {self.seq_len})."
            )

        return self._reshape_back(x_ct_np, tuple(original_shape)).astype(np.float32, copy=False)
