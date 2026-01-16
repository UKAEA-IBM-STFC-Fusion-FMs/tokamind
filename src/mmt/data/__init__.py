"""
Public API for mmt.data.

Re-exports key classes and functions so users can import them directly
from `mmt.data` without knowing the internal submodule layout.
"""

# -----------------------------
# Datasets
# -----------------------------
from .datasets.window_cached_dataset import WindowCachedDataset
from .datasets.window_streamed_dataset import WindowStreamedDataset

# -----------------------------
# Transforms
# -----------------------------
from .transforms.chunk_windows import ChunkWindowsTransform
from .transforms.trim_chunks import TrimChunksTransform
from .transforms.select_valid_windows import SelectValidWindowsTransform
from .transforms.embed_chunks import EmbedChunksTransform
from .transforms.build_tokens import BuildTokensTransform
from .transforms.finalize_window import FinalizeWindowTransform
from .transforms.tune_dct3d import TuneDCT3DTransform
from .transforms.compose import ComposeTransforms


# -----------------------------
# Core utilities
# -----------------------------
from .collate import MMTCollate
from .data_loaders import initialize_mmt_dataloaders
from .signal_spec import (
    SignalSpec,
    build_signal_specs,
    infer_modality,
)

# -----------------------------
# Codec utils
# -----------------------------
from .embeddings.codec_utils import build_codecs

__all__ = [
    "WindowCachedDataset",
    "WindowStreamedDataset",
    "ChunkWindowsTransform",
    "TrimChunksTransform",
    "SelectValidWindowsTransform",
    "EmbedChunksTransform",
    "BuildTokensTransform",
    "FinalizeWindowTransform",
    "TuneDCT3DTransform",
    "ComposeTransforms",
    "MMTCollate",
    "initialize_mmt_dataloaders",
    "SignalSpec",
    "build_signal_specs",
    "infer_modality",
    "build_codecs",
]
