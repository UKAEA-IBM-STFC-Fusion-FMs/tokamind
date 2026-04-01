"""
Data pipeline utilities for MMT.

This package provides datasets, transforms, embeddings, and data loading utilities for multi-modal time-series data.

Key modules
-----------
- datasets.py       : window-cached dataset implementation
- signal_spec.py    : signal specification and modality inference
- collate.py        : batch collation for multi-modal data
- data_loaders.py   : dataloader initialization utilities
- transforms/       : data transformation pipeline components
- embeddings/       : codec utilities for signal compression/embedding
"""

# ----------------------------------------------------------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------------------------------------------------------

from .datasets import WindowCachedDataset

# ----------------------------------------------------------------------------------------------------------------------
# Transforms
# ----------------------------------------------------------------------------------------------------------------------

from .transforms.chunk_windows import ChunkWindowsTransform
from .transforms.trim_chunks import TrimChunksTransform
from .transforms.select_valid_windows import SelectValidWindowsTransform
from .transforms.embed_chunks import EmbedChunksTransform
from .transforms.build_tokens import BuildTokensTransform
from .transforms.finalize_window import FinalizeWindowTransform
from .transforms.tune_ranked_dct3d import TuneRankedDCT3DTransform
from .transforms.compose import ComposeTransforms

# ----------------------------------------------------------------------------------------------------------------------
# Core utilities
# ----------------------------------------------------------------------------------------------------------------------

from .collate import MMTCollate
from .data_loaders import initialize_mmt_dataloader
from .signal_spec import SignalSpec, build_signal_specs, infer_modality

# ----------------------------------------------------------------------------------------------------------------------
# Codec utils
# ----------------------------------------------------------------------------------------------------------------------

from .embeddings.codec_utils import build_codecs


# ----------------------------------------------------------------------------------------------------------------------

__all__ = [
    "WindowCachedDataset",
    "ChunkWindowsTransform",
    "TrimChunksTransform",
    "SelectValidWindowsTransform",
    "EmbedChunksTransform",
    "BuildTokensTransform",
    "FinalizeWindowTransform",
    "TuneRankedDCT3DTransform",
    "ComposeTransforms",
    "MMTCollate",
    "initialize_mmt_dataloader",
    "SignalSpec",
    "build_signal_specs",
    "infer_modality",
    "build_codecs",
]
