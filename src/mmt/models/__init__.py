"""
Model architecture for MMT.

This package provides the multi-modal transformer architecture and modality-specific components.

Key modules
-----------
- mmt.py              : main MultiModalTransformer model
- backbone.py         : transformer backbone implementation
- token_encoder.py    : token encoding and positional embeddings
- modality_heads.py   : modality-specific input/output heads
- output_adapters.py  : output adaptation layers
"""

from .mmt import MultiModalTransformer


# ----------------------------------------------------------------------------------------------------------------------

__all__ = ["MultiModalTransformer"]
