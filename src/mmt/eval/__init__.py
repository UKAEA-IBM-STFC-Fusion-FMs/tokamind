"""
Evaluation and inference utilities for MMT.

This package provides utilities for model evaluation and inference,
including forward passes and decoding operations.

Key modules
-----------
- forward.py : forward pass and native decoding utilities
- decode.py  : decoding and post-processing operations
"""

from .forward import forward_decode_native

__all__ = ["forward_decode_native"]
