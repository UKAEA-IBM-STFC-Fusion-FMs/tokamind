"""
General utilities for MMT.

This package provides common utilities for logging, seeding, AMP operations,
and configuration validation.

Key modules
-----------
- seed.py      : random seed management for reproducibility
- logger.py    : logging setup and configuration
- amp_utils.py : automatic mixed precision utilities
- config/      : configuration validation and schema utilities
"""

from .seed import set_seed
from .logger import setup_logging
from .amp_utils import sdpa_math_only_ctx

__all__ = [
    "set_seed",
    "setup_logging",
    "sdpa_math_only_ctx",
]
