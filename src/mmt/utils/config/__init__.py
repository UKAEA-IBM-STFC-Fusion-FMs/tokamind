"""
Configuration validation utilities for MMT.

This package provides configuration schema validation and utilities for ensuring experiment configurations are valid.

Key modules
-----------
- validator.py : configuration validation functions
- schema.py    : configuration schema definitions
"""

from .validator import validate_config


# ----------------------------------------------------------------------------------------------------------------------

__all__ = ["validate_config"]
