"""
Shared constants for the MMT token pipeline.

This module defines:
- integer codes for token roles (context/input, actuator, output),
- explicit PAD semantics for token-level fields (ID, role, modality, position).

These values are used consistently across transforms, collation, and model code to ensure padding and masking are
unambiguous and never collide with real signal IDs or semantic roles.
"""

# Floating-point tolerance for time-based window/chunk comparisons (seconds)
TIME_FLOAT_TOL = 1e-12

# Small epsilon added to denominators / used as a floor to avoid exact zero (numerical stability)
FLOAT_STABILITY_EPS = 1e-8

# Token roles
ROLE_CONTEXT = 0
ROLE_ACTUATOR = 1
ROLE_OUTPUT = 2

# Explicit PAD semantics for token-level fields
PAD_ID = -1  # never a real signal_id
PAD_ROLE = -1  # no semantic role
PAD_MOD = -1  # no modality
PAD_POS = 0  # arbitrary, unused by model for PAD
