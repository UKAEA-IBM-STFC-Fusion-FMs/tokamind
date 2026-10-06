"""
Schema definition for experiment configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ======================================================================================================================
@dataclass
class ExperimentConfig:
    """
    Dynamic experiment configuration object.

    - The full merged dictionary is stored in `cfg.raw`.
    - Every top-level key in `raw` is available as an attribute: cfg.model, cfg.train, cfg.paths, ...

    """

    raw: dict[str, Any]

    # ------------------------------------------------------------------------------------------------------------------
    def __getattr__(self, key: str) -> Any:
        if key in self.raw:
            return self.raw[key]
        raise AttributeError(f"'ExperimentConfig' has no attribute '{key}'.")

    # ------------------------------------------------------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Return underlying raw dict value by key (None for unexisting keys)."""
        return self.raw.get(key, default)

    # ------------------------------------------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the underlying raw dict (by reference)."""
        return self.raw

    # ------------------------------------------------------------------------------------------------------------------
