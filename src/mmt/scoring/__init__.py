"""
Scoring rules for MMT.

CRPS and related scoring rules are used both as *training losses* and as *evaluation metrics* — they are not
loss-only or metric-only, so they live in a neutral ``scoring`` module that both ``mmt.train.losses`` and
``mmt.eval`` import from (rather than eval reaching into training losses, or the formula being duplicated).
"""

from .crps import gaussian_crps, point_crps, sample_crps


__all__ = ["gaussian_crps", "sample_crps", "point_crps"]
