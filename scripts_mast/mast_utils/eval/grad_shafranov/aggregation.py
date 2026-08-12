"""
Small finite-value aggregation mechanics shared by Grad-Shafranov diagnostics.

These helpers intentionally share *mechanics*, not metric schemas.
Residual, smoothness, and topology diagnostics expose different quantities,
coverage rules, and CSV layouts, but all need the same treatment of finite
values: accumulate a per-shot mean, aggregate equal-weight shot means into a
task mean and population standard deviation, and render unavailable values as
``nan`` in companion CSVs.

No metric base class lives here. Each metric keeps its own ``add_batch``,
``is_empty``, and ``write_csvs`` implementation, while ``RunningMean``,
``task_stats``, and ``format_value`` prevent the numerical bookkeeping from
drifting between them. ``warn_once`` similarly centralizes rate-limited logging
for optional reference data without coupling the metrics' validation policies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class RunningMean:
    """Finite-value sum and count with a NaN mean when no values were observed."""

    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        """Accumulate one finite value and ignore non-finite values."""

        if np.isfinite(value):
            self.total += float(value)
            self.count += 1

    @property
    def mean(self) -> float:
        """Return the finite-value mean, or ``NaN`` when no value was observed."""

        return self.total / self.count if self.count > 0 else float("nan")


def format_value(value: float) -> str:
    """Format a finite metric value for CSV output, preserving the established ``nan`` representation."""

    return f"{value:.8e}" if np.isfinite(value) else "nan"


def task_stats(values: Iterable[float]) -> tuple[float, float, int]:
    """Return equal-weight mean, population standard deviation, and count for finite per-shot values."""

    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return float("nan"), float("nan"), 0
    return float(array.mean()), float(array.std()), int(array.size)


def warn_once(seen: set[str], key: str, logger: logging.Logger, message: str, *args: object) -> None:
    """Emit one warning per ``key`` for a metric instance."""

    if key not in seen:
        logger.warning(message, *args)
        seen.add(key)
