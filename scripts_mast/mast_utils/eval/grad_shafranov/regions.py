"""
Named evaluation-domain conventions shared by Grad-Shafranov diagnostics.

The GS equation/operator and smoothness metrics both report results over the same two
domains:

- ``limiter``: the fixed, eroded vessel-interior baseline;
- ``plasma_mask``: the time-dependent ground-truth current region inside that
  limiter baseline.

This module owns only their stable keys and output suffixes. It does *not*
construct masks, define a metric's base column names, or write CSV schemas:
``GSEvalGrid`` constructs the physical masks and each metric remains responsible
for its own quantities and historical CSV names. Keeping that boundary explicit
lets a new domain be added consistently without forcing otherwise different
metrics into a shared output format.

``PlasmaMaskCoverage`` is deliberately narrower than a generic region
accumulator. It records the provenance of the optional ground-truth ``j_tor``
reference used for the plasma mask: how many scored fields had that reference
and how many produced an empty mask.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EvalRegion:
    """Stable key and output suffix for one Grad-Shafranov evaluation domain."""

    key: str
    suffix: str

    def column(self, base: str) -> str:
        """Return the metric-local column name formed from ``base`` and this region's suffix."""

        return f"{base}{self.suffix}"


LIMITER = EvalRegion(key="limiter", suffix="")
PLASMA_MASK = EvalRegion(key="plasma_mask", suffix="_plasma_mask")
EVAL_REGIONS = (LIMITER, PLASMA_MASK)


@dataclass
class PlasmaMaskCoverage:
    """Track scored fields with a plasma reference and fields whose resulting plasma mask is empty."""

    _by_shot: dict[int, list[int]] = field(default_factory=dict)

    def observe(self, shot_id: int, plasma_mask: np.ndarray, field_index: int) -> None:
        """Record one scored field against its validated plasma mask."""

        counts = self._by_shot.setdefault(int(shot_id), [0, 0])
        counts[0] += 1
        counts[1] += int(not bool(plasma_mask[field_index].any()))

    def per_shot(self, shot_id: int) -> tuple[int, int]:
        """Return ``(with_reference, empty_mask)`` counts for one shot."""

        counts = self._by_shot.get(int(shot_id), [0, 0])
        return counts[0], counts[1]

    def totals(self) -> tuple[int, int]:
        """Return total ``(with_reference, empty_mask)`` counts across scored shots."""

        return (
            sum(counts[0] for counts in self._by_shot.values()),
            sum(counts[1] for counts in self._by_shot.values()),
        )
