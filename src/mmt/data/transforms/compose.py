"""
Lightweight transform composer for the MMT data pipeline.

ComposeTransforms chains a sequence of shot/window transforms into a single
callable, stopping early if any transform returns None (drop semantics).
"""

from __future__ import annotations
from typing import Any, Iterable, List, Callable


class ComposeTransforms:
    """
    Compose a sequence of MMT data transforms into a single callable.

    Each transform is applied in order:

        x_out = t_n(... t_2(t_1(x)) ...)

    If any transform returns ``None``, the entire pipeline stops early and
    ``None`` is returned. This matches the semantics of the MMT pipeline,
    where a shot or window may be discarded at intermediate steps
    (e.g., invalid windows after filtering).

    Typical usage:

        model_specific_transform = ComposeTransforms([
            ChunkWindowsTransform(...),
            SelectValidWindowsTransform(...),
            TrimChunksTransform(...),
            EmbedChunksTransform(...),
            BuildTokensTransform(...),
        ])

    Notes
    -----
    - Transforms may operate on shots or windows; Compose makes no
      assumptions about intermediate types.
    - The class is intentionally lightweight, similar to
      ``torchvision.transforms.Compose``, but tailored for MMT.
    """

    def __init__(self, transforms: Iterable[Callable[[Any], Any]]) -> None:
        self.transforms: List[Callable[[Any], Any]] = list(transforms)

    def __call__(self, x: Any) -> Any:
        for t in self.transforms:
            x = t(x)
            if x is None:
                return None
        return x
