"""
Lightweight transform composer for the MMT data pipeline.

ComposeTransforms chains a sequence of shot/window transforms into a single callable, stopping early if any transform
returns None (drop semantics).
"""

from __future__ import annotations

from typing import Any, Iterable
from collections.abc import Callable


# ======================================================================================================================
class ComposeTransforms:
    """
    Compose a sequence of MMT data transforms into a single callable.

    Each transform is applied in order:

        x_out = t_n(... t_2(t_1(x)) ...)

    If any transform returns `None`, the entire pipeline stops early and `None` is returned. This matches the semantics
    of the MMT pipeline, where a shot or window may be discarded at intermediate steps (e.g., invalid windows after
    filtering).

    Typical usage:

        model_specific_transform = ComposeTransforms([
            ChunkWindowsTransform(...),
            SelectValidWindowsTransform(...),
            TrimChunksTransform(...),
            EmbedChunksTransform(...),
            BuildTokensTransform(...)
        ])

    Attributes
    ----------
    self.transforms: list[Callable[[Any], Any]]
        List of transforms to be iteratively applied on a given object.

    Methods
    -------
    __call__(x)
        Call method for the class instances to behave like a function.

    Notes
    -----
    - Transforms may operate on shots or windows; Compose makes no assumptions about intermediate types.
    - The class is intentionally lightweight, similar to `torchvision.transforms.Compose`, but tailored for MMT.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, transforms: Iterable[Callable[[Any], Any]]) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        transforms : Iterable[Callable[[Any], Any]]
            Input sequence of transforms.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes, as this is a callable class.

        """

        self.transforms: list[Callable[[Any], Any]] = list(transforms)

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, x: Any) -> Any:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        x : Any
            Object on which the composed transform is applied.

        Returns
        -------
        Any
            Composed transform applied to `x`.

        """

        for t in self.transforms:
            x = t(x)
            if x is None:
                return None

        return x
