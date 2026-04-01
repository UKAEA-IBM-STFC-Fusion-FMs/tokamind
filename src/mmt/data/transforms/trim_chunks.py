"""
TrimChunksTransform
===================

Trim input/actuator chunk histories to a maximum number of chunks and compute relative positional indices ("pos") using
index logic (no time-based fields).

Definitions (index-based)
-------------------------
For a role with N chunks indexed oldest→newest:

    chunk_index_in_window = 0,1,...,N-1

We define positional index (1 = closest to output, i.e., newest):

    pos = (N - 1 - chunk_index_in_window) + 1

Behavior
--------
- Adds integer `pos` to each chunk.
- Keeps at most `max_chunks` chunks per role ("input" and "actuator"), choosing chunks with the smallest `pos` (closest
  to output).
- Returns the window unchanged except for the trimmed/enriched chunk lists.

Expected input
--------------
window must contain:
  - "chunks": {"input": [chunk...], "actuator": [chunk...]}

each chunk must contain:
  - "chunk_index_in_window": int
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import logging


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.TrimChunks")


# ======================================================================================================================
class TrimChunksTransform:
    """
    Trim chunk histories and compute relative positions (pos) using indexes.

    Input
    -----
    window : dict
        Dictionary with keys described in module docstring.

    Output
    ------
    window: dict
        Dictionary with trimmed window["chunks"]["input"/"actuator"] and added "pos" field per chunk. Returns None if
        input window is None.

    Attributes
    ----------
    self.max_chunks : int
        Maximum number of chunks.

    Methods
    -------
    _compute_pos(n_chunks, chunk_index_in_window)
        Compute chunk position.
    _enrich_and_trim(chunks)
        Enrich and trim a list of chunks.
    __call__(window)
        Call method for the class instances to behave like a function.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, *, max_chunks: int) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        max_chunks : int
            Maximum number of chunks.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes, as this is a callable class.

        Raises
        ------
        ValueError
            If `max_chunks` is not greater than 0.

        """

        if max_chunks <= 0:
            raise ValueError("[TrimChunksTransform] `max_chunks` must be > 0.")

        self.max_chunks = int(max_chunks)

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_pos(*, n_chunks: int, chunk_index_in_window: int) -> int:
        """
        Compute chunk position.

        Parameters
        ----------
        n_chunks : int
            Number of chunks.
        chunk_index_in_window : int
            Chunk index in window.

        Returns
        -------
        int
            Computed chunk position.

        """

        # pos=1 for newest (chunk_index_in_window == n_chunks-1)
        pos = (int(n_chunks) - 1 - int(chunk_index_in_window)) + 1
        if pos < 1:
            pos = 1

        return int(pos)

    # ------------------------------------------------------------------------------------------------------------------
    def _enrich_and_trim(self, chunks: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """
        Enrich and trim a list of chunks.

        Parameters
        ----------
        chunks : list[Mapping[str, Any]]
            List of chunks to be enriched/trimmed.

        Returns
        -------
        list[dict[str, Any]]
            Enriched/trimmed chunks, reordered in chronological order (oldest→newest).

        Raises
        ------
        KeyError
            If a given chunk does not have a key "chunk_index_in_window".

        """

        if not chunks:
            return []

        n = len(chunks)

        enriched: list[dict[str, Any]] = []
        for ch in chunks:
            if "chunk_index_in_window" not in ch:
                raise KeyError("[TrimChunksTransform] chunk missing 'chunk_index_in_window'.")

            pos = self._compute_pos(n_chunks=n, chunk_index_in_window=int(ch["chunk_index_in_window"]))

            ch2 = dict(ch)
            ch2["pos"] = int(pos)
            enriched.append(ch2)

        # Keep chunks closest to output (smallest pos). Tie-break deterministically.
        closest = sorted(
            enriched,
            key=lambda c: (int(c["pos"]), int(c["chunk_index_in_window"])),
        )[: self.max_chunks]

        # Return in chronological order (oldest→newest) for determinism/readability.
        return sorted(closest, key=lambda c: int(c["chunk_index_in_window"]))

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, window: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        window : dict[str, Any]
            Window on which the transform is applied.

        Returns
        -------
        dict[str, Any] | None
            Transformed window with updated values for "input" and "actuator" keys.

        Raises
        ------
        KeyError
            If `window` does not have the required key "chunks".

        """

        if window is None:
            return None

        if "chunks" not in window:
            raise KeyError("[TrimChunksTransform] `window` missing required key 'chunks'.")

        chunks = window.get("chunks") or {}
        in_chunks = chunks.get("input") or []
        act_chunks = chunks.get("actuator") or []

        trimmed_in = self._enrich_and_trim(chunks=in_chunks)
        trimmed_act = self._enrich_and_trim(chunks=act_chunks)

        logger.debug(
            "win=%s (shot=%s) | input %d→%d | actuator %d→%d | max=%d",
            window.get("window_index"),
            window.get("shot_id"),
            len(in_chunks),
            len(trimmed_in),
            len(act_chunks),
            len(trimmed_act),
            self.max_chunks,
        )

        out = dict(window)
        out["chunks"] = {"input": trimmed_in, "actuator": trimmed_act}

        return out
