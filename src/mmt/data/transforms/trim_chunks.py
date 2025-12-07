from __future__ import annotations

from typing import Any, Dict, List
import logging

logger = logging.getLogger("mmt.TrimChunks")


class TrimChunksTransform:
    """
    Remove older input+actuator chunks, keeping only the most recent
    `max_chunks` based on temporal position relative to the output window.

    This is used BEFORE embedding, to avoid computing embeddings for
    chunks that will be discarded anyway.

    Expected input (after SelectValidWindows)
    --------------------------------------------
        window = {
            "t_cut": float,
            "chunks": {
                "input":    [chunk_dict, ...],
                "actuator": [chunk_dict, ...],
            },
            ...
        }

    Each chunk_dict must contain:
        "chunk_start_time": float    # absolute time of chunk start

    Output
    ------
    The same window dict, but with chunk lists trimmed:

        window["chunks"]["input"]    = trimmed list
        window["chunks"]["actuator"] = trimmed list

    Parameters
    ----------
    chunk_length_sec : float
        Duration of each chunk in seconds.

    delta : float
        Offset between t_cut and output window start.

    output_length : float
        Duration of the output window.

    max_chunks : int
        Maximum number of chunks per role (input / actuator) to keep.
        We retain only chunks whose positive position index satisfies
        1 ≤ pos ≤ max_chunks, where:

            pos = 1  → chunk ending exactly at t_out_end
            pos = 2  → one chunk earlier
            pos = 3  → two chunks earlier
            ...

        This convention matches BuildTokensTransform.
    """

    def __init__(
        self,
        chunk_length_sec: float,
        delta: float,
        output_length: float,
        max_chunks: int,
    ) -> None:
        self.chunk_length_sec = float(chunk_length_sec)
        self.delta = float(delta)
        self.output_length = float(output_length)

        self.max_chunks = int(max_chunks)
        if self.max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")

    # ------------------------------------------------------------------ #
    def _compute_pos(self, t_out_end: float, t_start: float) -> int:
        """
        Compute positive temporal index `pos` for a chunk, using the same
        convention as BuildTokensTransform.

        Let:

            L         = chunk_length_sec
            chunk_end = t_start + L
            steps     = round((t_out_end - chunk_end) / L)
            pos       = steps + 1

        Then:

            • pos = 1 → chunk that ends exactly at t_out_end
            • pos = 2 → one chunk earlier
            • pos = 3 → two chunks earlier
            • ...

        Chunks strictly in the future of t_out_end (if any) will yield
        pos ≤ 0; these are always discarded by the trimming logic.
        """
        if t_start is None:
            raise ValueError(
                "TrimChunksTransform: chunk_start_time is None; "
                "ChunkingTransform must set it."
            )

        L = self.chunk_length_sec
        chunk_end = float(t_start) + L
        steps = round((t_out_end - chunk_end) / L)
        pos = steps + 1
        return int(pos)

    # ------------------------------------------------------------------ #
    def _trim_role(
        self,
        chunks: List[Dict[str, Any]],
        t_out_end: float,
    ) -> List[Dict[str, Any]]:
        """
        Trim chunks of a given role (input/actuator).

        We keep only chunks whose positive position satisfies:

            1 ≤ pos ≤ max_chunks

        i.e. at most `max_chunks` chunks of most recent history for that
        role (context or actuator).
        """
        if not chunks:
            return chunks

        kept: List[Dict[str, Any]] = []
        for ch in chunks:
            t_start = ch.get("chunk_start_time", None)
            pos = self._compute_pos(t_out_end, t_start)

            # Keep only valid "past" chunks up to max_chunks steps back
            if 1 <= pos <= self.max_chunks:
                kept.append(ch)

        return kept

    # ------------------------------------------------------------------ #
    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply trimming to a single window.
        """
        if window is None:
            return None

        t_cut = window.get("t_cut", None)
        if t_cut is None:
            raise ValueError("TrimChunksTransform: window missing 't_cut'")

        # compute t_out_end ONCE
        t_out_end = float(t_cut) + self.delta + self.output_length

        chunks = window.get("chunks") or {}
        input_chunks = chunks.get("input", []) or []
        actuator_chunks = chunks.get("actuator", []) or []

        trimmed_input = self._trim_role(input_chunks, t_out_end)
        trimmed_act = self._trim_role(actuator_chunks, t_out_end)

        logger.debug(
            "[TrimChunks] t_cut=%.6f | input %d→%d | actuator %d→%d | max=%d",
            t_cut,
            len(input_chunks),
            len(trimmed_input),
            len(actuator_chunks),
            len(trimmed_act),
            self.max_chunks,
        )

        window["chunks"]["input"] = trimmed_input
        window["chunks"]["actuator"] = trimmed_act

        return window
