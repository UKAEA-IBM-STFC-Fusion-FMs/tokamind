"""
TrimChunksTransform
===================

Trim chunk histories to a fixed maximum number of chunks and compute
relative positions with respect to the output reference time.

Design principles
-----------------
- TS-first (timestamps / samples): all logic is performed in integer
  sample indices.
- PURE transform: no state is kept across windows.
- No fallback or legacy behavior: required metadata must be present.

This transform assumes that `ChunkWindowsTransform` has already populated
each chunk with:
  - chunk_start_sample
  - chunk_size_samples

and that the window dict contains:
  - t_cut (seconds)
  - output signals with known dt via dict_metadata
"""

from __future__ import annotations


from typing import Any, Dict, List
import logging

logger = logging.getLogger("mmt.TrimChunks")


class TrimChunksTransform:
    """
    Trim chunk histories and compute relative positions (pos) in TS-space.

    Responsibilities
    ----------------
    1) Trim input and actuator chunks to `max_chunks` (keep most recent).
    2) Compute per-chunk relative position indices with respect to the
       output reference time (end of output window).
    """

    def __init__(
        self,
        *,
        dict_metadata: Dict[str, Any],
        max_chunks: int,
        delta: float,
        output_length: float,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError("max_chunks must be > 0")

        self.dict_metadata = dict_metadata
        self.max_chunks = int(max_chunks)
        self.delta = float(delta)
        self.output_length = float(output_length)

    # ------------------------------------------------------------------

    @staticmethod
    def _sec_to_ts(sec: float, dt: float) -> int:
        """Convert seconds → samples using round()."""
        return int(round(sec / dt))

    def _trim_and_pos(
        self,
        *,
        chunks: List[Dict[str, Any]],
        t_out_end_ts: int,
    ) -> List[Dict[str, Any]]:
        """
        Trim chunks and compute pos indices in TS-space.
        """
        if not chunks:
            return []

        # Keep most recent chunks
        trimmed = chunks[-self.max_chunks :]

        out: List[Dict[str, Any]] = []
        for ch in trimmed:
            start_ts = int(ch["chunk_start_sample"])
            size_ts = int(ch["chunk_size_samples"])
            end_ts = start_ts + size_ts

            # pos = how many chunk-lengths BEFORE output end
            pos = t_out_end_ts - end_ts

            ch2 = dict(ch)
            ch2["pos"] = int(pos)
            out.append(ch2)

        return out

    # ------------------------------------------------------------------

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply trimming and position computation to a single window.
        """
        if "chunks" not in window:
            raise KeyError("[TrimChunksTransform] window missing 'chunks'")
        if "t_cut" not in window:
            raise KeyError("[TrimChunksTransform] window missing 't_cut'")

        t_cut = float(window["t_cut"])

        # Use any output signal to define dt (they must be consistent)
        output_group = window.get("output") or {}
        if not output_group:
            raise ValueError("[TrimChunksTransform] window has no output signals")

        first_out_sig = next(iter(output_group.keys()))
        dt = float(self.dict_metadata[first_out_sig]["dt"])

        # Output reference time (END of output window), in TS
        t_out_end_sec = t_cut + self.delta + self.output_length
        t_out_end_ts = self._sec_to_ts(t_out_end_sec, dt)

        chunks_in = window["chunks"].get("input", [])
        chunks_act = window["chunks"].get("actuator", [])

        trimmed_in = self._trim_and_pos(
            chunks=chunks_in,
            t_out_end_ts=t_out_end_ts,
        )
        trimmed_act = self._trim_and_pos(
            chunks=chunks_act,
            t_out_end_ts=t_out_end_ts,
        )

        logger.debug(
            "win %s (shot %s) | input %d→%d, actuator %d→%d | max=%d",
            window.get("window_index"),
            window.get("shot_id"),
            len(chunks_in),
            len(trimmed_in),
            len(chunks_act),
            len(trimmed_act),
            self.max_chunks,
        )

        out = dict(window)
        out["chunks"] = {
            "input": trimmed_in,
            "actuator": trimmed_act,
        }
        return out
