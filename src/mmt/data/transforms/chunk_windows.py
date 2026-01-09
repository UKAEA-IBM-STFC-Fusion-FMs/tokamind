"""
ChunkWindowsTransform
=====================

Chunk input and actuator signals of a window into fixed-length segments ("chunks")
using a shared time grid, while allowing per-signal dt (different samples per chunk).

Key design (v0)
---------------
- Chunk grid is defined in *seconds* (chunk_length_sec and stride_sec), which is
  the only unit shared across signals with different dt.
- Chunk identity is expressed as integer indices:
    * chunk_index_in_window: 0..N-1 inside that window/role
    * chunk_index_global: stable slot id on the stride grid (used for caching)
- Per-signal dt is used only to map slot offsets -> sample indices.

Expected input window
---------------------
From TaskModelTransformWrapper:

    window["input"][key]    = {"time": ..., "values": np.ndarray[..., T]}
    window["actuator"][key] = {"time": ..., "values": np.ndarray[..., T]}
    window["t_cut"]         = float
    window["shot_id"]       = any
    window["window_index"]  = int

dict_metadata (new structure)
-----------------------------
    dict_metadata = {
        "sec_stride": float,
        "input":    { "<source>-<signal>": {"dt": float, "sec_length": float, ...}, ... },
        "actuator": { "<source>-<signal>": {"dt": float, "sec_length": float, ...}, ... },
        "output":   { ... },
    }

Output
------
Adds:

    window["chunks"] = {
        "input":    [chunk, ...],
        "actuator": [chunk, ...],
    }

Each chunk is:

    {
      "role": "input" | "actuator",
      "chunk_index_in_window": int,
      "chunk_index_global": int,
      "signals": { key: np.ndarray[..., T_chunk] or None, ... }
    }

Notes
-----
- Input and actuator may have different number of chunks (role spans differ).
- Within a role, all signals share the SAME number of chunk slots.
- Each signal can have different sample count per chunk due to dt.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Mapping
import logging
import numpy as np

logger = logging.getLogger("mmt.Chunking")


class ChunkWindowsTransform:
    def __init__(
        self,
        *,
        dict_metadata: Mapping[str, Any],
        chunk_length_sec: float,
        stride_sec: Optional[float] = None,
    ) -> None:
        if chunk_length_sec <= 0:
            raise ValueError("chunk_length_sec must be > 0")

        self.dict_metadata = dict_metadata
        self.chunk_length_sec = float(chunk_length_sec)
        self.stride_sec = (
            float(stride_sec) if stride_sec is not None else float(chunk_length_sec)
        )

        if self.stride_sec <= 0:
            raise ValueError("stride_sec must be > 0")

        for k in ("input", "actuator", "output"):
            if k not in self.dict_metadata:
                raise KeyError(
                    f"ChunkWindowsTransform: dict_metadata missing top-level key {k!r}"
                )

    # ------------------------------------------------------------------

    @staticmethod
    def _slice_with_pad(arr: np.ndarray, start: int, length: int) -> np.ndarray:
        """
        Slice arr[..., start:start+length] and right-pad with NaN if out-of-range.
        Assumes float-like arrays.
        """
        if length <= 0:
            raise ValueError("length must be > 0")

        T = int(arr.shape[-1])
        end = start + length

        s0 = max(0, min(int(start), T))
        e0 = max(0, min(int(end), T))

        sub = arr[..., s0:e0]
        got = int(sub.shape[-1])
        if got == length:
            return sub

        pad = length - got
        pad_shape = sub.shape[:-1] + (pad,)
        pad_arr = np.full(pad_shape, np.nan, dtype=sub.dtype)
        return np.concatenate([sub, pad_arr], axis=-1)

    def _role_sec_length(self, role: str, group: Dict[str, Any]) -> float:
        """
        Role span (seconds) is taken from metadata for the first signal key.
        Assumed consistent within the role (by construction from get_metadata()).
        """
        if not group:
            return 0.0
        first_key = next(iter(group.keys()))
        md = self.dict_metadata[role].get(first_key)
        if md is None:
            raise KeyError(
                f"ChunkWindowsTransform: missing metadata for {role}:{first_key}"
            )
        return float(md["sec_length"])

    def _num_chunks(self, role_span_sec: float) -> int:
        """
        Number of chunk slots given role span, chunk_length, stride (all seconds).
        """
        L = self.chunk_length_sec
        S = self.stride_sec
        if role_span_sec < L - 1e-12:
            return 0
        return int(np.floor((role_span_sec - L) / S + 1e-12)) + 1

    def _chunks_for_group(
        self,
        *,
        group: Optional[Dict[str, Any]],
        role: str,
        base_global_index: int,
    ) -> List[Dict[str, Any]]:
        if not group:
            return []

        role_span_sec = self._role_sec_length(role, group)
        n_chunks = self._num_chunks(role_span_sec)
        if n_chunks <= 0:
            return []

        chunks: List[Dict[str, Any]] = []

        for k in range(n_chunks):
            sigs: Dict[str, Any] = {}

            # Slot start offset (seconds) relative to role start
            start_off_sec = k * self.stride_sec

            for key, entry in group.items():
                md = self.dict_metadata[role].get(key)
                if md is None:
                    raise KeyError(
                        f"ChunkWindowsTransform: missing metadata for {role}:{key}"
                    )

                if not isinstance(entry, dict) or "values" not in entry:
                    raise TypeError(
                        f"ChunkWindowsTransform expects window[{role}][{key}] to be a dict with key 'values'"
                    )

                values = entry["values"]
                if values is None:
                    sigs[key] = None
                    continue

                arr = np.asarray(values)
                if arr.size == 0:
                    sigs[key] = None
                    continue

                dt = float(md["dt"])
                if dt <= 0:
                    raise ValueError(
                        f"ChunkWindowsTransform: non-positive dt for {role}:{key}: {dt}"
                    )

                # Per-signal sample counts (dt can differ per signal!)
                chunk_len_ts = int(np.round(self.chunk_length_sec / dt))
                if chunk_len_ts <= 0:
                    raise ValueError(
                        f"ChunkWindowsTransform: chunk_length_sec={self.chunk_length_sec} too small for {role}:{key} dt={dt}"
                    )

                # Start index in this signal's array (relative to role start)
                start_idx = int(np.round(start_off_sec / dt))

                sigs[key] = self._slice_with_pad(arr, start_idx, chunk_len_ts)

            chunks.append(
                {
                    "role": role,
                    "chunk_index_in_window": int(k),
                    "chunk_index_global": int(base_global_index + k),
                    "signals": sigs,
                }
            )

        return chunks

    # ------------------------------------------------------------------

    def __call__(self, window: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if window is None:
            return None
        if "t_cut" not in window:
            raise KeyError(
                "[ChunkWindowsTransform] window missing required key 't_cut'"
            )

        t_cut = float(window["t_cut"])
        input_group = window.get("input") or {}
        act_group = window.get("actuator") or {}

        if not isinstance(input_group, dict) or not input_group:
            raise ValueError("[ChunkWindowsTransform] window has no 'input' signals")

        # Role start for both input/actuator slices is the start of the input span
        first_input_key = next(iter(input_group.keys()))
        md0 = self.dict_metadata["input"].get(first_input_key)
        if md0 is None:
            raise KeyError(
                f"ChunkWindowsTransform: missing metadata for input:{first_input_key}"
            )
        input_len_sec = float(md0["sec_length"])

        # Shared global slot base: integer index on the stride grid
        t0_sec = t_cut - input_len_sec
        base_global_index = int(np.round(t0_sec / self.stride_sec))

        chunks_input = self._chunks_for_group(
            group=input_group, role="input", base_global_index=base_global_index
        )
        chunks_act = self._chunks_for_group(
            group=act_group, role="actuator", base_global_index=base_global_index
        )

        logger.debug(
            "win %s (shot %s) | t_cut=%.6f → input chunks=%d, actuator chunks=%d | base_global=%d",
            window.get("window_index"),
            window.get("shot_id"),
            t_cut,
            len(chunks_input),
            len(chunks_act),
            base_global_index,
        )

        out = dict(window)
        out["chunks"] = {"input": chunks_input, "actuator": chunks_act}
        return out
