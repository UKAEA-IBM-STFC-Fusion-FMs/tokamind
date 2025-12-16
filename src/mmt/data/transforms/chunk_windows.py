"""
ChunkWindowsTransform
=====================

Chunk input and actuator signals from a single *baseline window* into fixed-length
segments (“chunks”).

This transform is designed to run inside the baseline `TaskModelTransformWrapper`
as part of the model-specific transform chain. It receives one window dict at a
time and appends a `window["chunks"]` structure used downstream (SelectValidWindows,
TrimChunks, EmbedChunks, BuildTokens, ...).

Design: TS-first (timestamps / samples)
---------------------------------------
All operational logic is performed in *timestamp indices* (samples):

  - window span is validated using dict_metadata[signal]["ts_length"]
  - signal sampling grid step is dict_metadata[signal]["ts_stride"] (typically 1)
  - chunk boundaries are computed and sliced in samples

Time (seconds) is derived *only* for metadata/logging:

  chunk_start_time_sec = chunk_start_sample * dt

Required dict_metadata fields (no fallback)
-------------------------------------------
For every signal name that appears under window["input"] or window["actuator"],
dict_metadata must include:

  - dt: float
  - ts_length: int
  - ts_stride: int

Additionally, for window time origin computation we require, for at least one
input signal present in the window:

  - sec_length: float  (input window span in seconds)

Assumptions (intentionally simple)
----------------------------------
Within each role group ("input" / "actuator"), all present signals share:

  - the same dt
  - the same ts_length
  - the same ts_stride

If not, the transform raises a ValueError (no fallback, no implicit resampling).

Chunk dict schema
-----------------
Each produced chunk has:

  {
    "role": "input" | "actuator",
    "chunk_index_in_window": int,
    "chunk_start_sample": int,     # sample index on the role grid (absolute-like)
    "chunk_start_time": float,     # absolute time (sec)
    "chunk_size_samples": int,
    "signals": {signal_name: np.ndarray[..., T_chunk], ...},
  }

`shot_id` and `window_index` are handled upstream by the wrapper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import logging

logger = logging.getLogger("mmt.Chunking")


class ChunkWindowsTransform:
    """
    Chunk input and actuator signals from a single baseline window into segments.

    User-facing parameters are provided in seconds, but all slicing is performed
    in timestamp indices (samples) using metadata dt and ts_* fields.
    """

    def __init__(
        self,
        *,
        dict_metadata: Dict[str, Any],
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

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _slice_last_axis(vals: Any, start: int, end: int) -> Optional[np.ndarray]:
        """Slice along the last axis [start:end] in sample indices."""
        if vals is None:
            return None
        arr = np.asarray(vals)
        if arr.shape[-1] < end:
            return None
        idx = [slice(None)] * arr.ndim
        idx[-1] = slice(start, end)
        return arr[tuple(idx)]

    def _validate_group_grid(self, group: Dict[str, Any]) -> Tuple[float, int, int]:
        """
        Validate that all signals in a role group share the same sampling grid and length.

        Returns
        -------
        (dt, ts_length, ts_stride)

        Validation performed (no fallback):
        - dict_metadata[sig]["dt"], ["ts_length"], ["ts_stride"] must exist
        - values.shape[-1] must equal ts_length
        - all signals must match (dt, ts_length, ts_stride)
        """
        ref_dt: Optional[float] = None
        ref_T: Optional[int] = None
        ref_step: Optional[int] = None

        for sig_name, entry in group.items():
            if not isinstance(entry, dict) or "values" not in entry:
                continue

            meta = self.dict_metadata[sig_name]  # no fallback
            dt = float(meta["dt"])
            T_expected = int(meta["ts_length"])
            step = int(meta["ts_stride"])

            if step <= 0:
                raise ValueError(
                    f"[ChunkWindowsTransform] dict_metadata[{sig_name!r}]['ts_stride'] must be > 0"
                )

            arr = np.asarray(entry["values"])
            T_actual = int(arr.shape[-1])
            if T_actual != T_expected:
                raise ValueError(
                    f"[ChunkWindowsTransform] signal {sig_name!r} has T={T_actual}, "
                    f"but dict_metadata expects ts_length={T_expected}"
                )

            if ref_dt is None:
                ref_dt = dt
                ref_T = T_expected
                ref_step = step
            else:
                if dt != ref_dt:
                    raise ValueError(
                        f"[ChunkWindowsTransform] mixed dt in group: {sig_name} dt={dt} vs ref_dt={ref_dt}"
                    )
                if T_expected != ref_T:
                    raise ValueError(
                        f"[ChunkWindowsTransform] mixed ts_length in group: {sig_name} ts_length={T_expected} vs ref_T={ref_T}"
                    )
                if step != ref_step:
                    raise ValueError(
                        f"[ChunkWindowsTransform] mixed ts_stride in group: {sig_name} ts_stride={step} vs ref_step={ref_step}"
                    )

        if ref_dt is None or ref_T is None or ref_step is None:
            return 0.0, 0, 0

        return ref_dt, ref_T, ref_step

    @staticmethod
    def _sec_to_ts_len(sec: float, dt: float) -> int:
        """
        Convert seconds -> samples for a length.
        Use round() to align with historical behavior and reduce loss drift.
        """
        return int(round(sec / dt))

    def _chunks_for_group(
        self,
        *,
        group: Optional[Dict[str, Any]],
        role: str,
        t0_sec: float,
    ) -> List[Dict[str, Any]]:
        """
        Chunk all signals in one role group using ts_* grid.
        """
        if not group:
            return []

        dt, T, ts_step = self._validate_group_grid(group)
        if T <= 0:
            return []

        # Convert chunk specs (seconds) to samples on this dt grid.
        chunk_len_ts = self._sec_to_ts_len(self.chunk_length_sec, dt)
        stride_ts = self._sec_to_ts_len(self.stride_sec, dt)

        if chunk_len_ts <= 0:
            raise ValueError(
                f"[ChunkWindowsTransform] chunk_length_sec={self.chunk_length_sec} with dt={dt} gives chunk_len_ts={chunk_len_ts}"
            )
        if stride_ts <= 0:
            raise ValueError(
                f"[ChunkWindowsTransform] stride_sec={self.stride_sec} with dt={dt} gives stride_ts={stride_ts}"
            )

        # Enforce that we advance along the actual grid step.
        # Typical case: ts_step=1 so this is a no-op.
        if (chunk_len_ts % ts_step) != 0:
            raise ValueError(
                f"[ChunkWindowsTransform] chunk_len_ts={chunk_len_ts} not divisible by ts_stride={ts_step} for role={role}"
            )
        if (stride_ts % ts_step) != 0:
            raise ValueError(
                f"[ChunkWindowsTransform] stride_ts={stride_ts} not divisible by ts_stride={ts_step} for role={role}"
            )

        chunks: List[Dict[str, Any]] = []
        start = 0
        idx_in_window = 0

        while start + chunk_len_ts <= T:
            end = start + chunk_len_ts

            sigs: Dict[str, np.ndarray] = {}
            for sig_name, entry in group.items():
                if not isinstance(entry, dict):
                    continue
                vals = entry.get("values")
                if vals is None:
                    continue
                sub = self._slice_last_axis(vals, start, end)
                if sub is not None:
                    sigs[sig_name] = sub

            if sigs:
                # Compute absolute-ish sample index from time origin on this dt grid.
                # This matches the older “global-ish” semantics (time -> sample).
                chunk_start_time = float(t0_sec + start * dt)
                chunk_start_sample = int(round(chunk_start_time / dt))

                chunks.append(
                    {
                        "role": role,
                        "chunk_index_in_window": idx_in_window,
                        "chunk_start_sample": chunk_start_sample,
                        "chunk_start_time": chunk_start_time,
                        "chunk_size_samples": int(chunk_len_ts),
                        "signals": sigs,
                    }
                )
                idx_in_window += 1

            start += stride_ts

        return chunks

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply chunking to a single window dict.

        Required fields:
          - window["t_cut"] : float (seconds)
          - window["input"] : dict of signals -> {"values": ...}
          - window["actuator"] : dict of signals -> {"values": ...} (optional)

        Returns:
          - same window dict with an added "chunks" field.
        """
        if "t_cut" not in window:
            raise KeyError(
                "[ChunkWindowsTransform] window missing required key 't_cut'"
            )

        t_cut = float(window["t_cut"])
        input_group = window.get("input") or {}
        act_group = window.get("actuator") or {}

        if not input_group:
            raise ValueError("[ChunkWindowsTransform] window has no 'input' signals")

        # Time origin: start of input window span
        first_input_sig = next(iter(input_group.keys()))
        input_sec_length = float(
            self.dict_metadata[first_input_sig]["sec_length"]
        )  # required
        t0_sec = t_cut - input_sec_length

        chunks_input = self._chunks_for_group(
            group=input_group, role="input", t0_sec=t0_sec
        )
        chunks_act = self._chunks_for_group(
            group=act_group, role="actuator", t0_sec=t0_sec
        )

        logger.debug(
            "win=%s shot=%s t_cut=%.6f → %d input chunks, %d actuator chunks",
            window.get("window_index"),
            window.get("shot_id"),
            t_cut,
            len(chunks_input),
            len(chunks_act),
        )

        out = dict(window)
        out["chunks"] = {"input": chunks_input, "actuator": chunks_act}
        return out
