from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import logging

logger = logging.getLogger("mmt.Chunking")


class ChunkingTransform:
    """
    Chunk input and actuator signals from a single baseline window into fixed-length segments.

    This transform is designed to run INSIDE TaskModelTransformWrapper's model_transform,
    so it sees one "raw" baseline window at a time, e.g.:

        window = {
            "input":   { var_name: {"time": ..., "values": ...}, ... },
            "actuator":{ var_name: {"time": ..., "values": ...}, ... },
            "output":  { var_name: {"time": ..., "values": ...}, ... },
            "t_cut":        float,   # absolute time of the window cut (s)
            "input_length": float,   # input window length (s)
        }

    It will:

      - chunk ONLY "input" and "actuator" along the time axis,
      - leave "output" as-is (no chunking),
      - attach a new field:

            window["chunks"] = {
                "input":    [chunk_dict, ...],
                "actuator": [chunk_dict, ...],
            }

        where each chunk_dict has:

            {
                "role": "input" | "actuator",
                "chunk_index_in_window": int,
                "chunk_index_global": int,
                "chunk_start_time": float,  # seconds
                "signals": { var_name: np.ndarray[chunk_size, ...], ... },
            }

    NOTE: shot_id and window_index are added OUTSIDE the transform by
          TaskModelTransformWrapper in its final yield, so they are not
          available here.
    """

    def __init__(
        self,
        chunk_length_sec: float,
        stride_sec: Optional[float] = None,
    ) -> None:
        if chunk_length_sec <= 0:
            raise ValueError("chunk_length_sec must be > 0")

        self.chunk_length_sec = float(chunk_length_sec)
        # If stride_sec is None → no overlap, stride == chunk_length
        self.stride_sec = (
            float(stride_sec) if stride_sec is not None else float(chunk_length_sec)
        )

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _time_axis_length(group: Optional[Dict[str, Any]]) -> Optional[int]:
        """
        Infer the length T of the time axis from one signal in the group.

        Convention: **time is the last axis**.
          - 1D: (T,)
          - 2D: (C, T)      # C = channels, T = time
          - 3D: (H, W, T)   # images/videos with time last
        """
        if not group:
            return None

        for entry in group.values():
            if not isinstance(entry, dict):
                continue
            vals = entry.get("values")
            if vals is None:
                continue
            arr = np.asarray(vals)
            if arr.ndim >= 1:
                return arr.shape[-1]  # <-- time dimension

        return None

    @staticmethod
    def _slice_time(vals: Any, start: int, end: int) -> Optional[np.ndarray]:
        """
        Slice vals[start:end] on the first axis, if possible.
        """
        if vals is None:
            return None
        arr = np.asarray(vals)
        if arr.shape[0] < end:
            return None
        return arr[start:end]

    def _build_chunks_for_group(
        self,
        group: Optional[Dict[str, Any]],
        role: str,
        t0: float,
        dt: float,
        chunk_size_samples: int,
        stride_samples: int,
    ) -> List[Dict[str, Any]]:
        """
        Chunk all signals in a given group ("input" or "actuator").

        index k along the time axis corresponds to absolute time:
            t_k = t0 + k * dt
        """
        if not group:
            return []

        T = self._time_axis_length(group)
        if T is None or T <= 0:
            return []

        chunks: List[Dict[str, Any]] = []
        start = 0
        chunk_index_in_window = 0

        while start + chunk_size_samples <= T:
            end = start + chunk_size_samples

            signals_chunk: Dict[str, np.ndarray] = {}
            for var_name, entry in group.items():
                if not isinstance(entry, dict):
                    continue
                vals = entry.get("values")
                if vals is None:
                    continue
                sub = self._slice_time(vals, start, end)
                if sub is not None:
                    signals_chunk[var_name] = sub

            if signals_chunk:
                # Absolute start time for this chunk
                chunk_start_time = t0 + start * dt
                # Global index along the shot timeline
                chunk_index_global = int(
                    np.floor(chunk_start_time / self.chunk_length_sec)
                )

                chunks.append(
                    {
                        "role": role,
                        "chunk_index_in_window": chunk_index_in_window,
                        "chunk_index_global": chunk_index_global,
                        "chunk_start_time": chunk_start_time,
                        "signals": signals_chunk,
                    }
                )
                chunk_index_in_window += 1

            start += stride_samples

        return chunks

    # ------------------------------------------------------------------ main API

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply chunking to a SINGLE baseline window dict.

        Returns the same window dict with an extra "chunks" field.
        """
        # Basic sanity checks for required metadata
        if "t_cut" not in window or "input_length" not in window:
            # If metadata is missing, leave the window unchanged.
            logger.debug(
                "[ChunkingTransform] window missing 't_cut' or 'input_length'; skipping chunking."
            )
            return window

        t_cut = float(window["t_cut"])
        input_length = float(window["input_length"])

        input_group = window.get("input")
        act_group = window.get("actuator")

        # Compute dt from INPUT only (baseline guarantees fixed input_length)
        T_input = self._time_axis_length(input_group)
        dt = input_length / float(T_input)

        # Convert chunk_length / stride from seconds → samples
        chunk_size_samples = int(round(self.chunk_length_sec / dt))
        if chunk_size_samples <= 0:
            raise ValueError(
                f"chunk_length_sec={self.chunk_length_sec} with dt={dt} → "
                f"chunk_size_samples={chunk_size_samples} (must be > 0)"
            )

        stride_samples = int(round(self.stride_sec / dt))
        if stride_samples <= 0:
            raise ValueError(
                f"stride_sec={self.stride_sec} with dt={dt} → "
                f"stride_samples={stride_samples} (must be > 0)"
            )

        # Time origin for index 0 in the window: start of the input span
        t0 = t_cut - input_length

        chunks_input = self._build_chunks_for_group(
            input_group,
            role="input",
            t0=t0,
            dt=dt,
            chunk_size_samples=chunk_size_samples,
            stride_samples=stride_samples,
        )
        chunks_act = self._build_chunks_for_group(
            act_group,
            role="actuator",
            t0=t0,
            dt=dt,
            chunk_size_samples=chunk_size_samples,
            stride_samples=stride_samples,
        )

        t_str = f"{t_cut:.6f}"
        logger.debug(
            f"[ChunkingTransform] t_cut={t_str} → "
            f"{len(chunks_input)} input chunks, {len(chunks_act)} actuator chunks"
        )

        # Attach chunks; keep original input/actuator/output untouched for now
        window = dict(window)  # shallow copy to avoid side effects if needed
        window["chunks"] = {
            "input": chunks_input,
            "actuator": chunks_act,
        }
        return window
