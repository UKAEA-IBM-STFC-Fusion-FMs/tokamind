"""
ChunkWindowsTransform
=====================

Chunk input and actuator signals of a window into fixed-length segments ("chunks") using a shared time grid, while
allowing per-signal dt (different samples per chunk).

Key design (v0)
---------------
- Chunk grid is defined in *seconds* (chunk_length_sec and stride_sec), which is the only unit shared across signals
with different dt.
- Chunk identity is expressed as integer indices:
    * chunk_index_in_window: 0..N-1 inside that window/role
    * chunk_index_global: stable slot ID on the stride grid (used for caching)
- Per-signal dt is used only to map slot offsets -> sample indices.

Expected input window
---------------------
From TokaMarkDataset (benchmark window iterable):

    window["input"][key]    = {"time": ..., "values": np.ndarray[..., T]}
    window["actuator"][key] = {"time": ..., "values": np.ndarray[..., T]}
    window["t_cut"]         = float
    window["shot_id"]       = str, int
    window["window_index"]  = str, int

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
- Role span is derived from current window arrays (len(values_time_axis) * dt).
"""

from __future__ import annotations

from typing import Any, Optional
from collections.abc import Mapping
import logging
import numpy as np

from mmt.constants import TIME_FLOAT_TOL


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.Chunking")


# ======================================================================================================================
class ChunkWindowsTransform:
    """
    Class for the Chunk Window Transform.

    Attributes
    ----------
    dict_metadata : Mapping[str, Any]
        FAIR MAST metadata dictionary.
    chunk_length_sec : float
        Chunk length in seconds.
    stride_sec : float
        Stride in seconds.

    Methods
    -------
    __call__(window)
        Call method for the class instances to behave like a function.
    _slice_with_pad(arr, start, length)
        Slice arr[..., start:start+length] and right-pad with NaN if out-of-range. Assumes float-like arrays.
    _role_sec_length(role, group)
        Get role span (seconds) from current window payload.
    _num_chunks(role_span_sec)
        Number of chunk slots given role span, chunk_length, stride (all seconds).
    _chunks_for_group(role, group, base_global_index_, shot_id, window_idx)
        Get chunks for a given group.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        dict_metadata: Mapping[str, Any],
        chunk_length_sec: float,
        stride_sec: Optional[float] = None,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        dict_metadata : Mapping[str, Any]
            FAIR MAST metadata dictionary.
        chunk_length_sec : float
            Chunk length in seconds.
        stride_sec : Optional[float]
            Stride in seconds.
            Optional. Default: None.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors, as this is a callable class.

        Raises
        ------
        KeyError
            If `dict_metadata` misses any top-level key.

        """

        if chunk_length_sec <= 0:
            raise ValueError("[ChunkWindowsTransform] `chunk_length_sec` must be > 0.")

        self.dict_metadata = dict_metadata
        self.chunk_length_sec = float(chunk_length_sec)
        if stride_sec is not None:
            self.stride_sec = float(stride_sec)
        else:
            self.stride_sec = float(chunk_length_sec)

        if self.stride_sec <= 0:
            raise ValueError("[ChunkWindowsTransform] `stride_sec` must be > 0.")

        for k in ("input", "actuator", "output"):
            if k not in self.dict_metadata:
                raise KeyError(f"[ChunkWindowsTransform] `dict_metadata` missing top-level key {k!r}.")

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _slice_with_pad(arr: np.ndarray, start: int, length: int) -> np.ndarray:
        """
        Slice arr[..., start:start+length] and right-pad with NaN if out-of-range. Assumes float-like arrays.

        Parameters
        ----------
        arr : np.ndarray
            Input array to be sliced.
        start : int
            Start of the slicing interval.
        length : int
            Length of the slicing interval.

        Returns
        -------
        np.ndarray
            Sliced interval.

        """

        if length <= 0:
            raise ValueError("[ChunkWindowsTransform] `length` must be > 0")

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
        pad_arr = np.full(shape=pad_shape, fill_value=np.nan, dtype=sub.dtype)

        return np.concatenate([sub, pad_arr], axis=-1)

    # ------------------------------------------------------------------------------------------------------------------
    def _role_sec_length(self, role: str, group: Mapping[str, Any]) -> float:
        """
        Get role span (seconds) from current window payload.

        We interpret a signal with T samples and dt as spanning T*dt seconds.

        Parameters
        ----------
        role : str
            Target rol.
        group : Mapping[str, Any]
            Group mapping (dict) associated with the provided `role`.

        Returns
        -------
        float
            Maximum span in seconds.

        Raises
        ------
        KeyError
            If `self.dict_metadata["role"]` does not have a given required key.
            If a non-positive for the key "dt" is obtained from a given (`role`, `group`) pair.

        """

        if not group:
            return 0.0

        spans_sec: list[float] = []

        for key, entry in group.items():
            md = self.dict_metadata[role].get(key)
            if md is None:
                raise KeyError(f"[ChunkWindowsTransform] missing metadata for {role}:{key}.")

            dt = float(md["dt"])
            if dt <= 0:
                raise ValueError(f"[ChunkWindowsTransform] Non-positive dt for {role}:{key}: {dt}.")

            if not isinstance(entry, dict):
                continue

            values = entry.get("values")
            if values is None:
                continue

            arr = np.asarray(values)
            if (arr.size == 0) or (arr.ndim < 1):
                continue

            spans_sec.append(float(arr.shape[-1]) * dt)

        if not spans_sec:
            return 0.0

        return float(max(spans_sec))

    # ------------------------------------------------------------------------------------------------------------------
    def _num_chunks(self, role_span_sec: float) -> int:
        """
        Number of chunk slots given role span, chunk_length, stride (all seconds).

        Parameters
        ----------
        role_span_sec : float
            Role span in seconds.

        Returns
        -------
        int
            Number of chunk slots.

        """

        if role_span_sec < (self.chunk_length_sec - TIME_FLOAT_TOL):
            return 0

        return int(np.floor((role_span_sec - self.chunk_length_sec) / self.stride_sec + TIME_FLOAT_TOL)) + 1

    # ------------------------------------------------------------------------------------------------------------------
    def _chunks_for_group(  # NOSONAR - Ignore cognitive complexity
        self,
        *,
        role: str,
        group: Optional[Mapping[str, Any]],
        base_global_index: int,
        shot_id: str | int,
        window_idx: str | int,
    ) -> list[dict[str, Any]]:
        """
        Get chunks for a given group.

        Parameters
        ----------
        role : str
            Target rol.
        group : Optional[Mapping[str, Any]]
            Optional group mapping (dict) associated with the provided `role`.
        base_global_index : int
            Best global index.
        shot_id : str | int
            Shot ID of the chunks.
        window_idx : str | int
            Window index of the chunks.

        Returns
        -------
        list[dict[str, Any]]
            List of resulting chunks

        Raises
        ------
        KeyError
             If `self.dict_metadata[role]` does not have a given required key.
        TypeError
             If a given `group` is not a mapping (dict) or does not have a key "value".
        ValueError
            If a non-positive for the key "dt" is obtained from a given (`role`, `group`) pair.
            If `self.chunk_length_sec` is too small for the resulting dt from a given (`role`, `group`) pair.

        """

        if not group:
            return []

        for entry in group.values():
            if not isinstance(entry, dict) or ("values" not in entry):
                raise TypeError("[ChunkWindowsTransform] `group` expected to be a dict with key 'values'.")

        role_span_sec = self._role_sec_length(role=role, group=group)
        n_chunks = self._num_chunks(role_span_sec=role_span_sec)
        if n_chunks <= 0:
            return []

        logger.debug(
            "win=%s (shot=%s) role=%s | span=%.6fs -> chunks=%d",
            window_idx,
            shot_id,
            role,
            role_span_sec,
            n_chunks,
        )

        chunks: list[dict[str, Any]] = []

        for k in range(n_chunks):
            sigs: dict[str, Any] = {}

            # Slot start offset (seconds) relative to role start
            start_off_sec = k * self.stride_sec

            for key, entry in group.items():
                md = self.dict_metadata[role].get(key)
                if md is None:
                    raise KeyError(f"[ChunkWindowsTransform] Missing metadata for {role}:{key}.")

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
                    raise ValueError(f"[ChunkWindowsTransform] Non-positive dt for {role}:{key}: {dt}.")

                # Per-signal sample counts (dt can differ per signal!)
                chunk_len_ts = int(np.round(self.chunk_length_sec / dt))
                if chunk_len_ts <= 0:
                    raise ValueError(
                        f"[ChunkWindowsTransform] `chunk_length_sec={self.chunk_length_sec}` too small for "
                        f"{role}:{key} dt={dt}."
                    )

                # Start index in this signal's array (relative to role start)
                start_idx = int(np.round(start_off_sec / dt))

                sigs[key] = self._slice_with_pad(arr=arr, start=start_idx, length=chunk_len_ts)

            chunks.append(
                {
                    "role": role,
                    "chunk_index_in_window": int(k),
                    "chunk_index_global": int(base_global_index + k),
                    "signals": sigs,
                }
            )

        return chunks

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, window: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """
        Call method for the class instances to behave like a function.

        Parameters
        ----------
        window : Optional[dict[str, Any]]
            Chunk window on which the transform is applied.

        Returns
        -------
        Optional[dict[str, Any]]
            Chunk window extended with input/actuator chunks for a valid `window`, otherwise None.

        Raises
        ------
        KeyError
            If `window` mapping does not have the required key 't_cut'.
        ValueError
            If `window` is not a mapping (dict) or does not have key "input".

        """

        if window is None:
            return None

        if "t_cut" not in window:
            raise KeyError("[ChunkWindowsTransform] `window` missing required key 't_cut'.")

        t_cut = float(window["t_cut"])
        input_group = window.get("input") or {}
        act_group = window.get("actuator") or {}

        if (not isinstance(input_group, dict)) or (not input_group):
            raise ValueError("[ChunkWindowsTransform] `window` has no 'input' signals.")

        # Role start for both input/actuator slices is the start of the input span.
        # Use current window span when available (dynamic history support).
        input_len_sec = self._role_sec_length(role="input", group=input_group)

        # Shared global slot base: integer index on the stride grid
        t0_sec = t_cut - input_len_sec
        base_global_index = int(np.round(t0_sec / self.stride_sec))

        shot_id: str | int = window.get("shot_id", -1)
        window_idx: str | int = window.get("window_index", -1)

        chunks_input = self._chunks_for_group(
            role="input",
            group=input_group,
            base_global_index=base_global_index,
            shot_id=shot_id,
            window_idx=window_idx,
        )
        chunks_act = self._chunks_for_group(
            role="actuator",
            group=act_group,
            base_global_index=base_global_index,
            shot_id=shot_id,
            window_idx=window_idx,
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
