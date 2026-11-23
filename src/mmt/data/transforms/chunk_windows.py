"""Physical chunking for input and actuator histories.

``ChunkWindowsTransform`` owns the canonical ``preprocess.chunks`` contract:
input and actuator histories may use independent chunk durations, strides, and
retained depths.  Every emitted chunk carries a physical ``time_center`` for
coordinate-aware models and the standard ``window['chunks']`` layout consumed
by embedding and collation.

The previous single-duration constructor remains accepted for older callers.
That compatibility path leaves trimming to ``TrimChunksTransform`` exactly as
before; current configurations use the role-aware form.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mmt.constants import TIME_FLOAT_TOL


class ChunkWindowsTransform:
    """Create physical chunks for independently configured input and actuator roles."""

    def __init__(
        self,
        *,
        dict_metadata: Mapping[str, Mapping[str, Any]],
        chunks_cfg: Mapping[str, Mapping[str, Any]] | None = None,
        chunk_length_sec: float | None = None,
        stride_sec: float | None = None,
    ) -> None:
        """Initialize canonical role chunks or the compatible legacy single-duration mode.

        Parameters
        ----------
        dict_metadata : Mapping[str, Mapping[str, Any]]
            Metadata keyed by role and signal name. Each dynamic signal needs a
            positive ``dt`` entry.
        chunks_cfg : Mapping[str, Mapping[str, Any]] | None
            Canonical configuration with ``input`` and ``actuator`` entries,
            each containing ``chunk_length``, ``stride``, and ``max_chunks``.
        chunk_length_sec : float | None
            Legacy shared chunk length. Mutually exclusive with ``chunks_cfg``.
        stride_sec : float | None
            Legacy shared stride. Defaults to ``chunk_length_sec``.
        """

        self.dict_metadata = dict_metadata
        self._legacy_mode = chunks_cfg is None
        if chunks_cfg is None:
            if chunk_length_sec is None or float(chunk_length_sec) <= 0.0:
                raise ValueError("chunk_length_sec must be positive when chunks_cfg is not supplied.")
            stride = float(stride_sec if stride_sec is not None else chunk_length_sec)
            if stride <= 0.0:
                raise ValueError("stride_sec must be positive when chunks_cfg is not supplied.")
            self.chunks_cfg: dict[str, dict[str, float | int | None]] = {
                role: {"chunk_length": float(chunk_length_sec), "stride": stride, "max_chunks": None}
                for role in ("input", "actuator")
            }
            return

        self.chunks_cfg = {role: dict(chunks_cfg[role]) for role in ("input", "actuator")}
        for role, cfg in self.chunks_cfg.items():
            for key in ("chunk_length", "stride", "max_chunks"):
                if key not in cfg:
                    raise KeyError(f"preprocess.chunks.{role}.{key} is required.")
            if float(cfg["chunk_length"]) <= 0.0 or float(cfg["stride"]) <= 0.0:
                raise ValueError(f"preprocess.chunks.{role} chunk_length and stride must be positive.")
            if not isinstance(cfg["max_chunks"], int) or int(cfg["max_chunks"]) <= 0:
                raise ValueError(f"preprocess.chunks.{role}.max_chunks must be a positive integer.")

    def __call__(self, window: dict[str, Any] | None) -> dict[str, Any] | None:
        """Attach independently chunked input and actuator histories to a window."""

        if window is None:
            return None
        out = dict(window)
        t_cut = float(window.get("t_cut", 0.0))
        out["chunks"] = {
            role: self._chunks_for_role(role=role, group=window.get(role) or {}, t_cut=t_cut)
            for role in ("input", "actuator")
        }
        return out

    def _chunks_for_role(self, *, role: str, group: Mapping[str, Any], t_cut: float) -> list[dict[str, Any]]:
        """Split one role into physical chunk slots and retain configured history."""

        if not group:
            return []
        cfg = self.chunks_cfg[role]
        chunk_length_raw = cfg["chunk_length"]
        stride_raw = cfg["stride"]
        if (
            isinstance(chunk_length_raw, bool)
            or not isinstance(chunk_length_raw, (int, float))
            or isinstance(stride_raw, bool)
            or not isinstance(stride_raw, (int, float))
        ):
            raise TypeError(f"Chunk configuration for role={role!r} must use numeric chunk_length and stride.")
        chunk_length = float(chunk_length_raw)
        stride = float(stride_raw)
        span = self._span_seconds(role=role, group=group)
        if span < chunk_length - TIME_FLOAT_TOL:
            return []
        n_chunks = int(np.floor((span - chunk_length) / stride + TIME_FLOAT_TOL)) + 1
        chunks: list[dict[str, Any]] = []
        for chunk_index in range(n_chunks):
            offset = float(chunk_index) * stride
            signals: dict[str, np.ndarray | None] = {}
            centres: list[float] = []
            for name, payload in group.items():
                if not isinstance(payload, Mapping) or payload.get("values") is None:
                    signals[str(name)] = None
                    continue
                metadata = self.dict_metadata[role].get(str(name))
                if metadata is None:
                    raise KeyError(f"Missing metadata for {role}:{name}.")
                dt = float(metadata["dt"])
                if dt <= 0.0:
                    raise ValueError(f"Non-positive dt for {role}:{name}: {dt}.")
                length = int(np.rint(chunk_length / dt))
                if length <= 0:
                    raise ValueError(f"Chunk length {chunk_length} is too short for {role}:{name} dt={dt}.")
                start = int(np.rint(offset / dt))
                values = np.asarray(payload["values"])
                signals[str(name)] = self._slice_with_pad(values, start=start, length=length)
                raw_time = np.asarray(payload.get("time", []), dtype=np.float64).reshape(-1)
                selected = raw_time[max(0, start) : min(start + length, raw_time.size)]
                selected = selected[np.isfinite(selected)]
                if selected.size:
                    centres.append(float(np.mean(selected)))
            chunks.append(
                {
                    "role": role,
                    "chunk_index_in_window": int(chunk_index),
                    "chunk_index_global": int(np.rint((t_cut - span + offset) / stride)),
                    "time_center": float(np.mean(centres)) if centres else t_cut - span + offset + 0.5 * chunk_length,
                    "signals": signals,
                }
            )

        max_chunks = cfg["max_chunks"]
        retained = chunks if max_chunks is None else chunks[-int(max_chunks) :]
        for index, chunk in enumerate(retained):
            chunk["pos"] = int(len(retained) - index)
        return retained

    def _span_seconds(self, *, role: str, group: Mapping[str, Any]) -> float:
        """Return the maximum available physical span for one role."""

        spans: list[float] = []
        for name, payload in group.items():
            if not isinstance(payload, Mapping) or payload.get("values") is None:
                continue
            time = np.asarray(payload.get("time", []), dtype=np.float64).reshape(-1)
            if time.size >= 2 and np.all(np.isfinite(time)):
                dt_values = np.diff(time)
                dt_values = dt_values[np.isfinite(dt_values) & (dt_values > 0.0)]
                if dt_values.size:
                    spans.append(float(time[-1] - time[0] + np.median(dt_values)))
                    continue
            # Padded MAST payloads may retain the intended values length while
            # carrying non-finite timestamps near a shot boundary.
            metadata = self.dict_metadata[role].get(str(name))
            if metadata is not None:
                spans.append(float(np.asarray(payload["values"]).shape[-1]) * float(metadata["dt"]))
                continue
        return max(spans, default=0.0)

    @staticmethod
    def _slice_with_pad(values: np.ndarray, *, start: int, length: int) -> np.ndarray:
        """Slice a time-last array and right-pad unavailable samples with NaN."""

        lower = max(0, min(int(start), int(values.shape[-1])))
        upper = max(0, min(int(start + length), int(values.shape[-1])))
        out = values[..., lower:upper]
        if int(out.shape[-1]) == length:
            return out
        padding = np.full(out.shape[:-1] + (length - int(out.shape[-1]),), np.nan, dtype=out.dtype)
        return np.concatenate([out, padding], axis=-1)


__all__ = ["ChunkWindowsTransform"]
