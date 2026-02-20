"""
SelectValidWindowsTransform
===========================

Window-level filtering + optional subsampling for the MMT preprocessing pipeline.

This transform operates on a *single window dict* and returns either:
  - a (shallow) masked copy of the window if it should be kept, or
  - None if it should be dropped.

Expected pipeline position
--------------------------
    ChunkWindowsTransform
        → SelectValidWindowsTransform
            → TrimChunksTransform
                → EmbedChunksTransform
                    → BuildTokensTransform

Window subsampling (minimum spacing)
----------------------------------------------
If `window_stride_sec` is set, the transform keeps windows such that the time
between consecutive *kept* windows for the same shot is at least
`window_stride_sec`:

    keep window i if (t_cut_i - t_cut_last_kept) >= window_stride_sec

Notes:
- This logic is per-shot and requires minimal internal state (last kept t_cut).
- The stride rule is applied *after* validity checks; only kept windows update
  the last-kept state.

Validity
--------
- Chunk-level: input/actuator chunk values are masked (set to None) if invalid
  (NaN/inf/empty) according to `accept_na`.
- Signals: input/actuator signals count as valid if they have at least
  `min_valid_chunks` valid chunks.
- Outputs: output signals count as valid if their values are valid (after masking).

Returns
-------
- window dict (masked copy) if thresholds are met and optional stride constraint passes
- None otherwise
"""

from __future__ import annotations

from typing import Any, Dict, Hashable, Tuple
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.SelectValidWindows")


class SelectValidWindowsTransform:
    """
    Window-level filter and optional per-shot subsampler.

    Input
    -----
    window: dict containing at least:
      - "shot_id"
      - "window_index"
      - "t_cut"
      - "chunks": {"input": [...], "actuator": [...]}
      - "output": {name: {"values": ...}, ...}

    Output
    ------
    - Returns a masked copy of `window` if kept
    - Returns None if dropped
    """

    def __init__(
        self,
        *,
        min_valid_inputs_actuators: int = 1,
        min_valid_chunks: int = 1,
        min_valid_outputs: int = 1,
        accept_na: bool = False,
        window_stride_sec: float | None = None,
    ) -> None:
        self.min_valid_inputs_actuators = int(min_valid_inputs_actuators)
        self.min_valid_chunks = int(min_valid_chunks)
        self.min_valid_outputs = int(min_valid_outputs)

        self.accept_na = bool(accept_na)
        self.window_stride_sec = (
            float(window_stride_sec) if window_stride_sec is not None else None
        )
        if self.window_stride_sec is not None and self.window_stride_sec <= 0:
            raise ValueError("window_stride_sec must be > 0")

        # shot_key -> (last_kept_t_cut, last_kept_window_index)
        self._last_kept_by_shot: Dict[Hashable, Tuple[float | None, int | None]] = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _mask_if_bad(self, values):
        """
        Returns (mask: bool, cleaned: ndarray|None)
        - mask=True means treat as invalid → set to None
        - accept_na controls whether partial NaNs are allowed
        """
        if values is None:
            return True, None

        arr = np.asarray(values)
        if arr.size == 0:
            return True, None

        finite = np.isfinite(arr)
        if not finite.any():
            return True, None

        if not finite.all():
            if not self.accept_na:
                return True, None
            return False, arr

        return False, arr

    def _passes_window_stride(self, shot_id: Any, w_idx: Any, t_cut: Any) -> bool:
        """
        Keep iff distance from last kept t_cut (same shot) >= window_stride_sec.
        """
        if self.window_stride_sec is None:
            return True

        if shot_id is None:
            raise ValueError(
                "[SelectValidWindows] missing shot_id while window_stride_sec is set"
            )
        if t_cut is None:
            raise ValueError(
                "[SelectValidWindows] missing t_cut while window_stride_sec is set"
            )

        key = shot_id
        last_t_cut, last_w_idx = self._last_kept_by_shot.get(key, (None, None))

        # If window indices go backwards (new epoch / reset), reset state for that shot
        if (
            last_w_idx is not None
            and w_idx is not None
            and int(w_idx) <= int(last_w_idx)
        ):
            self._last_kept_by_shot.pop(key, None)
            last_t_cut, last_w_idx = None, None

        if last_t_cut is None:
            return True

        dt = float(t_cut) - float(last_t_cut)
        return dt >= (self.window_stride_sec - 1e-12)

    def _commit_window_stride(self, shot_id: Any, w_idx: Any, t_cut: Any) -> None:
        if self.window_stride_sec is None:
            return
        if shot_id is None or t_cut is None:
            return
        self._last_kept_by_shot[shot_id] = (
            float(t_cut),
            int(w_idx) if w_idx is not None else None,
        )

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def __call__(self, window: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if window is None:
            return None

        w_idx = window.get("window_index")
        shot_id = window.get("shot_id")
        t_cut = window.get("t_cut")

        # ------------------------------------------------------------------
        # 1) Chunk-level validation (copy before masking)
        # ------------------------------------------------------------------
        chunks = window.get("chunks") or {}
        valid_chunks_by_sig = {
            "input": defaultdict(int),
            "actuator": defaultdict(int),
        }

        new_chunks = dict(chunks)

        for role in ("input", "actuator"):
            new_role_chunks = []
            for ch in chunks.get(role) or []:
                ch2: Dict[str, Any] = dict(ch)
                sigs = ch.get("signals") or {}

                sigs2: Dict[str, Any] = {}
                for name, val in sigs.items():
                    mask, cleaned = self._mask_if_bad(val)
                    if mask:
                        sigs2[name] = None
                    else:
                        sigs2[name] = cleaned
                        valid_chunks_by_sig[role][name] += 1

                ch2["signals"] = sigs2
                new_role_chunks.append(ch2)

            new_chunks[role] = new_role_chunks

        valid_x_signals = set()
        for role in ("input", "actuator"):
            for name, count in valid_chunks_by_sig[role].items():
                if count >= self.min_valid_chunks:
                    valid_x_signals.add(name)

        n_inputs_actuators = len(valid_x_signals)

        # ------------------------------------------------------------------
        # 2) Output-level validation (copy before masking)
        # ------------------------------------------------------------------
        output = window.get("output") or {}
        output2: Dict[str, Any] = {}
        n_outputs_valid = 0

        for name, entry in output.items():
            if not isinstance(entry, dict):
                mask, cleaned = self._mask_if_bad(entry)
                output2[name] = {"values": None if mask else cleaned}
                if not mask:
                    n_outputs_valid += 1
                continue

            entry2 = dict(entry)
            mask, cleaned = self._mask_if_bad(entry.get("values"))
            entry2["values"] = None if mask else cleaned
            output2[name] = entry2
            if not mask:
                n_outputs_valid += 1

        # ------------------------------------------------------------------
        # 3) Validity decision
        # ------------------------------------------------------------------
        keep = True
        if n_inputs_actuators < self.min_valid_inputs_actuators:
            keep = False
        if n_outputs_valid < self.min_valid_outputs:
            keep = False

        # ------------------------------------------------------------------
        # 4) Optional subsampling — only among kept windows
        # ------------------------------------------------------------------
        if keep and not self._passes_window_stride(shot_id, w_idx, t_cut):
            keep = False

        logger.debug(
            "win %s (shot %s) | valid_inputs_actuators=%d, valid_outputs=%d "
            "(min_x=%d, min_y=%d, min_chunks=%d, stride=%s) → %s",
            w_idx,
            shot_id,
            n_inputs_actuators,
            n_outputs_valid,
            self.min_valid_inputs_actuators,
            self.min_valid_outputs,
            self.min_valid_chunks,
            "none"
            if self.window_stride_sec is None
            else f"{self.window_stride_sec:.6f}s",
            "KEEP" if keep else "DROP",
        )

        if not keep:
            return None

        # Commit stride state only on KEEP
        self._commit_window_stride(shot_id, w_idx, t_cut)

        # Return a masked copy
        out = dict(window)
        out["chunks"] = new_chunks
        out["output"] = output2
        return out
