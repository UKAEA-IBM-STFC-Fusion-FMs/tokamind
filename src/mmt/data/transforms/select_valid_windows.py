"""
SelectValidWindowsTransform
===========================

Pure window-level filtering and subsampling transform for the MMT pipeline.

This transform operates on a *single window dict* and decides whether the
window should be kept or dropped based on:

  - chunk-level validity of input and actuator signals
  - window-level validity of output signals
  - optional, stateless window subsampling

Design principles
-----------------
- PURE: stateless and deterministic (safe for cached/streamed datasets and multiprocessing).
- No reliance on deprecated fields such as `window["stride"]`.
- TS-first: subsampling decisions are made in *timestamps / samples* using `dt`
  from `dict_metadata`.

Expected pipeline position
--------------------------
    ChunkWindowsTransform
        → SelectValidWindowsTransform
            → TrimChunksTransform
                → EmbedChunksTransform
                    → BuildTokensTransform
"""

from __future__ import annotations

from typing import Any, Dict
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.SelectValidWindows")


class SelectValidWindowsTransform:
    """
    Pure window-level filter and subsampler for the MMT preprocessing pipeline.

    The transform is **pure and stateless**:
    - no retained memory across windows,
    - no dependence on call order,
    - safe with cached datasets, streamed datasets, multiprocessing, and multiple
      dataset splits (train / val / test).

    Window subsampling (TS-first)
    -----------------------------
    If `window_stride_sec` is set, we subsample windows deterministically using
    `window_index`, interpreted as an index on the *base dt grid*.

    Let `dt` be taken from `dict_metadata[sig]["dt"]` (for any signal in the window).
    Then:

        window_stride_ts = round(window_stride_sec / dt)

    A window is kept iff:

        window_index % window_stride_ts == 0

    This avoids any dependence on `sec_stride` and prevents unit-mismatch bugs
    when the generator operates in timestamp space.

    Validity
    --------
    - Chunks: invalid values are masked (None) based on NaN/inf/empty and `accept_na`.
    - Signals: input/actuator signals count as valid if they have at least
      `min_valid_chunks` valid chunks.
    - Outputs: output signals count as valid if their values are valid (after masking).

    Returns the window dict (possibly masked) if thresholds are met; otherwise None.
    """

    def __init__(
        self,
        *,
        dict_metadata: Dict[str, Any],
        min_valid_inputs_actuators: int = 1,
        min_valid_chunks: int = 1,
        min_valid_outputs: int = 1,
        accept_na: bool = False,
        window_stride_sec: float | None = None,
    ) -> None:
        self.dict_metadata = dict_metadata
        self.min_valid_inputs_actuators = int(min_valid_inputs_actuators)
        self.min_valid_chunks = int(min_valid_chunks)
        self.min_valid_outputs = int(min_valid_outputs)
        self.accept_na = bool(accept_na)
        self.window_stride_sec = (
            float(window_stride_sec) if window_stride_sec is not None else None
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _mask_if_bad(self, values):
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

    def _pick_any_signal_name(self, window: Dict[str, Any]) -> str:
        for role in ("input", "actuator", "output"):
            group = window.get(role) or {}
            if isinstance(group, dict) and group:
                return next(iter(group.keys()))
        raise ValueError(
            "[SelectValidWindows] window_stride_sec set but no signals found in window."
        )

    @staticmethod
    def _sec_to_ts_len(sec: float, dt: float) -> int:
        """Convert seconds -> samples using round() for stability."""
        return int(round(sec / dt))

    # ------------------------------------------------------------------
    # main API
    # ------------------------------------------------------------------

    def __call__(self, window: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if window is None:
            return None

        w_idx = window.get("window_index")
        shot_id = window.get("shot_id")

        # --------------------------------------------------------------
        # 0) Stateless window subsampling (TS-first)
        # --------------------------------------------------------------
        if self.window_stride_sec is not None:
            if w_idx is None:
                raise ValueError(
                    "[SelectValidWindows] window_stride_sec is set but window_index is None."
                )

            sig = self._pick_any_signal_name(window)
            dt = float(self.dict_metadata[sig]["dt"])
            if dt <= 0:
                raise ValueError(
                    f"[SelectValidWindows] dict_metadata[{sig}]['dt'] must be > 0"
                )

            window_stride_ts = max(1, self._sec_to_ts_len(self.window_stride_sec, dt))

            if (int(w_idx) % window_stride_ts) != 0:
                return None

        # --------------------------------------------------------------
        # 1) Chunk-level validation (PURE: copy before masking)
        # --------------------------------------------------------------
        chunks = window.get("chunks") or {}
        valid_chunks_by_sig = {
            "input": defaultdict(int),
            "actuator": defaultdict(int),
        }

        new_chunks = dict(chunks)

        for role in ("input", "actuator"):
            new_role_chunks = []

            for ch in chunks.get(role, []) or []:
                ch2 = dict(ch)  # copy chunk dict
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

        # --------------------------------------------------------------
        # 2) Output-level validation (PURE: copy before masking)
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 3) Final decision
        # --------------------------------------------------------------
        keep = True
        if n_inputs_actuators < self.min_valid_inputs_actuators:
            keep = False
        if n_outputs_valid < self.min_valid_outputs:
            keep = False

        logger.debug(
            "win=%s shot=%s: "
            "valid_inputs_actuators=%d, valid_outputs=%d "
            "(min_x=%d, min_y=%d) → %s",
            w_idx,
            shot_id,
            n_inputs_actuators,
            n_outputs_valid,
            self.min_valid_inputs_actuators,
            self.min_valid_outputs,
            "KEEP" if keep else "DROP",
        )

        if not keep:
            return None

        # Return a masked COPY (PURE)
        out = dict(window)
        out["chunks"] = new_chunks
        out["output"] = output2
        return out
