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
If `window_stride_sec` is set, the transform keeps windows such that the time between consecutive *kept* windows for
the same shot is at least `window_stride_sec`:

    keep window i if (t_cut_i - t_cut_last_kept) >= window_stride_sec

Notes:
- This logic is per-shot and requires minimal internal state (last kept t_cut).
- The stride rule is applied *after* validity checks; only kept windows update the last-kept state.

Validity
--------
- Chunk-level: input/actuator chunk values are masked (set to None) if invalid (NaN/inf/empty) according to
  `accept_nan`.
- Signals: input/actuator signals count as valid if they have at least `min_valid_chunks` valid chunks.
- Outputs: output signals count as valid if their values are valid (after masking).

Returns
-------
- window dict (masked copy) if thresholds are met and optional stride constraint passes
- None otherwise
"""

from __future__ import annotations

from typing import Any, Hashable, Union
from collections import defaultdict
import logging
import numpy as np

from mmt.constants import TIME_FLOAT_TOL


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.SelectValidWindows")


# ======================================================================================================================
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

    Attributes
    ----------
    min_valid_inputs_actuators : int
        Minimum valid number of inputs and actuators.
    min_valid_chunks : int
        Minimum valid number of chunks.
    min_valid_outputs : int
        Minimum valid number of outputs.
    accept_nan : bool
        Whether to accept NaN/inf/empty values.
    window_stride_sec : float | None
        Optional window stride in seconds.
    _last_tcut_and_widx_kept_by_shot : dict[Hashable, tuple[Union[float, None], Union[int, None]]]
        Supporting variable to hold last t_cut and last w_idx kept under a given shot ID.

    Methods
    -------
    _mask_if_bad(values)
        Mask values if considered bad (i.e., if empty values or not finite).
    _is_window_stride_satisfied(shot_id, w_idx, t_cut)
        Return True if the window satisfies the minimum stride spacing since the last kept window for the same shot.
    _commit_window_stride(shot_id, w_idx, t_cut)
        Record the window as kept, updating per-shot stride state for subsequent stride checks.
    __call__(window)
        Validate and mask a window, returning a cleaned copy or None if the window should be dropped.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        min_valid_inputs_actuators: int = 1,
        min_valid_chunks: int = 1,
        min_valid_outputs: int = 1,
        accept_nan: bool = False,
        window_stride_sec: float | None = None,
    ) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        min_valid_inputs_actuators : int
            Minimum valid number of inputs and actuators.
            Optional. Default: 1.
        min_valid_chunks : int
            Minimum valid number of chunks.
            Optional. Default: 1.
        min_valid_outputs : int
            Minimum valid number of outputs.
            Optional. Default: 1.
        accept_nan : bool
            Whether to accept NaN/inf/empty values.
            Optional. Default: False.
        window_stride_sec : float | None
            Window stride in seconds.
            Optional. Default: None.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes, as this is a callable class.

        """

        self.min_valid_inputs_actuators = int(min_valid_inputs_actuators)
        self.min_valid_chunks = int(min_valid_chunks)
        self.min_valid_outputs = int(min_valid_outputs)

        self.accept_nan = bool(accept_nan)
        self.window_stride_sec = float(window_stride_sec) if (window_stride_sec is not None) else None
        if (self.window_stride_sec is not None) and (self.window_stride_sec <= 0):
            raise ValueError("[SelectValidWindowsTransform] `window_stride_sec` must be > 0.")

        # Supporting variable: shot_key -> (last_kept_t_cut, last_kept_window_index)
        self._last_tcut_and_widx_kept_by_shot: dict[Hashable, tuple[Union[float, None], Union[int, None]]] = {}

    # ------------------------------------------------------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def _mask_if_bad(self, values: Any) -> tuple[bool, Union[np.ndarray, None]]:
        """
        Mask values if considered bad (i.e., if empty values or not finite).

        Parameters
        ----------
        values : Any
            Values to be processed.

        Returns
        -------
        tuple[bool, Union[np.ndarray, None]]
            Returns a tuple (mask: bool, cleaned: ndarray|None) where:
            - mask=True means treat as invalid → set cleaned to None
            - mask=False means treat as valid → set cleaned to `values`
            - REMARK: `self.accept_nan` controls whether partial NaNs are allowed

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
            if not self.accept_nan:
                return True, None

            return False, arr

        return False, arr

    # ------------------------------------------------------------------------------------------------------------------
    def _is_window_stride_satisfied(
        self, shot_id: int | str | None, w_idx: int | str | None, t_cut: float | None
    ) -> bool:
        """
        Return True if this window satisfies the minimum stride spacing, False if it should be dropped.

        A window passes if the time elapsed since the last kept window for the same shot is at least
        `self.window_stride_sec`. Returns True unconditionally when:
        - `window_stride_sec` is None (subsampling disabled), or
        - no previous window has been kept for this shot yet (first window is always kept).

        Window index is used to detect epoch resets: if `w_idx` goes backwards relative to the last kept window,
        the per-shot state is cleared and the window is treated as a first window.

        Parameters
        ----------
        shot_id : int | str | None
            Shot ID used to look up per-shot stride state.
        w_idx : int | str | None
            Window index within the shot, used to detect epoch resets.
        t_cut : float | None
            Cut time of the current window in seconds.

        Returns
        -------
        bool
            True if the stride constraint is satisfied (window should be kept),
            False if the window is too close to the previous kept window (window should be dropped).

        Raises
        ------
        ValueError
            If `shot_id` is None while `window_stride_sec` is set.
        ValueError
            If `t_cut` is None while `window_stride_sec` is set.

        """

        if self.window_stride_sec is None:
            return True

        if shot_id is None:
            raise ValueError("[SelectValidWindowsTransform] missing `shot_id` while `self.window_stride_sec` is set.")

        if t_cut is None:
            raise ValueError("[SelectValidWindowsTransform] missing `t_cut` while `self.window_stride_sec` is set.")

        last_t_cut, last_w_idx = self._last_tcut_and_widx_kept_by_shot.get(shot_id, (None, None))

        # If window indices go backwards (new epoch / reset), reset state for that shot
        if (last_w_idx is not None) and (w_idx is not None) and (int(w_idx) <= int(last_w_idx)):
            self._last_tcut_and_widx_kept_by_shot.pop(shot_id, None)
            last_t_cut, last_w_idx = None, None

        if last_t_cut is None:
            return True

        dt = float(t_cut) - float(last_t_cut)

        return dt >= (self.window_stride_sec - TIME_FLOAT_TOL)

    # ------------------------------------------------------------------------------------------------------------------
    def _commit_window_stride(self, shot_id: int | str | None, w_idx: int | str | None, t_cut: float | None) -> None:
        """
        Record `(t_cut, w_idx)` as the last kept window for `shot_id`, used by subsequent stride checks.

        Must be called only when a window has been accepted (after all validity checks pass). No-op if
        `window_stride_sec` is None, or if `shot_id` or `t_cut` are None.

        Parameters
        ----------
        shot_id : int | str | None
            Shot ID identifying which per-shot state to update.
        w_idx : int | str | None
            Window index of the kept window, stored for epoch-reset detection.
        t_cut : float | None
            Cut time of the kept window in seconds.

        Returns
        -------
        None

        """

        if self.window_stride_sec is None:
            return
        if (shot_id is None) or (t_cut is None):
            return

        self._last_tcut_and_widx_kept_by_shot[shot_id] = (
            float(t_cut),
            int(w_idx) if (w_idx is not None) else None,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(  # NOSONAR - Ignore cognitive complexity
        self, window: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """
        Validate and mask a window, returning a cleaned copy or None if the window should be dropped.

        Applies three sequential checks (with early-exit in non-debug mode):
        0. Stride check: `window` must be at least `window_stride_sec` after the last kept window for the same shot.
        1. Chunk-level: masks bad signal values; requires at least `min_valid_inputs_actuators` signals with
           >= `min_valid_chunks` valid chunks across input/actuator roles.
        2. Output-level: masks bad output values; requires at least `min_valid_outputs` valid output signals.

        A window is kept (stride state committed, masked copy returned) only if all three checks pass.
        In debug mode, all checks run even after a failure so the debug log captures the full validation picture.

        Parameters
        ----------
        window : dict[str, Any] | None
            Window to validate and mask. Passed through unchanged structure, with signal values masked in-place.

        Returns
        -------
        dict[str, Any] | None
            Masked copy of the window with updated "chunks" and "output" values, or None if the window is dropped.

        """

        if window is None:
            return None

        w_idx = window.get("window_index")
        shot_id = window.get("shot_id")
        t_cut = window.get("t_cut")

        NOT_IN_DEBUG_MODE = not logger.isEnabledFor(logging.DEBUG)  # noqa - Ignore lowercase warning

        # ..............................................................................................................
        # 0) Preliminary check: Optional subsampling — only among kept windows
        # ..............................................................................................................

        keep = True
        # Drop if stride not satisfied
        if not self._is_window_stride_satisfied(shot_id=shot_id, w_idx=w_idx, t_cut=t_cut):
            keep = False
            if NOT_IN_DEBUG_MODE:
                return None

        # ..............................................................................................................
        # 1) Chunk-level validation (copy before masking)
        # ..............................................................................................................

        chunks = window.get("chunks") or {}
        valid_chunks_by_sig = {"input": defaultdict(int), "actuator": defaultdict(int)}

        new_chunks = dict(chunks)
        for role in ("input", "actuator"):
            new_role_chunks = []
            for ch in chunks.get(role) or []:
                ch2: dict[str, Any] = dict(ch)  # type: ignore[arg-type]
                sigs = ch.get("signals") or {}

                sigs2: dict[str, Any] = {}
                for name, val in sigs.items():
                    mask, cleaned = self._mask_if_bad(values=val)
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

        # Validity decision
        if n_inputs_actuators < self.min_valid_inputs_actuators:
            keep = False
            if NOT_IN_DEBUG_MODE:
                return None

        # ..............................................................................................................
        # 2) Output-level validation (copy before masking)
        # ..............................................................................................................

        output = window.get("output") or {}
        output2: dict[str, Any] = {}
        n_outputs_valid = 0

        for name, entry in output.items():
            if not isinstance(entry, dict):
                mask, cleaned = self._mask_if_bad(values=entry)
                output2[name] = {"values": None if mask else cleaned}
                if not mask:
                    n_outputs_valid += 1
                continue

            entry2 = dict(entry)
            mask, cleaned = self._mask_if_bad(values=entry.get("values"))
            entry2["values"] = None if mask else cleaned
            output2[name] = entry2
            if not mask:
                n_outputs_valid += 1

        # Validity decision
        if n_outputs_valid < self.min_valid_outputs:
            keep = False
            if NOT_IN_DEBUG_MODE:
                return None

        # ..............................................................................................................

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
            "none" if self.window_stride_sec is None else f"{self.window_stride_sec:.6f}s",
            "KEEP" if keep else "DROP",
        )

        if not keep:
            return None

        # Commit stride state only on KEEP
        self._commit_window_stride(shot_id=shot_id, w_idx=w_idx, t_cut=t_cut)

        # Return a masked copy
        out = dict(window)
        out["chunks"] = new_chunks
        out["output"] = output2

        return out
