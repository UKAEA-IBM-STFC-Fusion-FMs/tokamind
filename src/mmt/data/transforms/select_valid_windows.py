from __future__ import annotations

from typing import Any, Dict, Tuple, Hashable
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.SelectValidWindows")


class SelectValidWindowsTransform:
    """
    Select (or drop) a single window based on validity criteria at *chunk* level,
    with an optional additional stride in time between *kept* windows.

    This transform is designed to run **after** ChunkWindowsTransform, as part
    of the model_transform chain inside TaskModelTransformWrapper. It operates
    on a **single window dict** and returns either:

      - the window (with some signals masked to None), or
      - None, if the window should be dropped entirely.

    Expected input (after ChunkWindowsTransform)
    --------------------------------------------
    A window dict with at least:

        {
            "shot_id": ...,
            "window_index": ...,
            "t_cut": float,       # window cut time (seconds, relative)
            "input_length": float,
            "stride": float,      # stride used by TaskModelTransformWrapper (informative)
            "chunks": {
                "input":    [chunk_dict, ...],
                "actuator": [chunk_dict, ...],
            },
            "output": {
                var_name: {"values": np.ndarray, ...},
                ...
            },
            ...
        }

    Each chunk_dict has:

        {
            "role": "input" | "actuator",
            "chunk_index_in_window": int,
            "chunk_index_global": int | None,
            "chunk_start_time": float,  # optional, for debugging
            "signals": {
                var_name: np.ndarray,  # time axis is the chunk
                ...
            },
        }

    Behaviour (per window)
    ----------------------
    The goal is to:

      1. Optionally subsample windows in time at *window* level (via
         `window_stride_sec`), dropping windows that are too close in t_cut
         to the last kept window for the same shot.

      2. Apply chunk-level masking and signal validity checks, and decide
         whether the window is worth keeping based on:

         • number of valid input/actuator signals
         • number of valid output signals

    The steps are:

    0. Optional window-level subsampling (window_stride_sec)
       -----------------------------------------------------
       If `window_stride_sec` is not None, the transform enforces a minimum
       time separation between *kept* windows within each shot:

         • Windows are assumed to be processed in (shot_id, window_index)
           order, with strictly increasing t_cut within a shot.

         • For each shot_id, the transform keeps track of the last kept
           t_cut and window_index.

         • For a new window:

             - If this looks like a new pass / epoch over the same shot
               (window_index <= last_window_index), the state for that
               shot_id is reset.

             - Otherwise, if (t_cut - last_kept_t_cut) < window_stride_sec,
               the window is **dropped early** (returns None immediately).

             - If kept, the last_kept_t_cut and last_window_index are
               updated for that shot.

       This is used to aggressively reduce the number of windows per shot
       for training, without modifying the baseline TaskModelTransformWrapper
       or the underlying task definition. For evaluation, set
       `window_stride_sec = None` so that *all* windows are kept.

    1. Chunk-level masking for inputs/actuators
       ----------------------------------------
       For all chunks under "input" and "actuator":
         • values is None or empty      → treated as missing.
         • if accept_na == False:
               any NaN / ±inf present  → the signal is masked in that chunk
                                        (signals[var_name] = None).
           accept_na == True:
               - all non-finite        → masked (treated as missing);
               - mixed finite/nonfinite→ kept as-is (including NaNs).
         • all finite, non-empty array → kept and counted as a "valid chunk"
                                        for that (role, signal_name).

       This produces, for each role ("input"/"actuator") and signal_name,
       a count of valid chunks: valid_chunks_by_sig[role][signal_name].

    2. Per-signal chunk threshold
       --------------------------
       For each role in {"input", "actuator"} and each signal_name:
         • if number of valid chunks < min_valid_chunks:
               → the signal is considered invalid for the entire window:
                 signals[var_name] = None in *all* chunks of that role.

       After this step, a signal counts as "present" on the X-side only if it
       has at least `min_valid_chunks` valid chunks.

    3. Window-level counts for inputs + actuators
       ------------------------------------------
       Count how many (role, signal_name) pairs still have at least one
       non-None value across all chunks. This is:

           n_inputs_actuators

       This number must be >= min_valid_inputs_actuators for the window to
       be considered valid.

    4. Outputs (not chunked)
       ---------------------
       For each output variable:
         • values is None / empty / all NaN/inf → treated as missing
           (entry["values"] = None).
         • if accept_na == False:
               any NaN / ±inf  → entire signal masked (entry["values"] = None).
           accept_na == True:
               - all non-finite → masked as missing;
               - mixed          → kept as-is (including NaNs).
         • all finite          → kept and counted as a valid output.

       This gives:

           n_outputs_valid

    5. Drop / keep decision
       --------------------
       If either:
           n_inputs_actuators < min_valid_inputs_actuators
           OR
           n_outputs_valid    < min_valid_outputs

       then the window is **dropped** (the transform returns None).
       Otherwise, the window is kept and returned (with some signals possibly
       masked to None at chunk or output level).

    Parameters
    ----------
    min_valid_inputs_actuators:
        Minimum number of distinct (role, signal_name) pairs across "input"
        and "actuator" that must remain valid after chunk-level masking and
        min_valid_chunks enforcement.

    min_valid_chunks:
        Minimum number of valid chunks per (role, signal_name). Signals that
        do not reach this threshold are masked to None in all chunks.

    min_valid_outputs:
        Minimum number of valid output signals (after masking) required for
        the window to be kept.

    accept_na:
        Controls how mixed finite / non-finite arrays are treated.

        - If False (default):
            Any NaN or ±inf in a signal causes the entire signal to be masked
            (treated as missing) for that chunk or output.

        - If True:
            Only signals that are entirely non-finite (all NaN/inf) are masked.
            Mixed finite/non-finite arrays are kept as-is and may contain NaNs.
            Use this only if downstream encoders / models can cope with NaNs.

    window_stride_sec:
        Optional minimum time separation, in seconds, between *kept* windows
        for the same shot. If None (default), no additional subsampling is
        performed and every window that passes the validity checks is kept.

        If set to a positive float, windows are processed in temporal order
        (t_cut increasing) and only windows whose `t_cut` is at least
        `window_stride_sec` after the last *kept* window for that `shot_id`
        are retained. This is typically used in training configs to reduce
        the number of windows per shot, while evaluation configs can set
        window_stride_sec = null to keep all windows.
    """

    def __init__(
        self,
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

        # Window-level subsampling configuration
        self.window_stride_sec = (
            float(window_stride_sec) if window_stride_sec is not None else None
        )
        # Per-shot state: shot_key -> (last_t_cut, last_window_index)
        self._last_kept_by_shot: Dict[Hashable, Tuple[float, int | None]] = {}

    # ------------------------------------------------------------------ helpers
    def _mask_if_bad(self, values: Any) -> Tuple[bool, Any]:
        """
        Decide whether a signal should be masked based on its values and
        the `accept_na` policy.

        Returns
        -------
        (mask, new_values)
        mask = True  → caller should treat the signal as missing.
        mask = False → new_values is the (possibly converted) array.
        """
        if values is None:
            return True, None

        arr = np.asarray(values)
        if arr.size == 0:
            return True, None

        finite = np.isfinite(arr)

        # Entirely NaN/inf → always treat as missing
        if not finite.any():
            return True, None

        # Mixed finite / non-finite:
        if not finite.all():
            if not self.accept_na:
                # strict mode: any NaN/inf invalidates the whole signal
                return True, None
            # relaxed mode: keep as-is (NaNs remain in the array)
            return False, arr

        # All finite and non-empty: keep as-is
        return False, arr

    # ------------------------------------------------------------------ main API
    def __call__(self, window: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """
        Apply optional window-level subsampling, then chunk-level masking and
        window selection to a single window.

        Parameters
        ----------
        window : Dict[str, Any] | None
            Window dict as produced by ChunkWindowsTransform. If None is
            passed, returns None directly (for compatibility with composed
            transforms).

        Returns
        -------
        window or None
            The possibly-masked window dict, or None if the window does not
            meet the stride and validity criteria and should be dropped.
        """
        if window is None:
            return None

        # --- safe extraction of window fields ---------------------------------------
        shot_id = window.get("shot_id")
        if shot_id is not None:
            shot_id = int(shot_id)

        w_idx = window.get("window_index")
        if w_idx is not None:
            w_idx = int(w_idx)

        t_cut = window.get("t_cut")
        if t_cut is not None:
            t_cut = float(t_cut)

        # ---------------------------------------------------------------------------
        if self.window_stride_sec is not None:
            if t_cut is None:
                raise KeyError(
                    "SelectValidWindowsTransform(window_stride_sec=...) "
                    "requires window['t_cut'] to be present."
                )

            key: Hashable = shot_id if shot_id is not None else "__global__"

            last = self._last_kept_by_shot.get(key)
            if last is not None:
                last_t_cut, last_w_idx = last

                # Detect window index reset
                if (
                    (w_idx is not None)
                    and (last_w_idx is not None)
                    and (w_idx <= last_w_idx)
                ):
                    # new epoch → ignore previous t_cut for stride check
                    last_t_cut = None

                # Subsampling
                if (
                    last_t_cut is not None
                    and (t_cut - last_t_cut) < self.window_stride_sec
                ):
                    logger.debug(
                        "[SelectValidWindows] dropping window %s (shot %s) due to "
                        "window_stride_sec=%.4f (t_cut=%.6f last_t=%.6f)",
                        w_idx,
                        shot_id,
                        self.window_stride_sec,
                        t_cut,
                        last_t_cut,
                    )
                    return None

            # Keep
            self._last_kept_by_shot[key] = (
                t_cut,
                w_idx,
            )

        chunks_dict = window.get("chunks") or {}
        input_chunks = chunks_dict.get("input") or []
        act_chunks = chunks_dict.get("actuator") or []
        output_group = window.get("output") or {}

        # ------------------------------------------------------------------
        # 1) First pass: per-chunk masking + count valid chunks per signal
        # ------------------------------------------------------------------
        valid_chunks_by_sig = {
            "input": defaultdict(int),
            "actuator": defaultdict(int),
        }

        def _process_role_chunks(role: str, chunks) -> None:
            for ch in chunks:
                sigs = ch.get("signals") or {}
                for sig_name, val in list(sigs.items()):
                    should_mask, cleaned_val = self._mask_if_bad(val)
                    if should_mask:
                        sigs[sig_name] = None
                    else:
                        sigs[sig_name] = cleaned_val
                        valid_chunks_by_sig[role][sig_name] += 1

        _process_role_chunks("input", input_chunks)
        _process_role_chunks("actuator", act_chunks)

        # ------------------------------------------------------------------
        # 2) Enforce min_valid_chunks per signal
        # ------------------------------------------------------------------
        def _mask_signals_with_too_few_chunks(role: str, chunks) -> None:
            if not chunks:
                return
            all_sig_names = set()
            for ch in chunks:
                sigs = ch.get("signals") or {}
                all_sig_names.update(sigs.keys())

            for sig_name in all_sig_names:
                n_chunks = valid_chunks_by_sig[role].get(sig_name, 0)
                if n_chunks < self.min_valid_chunks:
                    for ch in chunks:
                        sigs = ch.get("signals") or {}
                        if sig_name in sigs:
                            sigs[sig_name] = None

        _mask_signals_with_too_few_chunks("input", input_chunks)
        _mask_signals_with_too_few_chunks("actuator", act_chunks)

        # ------------------------------------------------------------------
        # 3) Count how many input+actuator signals remain (window-level)
        # ------------------------------------------------------------------
        valid_x_signals = set()  # (role, name)

        def _collect_valid_signals(role: str, chunks) -> None:
            for ch in chunks:
                sigs = ch.get("signals") or {}
                for sig_name, val in sigs.items():
                    if val is not None:
                        valid_x_signals.add((role, sig_name))

        _collect_valid_signals("input", input_chunks)
        _collect_valid_signals("actuator", act_chunks)

        n_inputs_actuators = len(valid_x_signals)

        # ------------------------------------------------------------------
        # 4) Outputs: window-level masking and counting
        # ------------------------------------------------------------------
        n_outputs_valid = 0
        for name, entry in (output_group or {}).items():
            if not isinstance(entry, dict):
                # Allow shorthand "name: values"
                values = entry
                mask, new_val = self._mask_if_bad(values)
                if mask:
                    output_group[name] = {"values": None}
                else:
                    output_group[name] = {"values": new_val}
                    n_outputs_valid += 1
                continue

            values = entry.get("values", None)
            mask, new_val = self._mask_if_bad(values)
            if mask:
                entry["values"] = None
            else:
                entry["values"] = new_val
                n_outputs_valid += 1

        # ------------------------------------------------------------------
        # 5) Drop / keep decision for this window
        # ------------------------------------------------------------------
        keep = True
        if n_inputs_actuators < self.min_valid_inputs_actuators:
            keep = False
        if n_outputs_valid < self.min_valid_outputs:
            keep = False

        logger.debug(
            "[SelectValidWindows] window %s (shot %s): "
            "valid_inputs_actuators=%d, valid_outputs=%d → %s",
            w_idx,
            shot_id,
            n_inputs_actuators,
            n_outputs_valid,
            "KEEP" if keep else "DROP",
        )

        if not keep:
            return None

        return window
