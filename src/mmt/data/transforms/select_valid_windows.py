from __future__ import annotations

from typing import Any, Dict, Tuple
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.SelectValidWindows")


class SelectValidWindowsTransform:
    """
    Select (or drop) a single window based on validity criteria at *chunk* level.

    This transform is designed to run **after** ChunkingTransform, as part of the
    model_transform chain inside TaskModelTransformWrapper. It operates on a
    **single window dict** and returns either:

      - the window (with some signals masked to None), or
      - None, if the window should be dropped entirely.

    Expected input (after ChunkingTransform)
    ----------------------------------------
    A window dict with at least:

        {
            "shot_id": ...,
            "window_index": ...,
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
    The goal is to **select valid windows** based on how many usable inputs /
    actuators / outputs they contain, while masking bad signals.

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
    """

    def __init__(
        self,
        min_valid_inputs_actuators: int = 1,
        min_valid_chunks: int = 1,
        min_valid_outputs: int = 1,
        accept_na: bool = False,
    ) -> None:
        self.min_valid_inputs_actuators = int(min_valid_inputs_actuators)
        self.min_valid_chunks = int(min_valid_chunks)
        self.min_valid_outputs = int(min_valid_outputs)
        self.accept_na = bool(accept_na)

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
        Apply chunk-level masking and window selection to a single window.

        Parameters
        ----------
        window : Dict[str, Any] | None
            Window dict as produced by ChunkingTransform. If None is passed,
            returns None directly (for compatibility with composed transforms).

        Returns
        -------
        window or None
            The possibly-masked window dict, or None if the window does not
            meet the validity criteria and should be dropped.
        """
        if window is None:
            return None

        shot_id = window.get("shot_id", None)
        w_idx = window.get("window_index", None)

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
                for name, val in list(sigs.items()):
                    mask, new_val = self._mask_if_bad(val)
                    if mask:
                        sigs[name] = None
                    else:
                        sigs[name] = new_val
                        valid_chunks_by_sig[role][name] += 1

        _process_role_chunks("input", input_chunks)
        _process_role_chunks("actuator", act_chunks)

        # ------------------------------------------------------------------
        # 2) Enforce min_valid_chunks per signal
        # ------------------------------------------------------------------
        def _mask_signals_with_too_few_chunks(role: str, chunks) -> None:
            if not chunks:
                return
            all_names = set()
            for ch in chunks:
                sigs = ch.get("signals") or {}
                all_names.update(sigs.keys())

            for name in all_names:
                n_chunks = valid_chunks_by_sig[role].get(name, 0)
                if n_chunks < self.min_valid_chunks:
                    for ch in chunks:
                        sigs = ch.get("signals") or {}
                        if name in sigs:
                            sigs[name] = None

        _mask_signals_with_too_few_chunks("input", input_chunks)
        _mask_signals_with_too_few_chunks("actuator", act_chunks)

        # ------------------------------------------------------------------
        # 3) Count how many input+actuator signals remain (window-level)
        # ------------------------------------------------------------------
        valid_x_signals = set()  # (role, name)

        def _collect_valid_signals(role: str, chunks) -> None:
            for ch in chunks:
                sigs = ch.get("signals") or {}
                for name, val in sigs.items():
                    if val is not None:
                        valid_x_signals.add((role, name))

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
