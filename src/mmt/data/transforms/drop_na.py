from __future__ import annotations

from typing import Any, Dict, Tuple
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.DropNa")


class DropNaChunksTransform:
    """
    Drop or mask a single window based on NaNs / non-finite values at *chunk* level.

    This transform is designed to run **after** ChunkingTransform, as part of the
    model_transform chain inside TaskModelTransformWrapper. It operates on a
    **single window dict** and returns either:

      - the window (with some signals masked), or
      - None, if the window should be dropped.

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
    1. For all chunks under "input" and "actuator":
       - For each signal:
         * values is None or empty      → treated as missing.
         * any NaN / ±inf present       → the signal is masked in that chunk
                                         (signals[var_name] = None).
         * all finite, non-empty array  → kept and counted as a "valid chunk"
                                         for that (role, signal_name).

       This produces, for each role ("input"/"actuator") and signal_name,
       a count of valid chunks: valid_chunks_by_sig[role][signal_name].

    2. Per-signal chunk threshold:
       - For each role in {"input", "actuator"} and each signal_name:
         * if number of valid chunks < min_valid_chunks:
             → the signal is considered invalid for the entire window:
               signals[var_name] = None in *all* chunks of that role.

       After this step, a signal counts as "present" on the X-side only if it
       has at least `min_valid_chunks` valid chunks.

    3. Window-level counts for inputs + actuators:
       - Count how many (role, signal_name) pairs still have at least one
         non-None value across all chunks. This is:

            n_inputs_actuators

    4. Outputs (not chunked):
       - For each output variable:
         * values is None / empty / all NaN/inf → treated as missing
           (entry["values"] = None).
         * any NaN / ±inf → entire signal masked (entry["values"] = None).
         * all finite    → kept and counted as a valid output.

       This gives:

            n_outputs_valid

    5. Drop / keep decision:
       - If:
           n_inputs_actuators < min_valid_inputs_actuators
           OR
           n_outputs_valid    < min_valid_outputs
         → the window is DROPPED (return None).
       - Otherwise, the window is kept and returned, with some signals masked
         to None at chunk or output level.
    """

    def __init__(
        self,
        min_valid_inputs_actuators: int = 1,
        min_valid_chunks: int = 1,
        min_valid_outputs: int = 1,
    ) -> None:
        self.min_valid_inputs_actuators = int(min_valid_inputs_actuators)
        self.min_valid_chunks = int(min_valid_chunks)
        self.min_valid_outputs = int(min_valid_outputs)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mask_if_bad(values: Any) -> Tuple[bool, Any]:
        """
        Decide whether a signal should be masked based on its values.

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

        # Entirely NaN/inf → treat as missing
        if not finite.any():
            return True, None

        # Mixed finite / non-finite: simple policy for now: mask whole signal
        if not finite.all():
            return True, None

        # All finite and non-empty: keep as-is
        return False, arr

    # ------------------------------------------------------------------ main API
    def __call__(self, window: Dict[str, Any] | None) -> Dict[str, Any] | None:
        """
        Apply chunk-level NaN masking and window dropping to a single window.

        Parameters
        ----------
        window : Dict[str, Any] | None
            Window dict as produced by ChunkingTransform. If None is passed,
            returns None directly (for compatibility with composed transforms).
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
            "[DropNaChunks] window %s (shot %s): "
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
