from __future__ import annotations

from typing import Any, Dict, Tuple, Hashable
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger("mmt.SelectValidWindows")


class SelectValidWindowsTransform:
    """
    Select (or drop) a single window based on validity criteria at *chunk*
    level, with an optional additional stride between *kept* windows.

    This transform is designed to run **after** ChunkWindowsTransform, as part
    of the `model_transform` chain inside `TaskModelTransformWrapper`. It
    operates on a **single window dict** and returns either:

      - the window (possibly with some signals masked to None), or
      - None, if the window should be dropped entirely.

    Expected input (after ChunkWindowsTransform)
    --------------------------------------------
    A window dict with at least:

        window = {
            "shot_id": Hashable,
            "window_index": int,
            "t_cut": float,          # center time of the window
            "stride": float,         # base time step between consecutive windows
            "chunks": {
                "input":    [chunk_dict, ...],
                "actuator": [chunk_dict, ...],
            },
            "output": {
                <signal_name>: {"values": array_like} or array_like,
                ...
            },
        }

    Chunk dictionaries are expected to have the form:

        chunk = {
            "chunk_start_sample": int,     # global sample index on shot timeline
            "signals": {
                <signal_name>: array_like or None,
                ...
            },
        }

    High-level behaviour
    --------------------
    1. Optional window-level subsampling:
       If `window_stride_sec` is not None, windows are subsampled *per shot*
       so that the difference in `window_index` between consecutive **kept**
       windows is at least:

           steps_per_window = round(window_stride_sec / stride)

       where `stride` is the base time step (in seconds) used by
       `TaskModelTransformWrapper` to generate `t_cuts`. This makes
       subsampling robust to small floating-point jitter in `t_cut`, and
       aligns the kept windows to the same underlying grid as chunks.


    2. Chunk-level masking and validity:
       For all chunks under "input" and "actuator":

         • if values is None or empty → signal is treated as missing.
         • values are converted to numpy arrays; NaN/inf handling depends on
           `accept_na`:

             - accept_na = False:
                   any NaN/inf → the entire signal is masked (treated missing)
             - accept_na = True:
                   NaNs are allowed to remain; only the case where *all*
                   values are non-finite causes the signal to be masked.

       We count, for each role ("input", "actuator"), how many chunks contain
       at least `min_valid_chunks` valid entries for a given signal name.
       All such signal names across input+actuator are pooled into a single
       set `valid_x_signals`, and we define:

           n_inputs_actuators = len(valid_x_signals)

    3. Output-level masking and validity:
       For each entry under window["output"]:

           - if the entry is not a dict, it is treated as raw `values`;
           - if dict, we read `entry["values"]`.

       The same `_mask_if_bad` logic is used. A signal is counted as a valid
       output if it is not masked. The total is:

           n_outputs_valid

    4. Final decision:
       The window is **dropped** (returns None) if either:

           n_inputs_actuators < min_valid_inputs_actuators
           OR
           n_outputs_valid    < min_valid_outputs

       Otherwise, the window is kept and returned, with any masked signals
       set to None in the chunk/output structures.

    Parameters
    ----------
    min_valid_inputs_actuators : int, default=1
        Minimum number of distinct signals across input+actuator that must
        have at least `min_valid_chunks` valid chunks.
    min_valid_chunks : int, default=1
        Minimum number of valid chunks a signal must appear in, to be
        counted as "valid" for inputs/actuators.
    min_valid_outputs : int, default=1
        Minimum number of valid output signals required to keep the window.
    accept_na : bool, default=False
        If False, any NaN/inf in a non-empty array invalidates the entire
        signal. If True, NaNs are allowed as long as there is at least one
        finite value.
    window_stride_sec : float or None, default=None
        If not None, windows are subsampled in time per shot. Subsampling is
        implemented via an index-based stride using the `stride` and
        `window_index` fields:

            steps_per_window = round(window_stride_sec / stride)

        Only windows whose `window_index` differs from the last kept window
        by at least `steps_per_window` are retained. If `stride` is missing
        or non-positive, a fallback time-based check on `t_cut` is used.
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

        # Window-level subsampling configuration (in seconds)
        self.window_stride_sec = (
            float(window_stride_sec) if window_stride_sec is not None else None
        )
        # Per-shot state: shot_key -> (last_t_cut, last_window_index)
        self._last_kept_by_shot: Dict[Hashable, Tuple[float | None, int | None]] = {}

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

        shot_id = window.get("shot_id")
        w_idx = window.get("window_index")
        t_cut = window.get("t_cut")  # kept only for logging / debugging
        stride = window.get("stride")  # base dt from TaskModelTransformWrapper

        # ------------------------------------------------------------------
        # 0) Optional window-level subsampling (index-based, using stride)
        # ------------------------------------------------------------------
        if self.window_stride_sec is not None:
            # We *require* stride and window_index when subsampling is requested
            if stride is None or stride <= 0.0:
                raise ValueError(
                    "[SelectValidWindows] window_stride_sec is set (%.4f), "
                    "but window['stride'] is %r for shot_id=%r, window_index=%r. "
                    "TaskModelTransformWrapper must provide a positive 'stride'."
                    % (self.window_stride_sec, stride, shot_id, w_idx)
                )
            if w_idx is None:
                raise ValueError(
                    "[SelectValidWindows] window_stride_sec is set (%.4f), "
                    "but window['window_index'] is None for shot_id=%r. "
                    "TaskModelTransformWrapper must provide 'window_index'."
                    % (self.window_stride_sec, shot_id)
                )

            key: Hashable = shot_id if shot_id is not None else "__global__"

            # Retrieve last kept info for this shot (if any)
            last_t_cut, last_w_idx = self._last_kept_by_shot.get(key, (None, None))

            # Detect epoch wrap / reset: if window_index goes backwards or repeats,
            # we reset the kept state for this shot.
            if last_w_idx is not None and w_idx <= last_w_idx:
                last_t_cut = None
                last_w_idx = None

            # Convert desired time stride into an integer number of base steps
            steps_per_window = max(
                1, int(round(self.window_stride_sec / float(stride)))
            )

            if (last_w_idx is not None) and ((w_idx - last_w_idx) < steps_per_window):
                return None

            # Keep this window and update state for this shot
            self._last_kept_by_shot[key] = (
                float(t_cut) if t_cut is not None else None,
                int(w_idx),
            )

        # ------------------------------------------------------------------
        # 1) First pass: per-chunk masking + count valid chunks per signal
        # ------------------------------------------------------------------
        input_chunks = (window.get("chunks") or {}).get("input") or []
        act_chunks = (window.get("chunks") or {}).get("actuator") or []

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
        # 2) Count valid input/actuator signals
        # ------------------------------------------------------------------
        valid_x_signals = set()
        for role in ("input", "actuator"):
            for sig_name, count in valid_chunks_by_sig[role].items():
                if count >= self.min_valid_chunks:
                    valid_x_signals.add(sig_name)

        n_inputs_actuators = len(valid_x_signals)

        # ------------------------------------------------------------------
        # 3) Outputs: window-level masking and counting
        # ------------------------------------------------------------------
        output_group = window.get("output") or {}
        n_outputs_valid = 0

        for name, entry in list(output_group.items()):
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
        # 4) Drop / keep decision for this window
        # ------------------------------------------------------------------
        keep = True
        if n_inputs_actuators < self.min_valid_inputs_actuators:
            keep = False
        if n_outputs_valid < self.min_valid_outputs:
            keep = False

        logger.info(
            "win %s (shot %s): valid_inputs_actuators=%d, valid_outputs=%d → %s",
            w_idx,
            shot_id,
            n_inputs_actuators,
            n_outputs_valid,
            "KEEP" if keep else "DROP",
        )

        if not keep:
            return None

        return window
