"""
TuneDCT3DTransform
==================

Terminal transform used to tune DCT3D embedding parameters.

This transform:
- observes window-level data (after Chunk → SelectValidWindows → TrimChunks),
- evaluates multiple DCT3D parameter combinations per signal and role,
- accumulates native-space reconstruction errors,
- does NOT forward any data downstream.

It is intended to be used only by the tune_dct3d script.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

import logging

logger = logging.getLogger("mmt.TuneDCT3D")


class TuneDCT3DTransform:
    """
    Terminal transform that accumulates reconstruction errors for different
    DCT3D configurations.

    Notes
    -----
    - This transform has side effects only.
    - It always returns `None`.
    - The caller is responsible for stopping iteration once budgets are reached.
    """

    def __init__(
        self,
        *,
        signal_specs,
        codecs,
        keep_h: Iterable[int],
        keep_w: Iterable[int],
        keep_t: Iterable[int],
        thresholds: Dict[str, float],
        max_coeffs: int | None = None,
    ):
        """
        Parameters
        ----------
        signal_specs
            Output of build_signal_specs(); used to determine signals, roles,
            modalities, and native shapes.

        codecs
            Dict[name, codec] built from signal_specs.

        keep_h / keep_w / keep_t
            Candidate truncation values for DCT3D.

        thresholds
            Role-specific RMSE thresholds, e.g.:
              { "input": 0.02, "actuator": 0.05, "output": 0.01 }

        max_coeffs
            Optional hard limit on keep_h * keep_w * keep_t. Combinations above
            this value are skipped.
        """
        self.signal_specs = signal_specs
        self.codecs = codecs
        self.thresholds = thresholds
        self.max_coeffs = max_coeffs

        # ------------------------------------------------------------
        # Build ordered candidate list (smallest first)
        # ------------------------------------------------------------
        candidates: List[Tuple[int, int, int]] = []
        for h, w, t in product(keep_h, keep_w, keep_t):
            n_coeffs = h * w * t
            if max_coeffs is not None and n_coeffs > max_coeffs:
                continue
            candidates.append((h, w, t))

        # Sort by compactness (primary objective)
        self.candidates = sorted(candidates, key=lambda x: x[0] * x[1] * x[2])

        if not self.candidates:
            raise ValueError("No valid DCT3D candidate configurations found.")

        # ------------------------------------------------------------
        # Accumulators:
        #   stats[role][signal_name][(h,w,t)] = [sum_sq_error, count]
        # ------------------------------------------------------------
        self.stats: Dict[str, Dict[str, Dict[Tuple[int, int, int], List[float]]]] = {}

        for spec in self.signal_specs.values():
            role = spec.role
            name = spec.name

            self.stats.setdefault(role, {})
            self.stats[role].setdefault(name, {})

            for cfg in self.candidates:
                self.stats[role][name][cfg] = [0.0, 0.0]

        logger.info(
            "Initialized TuneDCT3DTransform with %d candidate configs",
            len(self.candidates),
        )

    # ------------------------------------------------------------------
    # Transform API
    # ------------------------------------------------------------------

    def __call__(self, window: Dict[str, Any]) -> None:
        """
        Observe a single window and accumulate reconstruction errors.

        Parameters
        ----------
        window
            Dict emitted by TrimChunksTransform. Expected to contain:
              - window["input"], window["actuator"], window["output"]
              - each maps signal_name -> native NumPy array
        """
        for role in ("input", "actuator", "output"):
            if role not in window:
                continue

            data_by_signal = window[role]
            if not isinstance(data_by_signal, dict):
                continue

            for name, native_arr in data_by_signal.items():
                if name not in self.codecs:
                    continue

                codec = self.codecs[name]
                spec = self.signal_specs[name]

                # native_arr shape: (...), batch dim already collapsed
                native_arr = np.asarray(native_arr)

                for h, w, t in self.candidates:
                    # Encode → decode for this configuration
                    coeffs = codec.encode(
                        native_arr,
                        keep_h=h,
                        keep_w=w,
                        keep_t=t,
                    )
                    recon = codec.decode(coeffs, native_arr.shape)

                    # RMSE in native space
                    diff = recon - native_arr
                    rmse = float(np.sqrt(np.mean(diff**2)))

                    acc = self.stats[role][name][(h, w, t)]
                    acc[0] += rmse**2
                    acc[1] += 1

    # ------------------------------------------------------------------
    # Selection logic
    # ------------------------------------------------------------------

    def select_best(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Select the best configuration per (role, signal).

        Strategy
        --------
        - Iterate candidates in increasing size order
        - Pick the first configuration whose mean RMSE <= threshold
        - If none satisfies the threshold, pick the configuration with
          minimum mean RMSE

        Returns
        -------
        result : dict
            Nested dict:
              result[role][signal_name] = {
                  "keep_h": int,
                  "keep_w": int,
                  "keep_t": int,
                  "rmse": float,
                  "num_coeffs": int,
              }
        """
        result: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for role, signals in self.stats.items():
            role_threshold = self.thresholds.get(role, None)
            if role_threshold is None:
                raise KeyError(f"No RMSE threshold defined for role={role!r}")

            result.setdefault(role, {})

            for name, cfg_stats in signals.items():
                best_cfg = None
                best_rmse = float("inf")

                # First pass: smallest config under threshold
                for h, w, t in self.candidates:
                    sum_sq, count = cfg_stats[(h, w, t)]
                    if count == 0:
                        continue

                    mean_rmse = np.sqrt(sum_sq / count)
                    if mean_rmse <= role_threshold:
                        best_cfg = (h, w, t, mean_rmse)
                        break

                    if mean_rmse < best_rmse:
                        best_rmse = mean_rmse
                        best_cfg = (h, w, t, mean_rmse)

                if best_cfg is None:
                    raise RuntimeError(f"No statistics collected for {role}:{name}")

                h, w, t, rmse = best_cfg
                result[role][name] = {
                    "keep_h": h,
                    "keep_w": w,
                    "keep_t": t,
                    "rmse": float(rmse),
                    "num_coeffs": int(h * w * t),
                }

        return result
