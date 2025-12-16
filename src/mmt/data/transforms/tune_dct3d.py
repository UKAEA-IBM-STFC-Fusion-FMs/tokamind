"""
TuneDCT3DTransform
==================

Tune DCT3D truncation parameters (keep_h, keep_w, keep_t) per (role, signal).

This is a pass-through transform designed to be inserted in the model_transform
chain after TrimChunksTransform. It observes each kept window, accumulates
per-window RMSE for each candidate configuration, and later selects the smallest
effective configuration under a per-role threshold.

Objective
---------
For each window:
    rmse_win = sqrt(mean((x_hat - x)^2))

Aggregate per (role, signal, cfg) with equal weight per window:
    score(cfg) = mean_windows(rmse_win)

Selection
---------
Candidates are ranked by effective dimension:

    D_eff = min(keep_h, H) * min(keep_w, W) * min(keep_t, T)

where native shapes map to (H, W, T) using the same convention as DCT3D:
  (T,)    -> values_shape=()      -> H=1, W=1
  (C, T)  -> values_shape=(C,)    -> H=C, W=1
  (H,W,T) -> values_shape=(H, W)  -> H=H, W=W

Then for each (role, signal):
  - pick the first cfg with score(cfg) <= threshold[role]
  - if none, pick cfg with minimum score(cfg)

Expected window format
----------------------
This transform expects the same "signal entry" format used in the preprocessing
pipeline:

    window[role][name] == {"values": np.ndarray or array_like}

No fallback formats are supported on purpose.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.embeddings.dct3d_codec import DCT3DCodec
from mmt.data.embeddings.codec_utils import infer_hw_from_values_shape

logger = logging.getLogger("mmt.TuneDCT3D")


def _effective_dim(native_shape: Tuple[int, ...], cfg: Tuple[int, int, int]) -> int:
    """
    Effective latent dimension for native shape and keep config.
    native_shape must be (T,), (C,T), or (H,W,T).
    """
    if len(native_shape) not in (1, 2, 3):
        raise ValueError(f"Unsupported native_shape={native_shape!r}")

    values_shape = tuple(native_shape[:-1])  # exclude time axis
    H, W = infer_hw_from_values_shape(values_shape)
    T = int(native_shape[-1])

    keep_h, keep_w, keep_t = cfg
    return min(keep_h, H) * min(keep_w, W) * min(keep_t, T)


class TuneDCT3DTransform:
    """
    Tune DCT3D truncation parameters (keep_h, keep_w, keep_t) per signal.

    This is a **pass-through, stateful transform** designed to run inside
    `ComposeTransforms`, typically after `TrimChunksTransform`. It observes
    each kept window, evaluates multiple DCT3D configurations for every
    (role, signal), and accumulates reconstruction error statistics.

    Input
    -----
    A single window dict with native (non-embedded) signal values:

        window = {
            "input":    { signal_name: np.ndarray, ... },
            "actuator": { signal_name: np.ndarray, ... },
            "output":   { signal_name: np.ndarray, ... },
            ...
        }

    Arrays are expected to have shape:
        (T,), (C, T), or (H, W, T), with time on the last axis.

    Behaviour
    ---------
    For each window and each signal:
      • apply all candidate DCT3D keep configurations,
      • compute per-window RMSE between original and reconstructed signal,
      • aggregate RMSE with equal weight per window.

    The transform does **not** modify or drop windows; it returns the input
    window unchanged.

    Output
    ------
    The window dict, unchanged (pass-through).

    Final tuning results are retrieved by calling `select_best()`, which
    returns the selected DCT3D configuration per (role, signal) based on
    per-role error thresholds and minimal effective latent dimension.
    """

    def __init__(
        self,
        *,
        signal_specs: SignalSpecRegistry,
        keep_h: Iterable[int],
        keep_w: Iterable[int],
        keep_t: Iterable[int],
        thresholds: Mapping[str, float],
        max_coeffs: int | None = None,
    ) -> None:
        self.signal_specs = signal_specs
        self.thresholds = {k: float(v) for k, v in thresholds.items()}

        # candidate keep configs
        self.candidates: List[Tuple[int, int, int]] = []
        for h, w, t in product(keep_h, keep_w, keep_t):
            h, w, t = int(h), int(w), int(t)
            if h <= 0 or w <= 0 or t <= 0:
                continue
            if max_coeffs is not None and (h * w * t) > int(max_coeffs):
                continue
            self.candidates.append((h, w, t))

        if not self.candidates:
            raise ValueError("TuneDCT3D: no candidate configurations found")

        # one codec per candidate
        self._codecs: Dict[Tuple[int, int, int], DCT3DCodec] = {
            cfg: DCT3DCodec(cfg[0], cfg[1], cfg[2]) for cfg in self.candidates
        }

        # stats[role][name][cfg] = [sum_rmse, n_windows]
        self.stats: Dict[str, Dict[str, Dict[Tuple[int, int, int], List[float]]]] = {}
        self.rep_shape: Dict[Tuple[str, str], Tuple[int, ...]] = {}

        # pre-create slots for all specs (ignore original encoder_name on purpose)
        for spec in self.signal_specs.specs:
            role, name = spec.role, spec.name
            self.stats.setdefault(role, {}).setdefault(name, {})
            for cfg in self.candidates:
                self.stats[role][name][cfg] = [0.0, 0.0]

        logger.info("TuneDCT3D initialized | candidates=%d", len(self.candidates))

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        for role in ("input", "actuator", "output"):
            group = window.get(role)
            if not isinstance(group, dict):
                continue

            for name, entry in group.items():
                if self.signal_specs.get(role, name) is None:
                    continue

                if not isinstance(entry, dict) or "values" not in entry:
                    raise TypeError(
                        f"TuneDCT3D expected window[{role}][{name}] to be a dict with key 'values'"
                    )

                x = np.asarray(entry["values"])
                if x.size == 0 or x.ndim not in (1, 2, 3):
                    continue

                # Skip non-finite windows (prevents rmse=nan poisoning)
                if not np.isfinite(x).all():
                    continue

                self.rep_shape.setdefault((role, name), tuple(x.shape))

                for cfg in self.candidates:
                    codec = self._codecs[cfg]
                    z = codec.encode(x)
                    x_hat = codec.decode(z, original_shape=x.shape)

                    # Also skip if codec produces non-finite output (rare, but safe)
                    if not np.isfinite(x_hat).all():
                        continue

                    diff = (x_hat - x).astype(np.float64, copy=False)
                    rmse = float(np.sqrt(np.mean(diff * diff)))

                    acc = self.stats[role][name][cfg]
                    acc[0] += rmse
                    acc[1] += 1.0

        return window

    def select_best(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Return per-(role, signal) selected keep config and score.
        Only includes signals that were actually observed (rep_shape exists).
        """
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for role, by_sig in self.stats.items():
            if role not in self.thresholds:
                raise KeyError(f"TuneDCT3D missing threshold for role={role!r}")
            thr = float(self.thresholds[role])

            for name, by_cfg in by_sig.items():
                shape = self.rep_shape.get((role, name))
                if shape is None:
                    continue

                ranked = sorted(
                    self.candidates, key=lambda c: (_effective_dim(shape, c), c)
                )

                # Compute score = mean RMSE over windows
                scores: Dict[Tuple[int, int, int], float] = {}
                for cfg in ranked:
                    s, n = by_cfg[cfg]
                    if n == 0:
                        scores[cfg] = float("inf")
                    else:
                        scores[cfg] = float(s / n)

                # Choose smallest under threshold (ignore inf/nan)
                chosen = None
                for cfg in ranked:
                    v = scores[cfg]
                    if np.isfinite(v) and v <= thr:
                        chosen = cfg
                        break

                if chosen is None:
                    chosen = min(ranked, key=lambda config: scores[config])

                # Also compute effective keep_* (clipped by shape)
                values_shape = tuple(shape[:-1])
                H, W = infer_hw_from_values_shape(values_shape)
                T = int(shape[-1])
                eff_keep_h = min(int(chosen[0]), H)
                eff_keep_w = min(int(chosen[1]), W)
                eff_keep_t = min(int(chosen[2]), T)

                out.setdefault(role, {})[name] = {
                    # requested keep (from search space)
                    "keep_h": int(chosen[0]),
                    "keep_w": int(chosen[1]),
                    "keep_t": int(chosen[2]),
                    # effective keep (actually used given the signal shape)
                    "eff_keep_h": int(eff_keep_h),
                    "eff_keep_w": int(eff_keep_w),
                    "eff_keep_t": int(eff_keep_t),
                    "rmse_mean_windows": float(scores[chosen]),
                    "threshold": float(thr),
                    "effective_dim": int(_effective_dim(shape, chosen)),
                    "rep_shape": shape,
                    "n_windows": int(by_cfg[chosen][1]),
                }

        return out
