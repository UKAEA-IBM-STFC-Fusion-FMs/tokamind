"""
TuneDCT3DTransform
==================

Tune DCT3D truncation parameters (keep_h, keep_w, keep_t) per (role, signal).

This is a pass-through transform designed to be inserted in the model_transform
chain after TrimChunksTransform and before EmbedChunksTransform. It observes each
kept window, accumulates per-window RMSE for each candidate configuration, and
later selects the smallest effective configuration under a per-role threshold.

Objective
---------
For each window:
    rmse_win = mean_over_chunks( sqrt(mean_over_finite((x_hat - x)^2)) )

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

Expected window format (v0)
---------------------------
- input/actuator are read from window["chunks"][role][i]["signals"][name]
- output is read from window["output"][name]["values"]

No fallback formats are supported on purpose.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from collections import defaultdict

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

    def _rmse_on_finite(self, x: np.ndarray, x_hat: np.ndarray) -> float | None:
        """
        RMSE computed only over finite entries of x.
        Returns None if there are no finite entries.
        """
        finite = np.isfinite(x)
        if not finite.any():
            return None

        # Compare against x_clean (finite entries unchanged; non-finite replaced with 0)
        x_clean = np.where(finite, x, 0.0)

        diff = (x_hat - x_clean).astype(np.float64, copy=False)
        diff_f = diff[finite]
        if diff_f.size == 0:
            return None

        return float(np.sqrt(np.mean(diff_f * diff_f)))

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        if "chunks" not in window:
            raise KeyError(
                "TuneDCT3D expects window['chunks'] (run after ChunkWindows/TrimChunks)"
            )

        # Per-window accumulators so each window counts once (mean over chunks).
        # win_sum[role][name][cfg] = sum_rmse_over_chunks
        # win_n[role][name][cfg]   = n_chunks_used
        win_sum: Dict[str, Dict[str, Dict[Tuple[int, int, int], float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        win_n: Dict[str, Dict[str, Dict[Tuple[int, int, int], int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # 1) input/actuator: read from chunks
        chunks = window.get("chunks") or {}
        for role in ("input", "actuator"):
            role_chunks = chunks.get(role) or []
            for ch in role_chunks:
                sigs = ch.get("signals") or {}
                for name, values in sigs.items():
                    if values is None:
                        continue
                    if self.signal_specs.get(role, name) is None:
                        continue

                    x = np.asarray(values)
                    if x.size == 0 or x.ndim not in (1, 2, 3):
                        continue

                    # store representative shape for effective_dim ranking
                    self.rep_shape.setdefault((role, name), tuple(x.shape))

                    finite = np.isfinite(x)
                    if not finite.any():
                        continue
                    x_clean = np.where(finite, x, 0.0)

                    for cfg in self.candidates:
                        codec = self._codecs[cfg]
                        z = codec.encode(x_clean)
                        x_hat = codec.decode(z, original_shape=x_clean.shape)

                        rmse = self._rmse_on_finite(x, x_hat)
                        if rmse is None or not np.isfinite(rmse):
                            continue

                        win_sum[role][name][cfg] += rmse
                        win_n[role][name][cfg] += 1

        # 2) outputs: read from window["output"][name]["values"]
        out_group = window.get("output")
        if isinstance(out_group, dict):
            for name, entry in out_group.items():
                if self.signal_specs.get("output", name) is None:
                    continue
                if not isinstance(entry, dict) or "values" not in entry:
                    raise TypeError(
                        f"TuneDCT3D expected window['output'][{name!r}] to be a dict with key 'values'"
                    )
                values = entry["values"]
                if values is None:
                    continue

                x = np.asarray(values)
                if x.size == 0 or x.ndim not in (1, 2, 3):
                    continue

                self.rep_shape.setdefault(("output", name), tuple(x.shape))

                finite = np.isfinite(x)
                if not finite.any():
                    continue
                x_clean = np.where(finite, x, 0.0)

                for cfg in self.candidates:
                    codec = self._codecs[cfg]
                    z = codec.encode(x_clean)
                    x_hat = codec.decode(z, original_shape=x_clean.shape)

                    rmse = self._rmse_on_finite(x, x_hat)
                    if rmse is None or not np.isfinite(rmse):
                        continue

                    win_sum["output"][name][cfg] += rmse
                    win_n["output"][name][cfg] += 1

        # 3) Commit: per-window mean over chunks, then accumulate per-window
        for role, by_sig in win_sum.items():
            for name, by_cfg in by_sig.items():
                for cfg, s in by_cfg.items():
                    n = win_n[role][name][cfg]
                    if n <= 0:
                        continue
                    rmse_win = float(s / n)

                    acc = self.stats[role][name][cfg]
                    acc[0] += rmse_win
                    acc[1] += 1.0

        return window

    def select_best(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
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

                scores: Dict[Tuple[int, int, int], float] = {}
                for cfg in ranked:
                    s, n = by_cfg[cfg]
                    scores[cfg] = float("inf") if n == 0 else float(s / n)

                chosen = None
                for cfg in ranked:
                    v = scores[cfg]
                    if np.isfinite(v) and v <= thr:
                        chosen = cfg
                        break
                if chosen is None:
                    chosen = min(ranked, key=lambda config: scores[config])

                values_shape = tuple(shape[:-1])
                H, W = infer_hw_from_values_shape(values_shape)
                T = int(shape[-1])

                out.setdefault(role, {})[name] = {
                    "keep_h": int(chosen[0]),
                    "keep_w": int(chosen[1]),
                    "keep_t": int(chosen[2]),
                    "eff_keep_h": int(min(int(chosen[0]), H)),
                    "eff_keep_w": int(min(int(chosen[1]), W)),
                    "eff_keep_t": int(min(int(chosen[2]), T)),
                    "rmse_mean_windows": float(scores[chosen]),
                    "threshold": float(thr),
                    "effective_dim": int(_effective_dim(shape, chosen)),
                    "rep_shape": shape,
                    "n_windows": int(by_cfg[chosen][1]),
                }

        return out
