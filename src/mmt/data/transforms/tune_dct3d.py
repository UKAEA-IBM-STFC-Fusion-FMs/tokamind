"""
TuneDCT3DTransform
==================

Streaming evaluator used by `scripts_mast/run_tune_dct3d.py` to pick per-signal
DCT3D truncation parameters (keep_h, keep_w, keep_t).

This transform **does not** modify the window payload for downstream training.
It only accumulates reconstruction *explained energy* statistics over streamed
windows and later selects a best configuration per (role, signal).

Explained energy
----------------
For a signal chunk `x` (with non-finite entries replaced by 0) and a truncated
DCT3D code `z` (the kept DCT coefficients), the explained energy is:

    explained_energy = sum(z^2) / sum(x^2)

Because the codec uses an orthonormal DCT ("ortho" normalization), this ratio is
equivalent to the fraction of signal energy preserved by the kept coefficients.

For chunk-based roles (input/actuator), we compute the metric per chunk and then
average across chunks so each window contributes one value.

Selection
---------
Candidates are ranked by effective dimension:

    D_eff = min(keep_h, H) * min(keep_w, W) * min(keep_t, T)

Budget
------
Optionally, provide a per-role coefficient budget `max_budget` (effective
dimension cap). Budgets are expressed in the same units as D_eff, i.e., the
number of kept coefficients after clamping keep_h/keep_w/keep_t to the native
signal shape.

For each (role, signal), we pick the smallest candidate (by D_eff) whose mean
explained_energy across windows is >= target[role].

If a role-specific `max_budget` is provided, we only consider candidates with
D_eff <= max_budget[role]. If no candidates fit within the budget, we fall back
to the smallest candidate overall.

If none meet the target (within budget), we pick the candidate with the highest
mean explained_energy (ties broken by smaller D_eff).

Expected window format (v0)
---------------------------
- input/actuator are read from window["chunks"][role][i]["signals"][name]
- output is read from window["output"][name]["values"]

Roles
-----
By default, the tuner evaluates all roles: ("input", "actuator", "output").
You can restrict computation by passing a subset via `roles`, e.g.:

    TuneDCT3DTransform(..., roles=("output",))

Only the specified roles are processed and returned by `select_best()`.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.embeddings.dct3d_codec import DCT3DCodec
from mmt.data.embeddings.codec_utils import infer_hw_from_values_shape

logger = logging.getLogger("mmt.TuneDCT3D")


def _effective_dim(native_shape: Tuple[int, ...], cfg: Tuple[int, int, int]) -> int:
    """Effective latent dimension for native shape and keep config.

    Parameters
    ----------
    native_shape:
        Must be (T,), (C, T), or (H, W, T).
    cfg:
        (keep_h, keep_w, keep_t)
    """
    if len(native_shape) not in (1, 2, 3):
        raise ValueError(f"Unsupported native_shape={native_shape!r}")

    values_shape = tuple(native_shape[:-1])  # exclude time axis
    H, W = infer_hw_from_values_shape(values_shape)
    T = int(native_shape[-1])

    keep_h, keep_w, keep_t = cfg
    return min(keep_h, H) * min(keep_w, W) * min(keep_t, T)


def _explained_energy_from_total(total_energy: float, z: np.ndarray) -> float:
    """Compute explained energy fraction given total energy and truncated code.

    Parameters
    ----------
    total_energy:
        Sum of squares of the cleaned signal (float). Must be >= 0.
    z:
        Flattened cropped DCT coefficient vector returned by the codec.

    Returns a value in [0, 1]. If total_energy is 0 (all-zeros signal), returns 1.
    """
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        return 1.0

    z64 = np.asarray(z, dtype=np.float64).reshape(-1)
    kept = float(np.sum(z64 * z64))
    if not np.isfinite(kept) or kept <= 0.0:
        return 0.0

    # Numerical guard: kept energy should not exceed total energy.
    if kept > total_energy:
        kept = total_energy

    frac = kept / total_energy
    # Clamp to [0, 1] for stability.
    if frac < 0.0:
        return 0.0
    if frac > 1.0:
        return 1.0
    return float(frac)


class TuneDCT3DTransform:
    def __init__(
        self,
        *,
        signal_specs: SignalSpecRegistry,
        keep_h: Iterable[int],
        keep_w: Iterable[int],
        keep_t: Iterable[int],
        thresholds: Mapping[str, float],
        roles: Iterable[str] = ("input", "actuator", "output"),
        max_budget: int | Mapping[str, int] | None = None,
    ) -> None:
        """Create a DCT3D tuner.

        Parameters
        ----------
        signal_specs:
            Registry of all signals for the current task.
        keep_h, keep_w, keep_t:
            Candidate truncation values along H/W/T.
        thresholds:
            Role-specific *explained energy targets*, e.g.
            {"input": 0.99, "output": 0.98, ...}. Values should be in [0, 1].
        roles:
            Which roles to tune. Subset of {"input", "actuator", "output"}.
            Default: all three.
        max_budget:
            Optional coefficient budget (effective dimension cap). May be a
            single int (applied to all tuned roles) or a mapping per role, e.g.
            {"input": 4096, "output": 16384}. Candidates with
            D_eff > max_budget[role] are ignored during selection (and skipped
            during evaluation for speed).
        """
        self.signal_specs = signal_specs

        # Normalize + validate roles.
        allowed = {"input", "actuator", "output"}
        roles_norm: List[str] = []
        for r in roles:
            rr = str(r).strip()
            if not rr:
                continue
            if rr not in allowed:
                raise ValueError(
                    f"TuneDCT3D: invalid role {rr!r}. Allowed roles: {sorted(allowed)}"
                )
            if rr not in roles_norm:
                roles_norm.append(rr)
        if not roles_norm:
            raise ValueError("TuneDCT3D: roles must contain at least one role")
        self.roles = tuple(roles_norm)

        # Role targets (only for tuned roles).
        self.targets = {k: float(v) for k, v in thresholds.items() if k in self.roles}

        self.max_budget: Dict[str, int | None] = {r: None for r in self.roles}
        if max_budget is not None:
            if isinstance(max_budget, Mapping):
                for r in self.roles:
                    b = max_budget.get(r)
                    if b is None:
                        continue
                    b_int = int(b)
                    if b_int <= 0:
                        raise ValueError(
                            f"TuneDCT3D: max_budget for role={r!r} must be > 0 (got {b!r})"
                        )
                    self.max_budget[r] = b_int
            else:
                b_int = int(max_budget)
                if b_int <= 0:
                    raise ValueError(
                        f"TuneDCT3D: max_budget must be > 0 (got {max_budget!r})"
                    )
                for r in self.roles:
                    self.max_budget[r] = b_int

        # Candidate keep configs.
        self.candidates: List[Tuple[int, int, int]] = []
        for h, w, t in product(keep_h, keep_w, keep_t):
            h, w, t = int(h), int(w), int(t)
            if h <= 0 or w <= 0 or t <= 0:
                continue
            self.candidates.append((h, w, t))
        if not self.candidates:
            raise ValueError("TuneDCT3D: no candidate configurations found")

        # One codec per candidate.
        self._codecs: Dict[Tuple[int, int, int], DCT3DCodec] = {
            cfg: DCT3DCodec(cfg[0], cfg[1], cfg[2]) for cfg in self.candidates
        }

        # Cache: per (role, signal, shape) list of candidates that fit within budget.
        self._candidates_cache: Dict[
            Tuple[str, str, Tuple[int, ...]], List[Tuple[int, int, int]]
        ] = {}

        # stats[role][name][cfg] = [sum_explained_energy, n_windows]
        self.stats: Dict[str, Dict[str, Dict[Tuple[int, int, int], List[float]]]] = {}
        self.rep_shape: Dict[Tuple[str, str], Tuple[int, ...]] = {}

        # Pre-create slots for all specs in selected roles.
        for spec in self.signal_specs.specs:
            role, name = spec.role, spec.name
            if role not in self.roles:
                continue
            self.stats.setdefault(role, {}).setdefault(name, {})
            for cfg in self.candidates:
                self.stats[role][name][cfg] = [0.0, 0.0]

        logger.info(
            "TuneDCT3D initialized | roles=%s | candidates=%d | budgets=%s",
            ",".join(self.roles),
            len(self.candidates),
            {r: self.max_budget.get(r) for r in self.roles},
        )

    def _candidates_for(
        self, role: str, name: str, native_shape: Tuple[int, ...]
    ) -> List[Tuple[int, int, int]]:
        """Return candidate configs for this (role, signal, shape), respecting budget."""
        budget = self.max_budget.get(role)
        if budget is None:
            return self.candidates

        key = (role, name, tuple(native_shape))
        cached = self._candidates_cache.get(key)
        if cached is not None:
            return cached

        within = [
            cfg
            for cfg in self.candidates
            if _effective_dim(native_shape, cfg) <= int(budget)
        ]
        if not within:
            # Always evaluate at least one candidate (the smallest overall).
            smallest = min(
                self.candidates, key=lambda cfg: _effective_dim(native_shape, cfg)
            )
            within = [smallest]

        self._candidates_cache[key] = within
        return within

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        # Only require chunks if we're tuning chunk-based roles.
        if (
            any(r in self.roles for r in ("input", "actuator"))
            and "chunks" not in window
        ):
            raise KeyError(
                "TuneDCT3D expects window['chunks'] (run after ChunkWindows/TrimChunks)"
            )

        # Per-window accumulators so each window counts once (mean over chunks).
        # win_sum[role][name][cfg] = sum_explained_energy_over_chunks
        # win_n[role][name][cfg]   = n_chunks_used
        win_sum: Dict[str, Dict[str, Dict[Tuple[int, int, int], float]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(float))
        )
        win_n: Dict[str, Dict[str, Dict[Tuple[int, int, int], int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )

        # 1) input/actuator: read from chunks
        if any(r in self.roles for r in ("input", "actuator")):
            chunks = window.get("chunks") or {}
            for role in ("input", "actuator"):
                if role not in self.roles:
                    continue
                for ch in chunks.get(role) or []:
                    sigs = ch.get("signals") or {}
                    for name, values in sigs.items():
                        if values is None:
                            continue
                        if self.signal_specs.get(role, name) is None:
                            continue

                        x = np.asarray(values)
                        if x.size == 0 or x.ndim not in (1, 2, 3):
                            continue

                        # Representative shape for effective_dim ranking
                        self.rep_shape.setdefault((role, name), tuple(x.shape))

                        finite = np.isfinite(x)
                        if not finite.any():
                            continue
                        x_clean = np.where(finite, x, 0.0)

                        # Fast path: zero-energy signals are perfectly represented by any cfg.
                        x64 = np.asarray(x_clean, dtype=np.float64)
                        total = float(np.sum(x64 * x64))
                        if not np.isfinite(total) or total <= 0.0:
                            for cfg in self._candidates_for(role, name, tuple(x.shape)):
                                win_sum[role][name][cfg] += 1.0
                                win_n[role][name][cfg] += 1
                            continue

                        for cfg in self._candidates_for(role, name, tuple(x.shape)):
                            z = self._codecs[cfg].encode(x_clean)
                            ee = _explained_energy_from_total(total, z)
                            win_sum[role][name][cfg] += ee
                            win_n[role][name][cfg] += 1

        # 2) outputs: read from window["output"][name]["values"]
        if "output" in self.roles:
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

                    x64 = np.asarray(x_clean, dtype=np.float64)
                    total = float(np.sum(x64 * x64))
                    if not np.isfinite(total) or total <= 0.0:
                        for cfg in self._candidates_for("output", name, tuple(x.shape)):
                            win_sum["output"][name][cfg] += 1.0
                            win_n["output"][name][cfg] += 1
                        continue

                    for cfg in self._candidates_for("output", name, tuple(x.shape)):
                        z = self._codecs[cfg].encode(x_clean)
                        ee = _explained_energy_from_total(total, z)
                        win_sum["output"][name][cfg] += ee
                        win_n["output"][name][cfg] += 1

        # 3) Commit: per-window mean over chunks, then accumulate per-window.
        for role, by_sig in win_sum.items():
            for name, by_cfg in by_sig.items():
                for cfg, s in by_cfg.items():
                    n = win_n[role][name][cfg]
                    if n <= 0:
                        continue
                    ee_win = float(s / n)

                    acc = self.stats[role][name][cfg]
                    acc[0] += ee_win
                    acc[1] += 1.0

        return window

    def select_best(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for role in self.roles:
            by_sig = self.stats.get(role, {})
            if role not in self.targets:
                raise KeyError(f"TuneDCT3D missing target for role={role!r}")
            target = float(self.targets[role])

            for name, by_cfg in by_sig.items():
                shape = self.rep_shape.get((role, name))
                if shape is None:
                    continue

                ranked = sorted(
                    self.candidates, key=lambda c: (_effective_dim(shape, c), c)
                )

                # Apply role-specific budget (effective_dim cap) if provided.
                budget = self.max_budget.get(role)
                if budget is not None:
                    ranked_budget = [
                        cfg
                        for cfg in ranked
                        if _effective_dim(shape, cfg) <= int(budget)
                    ]
                    if ranked_budget:
                        ranked = ranked_budget
                    else:
                        # Fall back to smallest candidate overall.
                        ranked = [ranked[0]]

                scores: Dict[Tuple[int, int, int], float] = {}
                for cfg in ranked:
                    s, n = by_cfg[cfg]
                    scores[cfg] = float("-inf") if n == 0 else float(s / n)

                # Skip signals that never produced any metric.
                if not any(np.isfinite(v) for v in scores.values()):
                    continue

                chosen = None
                for cfg in ranked:
                    v = scores[cfg]
                    if np.isfinite(v) and v >= target:
                        chosen = cfg
                        break

                if chosen is None:
                    # Best explained energy, ties -> smaller effective_dim.
                    chosen = max(
                        ranked,
                        key=lambda cfg: (scores[cfg], -_effective_dim(shape, cfg)),
                    )

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
                    "explained_energy_mean_windows": float(scores[chosen]),
                    "target": float(target),
                    "effective_dim": int(_effective_dim(shape, chosen)),
                    "rep_shape": shape,
                    "n_windows": int(by_cfg[chosen][1]),
                    "max_budget": None
                    if self.max_budget.get(role) is None
                    else int(self.max_budget[role]),
                }

        return out
