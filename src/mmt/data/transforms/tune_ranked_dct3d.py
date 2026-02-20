"""
TuneRankedDCT3DTransform
========================

Streaming evaluator for variance-based (rank mode) DCT3D coefficient selection.

This transform accumulates per-coefficient energy E[c_i²] over a sample of windows
and selects the top-K coefficients by explained variance for each signal.

Unlike the old grid-search approach (TuneDCT3DTransform), this:
  • Computes the full DCT once per chunk
  • Accumulates energy for ALL coefficients
  • Selects top-K by variance (no grid search needed)
  • Outputs coefficient indices (.npy files) + config

Explained Energy
----------------
For an orthonormal DCT, the explained energy ratio is:

    explained_energy = sum(c_i² for i in selected) / sum(c_i² for all i)

Aggregation Strategy
--------------------
This transform uses a **pooled energy aggregation** approach:

  1. For each window/chunk, compute full DCT: z = DCT(x)
  2. Accumulate coefficient energies: acc_energy[i] += z[i]²
  3. After all windows, compute mean energy: E[c_i²] = acc_energy[i] / n_windows
  4. Compute explained energy: sum(E[c_i²] for selected) / sum(E[c_i²] for all)

This computes the **ratio of expected energies**: E[sum(z_selected²)] / E[sum(z_all²)]

**Note:** This differs from the old TuneDCT3DTransform which computed the
**expected ratio**: E[sum(z_selected²) / sum(z_all²)] by averaging per-window ratios.
The pooled approach is more robust to windows with varying signal energy and provides
a more accurate estimate of compression performance on the overall dataset.

Selection Strategy
------------------
For each (role, signal):
  1. Compute mean(c_i²) across all windows (pooled energy)
  2. Sort coefficients descending by energy
  3. Find smallest K where cumsum(energy[:K])/total >= threshold
  4. Apply max_budget cap if provided
  5. Return top-K indices

Expected Window Format
----------------------
- input/actuator: window["chunks"][role][i]["signals"][name]
- output: window["output"][name]["values"]

Usage
-----
    tuner = TuneRankedDCT3DTransform(
        signal_specs=specs,
        thresholds={"input": 0.999, "output": 0.995},
        max_budget={"input": 4096, "output": 8192},
        roles=("input", "output"),
    )

    for window in dataset:
        tuner(window)

    best = tuner.select_best()
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.embeddings.dct3d_codec import DCT3DCodec

logger = logging.getLogger("mmt.TuneRankedDCT3D")


def _dct_view_shape(native_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
    """Return the internal (H, W, T) view used by DCT3DCodec for a native shape."""
    if len(native_shape) == 1:
        return 1, 1, int(native_shape[0])
    if len(native_shape) == 2:
        return int(native_shape[0]), 1, int(native_shape[1])
    if len(native_shape) == 3:
        return int(native_shape[0]), int(native_shape[1]), int(native_shape[2])
    raise ValueError(f"Unsupported native_shape={native_shape!r}")


class TuneRankedDCT3DTransform:
    """Variance-based DCT3D coefficient selector (rank mode)."""

    def __init__(
        self,
        *,
        signal_specs: SignalSpecRegistry,
        thresholds: Mapping[str, float],
        roles: Iterable[str] = ("input", "actuator", "output"),
        max_budget: int | Mapping[str, int] | None = None,
        progress_every_n_shots: int | None = 10,
        guardrails: Mapping[str, Any] | None = None,
    ) -> None:
        """Create a ranked DCT3D tuner.

        Parameters
        ----------
        signal_specs:
            Registry of all signals for the current task.
        thresholds:
            Role-specific explained energy targets, e.g.
            {"input": 0.999, "output": 0.995}. Values in [0, 1].
        roles:
            Which roles to tune. Subset of {"input", "actuator", "output"}.
        max_budget:
            Optional coefficient budget (max K). May be a single int (applied
            to all roles) or a mapping per role, e.g. {"input": 4096}.
        progress_every_n_shots:
            Log progress every N shots (based on window["shot_id"] changes).
        guardrails:
            Optional guardrails configuration to ensure minimum dimension coverage.
            Expected structure: {"enable": bool, "timeseries": {...}, "profile": {...}, "video": {...}}
        """
        self.signal_specs = signal_specs

        # Progress logging
        self.progress_every_n_shots: int | None
        if progress_every_n_shots is None:
            self.progress_every_n_shots = None
        else:
            n = int(progress_every_n_shots)
            if n <= 0:
                raise ValueError("progress_every_n_shots must be positive")
            self.progress_every_n_shots = n
            self._shots_seen: int = 0
            self._windows_seen: int = 0
            self._last_shot_id: Any | None = None

        # Validate roles
        allowed = {"input", "actuator", "output"}
        roles_norm: List[str] = []
        for r in roles:
            rr = str(r).strip()
            if not rr:
                continue
            if rr not in allowed:
                raise ValueError(f"Invalid role {rr!r}. Allowed: {sorted(allowed)}")
            if rr not in roles_norm:
                roles_norm.append(rr)
        if not roles_norm:
            raise ValueError("roles must contain at least one role")
        self.roles = tuple(roles_norm)

        # Role targets
        self.targets = {k: float(v) for k, v in thresholds.items() if k in self.roles}

        # Max budget per role
        self.max_budget: Dict[str, int | None] = {r: None for r in self.roles}
        if max_budget is not None:
            if isinstance(max_budget, Mapping):
                for r in self.roles:
                    b = max_budget.get(r)
                    if b is None:
                        continue
                    b_int = int(b)
                    if b_int <= 0:
                        raise ValueError(f"max_budget for {r!r} must be > 0")
                    self.max_budget[r] = b_int
            else:
                b_int = int(max_budget)
                if b_int <= 0:
                    raise ValueError("max_budget must be > 0")
                for r in self.roles:
                    self.max_budget[r] = b_int

        # Cache full-shape codecs (key: (H, W, T))
        self._full_codec_cache: Dict[Tuple[int, int, int], DCT3DCodec] = {}

        # Accumulate per-coefficient energy: coeff_energy[role][name][(H,W,T)] = (sum_energy, count)
        # sum_energy is a 1D array of shape (H*W*T,) containing sum of c_i² across windows
        self.coeff_energy: Dict[
            str, Dict[str, Dict[Tuple[int, int, int], Tuple[np.ndarray, int]]]
        ] = defaultdict(lambda: defaultdict(dict))

        # Representative shape per signal
        self.rep_shape: Dict[Tuple[str, str], Tuple[int, ...]] = {}

        # Guardrails configuration
        self.guardrails_enable = False
        self.guardrails_config: Dict[str, Dict[str, int]] = {
            "timeseries": {},
            "profile": {},
            "video": {},
        }
        if guardrails is not None and guardrails.get("enable"):
            self.guardrails_enable = True
            for signal_type in ["timeseries", "profile", "video"]:
                type_config = guardrails.get(signal_type, {})
                if isinstance(type_config, dict):
                    self.guardrails_config[signal_type] = dict(type_config)

        logger.info(
            "TuneRankedDCT3D initialized | roles=%s | budgets=%s | guardrails=%s",
            ",".join(self.roles),
            {r: self.max_budget.get(r) for r in self.roles},
            "enable" if self.guardrails_enable else "disabled",
        )

    def _full_codec(self, H: int, W: int, T: int) -> DCT3DCodec:
        """Get (or create) a codec that keeps the full DCT grid."""
        key = (int(H), int(W), int(T))
        codec = self._full_codec_cache.get(key)
        if codec is None:
            codec = DCT3DCodec(key[0], key[1], key[2], selection_mode="spatial")
            self._full_codec_cache[key] = codec
        return codec

    def __call__(self, window: Dict[str, Any]) -> Dict[str, Any]:
        """Process one window and accumulate coefficient energies."""
        # Progress logging
        if self.progress_every_n_shots is not None:
            self._windows_seen += 1
            shot_id = window.get("shot_id")
            if shot_id is not None and shot_id != self._last_shot_id:
                self._last_shot_id = shot_id
                self._shots_seen += 1
                if self._shots_seen % self.progress_every_n_shots == 0:
                    logger.info(
                        "TuneRankedDCT3D progress: shots=%d, windows=%d",
                        self._shots_seen,
                        self._windows_seen,
                    )

        # Process chunk-based roles (input, actuator)
        if any(r in self.roles for r in ("input", "actuator")):
            if "chunks" not in window:
                raise KeyError("Expected window['chunks'] for input/actuator roles")

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

                        # Store representative shape
                        self.rep_shape.setdefault((role, name), tuple(x.shape))

                        # Clean non-finite values
                        finite = np.isfinite(x)
                        if not finite.any():
                            continue
                        x_clean = np.where(finite, x, 0.0)

                        # Compute full DCT
                        native_shape = tuple(x.shape)
                        H, W, T = _dct_view_shape(native_shape)
                        codec_full = self._full_codec(H, W, T)

                        z_full = codec_full.encode(x_clean)  # (H*W*T,)
                        z64 = np.asarray(z_full, dtype=np.float64)
                        energy = z64 * z64  # per-coefficient energy

                        # Accumulate
                        key = (H, W, T)
                        if key not in self.coeff_energy[role][name]:
                            self.coeff_energy[role][name][key] = (
                                np.zeros(H * W * T, dtype=np.float64),
                                0,
                            )

                        acc_energy, acc_count = self.coeff_energy[role][name][key]
                        acc_energy += energy
                        self.coeff_energy[role][name][key] = (acc_energy, acc_count + 1)

        # Process output role
        if "output" in self.roles:
            out_group = window.get("output")
            if isinstance(out_group, dict):
                for name, entry in out_group.items():
                    if self.signal_specs.get("output", name) is None:
                        continue
                    if not isinstance(entry, dict) or "values" not in entry:
                        raise TypeError(
                            f"Expected window['output'][{name!r}] to have 'values' key"
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

                    native_shape = tuple(x.shape)
                    H, W, T = _dct_view_shape(native_shape)
                    codec_full = self._full_codec(H, W, T)

                    z_full = codec_full.encode(x_clean)
                    z64 = np.asarray(z_full, dtype=np.float64)
                    energy = z64 * z64

                    key = (H, W, T)
                    if key not in self.coeff_energy["output"][name]:
                        self.coeff_energy["output"][name][key] = (
                            np.zeros(H * W * T, dtype=np.float64),
                            0,
                        )

                    acc_energy, acc_count = self.coeff_energy["output"][name][key]
                    acc_energy += energy
                    self.coeff_energy["output"][name][key] = (acc_energy, acc_count + 1)

        return window

    def _determine_signal_type(self, shape: Tuple[int, ...]) -> str:
        """Determine signal type from shape.

        Parameters
        ----------
        shape : Tuple[int, ...]
            Native signal shape (T,), (C, T), or (H, W, T)

        Returns
        -------
        str
            One of: "timeseries", "profile", "video"
        """
        if len(shape) == 1:
            return "timeseries"
        elif len(shape) == 2:
            return "profile"
        elif len(shape) == 3:
            return "video"
        raise ValueError(f"Unsupported shape: {shape}")

    def _apply_dimension_guardrails(
        self,
        sorted_indices: np.ndarray,
        H: int,
        W: int,
        T: int,
        signal_type: str,
        role: str,
        name: str,
    ) -> int:
        """Find minimum K to satisfy dimension coverage requirements.

        This method ensures that the selected coefficients have at least
        min_unique_h/w/t distinct indices per dimension, while still
        prioritizing high-energy coefficients.

        Parameters
        ----------
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending)
        H, W, T : int
            DCT grid dimensions
        signal_type : str
            One of: "timeseries", "profile", "video"
        role : str
            Signal role (for logging)
        name : str
            Signal name (for logging)

        Returns
        -------
        int
            Minimum number of coefficients needed to satisfy guardrails.
            Returns 0 if no guardrails apply.
        """
        config = self.guardrails_config.get(signal_type, {})

        # Map signal type to dimension requirements
        # Format: {signal_type: (h_key, w_key, t_key)}
        # Note: All signals use canonical (H, W, T) representation internally
        dim_keys = {
            "timeseries": (None, None, "min_unique_t"),
            "profile": ("min_unique_h", None, "min_unique_t"),  # C maps to H
            "video": ("min_unique_h", "min_unique_w", "min_unique_t"),
        }

        h_key, w_key, t_key = dim_keys.get(signal_type, (None, None, None))

        # Extract and clamp minimums to actual dimensions
        min_h = config.get(h_key, 0) if h_key else 0
        min_w = config.get(w_key, 0) if w_key else 0
        min_t = config.get(t_key, 0) if t_key else 0

        eff_min_h = min(min_h, H) if min_h > 0 else 0
        eff_min_w = min(min_w, W) if min_w > 0 else 0
        eff_min_t = min(min_t, T) if min_t > 0 else 0

        # If no guardrails needed, return 0
        if eff_min_h == 0 and eff_min_w == 0 and eff_min_t == 0:
            return 0

        # Track unique indices per dimension
        unique_h = set()
        unique_w = set()
        unique_t = set()

        # Iterate through sorted indices
        for k in range(len(sorted_indices)):
            idx = int(sorted_indices[k])
            # Convert flat index to (h, w, t)
            h = idx // (W * T)
            w = (idx % (W * T)) // T
            t = idx % T

            unique_h.add(h)
            unique_w.add(w)
            unique_t.add(t)

            # Check if requirements met
            if (
                len(unique_h) >= eff_min_h
                and len(unique_w) >= eff_min_w
                and len(unique_t) >= eff_min_t
            ):
                min_k = k + 1  # +1 because k is 0-indexed
                logger.debug(
                    "Guardrails for %s:%s (%s) require min K=%d "
                    "(unique: H=%d/%d, W=%d/%d, T=%d/%d)",
                    role,
                    name,
                    signal_type,
                    min_k,
                    len(unique_h),
                    eff_min_h,
                    len(unique_w),
                    eff_min_w,
                    len(unique_t),
                    eff_min_t,
                )
                return min_k

        # If we get here, use all coefficients
        logger.warning(
            "Guardrails for %s:%s (%s) cannot be satisfied with all %d coefficients "
            "(unique: H=%d/%d, W=%d/%d, T=%d/%d)",
            role,
            name,
            signal_type,
            len(sorted_indices),
            len(unique_h),
            eff_min_h,
            len(unique_w),
            eff_min_w,
            len(unique_t),
            eff_min_t,
        )
        return len(sorted_indices)

    def select_best(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Select top-K coefficients by variance for each signal.

        Returns
        -------
        Dict with structure:
            {role: {signal_name: {
                "coeff_indices": np.ndarray,  # top-K indices
                "coeff_shape": (H, W, T),
                "num_coeffs": K,
                "explained_energy": float,
                "target": float,
                "n_windows": int,
                "max_budget": int | None,
            }}}
        """
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for role in self.roles:
            by_sig = self.coeff_energy.get(role, {})
            if role not in self.targets:
                raise KeyError(f"Missing target for role={role!r}")
            target = float(self.targets[role])
            budget = self.max_budget.get(role)

            for name, by_shape in by_sig.items():
                shape = self.rep_shape.get((role, name))
                if shape is None:
                    continue

                # Should have exactly one shape key per signal
                if len(by_shape) != 1:
                    logger.warning(
                        "Signal %s:%s has multiple shapes: %s. Using first.",
                        role,
                        name,
                        list(by_shape.keys()),
                    )

                key = list(by_shape.keys())[0]
                acc_energy, acc_count = by_shape[key]

                if acc_count == 0:
                    logger.warning("Signal %s:%s has no windows, skipping", role, name)
                    continue

                # Mean energy per coefficient
                mean_energy = acc_energy / acc_count
                total_energy = float(np.sum(mean_energy))

                if total_energy <= 0:
                    logger.warning(
                        "Signal %s:%s has zero total energy, skipping", role, name
                    )
                    continue

                # Sort coefficients by energy descending
                sorted_indices = np.argsort(mean_energy)[::-1]  # descending

                # Apply guardrails if enabled
                H, W, T = key
                guardrail_min_K = 0
                if self.guardrails_enable:
                    signal_type = self._determine_signal_type(shape)
                    guardrail_min_K = self._apply_dimension_guardrails(
                        sorted_indices, H, W, T, signal_type, role, name
                    )

                # Find smallest K where cumsum(energy[:K])/total >= target
                cumsum_energy = np.cumsum(mean_energy[sorted_indices])
                explained_ratio = cumsum_energy / total_energy

                # Find first K that meets target
                meets_target = np.where(explained_ratio >= target)[0]
                if len(meets_target) > 0:
                    K = int(meets_target[0]) + 1  # +1 because index is 0-based
                else:
                    # Target not achievable, use all coefficients
                    K = len(sorted_indices)
                    logger.warning(
                        "Signal %s:%s cannot reach target %.4f (max: %.4f), using all %d coeffs",
                        role,
                        name,
                        target,
                        explained_ratio[-1],
                        K,
                    )

                # Apply guardrails if enabled
                budget_violated = False
                if self.guardrails_enable and guardrail_min_K > 0:
                    # Guardrails enabled: compute K_req = max(K_target, K_guardrail_min)
                    K_req = max(K, guardrail_min_K)

                    if K < guardrail_min_K:
                        logger.info(
                            "Signal %s:%s: increasing K from %d to %d to meet guardrails",
                            role,
                            name,
                            K,
                            K_req,
                        )

                    # Strict guardrails: check if they exceed budget
                    if budget is not None and K_req > budget:
                        budget_violated = True
                        logger.warning(
                            "Signal %s:%s: guardrails require K=%d, exceeding budget=%d. "
                            "Keeping K=%d (guardrails take precedence).",
                            role,
                            name,
                            K_req,
                            budget,
                            K_req,
                        )

                    K = K_req
                else:
                    # Guardrails disabled: apply budget cap normally (hard cap)
                    if budget is not None and K > budget:
                        K = int(budget)
                        logger.info(
                            "Signal %s:%s capped at budget %d (target would need %d)",
                            role,
                            name,
                            budget,
                            len(meets_target) + 1
                            if len(meets_target) > 0
                            else len(sorted_indices),
                        )

                # Select top-K indices
                coeff_indices = sorted_indices[:K].astype(np.int32)
                achieved_energy = float(cumsum_energy[K - 1] / total_energy)

                out.setdefault(role, {})[name] = {
                    "coeff_indices": coeff_indices,
                    "coeff_shape": (H, W, T),
                    "num_coeffs": K,
                    "explained_energy": achieved_energy,
                    "target": target,
                    "n_windows": acc_count,
                    "max_budget": budget,
                    "budget_violated_for_guardrails": budget_violated,
                }

        return out
