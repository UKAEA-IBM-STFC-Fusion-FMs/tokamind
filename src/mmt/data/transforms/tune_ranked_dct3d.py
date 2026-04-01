"""
TuneRankedDCT3DTransform
========================

Streaming evaluator for variance-based (rank mode) DCT3D coefficient selection.

This transform accumulates per-coefficient energy E[c_i²] over a sample of windows and selects the top-K coefficients
by explained variance for each signal.

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

**Note:** This differs from the old TuneDCT3DTransform which computed the **expected ratio**:
E[sum(z_selected²) / sum(z_all²)] by averaging per-window ratios.
The pooled approach is more robust to windows with varying signal energy and provides a more accurate estimate of
compression performance on the overall dataset.

Selection Strategy
------------------
For each (role, signal):
  1. Compute mean(c_i²) across all windows (pooled energy)
  2. Sort coefficients descending by energy
  3. Find `K_target`: smallest K where cumsum(energy[:K])/total >= threshold
  4. If guardrails are enabled, compute `guardrail_min_k` from minimum unique index coverage by modality and lift K to
     `max(K_target, guardrail_min_k)`
  5. Apply `max_budget` as a hard cap (final K never exceeds budget)
  6. Return top-K indices and selection metadata

Selection Metadata
------------------
`select_best()` returns per-signal metadata used by artifact writers:
- coefficient payload: `coeff_indices`, `coeff_shape`, `num_coeffs`
- objective outcome: `explained_energy`, `target_energy`, `n_windows`
- policy breakdown:
  - `k_target`
  - `guardrail_min_k`
  - `k_after_guardrails`
  - `max_budget`
  - `guardrail_increased_k`
  - `budget_capped`
  - `budget_violated_for_guardrails`
- diagnostics:
  - `dim_distribution.{unique_h,unique_w,unique_t}`
  - `flags` (e.g., `guardrail_up`, `budget_cap`, `budget_cap_guardrail`)

Runtime Logging
---------------
At INFO level the transform reports:
- active guardrail rules (once at init, if enabled),
- per-signal guardrail lifts (`K old -> new`, with coverage before/after),
- budget-cap warnings when requested K exceeds role budget.

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
        roles=("input", "output")
    )

    for window in dataset:
        tuner(window)

    best = tuner.select_best()
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable
from collections.abc import Mapping
import logging
import numpy as np

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.embeddings.dct3d_codec import DCT3DCodec


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.TuneRankedDCT3D")

Shape3D = tuple[int, int, int]


# ======================================================================================================================
@dataclass(frozen=True)
class _KSelection:
    """Final K policy decision for one signal."""

    k_target: int
    """Raw K computed from the energy target, before guardrails or budget are applied."""

    guardrail_min_k: int
    """Minimum K enforced by guardrails to ensure adequate coefficient coverage."""

    k_after_guardrails: int
    """K after applying guardrails: max(k_target, guardrail_min_k) when guardrails are enabled."""

    k_final: int
    """Final selected K after applying the budget cap: min(k_after_guardrails, budget)."""

    guardrail_increased_k: bool
    """True if guardrails raised K above k_target."""

    budget_capped: bool
    """True if the budget cap reduced K below k_after_guardrails."""

    budget_violated_for_guardrails: bool
    """
    True if the budget cap prevented guardrails from being fully satisfied (guardrails needed higher K than budget 
    allows).
    """


# ----------------------------------------------------------------------------------------------------------------------
def _dct_view_shape(native_shape: tuple[int, ...]) -> tuple[int, int, int]:
    """
    Return the internal (H, W, T) view used by DCT3DCodec for a native shape.

    Parameters
    ----------
    native_shape : tuple[int, ...]
        Native shape.

    Returns
    -------
    tuple[int, int, int]
        Internal (H, W, T) view used by DCT3DCodec

    Raises
    ------
    ValueError
        If an unsupported `native_shape` is passed, i.e., if `len(native_shape)` not in [1, 2, 3].

    """

    if len(native_shape) == 1:
        return 1, 1, int(native_shape[0])
    if len(native_shape) == 2:
        return int(native_shape[0]), 1, int(native_shape[1])
    if len(native_shape) == 3:
        return int(native_shape[0]), int(native_shape[1]), int(native_shape[2])

    raise ValueError(f"[TuneRankedDCT3DTransform] Unsupported `native_shape={native_shape!r}`.")


# ----------------------------------------------------------------------------------------------------------------------
def _format_guardrail_rules(rules: Mapping[str, Any]) -> str:
    """
    Format active guardrail rules in stable key order.

    Parameters
    ----------
    rules : Mapping[str, Any]
        Mapping (dict) with active guardrail rules.

    Returns
    -------
    str
        Joined valid items in `rules` in stable key order.

    """

    ordered_keys = ("min_unique_h", "min_unique_w", "min_unique_t")
    parts: list[str] = []
    for key in ordered_keys:
        val = rules.get(key)
        if val is not None:
            parts.append(f"{key}={int(val)}")  # type: ignore[arg-type]
    for key in sorted(rules.keys()):
        if key in ordered_keys:
            continue
        val = rules.get(key)
        if val is not None:
            parts.append(f"{key}={val}")

    return ", ".join(parts) if parts else "-"


# ----------------------------------------------------------------------------------------------------------------------
def _coverage_shape_for_top_k_indices(
    sorted_indices: np.ndarray,
    *,
    K: int,  # noqa - Ignore lowercase warning
    W: int,  # noqa - Ignore lowercase warning
    T: int,  # noqa - Ignore lowercase warning
) -> tuple[int, int, int]:
    """
    Return unique-dimension coverage (h, w, t) for the first K indices.

    Parameters
    ----------
    sorted_indices : np.ndarray
        Coefficient indices sorted by energy (descending).
    K, W, T, : int
        Coefficient shape (matching DCT grid dimensions), with K being the number of target first indices.

    Returns
    -------
    tuple[int, int, int]
        Unique-dimension coverage for the first K indices.

    """

    if K <= 0:
        return 0, 0, 0

    top_k_indices = np.asarray(sorted_indices[:K], dtype=np.int64)
    h_idx = top_k_indices // (W * T)
    w_idx = (top_k_indices % (W * T)) // T
    t_idx = top_k_indices % T

    return (
        int(len(np.unique(h_idx))),
        int(len(np.unique(w_idx))),
        int(len(np.unique(t_idx))),
    )


# ======================================================================================================================
class TuneRankedDCT3DTransform:
    """
    Variance-based DCT3D coefficient selector (rank mode).

    Attributes
    ----------
    signal_specs : SignalSpecRegistry
        Registry of all signals for the current task.
    progress_every_n_shots : int | None
        Log progress every N shots (based on window["shot_id"] changes).
    roles : list[str]
        Signal roles.
    targets : Mapping[str, Any]
        Role targets.
    coeff_energy : dict[str, dict[str, dict[tuple[int, int, int], tuple[np.ndarray, int]]]]
        Mapping (dict) with accumulated per-coefficient energy.
    rep_shape : dict[tuple[str, str], tuple[int, ...]]
        Mapping (dict) of representative shape per signal.
    guardrails_enable : bool
        Whether to enable guardrails.
    guardrails_config : dict[str, dict[str, int]]
        Mapping (dict) with guardrails configuration for "timeseries", "profile", and "video".
    _shots_seen : int
        Supporting variable to hold shots seen.
    _windows_seen : int
        Supporting variable to hold windows seen.
    _last_shot_id : Any | None
        Supporting variable to hold last shot ID seen.
    max_budget : dict[str, int | None]
        Coefficient budget (max K). Maybe a single int (applied to all roles) or a mapping per role.
    _full_codec_cache : dict[tuple[int, int, int], DCT3DCodec]
        Mapping (dict) with cache full-shape codecs per (H, W, T) key.

    Methods
    -------
    _full_codec(H, W, T)
        Get (or create) a codec that keeps the full DCT grid.
    __call__(window)
        Call method for the class instances to behave like a function.
    _log_progress(window)
        Log progress for the passed window.
    _accumulate_chunk_roles(window)
        Accumulate chunks for "input" and "actuator" roles.
    _accumulate_output_role(window)
        Accumulate "output" role.
    _accumulate_signal_values(role, name, values)
        Accumulate signal values for a given role and name.
    _accumulate_energy(role, name, energy, key)
        Accumulate coefficient energy for a given signal role, signal name, and coefficient key.
    _determine_signal_type(shape)
        Determine signal type from native signal shape.
    _apply_dimension_guardrails(sorted_indices, H, W, T, signal_type, role, name)
        Find minimum K to satisfy dimension coverage requirements.
    _selection_flags(guardrail_increased_k, budget_capped, budget_violated_for_guardrails)
        Build stable selection flags for one tuned signal.
    summarize_selection(best)
        Return compact counters for tuning summary logs.
    _prepare_signal_selection(role, name, by_shape)
        Prepare mean-energy statistics and sorted indices for one signal.
    _compute_target_k(role, name, sorted_indices, mean_energy, total_energy, target_energy)
        Compute K required to meet explained-energy target.
    _compute_guardrail_min_k(shape, sorted_indices, H, W, T, role, name)
        Compute minimum K required by guardrails for one signal.
    _resolve_k_selection(role, name, sorted_indices, W, T, k_target, guardrail_min_k, budget, target_energy)
        Apply guardrail lift and budget cap to produce final K selection.
    _build_selection_payload(sorted_indices, cumsum_energy, total_energy, H, W, T, selection, target_energy, n_windows,
        budget)
        Build per-signal result payload from selection decision.
    select_best()
        Select top-K coefficients by variance for each signal.

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(  # NOSONAR - Ignore cognitive complexity
        self,
        *,
        signal_specs: SignalSpecRegistry,
        thresholds: Mapping[str, float],
        roles: Iterable[str] = ("input", "actuator", "output"),
        max_budget: int | Mapping[str, int] | None = None,
        progress_every_n_shots: int | None = 10,
        guardrails: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Create a ranked DCT3D tuner.

        Parameters
        ----------
        signal_specs : SignalSpecRegistry
            Registry of all signals for the current task.
        thresholds : Mapping[str, float]
            Role-specific explained energy targets, e.g., {"input": 0.999, "output": 0.995}. Values in [0, 1].
        roles : Iterable[str]
            Which roles to tune. Subset of {"input", "actuator", "output"}.
            Optional. Default: ("input", "actuator", "output").
        max_budget : int | Mapping[str, int] | None
            Optional coefficient budget (max K). Maybe a single int (applied to all roles) or a mapping per role,
            e.g., {"input": 4096}.
            Optional. Default: None.
        progress_every_n_shots : int | None
            Log progress every N shots (based on window["shot_id"] changes).
            Optional. Default: 10.
        guardrails : Mapping[str, Any] | None
            Guardrails configuration to ensure minimum dimension coverage.
            Expected structure: {"enable": bool, "timeseries": {...}, "profile": {...}, "video": {...}}
            Optional. Default: None.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking mistakes, as this is a callable class.

        Raises
        ------
        ValueError
            If `progress_every_n_shots` is not positive.
            If invalid role in `roles`, i.e., not in {"input", "actuator", "output"}.
            If `roles` is empty.
            If `max_budget` is a mapping (dict) does not have a positive value for a given role in `roles`.
            If `max_budget` is of type int and is not positive.

        """

        self.signal_specs = signal_specs

        # Progress logging
        self.progress_every_n_shots: int | None
        if progress_every_n_shots is None:
            self.progress_every_n_shots = None
        else:
            n = int(progress_every_n_shots)
            if n <= 0:
                raise ValueError("[TuneRankedDCT3DTransform] `progress_every_n_shots` must be positive.")

            self.progress_every_n_shots = n
            self._shots_seen: int = 0
            self._windows_seen: int = 0
            self._last_shot_id: Any | None = None

        # Validate roles
        allowed = {"input", "actuator", "output"}
        roles_norm: list[str] = []
        for r in roles:
            rr = str(r).strip()
            if not rr:
                continue
            if rr not in allowed:
                raise ValueError(f"[TuneRankedDCT3DTransform] Invalid role {rr!r}. Allowed: {sorted(allowed)}.")

            if rr not in roles_norm:
                roles_norm.append(rr)

        if not roles_norm:
            raise ValueError("[TuneRankedDCT3DTransform] `roles` must contain at least one role.")

        self.roles = tuple(roles_norm)

        # Role targets
        self.targets = {k: float(v) for k, v in thresholds.items() if k in self.roles}

        # Max budget per role
        self.max_budget: dict[str, int | None] = dict.fromkeys(self.roles, None)
        if max_budget is not None:
            if isinstance(max_budget, dict):
                for r in self.roles:
                    b = max_budget.get(r)
                    if b is None:
                        continue
                    b_int = int(b)  # type: ignore[arg-type]
                    if b_int <= 0:
                        raise ValueError(f"[TuneRankedDCT3DTransform] `max_budget` for {r!r} must be > 0.")

                    self.max_budget[r] = b_int

            elif isinstance(max_budget, int):
                b_int = int(max_budget)
                if b_int <= 0:
                    raise ValueError("[TuneRankedDCT3DTransform] `max_budget` must be > 0.")
                for r in self.roles:
                    self.max_budget[r] = b_int

        # Cache full-shape codecs (key: (H, W, T))
        self._full_codec_cache: dict[tuple[int, int, int], DCT3DCodec] = {}

        # Accumulate per-coefficient energy: coeff_energy[role][name][(H,W,T)] = (sum_energy, count).
        # sum_energy is a 1D array of shape (H*W*T,) containing sum of c_i² across windows.
        self.coeff_energy: dict[str, dict[str, dict[tuple[int, int, int], tuple[np.ndarray, int]]]] = defaultdict(
            lambda: defaultdict(dict)
        )  # type: ignore[assignment]

        # Representative shape per signal
        self.rep_shape: dict[tuple[str, str], tuple[int, ...]] = {}

        # Guardrails configuration
        self.guardrails_enable = False
        self.guardrails_config: dict[str, dict[str, int]] = {
            "timeseries": {},
            "profile": {},
            "video": {},
        }
        if (guardrails is not None) and guardrails.get("enable"):
            self.guardrails_enable = True
            for signal_type in ["timeseries", "profile", "video"]:
                type_config = guardrails.get(signal_type, {})
                if isinstance(type_config, dict):
                    self.guardrails_config[signal_type] = dict(type_config)

        logger.debug(
            "TuneRankedDCT3D initialized | roles=%s | budgets=%s | guardrails=%s",
            ",".join(self.roles),
            {r: self.max_budget.get(r) for r in self.roles},
            "enable" if self.guardrails_enable else "disabled",
        )
        if self.guardrails_enable:
            logger.info(
                "Active guardrails: timeseries[%s] profile[%s] video[%s]",
                _format_guardrail_rules(self.guardrails_config["timeseries"]),
                _format_guardrail_rules(self.guardrails_config["profile"]),
                _format_guardrail_rules(self.guardrails_config["video"]),
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _full_codec(
        self,
        H: int,  # noqa - Ignore lowercase warning
        W: int,  # noqa - Ignore lowercase warning
        T: int,  # noqa - Ignore lowercase warning
    ) -> DCT3DCodec:
        """
        Get (or create) a codec that keeps the full DCT grid.

        Parameters
        ----------
        H, W, T : int
            DCT grid dimensions.

        Returns
        -------
        DCT3DCodec
            Obtained or created DCT3DCodec instance.

        """

        key = (int(H), int(W), int(T))
        codec = self._full_codec_cache.get(key)
        if codec is None:
            codec = DCT3DCodec(keep_h=key[0], keep_w=key[1], keep_t=key[2], selection_mode="spatial")
            self._full_codec_cache[key] = codec

        return codec

    # ------------------------------------------------------------------------------------------------------------------
    def __call__(self, window: dict[str, Any]) -> dict[str, Any]:
        """
        Call method for the class instances to behave like a function.

        Process one window and accumulate per-coefficient energies.

        Parameters
        ----------
        window : dict[str, Any]
            Window on which the transform is applied.

        Returns
        -------
        dict[str, Any]
            The same window, unchanged (accumulation is a side effect).

        """

        self._log_progress(window=window)
        self._accumulate_chunk_roles(window=window)
        self._accumulate_output_role(window=window)

        return window

    # ------------------------------------------------------------------------------------------------------------------
    # Window ingestion helpers
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def _log_progress(self, window: dict[str, Any]) -> None:
        """
        Log progress for the passed window.

        Parameters
        ----------
        window : dict[str, Any]
            Window used for logging purposes.

        Returns
        -------
        None

        """

        if self.progress_every_n_shots is None:
            return

        self._windows_seen += 1
        shot_id = window.get("shot_id")
        if (shot_id is None) or (shot_id == self._last_shot_id):
            return

        self._last_shot_id = shot_id
        self._shots_seen += 1
        if self._shots_seen % self.progress_every_n_shots == 0:
            logger.info(
                "TuneRankedDCT3D progress: shots=%d, windows=%d",
                self._shots_seen,
                self._windows_seen,
            )

    # ------------------------------------------------------------------------------------------------------------------
    def _accumulate_chunk_roles(self, window: dict[str, Any]) -> None:
        """
        Accumulate chunks for "input" and "actuator" roles.

        Parameters
        ----------
        window : dict[str, Any]
            Window used for chunk accumulation, expected to have a required key "chunks".

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If `window`does not contain required key "chunks".

        """

        if not any(r in self.roles for r in ("input", "actuator")):
            return

        if "chunks" not in window:
            raise KeyError("[TuneRankedDCT3DTransform] Expected `window['chunks']` for input/actuator roles.")

        chunks = window.get("chunks") or {}
        for role in ("input", "actuator"):
            if role not in self.roles:
                continue
            for ch in chunks.get(role) or []:
                sigs = ch.get("signals") or {}
                for name, values in sigs.items():
                    self._accumulate_signal_values(role=role, name=name, values=values)

    # ------------------------------------------------------------------------------------------------------------------
    def _accumulate_output_role(self, window: dict[str, Any]) -> None:
        """
        Accumulate "output" role.

        Parameters
        ----------
        window : dict[str, Any]
            Window used for output accumulation.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If `window["output"]` contains an item without a required key "values".

        """

        if "output" not in self.roles:
            return

        out_group = window.get("output")
        if not isinstance(out_group, dict):
            return

        for name, entry in out_group.items():
            if self.signal_specs.get("output", name) is None:
                continue
            if not isinstance(entry, dict) or "values" not in entry:
                raise TypeError(f"[TuneRankedDCT3DTransform] Expected window['output'][{name!r}] to have 'values' key.")

            self._accumulate_signal_values(role="output", name=name, values=entry.get("values"))

    # ------------------------------------------------------------------------------------------------------------------
    def _accumulate_signal_values(self, *, role: str, name: str, values: Any) -> None:
        """
        Accumulate signal values for a given role and name.

        Parameters
        ----------
        role : str
            Signal role.
        name : str
            Canonical signal name.
        values : Any
            Signal values.

        Returns
        -------
        None

        """

        if values is None:
            return
        if self.signal_specs.get(role, name) is None:
            return

        x = np.asarray(values)
        if (x.size == 0) or (x.ndim not in [1, 2, 3]):
            return

        self.rep_shape.setdefault((role, name), tuple(x.shape))

        finite = np.isfinite(x)
        if not finite.any():
            return

        x_clean = np.where(finite, x, 0.0)

        H, W, T = _dct_view_shape(native_shape=tuple(x.shape))
        codec_full = self._full_codec(H=H, W=W, T=T)
        z_full = codec_full.encode(x=x_clean)  # (H*W*T,)
        z64 = np.asarray(z_full, dtype=np.float64)
        energy = z64 * z64

        self._accumulate_energy(role=role, name=name, key=(H, W, T), energy=energy)

    # ------------------------------------------------------------------------------------------------------------------
    def _accumulate_energy(self, *, role: str, name: str, key: Shape3D, energy: np.ndarray) -> None:
        """
        Accumulate coefficient energy for a given signal role, signal name, and coefficient key.

        Parameters
        ----------
        role : str
            Signal role.
        name : str
            Signal canonical name.
        key : Shape3D
            (int, int, int) tuple used as key to save coefficient energy for a given `role` and `name`.
        energy : np.ndarray
            Energy to be accumulated for a given coefficient key.

        Returns
        -------
        None

        """

        if key not in self.coeff_energy[role][name]:
            H, W, T = key
            self.coeff_energy[role][name][key] = (
                np.zeros(H * W * T, dtype=np.float64),
                0,
            )

        acc_energy, acc_count = self.coeff_energy[role][name][key]
        acc_energy += energy
        self.coeff_energy[role][name][key] = (acc_energy, acc_count + 1)

    # ------------------------------------------------------------------------------------------------------------------
    # Guardrail helpers
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _determine_signal_type(shape: tuple[int, ...]) -> str:
        """
        Determine signal type from native signal shape.

        Parameters
        ----------
        shape : tuple[int, ...]
            Native signal shape (T,), (C, T), or (H, W, T).

        Returns
        -------
        str
            One of: "timeseries", "profile", "video".

        Raises
        ------
        ValueError
            If `len(shape)` not in [1, 2, 3].

        """

        if len(shape) == 1:
            return "timeseries"
        elif len(shape) == 2:
            return "profile"
        elif len(shape) == 3:
            return "video"
        raise ValueError(f"[TuneRankedDCT3DTransform] Unsupported `shape` value: {shape}.")

    # ------------------------------------------------------------------------------------------------------------------
    def _apply_dimension_guardrails(
        self,
        sorted_indices: np.ndarray,
        H: int,  # noqa - Ignore lowercase warning
        W: int,  # noqa - Ignore lowercase warning
        T: int,  # noqa - Ignore lowercase warning
        signal_type: str,
        role: str,
        name: str,
    ) -> int:
        """
        Find minimum K to satisfy dimension coverage requirements.

        This method ensures that the selected coefficients have at least min_unique_h/w/t distinct indices per
        dimension, while still prioritizing high-energy coefficients.

        Parameters
        ----------
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending).
        H, W, T : int
            Coefficient shape (matching DCT grid dimensions).
        signal_type : str
            One of: "timeseries", "profile", "video".
        role : str
            Signal role (for logging).
        name : str
            Canonical signal name (for logging).

        Returns
        -------
        int
            Minimum number of coefficients needed to satisfy guardrails. Returns 0 if no guardrails apply.

        """

        config = self.guardrails_config.get(signal_type, {})

        # Map signal type to dimension requirements.
        # Format: {signal_type: (h_key, w_key, t_key)}
        # Note: All signals use canonical (H, W, T) representation internally.
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

        eff_min_h = min(min_h, H) if (min_h > 0) else 0
        eff_min_w = min(min_w, W) if (min_w > 0) else 0
        eff_min_t = min(min_t, T) if (min_t > 0) else 0

        # If no guardrails needed, return 0
        if (eff_min_h == 0) and (eff_min_w == 0) and (eff_min_t == 0):
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
            if (len(unique_h) >= eff_min_h) and (len(unique_w) >= eff_min_w) and (len(unique_t) >= eff_min_t):
                min_k = k + 1  # +1 because k is 0-indexed
                logger.debug(
                    "Guardrails for %s:%s (%s) require min K=%d (unique: H=%d/%d, W=%d/%d, T=%d/%d)",
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

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _selection_flags(
        *,
        guardrail_increased_k: bool,
        budget_capped: bool,
        budget_violated_for_guardrails: bool,
    ) -> list[str]:
        """
        Build stable selection flags for one tuned signal.

        Parameters
        ----------
        guardrail_increased_k : bool
            Whether to include "guardrail_up" in the resulting list of selection flags.
        budget_capped, budget_violated_for_guardrails : bool
            Whether to include budget-cap related flags ("budget_cap_guardrail" or "budget_cap") in the resulting list
            of selection flags.

        Returns
        -------
        list[str]
            Resulting list of selection flags.

        """

        flags: list[str] = []
        if guardrail_increased_k:
            flags.append("guardrail_up")
        if budget_capped:
            if budget_violated_for_guardrails:
                flags.append("budget_cap_guardrail")
            else:
                flags.append("budget_cap")

        return flags

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def summarize_selection(
        best: Mapping[str, Mapping[str, Mapping[str, Any]]],
    ) -> dict[str, int]:
        """
        Return compact counters for tuning summary logs.

        Parameters
        ----------
        best : Mapping[str, Mapping[str, Mapping[str, Any]]]
            Mapping (dict) with data for the selected top coefficients for each signal.

        Returns
        -------
        dict[str, int]
            Dictionary with counter values (calculated from the `best` mapping) for the "signals", "guardrail_up", and
            "budget_capped" counters.

        """

        n_signals = 0
        n_guardrail_up = 0
        n_budget_capped = 0
        for by_sig in best.values():
            for info in by_sig.values():
                n_signals += 1
                if bool(info.get("guardrail_increased_k", False)):
                    n_guardrail_up += 1
                if bool(info.get("budget_capped", False)):
                    n_budget_capped += 1

        return {
            "signals": int(n_signals),
            "guardrail_up": int(n_guardrail_up),
            "budget_capped": int(n_budget_capped),
        }

    # ------------------------------------------------------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------------------------------------------------------

    # ------------------------------------------------------------------------------------------------------------------
    def _prepare_signal_selection(
        self,
        *,
        role: str,
        name: str,
        by_shape: Mapping[Shape3D, tuple[np.ndarray, int]],
    ) -> tuple[tuple[int, ...], Shape3D, int, np.ndarray, float, np.ndarray] | None:
        """
        Prepare mean-energy statistics and sorted indices for one signal.

        Parameters
        ----------
        role : str
            Signal role.
        name : str
            Canonical signal name.
        by_shape : Mapping[Shape3D, tuple[np.ndarray, int]]
            Mapping (dict) with shape information for signal selection.

        Returns
        -------
        tuple[tuple[int, ...], Shape3D, int, np.ndarray, float, np.ndarray] | None
            Tuple consisting of the following elements:
            - Signal shape
            - Coefficient key
            - Energy accumulation count
            - Mean signal energy
            - Total signal energy
            - Sorted indices

        """

        shape = self.rep_shape.get((role, name))
        if shape is None:
            return None

        if len(by_shape) != 1:
            logger.warning(
                "Signal %s:%s has multiple shapes: %s. Using first.",
                role,
                name,
                list(by_shape.keys()),
            )

        key = next(iter(by_shape.keys()))
        acc_energy, acc_count = by_shape[key]
        if acc_count == 0:
            logger.warning("Signal %s:%s has no windows, skipping", role, name)

            return None

        mean_energy = acc_energy / acc_count
        total_energy = float(np.sum(mean_energy))
        if total_energy <= 0:
            logger.warning("Signal %s:%s has zero total energy, skipping", role, name)

            return None

        sorted_indices = np.argsort(mean_energy)[::-1]  # Descending

        return shape, key, acc_count, mean_energy, total_energy, sorted_indices

    # ------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def _compute_target_k(
        *,
        role: str,
        name: str,
        sorted_indices: np.ndarray,
        mean_energy: np.ndarray,
        total_energy: float,
        target_energy: float,
    ) -> tuple[int, np.ndarray]:
        """
        Compute K required to meet explained-energy target.

        Parameters
        ----------
        role : str
            Signal role.
        name : str
            Canonical signal name.
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending).
        mean_energy : np.ndarray
            Mean signal energy.
        total_energy : float
            Total signal energy.
        target_energy : float
            Target cumsum(mean_energy) / total_energy ratio.

        Returns
        -------
        tuple[int, np.ndarray]
            Target K and cumulatively summed signal energy.

        """

        cumsum_energy = np.cumsum(mean_energy[sorted_indices])
        explained_ratio = cumsum_energy / total_energy

        meets_target = np.nonzero(explained_ratio >= target_energy)[0]
        if len(meets_target) > 0:
            k_target = int(meets_target[0]) + 1  # +1 because index is 0-based
        else:
            k_target = len(sorted_indices)
            logger.warning(
                "Signal %s:%s cannot reach target %.4f (max: %.4f), using all %d coeffs",
                role,
                name,
                target_energy,
                explained_ratio[-1],
                k_target,
            )

        return k_target, cumsum_energy

    # ------------------------------------------------------------------------------------------------------------------
    def _compute_guardrail_min_k(
        self,
        *,
        shape: tuple[int, ...],
        sorted_indices: np.ndarray,
        H: int,  # noqa - Ignore lowercase warning
        W: int,  # noqa - Ignore lowercase warning
        T: int,  # noqa - Ignore lowercase warning
        role: str,
        name: str,
    ) -> int:
        """
        Compute minimum K required by guardrails for one signal.

        Parameters
        ----------
        shape : tuple[int, ...]
            Signal shape.
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending).
        H, W, T : int
            Coefficient shape (matching DCT grid dimensions).
        role : str
            Signal role.
        name : str
            Canonical signal name.

        Returns
        -------
        int
            Minimum K to satisfy dimension guardrail requirements.

        """

        if not self.guardrails_enable:
            return 0

        signal_type = self._determine_signal_type(shape)

        return self._apply_dimension_guardrails(
            sorted_indices=sorted_indices,
            H=H,
            W=W,
            T=T,
            signal_type=signal_type,
            role=role,
            name=name,
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _resolve_k_selection(
        self,
        *,
        role: str,
        name: str,
        sorted_indices: np.ndarray,
        W: int,  # noqa - Ignore lowercase warning
        T: int,  # noqa - Ignore lowercase warning
        k_target: int,
        guardrail_min_k: int,
        budget: int | None,
        target_energy: float,
    ) -> _KSelection:
        """
        Apply guardrail lift and budget cap to produce final K selection.

        Parameters
        ----------
        role : str
            Signal role.
        name : str
            Canonical signal name.
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending).
        W, T : int
            Coefficient shape dimensions (width and time).
        k_target : int
            Target K for the calculation of the final K selection policy.
        guardrail_min_k : int
            Minimum number of coefficients enforced by guardrails, regardless of the energy target.
        budget : int | None
            Maximum number of coefficients allowed. If None, no budget cap is applied.
        target_energy : float
            Target cumsum(mean_energy) / total_energy ratio (for logging).

        Returns
        -------
        _KSelection
            Resulting K selection policy for the passed parameters.

        """

        k_after_guardrails = k_target
        guardrail_increased_k = False
        if self.guardrails_enable and (guardrail_min_k > k_target):
            guardrail_increased_k = True
            k_after_guardrails = guardrail_min_k
            coverage_before = _coverage_shape_for_top_k_indices(sorted_indices=sorted_indices, K=k_target, W=W, T=T)
            coverage_after = _coverage_shape_for_top_k_indices(
                sorted_indices=sorted_indices, K=k_after_guardrails, W=W, T=T
            )
            logger.info(
                "Signal %s:%s hit guardrails: K %d -> %d | coverage(H,W,T) %s -> %s",
                role,
                name,
                k_target,
                k_after_guardrails,
                coverage_before,
                coverage_after,
            )

        k_final = k_after_guardrails
        budget_capped = False
        budget_violated_for_guardrails = False

        if (budget is not None) and (k_after_guardrails > budget):
            budget_capped = True
            k_final = int(budget)
            guardrail_driven = self.guardrails_enable and (guardrail_min_k > k_target)
            if guardrail_driven:
                budget_violated_for_guardrails = True
                logger.warning(
                    "Signal %s:%s: guardrails need K=%d but budget=%d. "
                    "Capping to budget=%d (guardrails not fully satisfied).",
                    role,
                    name,
                    k_after_guardrails,
                    budget,
                    k_final,
                )
            else:
                logger.warning(
                    "Signal %s:%s: target=%.4f needs K=%d but budget=%d. Capping to budget=%d.",
                    role,
                    name,
                    target_energy,
                    k_after_guardrails,
                    budget,
                    k_final,
                )

        return _KSelection(
            k_target=int(k_target),
            guardrail_min_k=int(guardrail_min_k),
            k_after_guardrails=int(k_after_guardrails),
            k_final=int(k_final),
            guardrail_increased_k=bool(guardrail_increased_k),
            budget_capped=bool(budget_capped),
            budget_violated_for_guardrails=bool(budget_violated_for_guardrails),
        )

    # ------------------------------------------------------------------------------------------------------------------
    def _build_selection_payload(
        self,
        *,
        sorted_indices: np.ndarray,
        cumsum_energy: np.ndarray,
        total_energy: float,
        H: int,  # noqa - Ignore lowercase warning
        W: int,  # noqa - Ignore lowercase warning
        T: int,  # noqa - Ignore lowercase warning
        selection: _KSelection,
        target_energy: float,
        n_windows: int,
        budget: int | None,
    ) -> dict[str, Any]:
        """
        Build per-signal result payload from selection decision.

        Parameters
        ----------
        sorted_indices : np.ndarray
            Coefficient indices sorted by energy (descending).
        cumsum_energy : np.ndarray
            Cumulatively summed signal energy.
        total_energy : float
            Total signal energy.
        H, W, T : int
            Coefficient shape (matching DCT grid dimensions).
        selection : _KSelection
            K selection policy.
        target_energy : float
            Target fraction of total energy to be explained (i.e., cumsum(mean_energy) / total_energy), in [0, 1].
        n_windows : int
            Number of windows used to accumulate energy statistics.
        budget : int | None
            Maximum number of coefficients allowed. If None, no budget cap is applied.

        Returns
        -------
        dict[str, Any]
            Resulting selection payload.

        """

        coeff_indices = sorted_indices[: selection.k_final].astype(np.int32)
        achieved_energy = float(cumsum_energy[selection.k_final - 1] / total_energy)
        indices_3d = np.unravel_index(coeff_indices, shape=(H, W, T))
        flags = self._selection_flags(
            guardrail_increased_k=selection.guardrail_increased_k,
            budget_capped=selection.budget_capped,
            budget_violated_for_guardrails=selection.budget_violated_for_guardrails,
        )

        return {
            "coeff_indices": coeff_indices,
            "coeff_shape": (H, W, T),
            "num_coeffs": selection.k_final,
            "explained_energy": achieved_energy,
            "target_energy": target_energy,
            "n_windows": n_windows,
            "max_budget": budget,
            "k_target": selection.k_target,
            "guardrail_min_k": selection.guardrail_min_k,
            "k_after_guardrails": selection.k_after_guardrails,
            "guardrail_increased_k": selection.guardrail_increased_k,
            "budget_capped": selection.budget_capped,
            "budget_violated_for_guardrails": selection.budget_violated_for_guardrails,
            "dim_distribution": {
                "unique_h": int(len(np.unique(indices_3d[0]))),
                "unique_w": int(len(np.unique(indices_3d[1]))),
                "unique_t": int(len(np.unique(indices_3d[2]))),
            },
            "flags": flags,
        }

    # ------------------------------------------------------------------------------------------------------------------
    def select_best(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Select top-K coefficients by variance for each signal.

        Returns
        -------
        dict[str, dict[str, dict[str, Any]]]
            Dictionary with structure:
                {role: {signal_name: {
                    "coeff_indices": np.ndarray,  # top-K indices
                    "coeff_shape": (H, W, T),
                    "num_coeffs": K,
                    "explained_energy": float,
                    "target_energy": float,
                    "n_windows": int,
                    "max_budget": int | None,
                    "k_target": int,
                    "guardrail_min_k": int,
                    "k_after_guardrails": int,
                    "guardrail_increased_k": bool,
                    "budget_capped": bool,
                    "budget_violated_for_guardrails": bool,
                    "dim_distribution": {"unique_h": int, "unique_w": int, "unique_t": int},
                    "flags": list[str]
                }}}

        Raises
        ------
        KeyError
            If a given role in `self.roles` is not listed in `self.targets`.

        """

        out: dict[str, dict[str, dict[str, Any]]] = {}

        for role in self.roles:
            by_sig = self.coeff_energy.get(role, {})
            if role not in self.targets:
                raise KeyError(f"[TuneRankedDCT3DTransform] Missing target for role={role!r}.")

            target_energy = float(self.targets[role])
            budget = self.max_budget.get(role)

            for name, by_shape in by_sig.items():
                prepared = self._prepare_signal_selection(role=role, name=name, by_shape=by_shape)
                if prepared is None:
                    continue
                shape, key, acc_count, mean_energy, total_energy, sorted_indices = prepared
                H, W, T = key

                guardrail_min_k = self._compute_guardrail_min_k(
                    shape=shape,
                    sorted_indices=sorted_indices,
                    H=H,
                    W=W,
                    T=T,
                    role=role,
                    name=name,
                )
                k_target, cumsum_energy = self._compute_target_k(
                    role=role,
                    name=name,
                    sorted_indices=sorted_indices,
                    mean_energy=mean_energy,
                    total_energy=total_energy,
                    target_energy=target_energy,
                )
                selection = self._resolve_k_selection(
                    role=role,
                    name=name,
                    sorted_indices=sorted_indices,
                    W=W,
                    T=T,
                    k_target=k_target,
                    guardrail_min_k=guardrail_min_k,
                    budget=budget,
                    target_energy=target_energy,
                )

                out.setdefault(role, {})[name] = self._build_selection_payload(
                    sorted_indices=sorted_indices,
                    cumsum_energy=cumsum_energy,
                    total_energy=total_energy,
                    H=H,
                    W=W,
                    T=T,
                    selection=selection,
                    target_energy=target_energy,
                    n_windows=acc_count,
                    budget=budget,
                )

        return out
