"""
Signal specification and registry utilities

This module defines the core "signal spec" abstraction used throughout MMT.

A SignalSpec is a compact, stable description of a signal:
  - canonical name (e.g., "pf_active-coil_current")
  - integer signal_id (stable within a task/config)
  - role: input / actuator / output
  - modality: timeseries / profile / video
  - embedding parameters (e.g., DCT keep coefficients)
  - window/chunk metadata used by the model and collate (dt, stride, etc.)

The SignalSpecRegistry is the single source of truth that maps between:
  - canonical signal names ↔ integer IDs
  - IDs ↔ role/modality
  - IDs ↔ embedding configuration

Builders
--------
This module provides builders that construct the registry deterministically from:
  - a role→signal mapping (signals_by_role) provided by the dataset adapter,
  - metadata (sampling rate, shapes),
  - embedding configuration.

Adapter contract
----------------
MMT core is dataset-agnostic, but it expects the dataset integration layer to provide two inputs:

1) signals_by_role:
   Mapping[str, Mapping[str, str]]
       role -> { canonical_signal_name -> modality }

   where role ∈ {"input", "actuator", "output"} and
         modality ∈ {"timeseries", "profile", "video"}.

2) dict_metadata:
   Mapping[str, Mapping[str, Any]]
       dict_metadata[role][canonical_signal_name] must contain at least:
         - "dt": float
         - "values_shape": tuple[int, ...]
       and for role == "output" it must also contain:
         - "sec_length": float  (the target window length in seconds)

Dataset-specific task YAML parsing (e.g., FAIR MAST config_task_*.yaml) lives outside the core package (e.g., in
scripts_mast/).

Motivation
----------
MMT routes and masks signals using integer IDs (fast, stable, compact). Names are used only at the boundaries (configs,
logging, saving results). The registry bridges those worlds consistently across training, evaluation, and warm-start.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, cast
from collections.abc import Mapping
import logging

from mmt.data.embeddings.codec_utils import (
    compute_embedding_dim_for_encoder,
    infer_hw_from_values_shape,
)


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.SignalSpec")


# ======================================================================================================================
# Signal specification
# ======================================================================================================================


# ======================================================================================================================
@dataclass(frozen=True)
class SignalSpec:
    """
    A specification for a single *role-specific* signal.

    Parameters
    ----------
    name : str
        Canonical signal name, e.g., "pf_active-coil_current".
    role : str
        One of {"input", "actuator", "output"}.
    modality : str
        One of {"timeseries", "profile", "video"}.
    encoder_name : str
        Name of the embedding codec to use (e.g., "dct3d", "identity").
    encoder_kwargs : dict[str, Any]
        Codec-specific parameters used to configure the codec.
    signal_id : int
        Unique integer identifier **per (role, name)**. This ID is used in the token pipeline and metadata arrays.
    embedding_dim : int
        Dimension of the codec output for a single chunk.

    Notes
    -----
    A single physical signal may appear in multiple roles (e.g., as input and as output). Each role receives its own
    SignalSpec and its own `signal_id` because the embedding parameters and output dimensions may differ.

    """

    name: str
    role: str
    modality: str
    encoder_name: str
    encoder_kwargs: dict[str, Any]
    signal_id: int
    embedding_dim: int

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def canonical_key(self) -> str:
        """
        Return a stable, human-readable key uniquely identifying this role-specific signal.

        This key is used to index learnable per-signal modules (e.g., projection layers, adapters) so that module names
        remain consistent and independent of the internal numeric `signal_id`.

        """

        return f"{self.role}:{self.name}"


# ======================================================================================================================
# Registry
# ======================================================================================================================


# ======================================================================================================================
class SignalSpecRegistry:
    """
    Container for all role-specific SignalSpec objects.

    Provides:
        - get_by_id(signal_id) → SignalSpec
        - get(role, name)      → SignalSpec
        - specs_for_role(role) → list of SignalSpec
        - num_signals, roles, modalities

    Attributes
    ----------
    _specs : list[SignalSpec]
        Flat list of all registered SignalSpec objects.
    _by_id : dict[int, SignalSpec]
        Lookup by numeric signal ID.
    _by_role_name : dict[tuple[str, str], SignalSpec]
        Lookup by (role, name) key.
    _by_role : dict[str, list[SignalSpec]]
        Lookup by role, returning all specs for that role.

    Methods
    -------
    get_by_id(signal_id)
        Return a SignalSpec by numeric signal ID, or None if not found.
    get(role, name)
        Return a SignalSpec by (role, name) key, or None if not found.
    specs_for_role(role)
        Return all SignalSpec objects for a given role.
    num_signals()
        Total number of registered signals (property).
    roles()
        Sorted list of roles present in the registry (property).
    modalities()
        Sorted list of distinct modalities present in the registry (property).

    """

    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self, specs: list[SignalSpec]) -> None:
        """
        Initialize class attributes.

        Parameters
        ----------
        specs : list[SignalSpec]
            List of SignalSpec objects.

        Returns
        -------
        # None  # REMARK: Commented out to avoid type checking errors.

        Raises
        ------
        ValueError
            If a duplicate signal ID is identified for different specs items.
            If a duplicate entry is identified for a (role, name) key for different specs items.
        """

        self._specs: list[SignalSpec] = list(specs)

        self._by_id: dict[int, SignalSpec] = {}
        self._by_role_name: dict[tuple[str, str], SignalSpec] = {}
        self._by_role: dict[str, list[SignalSpec]] = {}

        for spec in self._specs:
            # Each role-specific instance must have a unique ID.
            if spec.signal_id in self._by_id:
                raise ValueError(f"Duplicate signal_id={spec.signal_id} for {spec.role}:{spec.name}.")

            key = (spec.role, spec.name)
            if key in self._by_role_name:
                raise ValueError(f"Duplicate entry for (role, name)={key}.")

            self._by_id[spec.signal_id] = spec
            self._by_role_name[key] = spec
            self._by_role.setdefault(spec.role, []).append(spec)

    # ==================================================================================================================
    # Basic lookups
    # ==================================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def specs(self) -> list[SignalSpec]:
        """List all specifications."""
        return list(self._specs)

    # ------------------------------------------------------------------------------------------------------------------
    def get_by_id(self, signal_id: int) -> Optional[SignalSpec]:
        """Return specifications by signal ID key."""
        return self._by_id.get(signal_id)

    # ------------------------------------------------------------------------------------------------------------------
    def get(self, role: str, name: str) -> Optional[SignalSpec]:
        """Return specifications by role-name key."""
        return self._by_role_name.get((role, name))

    # ------------------------------------------------------------------------------------------------------------------
    def specs_for_role(self, role: str) -> list[SignalSpec]:
        """List specifications for a given role."""
        return list(self._by_role.get(role, []))

    # ==================================================================================================================
    # Cardinalities
    # ==================================================================================================================

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def num_signals(self) -> int:
        """Number of signals."""
        return len(self._specs)

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def roles(self) -> list[str]:
        """Sorted list of roles."""
        return sorted(self._by_role.keys())

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def modalities(self) -> list[str]:
        """Sorted set of modalities."""
        return sorted({spec.modality for spec in self._specs})


# ======================================================================================================================
# Registry Builder
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_signal_specs(  # NOSONAR - Ignore cognitive complexity
    embeddings_cfg: Mapping[str, Any],
    signals_by_role: Mapping[str, Mapping[str, str]],
    dict_metadata: Mapping[str, Mapping[str, Any]],
    chunk_length_sec: float,
    log_summary: bool = True,
) -> SignalSpecRegistry:
    """
    Build a SignalSpecRegistry from embedding config + adapter-provided signal lists.

    This function is part of the MMT core. It does not parse dataset-specific task YAMLs. Instead, it expects the
    dataset integration layer to provide:
      - signals_by_role: role -> {canonical_name -> modality}
      - dict_metadata:   role -> {canonical_name -> meta}

    Parameters
    ----------
    embeddings_cfg:
        Embedding configuration dictionary (defaults + optional per-signal overrides).
        Expected to follow the structure used in mmt/configs/embeddings_*.yaml.

    signals_by_role:
        Mapping of role -> {canonical_signal_name -> modality}.
        role must be one of {"input", "actuator", "output"}.

    dict_metadata:
        Mapping of role -> {canonical_signal_name -> meta}.
        For each (role, name), meta must include:
          - "dt": float
          - "values_shape": tuple[int, ...]
        Additionally, for role == "output", meta must include:
          - "sec_length": float  (target window length in seconds)

    chunk_length_sec:
        Chunk length used for input/actuator chunking (seconds). This is used to derive chunk counts/strides and to
        validate embeddings.
    log_summary:
        If True, print the grouped SignalSpec summary to `mmt.SignalSpec` logger.

    Returns
    -------
    SignalSpecRegistry
        Registry containing all SignalSpecs with stable IDs, roles, modalities, and embedding parameters.

    """

    # YAML may parse "null" as None -> normalize to {}.
    defaults = embeddings_cfg.get("defaults") or {}
    overrides = embeddings_cfg.get("per_signal_overrides") or {}

    specs: list[SignalSpec] = []
    native_shape_by_key: dict[tuple[str, str], tuple[int, int, int]] = {}
    next_id = 0  # Unique ID per (role, name).

    # Deterministic ordering: sorted by role, then by name.
    for role in sorted(signals_by_role.keys()):
        for name in sorted(signals_by_role[role].keys()):
            modality = signals_by_role[role][name]

            role_meta = dict_metadata.get(role)
            if role_meta is None:
                raise KeyError(f"`dict_metadata` missing role={role!r}")
            meta = role_meta.get(name)

            values_shape = tuple(meta.get("values_shape", ()))
            dt = float(meta.get("dt"))

            # ..........................................................................................................
            # Load default encoder settings for (role, modality)

            role_defaults = defaults.get(role, {})
            modality_defaults = role_defaults.get(modality)
            if modality_defaults is None:
                raise KeyError(f"No default embedding settings for role={role}, modality={modality}")

            encoder_name = modality_defaults.get("encoder_name")
            encoder_kwargs = dict(modality_defaults.get("encoder_kwargs", {}) or {})

            if encoder_name is None:
                raise KeyError(f"Missing encoder_name for role={role}, modality={modality}")

            # ..........................................................................................................
            # Apply per-signal overrides

            role_overrides = overrides.get(role) or {}
            sig_override = role_overrides.get(name)
            if isinstance(sig_override, dict):
                encoder_name = sig_override.get("encoder_name", encoder_name)
                encoder_kwargs = dict(sig_override.get("encoder_kwargs", encoder_kwargs) or {})

            # ..........................................................................................................
            # Embedding length: chunk-level for inputs/actuators, window-level for outputs

            length_sec = float(chunk_length_sec)
            if role == "output":
                # Output window is taken directly from metadata
                length_sec = float(meta["length_in_seconds"])

            # ..........................................................................................................
            # Compute embedding dimension

            embedding_dim = compute_embedding_dim_for_encoder(
                encoder_name=encoder_name,  # type: ignore[arg-type]
                encoder_kwargs=encoder_kwargs,
                values_shape=values_shape,
                dt=dt,
                chunk_length_sec=length_sec,
            )

            # Native shape used by codec logic: canonical (H, W, T).
            n_samples = max(1, int(round(float(length_sec) / float(dt))))
            h_native, w_native = infer_hw_from_values_shape(values_shape=values_shape)
            native_shape = (int(h_native), int(w_native), int(n_samples))
            native_shape_by_key[(role, name)] = native_shape

            # ..........................................................................................................
            # Create the SignalSpec

            spec = SignalSpec(
                name=name,
                role=role,
                modality=modality,
                encoder_name=encoder_name,  # type: ignore[arg-type]
                encoder_kwargs=encoder_kwargs,
                signal_id=next_id,
                embedding_dim=int(embedding_dim),
            )
            specs.append(spec)
            next_id += 1

    registry = SignalSpecRegistry(specs=specs)

    if log_summary:
        _log_signal_spec_summary(registry=registry, native_shape_by_key=native_shape_by_key)

    return registry


# ======================================================================================================================
# Helpers
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def _format_explained_energy(encoder_kwargs: Mapping[str, Any]) -> str:
    """
    Format explain energy from encoder mapping (dict).

    Parameters
    ----------
    encoder_kwargs : Mapping[str, Any]
        Input encoder mapping (dict).

    Returns
    -------
    str
        Formatted explained energy.

    """

    v = encoder_kwargs.get("explained_energy")
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"

    return "-"


# ----------------------------------------------------------------------------------------------------------------------
def _log_signal_spec_summary(
    registry: SignalSpecRegistry,
    *,
    native_shape_by_key: Mapping[tuple, tuple] | None = None,
) -> None:
    """
    Log signal spec summary.

    Parameters
    ----------
    registry : SignalSpecRegistry
        SignalSpecRegistry instance used for logging.
    native_shape_by_key : Mapping[tuple, tuple] | None
        Native shape mapping (dict) with role-name keys.
        Optional. Default: None.

    Returns
    -------
    None

    """

    logger.info("Built SignalSpecRegistry with %d role-specific signals", registry.num_signals)
    role_order = [r for r in ("input", "actuator", "output") if r in registry.roles]
    for role in role_order:
        specs = sorted(registry.specs_for_role(role), key=lambda s: s.signal_id)
        logger.info("%s:", role)
        logger.info("  id | name | modality | encoder | native_shape | encoded_dim | expl_energy")
        for spec in specs:
            encoder_kwargs = spec.encoder_kwargs if isinstance(spec.encoder_kwargs, dict) else {}
            native_shape = "-"
            if native_shape_by_key is not None:
                native_shape_by_key = cast(dict, native_shape_by_key)
                ns = native_shape_by_key.get((spec.role, spec.name))
                if ns is not None:
                    native_shape = str(tuple(ns))
            logger.info(
                "  id=%d | name=%s | modality=%s | encoder=%s | native_shape=%s | encoded_dim=%d | expl_energy=%s",
                int(spec.signal_id),
                str(spec.name),
                str(spec.modality),
                str(spec.encoder_name),
                native_shape,
                int(spec.embedding_dim),
                _format_explained_energy(encoder_kwargs=encoder_kwargs),
            )


# ----------------------------------------------------------------------------------------------------------------------
def infer_modality(values_shape: tuple[int, ...]) -> str:
    """
    Infer signal modality from its spatial shape.

    Conventions:
      () or (1,)          → timeseries
      (C,) with C > 1     → profile
      (H, W)              → video/map

    Parameters
    ----------
    values_shape : tuple[int, ...]
        Values shape used to infer modality.

    Returns
    -------
    str
        Inferred modality.

    """

    if len(values_shape) == 0:
        return "timeseries"
    if len(values_shape) == 1:
        return "timeseries" if values_shape[0] == 1 else "profile"
    if len(values_shape) == 2:
        return "video"
    raise ValueError(f"Unsupported values_shape={values_shape!r}")


# ----------------------------------------------------------------------------------------------------------------------
def canonical_key(role: str, name: str) -> str:
    """

    Build a stable, human-readable key for a (role, name) pair.

    This is used by the MMT model to key all per-signal learnable modules (input projections, missing token embeddings,
    output adapters) so that:
        • warm-starting is semantically correct,
        • adapters are uniquely associated with each physical+role signal,
        • ordering changes in SignalSpecRegistry do NOT affect loaded weights.

    Parameters
    ----------
    role : str
        Signal role.
    name : str
        Signal name.

    Returns
    -------
    str
        Resulting role-name key.

    """

    return f"{role}:{name}"
