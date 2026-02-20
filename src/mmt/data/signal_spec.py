"""
Signal specification and registry utilities

This module defines the core "signal spec" abstraction used throughout MMT.

A SignalSpec is a compact, stable description of a signal:
  - canonical name (e.g. "pf_active-coil_current")
  - integer signal_id (stable within a task/config)
  - role: input / actuator / output
  - modality: timeseries / profile / video
  - embedding parameters (e.g. DCT keep coefficients)
  - window/chunk metadata used by the model and collate (dt, stride, etc.)

The SignalSpecRegistry is the single source of truth that maps between:
  - canonical signal names ↔ integer ids
  - ids ↔ role/modality
  - ids ↔ embedding configuration

Builders
--------
This module provides builders that construct the registry deterministically from:
  - a role→signal mapping (signals_by_role) provided by the dataset adapter,
  - metadata (sampling rate, shapes),
  - embedding configuration.

Adapter contract
----------------
MMT core is dataset-agnostic, but it expects the dataset integration layer to
provide two inputs:

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

Dataset-specific task YAML parsing (e.g. FAIRMAST config_task_*.yaml) lives
outside the core package (e.g. in scripts_mast/).

Motivation
----------
MMT routes and masks signals using integer ids (fast, stable, compact). Names
are used only at the boundaries (configs, logging, saving results). The registry
bridges those worlds consistently across training, evaluation, and warm-start.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import logging

from mmt.data.embeddings.codec_utils import compute_embedding_dim_for_encoder

logger = logging.getLogger("mmt.SignalSpec")


# ------------------------------------------------------------------
# Signal specification
# ------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """
    A specification for a single *role-specific* signal.

    Parameters
    ----------
    name : str
        Canonical name, e.g. "pf_active-coil_current".

    role : str
        One of {"input", "actuator", "output"}.

    modality : str
        One of {"timeseries", "profile", "video"}.

    encoder_name : str
        Name of the embedding codec to use (e.g., "dct3d", "identity").

    encoder_kwargs : Dict[str, Any]
        Codec-specific parameters used to configure the codec.

    signal_id : int
        Unique integer identifier **per (role, name)**. This ID is used in the
        token pipeline and metadata arrays.

    embedding_dim : int
        Dimension of the codec output for a single chunk.

    Notes
    -----
    A single physical signal may appear in multiple roles (e.g. as input and
    as output). Each role receives its own SignalSpec and its own `signal_id`
    because the embedding parameters and output dimensions may differ.
    """

    name: str
    role: str
    modality: str
    encoder_name: str
    encoder_kwargs: Dict[str, Any]
    signal_id: int
    embedding_dim: int

    @property
    def canonical_key(self) -> str:
        """
        Return a stable, human-readable key uniquely identifying this
        role-specific signal.

        This key is used to index learnable per-signal modules
        (e.g. projection layers, adapters) so that module names remain
        consistent and independent of the internal numeric `signal_id`.
        """
        return f"{self.role}:{self.name}"


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------


class SignalSpecRegistry:
    """
    Container for all role-specific SignalSpec objects.

    Provides:
        - get_by_id(signal_id) → SignalSpec
        - get(role, name)      → SignalSpec
        - specs_for_role(role) → list of SignalSpec
        - num_signals, roles, modalities
    """

    def __init__(self, specs: List[SignalSpec]) -> None:
        self._specs: List[SignalSpec] = list(specs)

        self._by_id: Dict[int, SignalSpec] = {}
        self._by_role_name: Dict[Tuple[str, str], SignalSpec] = {}
        self._by_role: Dict[str, List[SignalSpec]] = {}

        for spec in self._specs:
            # Each role-specific instance must have a unique ID.
            if spec.signal_id in self._by_id:
                raise ValueError(
                    f"Duplicate signal_id={spec.signal_id} for {spec.role}:{spec.name}"
                )

            key = (spec.role, spec.name)
            if key in self._by_role_name:
                raise ValueError(f"Duplicate entry for (role,name)={key}")

            self._by_id[spec.signal_id] = spec
            self._by_role_name[key] = spec
            self._by_role.setdefault(spec.role, []).append(spec)

    # Basic lookups ---------------------------------------------------------

    @property
    def specs(self) -> List[SignalSpec]:
        return list(self._specs)

    def get_by_id(self, signal_id: int) -> Optional[SignalSpec]:
        return self._by_id.get(signal_id)

    def get(self, role: str, name: str) -> Optional[SignalSpec]:
        return self._by_role_name.get((role, name))

    def specs_for_role(self, role: str) -> List[SignalSpec]:
        return list(self._by_role.get(role, []))

    # Cardinalities ---------------------------------------------------------

    @property
    def num_signals(self) -> int:
        return len(self._specs)

    @property
    def roles(self) -> List[str]:
        return sorted(self._by_role.keys())

    @property
    def modalities(self) -> List[str]:
        return sorted({spec.modality for spec in self._specs})


# ------------------------------------------------------------------
# Registry Builder
# ------------------------------------------------------------------


def build_signal_specs(
    embeddings_cfg: Mapping[str, Any],
    signals_by_role: Mapping[str, Mapping[str, str]],
    dict_metadata: Mapping[str, Mapping[str, Any]],
    chunk_length_sec: float,
) -> SignalSpecRegistry:
    """
    Build a SignalSpecRegistry from embedding config + adapter-provided signal lists.

    This function is part of the MMT core. It does not parse dataset-specific task
    YAMLs. Instead, it expects the dataset integration layer to provide:
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
        Chunk length used for input/actuator chunking (seconds). This is used to
        derive chunk counts/strides and to validate embeddings.

    Returns
    -------
    SignalSpecRegistry
        Registry containing all SignalSpecs with stable ids, roles, modalities,
        and embedding parameters.
    """

    # YAML may parse "null" as None -> normalize to {}
    defaults = embeddings_cfg.get("defaults") or {}
    overrides = embeddings_cfg.get("per_signal_overrides") or {}

    specs: List[SignalSpec] = []
    next_id = 0  # Unique ID per (role, name)

    # Deterministic ordering: sorted by role, then by name
    for role in sorted(signals_by_role.keys()):
        for name in sorted(signals_by_role[role].keys()):
            modality = signals_by_role[role][name]

            role_meta = dict_metadata.get(role)
            if role_meta is None:
                raise KeyError(f"dict_metadata missing role={role!r}")
            meta = role_meta.get(name)

            values_shape = tuple(meta.get("values_shape", ()))
            dt = float(meta.get("dt"))

            # --- Load default encoder settings for (role, modality)
            role_defaults = defaults.get(role, {})
            modality_defaults = role_defaults.get(modality)
            if modality_defaults is None:
                raise KeyError(
                    f"No default embedding settings for role={role}, modality={modality}"
                )

            encoder_name = modality_defaults.get("encoder_name")
            encoder_kwargs = dict(modality_defaults.get("encoder_kwargs", {}) or {})

            if encoder_name is None:
                raise KeyError(
                    f"Missing encoder_name for role={role}, modality={modality}"
                )

            # --- Apply per-signal overrides
            role_overrides = overrides.get(role) or {}
            sig_override = role_overrides.get(name)
            if isinstance(sig_override, dict):
                encoder_name = sig_override.get("encoder_name", encoder_name)
                encoder_kwargs = dict(
                    sig_override.get("encoder_kwargs", encoder_kwargs) or {}
                )

            # --- Embedding length: chunk-level for inputs/actuators, window-level for outputs
            length_sec = float(chunk_length_sec)
            if role == "output":
                length_sec = float(meta.get("sec_length", length_sec))

            # --- Compute embedding dimension
            embedding_dim = compute_embedding_dim_for_encoder(
                encoder_name=encoder_name,
                encoder_kwargs=encoder_kwargs,
                values_shape=values_shape,
                dt=dt,
                chunk_length_sec=length_sec,
            )

            # --- Create the SignalSpec
            spec = SignalSpec(
                name=name,
                role=role,
                modality=modality,
                encoder_name=encoder_name,
                encoder_kwargs=encoder_kwargs,
                signal_id=next_id,
                embedding_dim=int(embedding_dim),
            )
            specs.append(spec)
            next_id += 1

    registry = SignalSpecRegistry(specs)

    # Logging summary
    logger.info(
        "Built SignalSpecRegistry with %d role-specific signals", registry.num_signals
    )
    for spec in registry.specs:
        logger.info(
            "  • %-30s | role=%-8s | id=%3d | modality=%-10s | encoder=%s | dim=%s",
            spec.name,
            spec.role,
            spec.signal_id,
            spec.modality,
            spec.encoder_name,
            spec.embedding_dim,
        )

    return registry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def infer_modality(values_shape: Tuple[int, ...]) -> str:
    """
    Infer signal modality from its spatial shape.

    Conventions:
      () or (1,)          → timeseries
      (C,) with C > 1     → profile
      (H, W)              → video/map
    """
    if len(values_shape) == 0:
        return "timeseries"
    if len(values_shape) == 1:
        return "timeseries" if values_shape[0] == 1 else "profile"
    if len(values_shape) == 2:
        return "video"
    raise ValueError(f"Unsupported values_shape={values_shape!r}")


def canonical_key(role: str, name: str) -> str:
    """
    Build a stable, human-readable key for a (role, name) pair.

    This is used by the MMT model to key all per-signal learnable modules
    (input projections, missing token embeddings, output adapters) so that:
        • warm-starting is semantically correct,
        • adapters are uniquely associated with each physical+role signal,
        • ordering changes in SignalSpecRegistry do NOT affect loaded weights.
    """
    return f"{role}:{name}"
