"""
Signal specification and registry utilities
===========================================

This module defines:

  • `SignalSpec`: a frozen dataclass describing one *role-specific* signal
                  in the MMT pipeline (e.g., input coil_current,
                  output coil_current).

  • `SignalSpecRegistry`: a container providing lookup by id, by (role,name),
                          and by role, used throughout the tokenization and
                          embedding stages.

  • Builders that construct the registry deterministically from:
        - task configuration (signals and roles),
        - metadata (sampling rate, shapes),
        - embedding configuration.

Motivation
----------
A single *physical* signal (e.g., "pf_active-coil_current") may appear in
multiple roles:

    - as INPUT (past context),
    - as ACTUATOR (control signal for prediction),
    - as OUTPUT (supervised target to reconstruct/forecast).

These roles can produce embeddings of *different* dimensionalities.
Therefore, each (role, name) pair must receive its own signal_id, so
that encoders, projection layers, and token embeddings remain consistent
and unambiguous.

This registry makes that explicit: **each (role, name) is a distinct SignalSpec.**

The model still learns that different roles of the same physical signal
are semantically related via modality embeddings, context, and attention.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple
import logging

from mmt.data.embeddings.codec_utils import compute_embedding_dim_for_encoder

logger = logging.getLogger("mmt.SignalSpec")


# ---------------------------------------------------------------------------
# Signal specification
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Registry Builder
# ---------------------------------------------------------------------------


def build_signal_specs(
    embeddings_cfg: Mapping[str, Any],
    signals_by_role: Mapping[str, Mapping[str, str]],
    dict_metadata: Mapping[str, Mapping[str, Any]],
    chunk_length_sec: float,
) -> SignalSpecRegistry:
    """
    Construct a SignalSpecRegistry from:

        - embedding defaults + per-signal overrides,
        - task role-to-signal mapping,
        - metadata describing dt and spatial shape.

    Each (role, name) pair becomes one SignalSpec with a unique ID.
    Roles do NOT share IDs, even if their names coincide.

    This avoids embedding-dimension collisions between input/actuator/output
    versions of the same physical variable.
    """

    defaults = embeddings_cfg.get("defaults", {})
    overrides = embeddings_cfg.get("per_signal_overrides", {})

    specs: List[SignalSpec] = []

    next_id = 0  # Unique ID per (role, name)

    # Deterministic ordering: sorted by role, then by name
    for role in sorted(signals_by_role.keys()):
        for name in sorted(signals_by_role[role].keys()):
            modality = signals_by_role[role][name]

            meta = dict_metadata.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} missing in dict_metadata")

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
            role_overrides = overrides.get(role, {})
            sig_override = role_overrides.get(name)
            if isinstance(sig_override, dict):
                encoder_name = sig_override.get("encoder_name", encoder_name)
                encoder_kwargs = dict(
                    sig_override.get("encoder_kwargs", encoder_kwargs) or {}
                )

            # --- Compute embedding dimension
            embedding_dim = compute_embedding_dim_for_encoder(
                encoder_name=encoder_name,
                encoder_kwargs=encoder_kwargs,
                values_shape=values_shape,
                dt=dt,
                chunk_length_sec=chunk_length_sec,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def build_signal_role_modality_map(
    cfg_task: Mapping[str, Any],
    dict_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    Build the signals_by_role mapping from the task configuration and metadata.

    Expected task config structure:
        sources_and_signals:
            input_name:    [[source, signal], ...]
            actuator_name: [[source, signal], ...]
            output_name:   [[source, signal], ...]

    Returns:
        {
          "input":    { "<name>": "<modality>", ... },
          "actuator": { ... },
          "output":   { ... }
        }
    """
    ss_cfg = cfg_task.get("sources_and_signals", {})
    signals_by_role: Dict[str, Dict[str, str]] = {}

    role_key_pairs = [
        ("input", "input_name"),
        ("actuator", "actuator_name"),
        ("output", "output_name"),
    ]

    for role, key in role_key_pairs:
        pairs = ss_cfg.get(key) or []
        role_map: Dict[str, str] = {}

        for source, signal in pairs:
            name = f"{source}-{signal}"  # canonical name
            meta = dict_metadata.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} missing in dict_metadata")

            val_shape = tuple(meta.get("values_shape", ()))
            modality = infer_modality(val_shape)
            role_map[name] = modality

        signals_by_role[role] = role_map

    return signals_by_role


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
