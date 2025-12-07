# src/mmt/data/signal_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import logging

from mmt.data.embeddings.codec_utils import compute_embedding_dim_for_encoder

logger = logging.getLogger("mmt.SignalSpec")


# ---------------------------------------------------------------------------
# Core dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalSpec:
    """
    Specification for a single signal in the MMT pipeline.

    Parameters
    ----------
    name : str
        Canonical signal name, e.g. "pf_active-solenoid_current".
        This should match the keys used in the baseline / datasets.

    role : str
        One of {"input", "actuator", "output"}.

    modality : str
        One of {"timeseries", "profile", "video"}.

    encoder_name : str
        Name of the encoder / codec to use, e.g. "dct3d", "identity".

    encoder_kwargs : Dict[str, Any]
        Encoder-specific parameters, e.g. {"keep_h": 1, "keep_w": 1, "keep_t": 10}
        for DCT3D.

    signal_id : int
        Stable integer ID for this signal. Used for signal-ID embeddings and
        codec lookup. The assignment is done in `build_signal_specs`.

    embedding_dim : int
        Dimension of the encoded representation for a single chunk of this
        signal. If `codec.encode(chunk)` returns a 1D array of shape (D,),
        then `embedding_dim == D`.
    """

    name: str
    role: str
    modality: str
    encoder_name: str
    encoder_kwargs: Dict[str, Any]
    signal_id: int
    embedding_dim: int


# ---------------------------------------------------------------------------
# Registry / helper container
# ---------------------------------------------------------------------------


class SignalSpecRegistry:
    """
    Small helper around a collection of SignalSpec.

    Provides convenient lookups by (role, name), by signal_id, and by role.
    """

    def __init__(self, specs: List[SignalSpec]) -> None:
        self._specs: List[SignalSpec] = list(specs)

        self._by_id: Dict[int, SignalSpec] = {}
        self._by_role_name: Dict[Tuple[str, str], SignalSpec] = {}
        self._by_role: Dict[str, List[SignalSpec]] = {}

        for spec in self._specs:
            if spec.signal_id in self._by_id:
                raise ValueError(
                    f"Duplicate signal_id={spec.signal_id} for {spec.name}"
                )

            key = (spec.role, spec.name)
            if key in self._by_role_name:
                raise ValueError(f"Duplicate (role, name)={key} in SignalSpecRegistry")

            self._by_id[spec.signal_id] = spec
            self._by_role_name[key] = spec
            self._by_role.setdefault(spec.role, []).append(spec)

    # Basic accessors -------------------------------------------------------

    @property
    def specs(self) -> List[SignalSpec]:
        return list(self._specs)

    def get_by_id(self, signal_id: int) -> Optional[SignalSpec]:
        return self._by_id.get(signal_id, None)

    def get(self, role: str, name: str) -> Optional[SignalSpec]:
        return self._by_role_name.get((role, name), None)

    def specs_for_role(self, role: str) -> List[SignalSpec]:
        return list(self._by_role.get(role, []))

    # Cardinalities useful for the model -----------------------------------

    @property
    def num_signals(self) -> int:
        return len(self._specs)

    @property
    def roles(self) -> List[str]:
        return sorted(set(s.role for s in self._specs))

    @property
    def modalities(self) -> List[str]:
        return sorted(set(s.modality for s in self._specs))


# ---------------------------------------------------------------------------
# Builder from config
# ---------------------------------------------------------------------------


def build_signal_specs(
    embeddings_cfg: Mapping[str, Any],
    signals_by_role: Mapping[str, Mapping[str, str]],
    dict_metadata: Mapping[str, Mapping[str, Any]],
    chunk_length_sec: float,
) -> SignalSpecRegistry:
    """
    Build a SignalSpecRegistry from the embeddings config and known signals.

    Parameters
    ----------
    embeddings_cfg : Mapping[str, Any]
        The *embeddings* block from the merged ExperimentConfig, i.e.
        something like ``cfg_mmt.embeddings``. It must have the structure:

            {
              "defaults": {
                "<role>": {
                  "<modality>": {
                    "encoder_name": "dct3d",
                    "encoder_kwargs": { ... },
                  },
                  ...
                },
                ...
              },
              "per_signal_overrides": {
                "<role>": {
                  "<signal_name>": {
                    "encoder_name": "dct3d",
                    "encoder_kwargs": { ... },
                  },
                  ...
                },
                ...
              },
            }

        where:
          - <role> is one of {"input", "actuator", "output"},
          - <modality> is one of {"timeseries", "profile", "video"}.

    signals_by_role : Mapping[str, Mapping[str, str]]
        A mapping describing which signals exist and their modality, e.g.:

            {
              "input": {
                "pf_active-solenoid_current": "timeseries",
                "pf_active-coil_current": "timeseries",
                ...
              },
              "actuator": {
                "pf_active-coil_voltage": "timeseries",
                ...
              },
              "output": {
                "pf_active-solenoid_current": "timeseries",
                ...
              },
            }

    dict_metadata : Mapping[str, Mapping[str, Any]]
        Metadata dict as returned by initialize_datasets_and_metadata_for_task, e.g.:

            {
              "pf_active-solenoid_current": {
                  "dt": 0.00025,
                  "values_shape": (1,),
              },
              ...
            }

        We use:
          - `dt`           : sampling period (seconds),
          - `values_shape` : spatial shape (excluding time) of each sample.

    chunk_length_sec : float
        Chunk length in seconds used by the chunking transform. Together with
        `dt` and `values_shape`, this allows each encoder to compute the
        dimension of its encoded output for a single chunk.

    Returns
    -------
    SignalSpecRegistry
        A registry containing one SignalSpec per known signal.
    """
    defaults = embeddings_cfg.get("defaults", {})
    overrides = embeddings_cfg.get("per_signal_overrides", {})

    specs: List[SignalSpec] = []
    next_id = 0

    # Deterministic ordering: sort by (role, name)
    for role in sorted(signals_by_role.keys()):
        role_signals = signals_by_role[role]
        for name in sorted(role_signals.keys()):
            modality = role_signals[name]

            meta = dict_metadata.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} not found in dict_metadata")

            values_shape = tuple(meta.get("values_shape", ()))
            dt = float(meta.get("dt"))

            # 1) Get default encoder settings for this (role, modality)
            role_defaults = defaults.get(role, {})
            modality_defaults = role_defaults.get(modality)
            if modality_defaults is None:
                raise KeyError(
                    f"No default embedding settings for role={role!r}, "
                    f"modality={modality!r}"
                )

            encoder_name = modality_defaults.get("encoder_name", None)
            encoder_kwargs = dict(modality_defaults.get("encoder_kwargs", {}) or {})

            if encoder_name is None:
                raise KeyError(
                    f"Missing 'encoder_name' in defaults for role={role!r}, "
                    f"modality={modality!r}"
                )

            # 2) Apply per-signal overrides, if present
            role_overrides = overrides.get(role, {})
            sig_override = role_overrides.get(name)

            if isinstance(sig_override, dict):
                encoder_name = sig_override.get("encoder_name", encoder_name)
                encoder_kwargs = dict(
                    sig_override.get("encoder_kwargs", encoder_kwargs) or {}
                )

            # 3) Compute embedding_dim via encoder-specific helper
            embedding_dim = compute_embedding_dim_for_encoder(
                encoder_name=encoder_name,
                encoder_kwargs=encoder_kwargs,
                values_shape=values_shape,
                dt=dt,
                chunk_length_sec=chunk_length_sec,
            )

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
    logger.info("Built SignalSpecRegistry with %d signals", registry.num_signals)
    for spec in registry.specs:
        logger.info(
            "  • %-30s | role=%-8s | modality=%-10s | encoder=%s | dim=%s",
            spec.name,
            spec.role,
            spec.modality,
            spec.encoder_name,
            spec.embedding_dim,
        )

    return registry


# ---------------------------------------------------------------------------
# Helpers to derive signals_by_role from task config + metadata
# ---------------------------------------------------------------------------


def infer_modality(values_shape: Tuple[int, ...]) -> str:
    """
    Infer the signal modality from its spatial shape (excluding time).

    Conventions
    ----------
    - timeseries: no spatial dims or single channel
        ()      -> timeseries
        (1,)    -> timeseries
    - profile:  1D with C > 1 channels
        (C,)    -> profile
    - video / map: 2D spatial grid
        (H, W)  -> video
    """
    if len(values_shape) == 0:
        return "timeseries"
    if len(values_shape) == 1:
        return "timeseries" if values_shape[0] == 1 else "profile"
    if len(values_shape) == 2:
        return "video"
    raise ValueError(
        f"Unsupported values_shape={values_shape!r} for modality inference"
    )


def build_signal_role_modality_map(
    cfg_task: Mapping[str, Any],
    dict_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Dict[str, str]]:
    """
    Build the signals_by_role mapping from the baseline task config
    and the metadata returned by initialize_datasets_and_metadata_for_task.

    Parameters
    ----------
    cfg_task : Mapping[str, Any]
        Baseline task config (YAML) used to build the datasets. It must
        contain a "sources_and_signals" block with keys "input_name",
        "actuator_name", "output_name", each being a list of [source, signal].

    dict_metadata : Mapping[str, Mapping[str, Any]]
        Metadata dict as returned by initialize_datasets_and_metadata_for_task.

    Returns
    -------
    signals_by_role : dict
        Mapping:

            {
              "input": {
                "<signal_name>": "<modality>",
                ...
              },
              "actuator": {
                ...
              },
              "output": {
                ...
              },
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
        pairs = ss_cfg.get(key) or []  # e.g. [["pf_active", "solenoid_current"], ...]
        role_map: Dict[str, str] = {}

        for source, signal in pairs:
            name = f"{source}-{signal}"  # canonical name used everywhere
            meta = dict_metadata.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} not found in dict_metadata")

            val_shape = tuple(meta.get("values_shape", ()))
            modality = infer_modality(val_shape)
            role_map[name] = modality

        signals_by_role[role] = role_map

    return signals_by_role
