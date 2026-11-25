"""
Load, stage, and merge embedding artifacts, plus strict inherited-artifact validation.

Handles everything touching on-disk rank artifacts: merging per-signal overrides into the live config, capturing the
explicit profile overrides that must win over computed ones, staging task-used rank artifacts (coefficient indices +
YAML) from a warm-start source run and strictly validating inherited overrides so malformed artifacts fail loudly
instead of silently loading.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mmt.data.signal_spec import SignalSpecRegistry

from ..tune_dct3d import load_embeddings_overrides

logger = logging.getLogger("mmt.EmbeddingResolution")


def _merge_embedding_overrides(embeddings_cfg: dict[str, Any], overrides: Mapping[str, Any]) -> None:
    """
    Merge ``per_signal_overrides`` from ``overrides`` into ``embeddings_cfg`` in place.

    For each role present in ``overrides``, the corresponding role dict inside
    ``embeddings_cfg["per_signal_overrides"]`` is updated with the incoming signal entries. Existing entries for
    signals not present in ``overrides`` are left unchanged.

    Parameters
    ----------
    embeddings_cfg : dict[str, Any]
        The ``embeddings`` sub-dict that is mutated in place (i.e. ``cfg_mmt.raw["embeddings"]``).
    overrides : Mapping[str, Any]
        ``per_signal_overrides`` to merge, keyed by role then signal name.

    """

    embeddings_cfg.setdefault("per_signal_overrides", {})
    per_signal = embeddings_cfg["per_signal_overrides"]

    for role, sigs in overrides.items():
        if not isinstance(sigs, Mapping):
            continue
        per_signal.setdefault(role, {})
        per_signal[role].update(sigs)


def _profile_embedding_overrides(embeddings_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """
    Return a deep copy of the explicit ``per_signal_overrides`` present in the profile YAML.

    Must be called before any DCT3D artifacts are merged into ``embeddings_cfg``, so that the snapshot reflects only
    the user-authored config choices. The copy is re-applied after all artifact merges via
    :func:`_merge_embedding_overrides`, ensuring that explicit profile overrides always win over computed defaults.

    Parameters
    ----------
    embeddings_cfg : Mapping[str, Any]
        The ``embeddings`` sub-dict from the resolved experiment config (i.e. ``cfg_mmt.raw["embeddings"]``).

    Returns
    -------
    dict[str, Any]
        Deep copy of ``embeddings_cfg["per_signal_overrides"]``, or an empty dict if the key is absent or not a
        mapping.

    """

    per_signal = embeddings_cfg.get("per_signal_overrides", {})
    if not isinstance(per_signal, dict):
        return {}
    return deepcopy(per_signal)


def stage_task_used_dct3d_artifacts_from_source(  # NOSONAR - Ignore cognitive complexity
    source_run_dir: Path, run_dir: Path, signals_by_role: Mapping[str, Any]
) -> bool:
    """
    Stage only task-used DCT3D artifacts from a source run.

    The destination finetune run receives only the source artifacts required for the current task. This includes
    task-used signals that may later be re-tuned, since their files and YAML entries are overwritten in-place by
    the tuning step.

    Parameters
    ----------
    source_run_dir : Path
        Source training run directory.
    run_dir : Path
        Destination finetune run directory.
    signals_by_role : Mapping[str, Any]
        Task-used signals keyed by role. Each role value may be either a mapping of signal name -> modality or an
        iterable of signal names.

    Returns
    -------
    bool
        True if the source `embeddings/` folder exists, else False.

    """

    src_emb = source_run_dir / "embeddings"
    if not src_emb.exists():
        return False

    src_overrides = load_embeddings_overrides(run_dir=source_run_dir)
    filtered_overrides: dict = {}
    dst_emb = run_dir / "embeddings"

    for role, role_signals in signals_by_role.items():
        role_overrides = src_overrides.get(role)
        if not isinstance(role_overrides, dict):
            continue

        if isinstance(role_signals, dict):
            signal_names = role_signals.keys()
        else:
            signal_names = role_signals

        for sig_name in signal_names:
            if sig_name not in role_overrides:
                continue

            sig_override = role_overrides[sig_name]
            filtered_overrides.setdefault(role, {})[sig_name] = sig_override

            if not isinstance(sig_override, dict):
                continue

            encoder_kwargs = sig_override.get("encoder_kwargs")
            if not isinstance(encoder_kwargs, dict):
                continue

            coeff_indices_path = encoder_kwargs.get("coeff_indices_path")
            if not isinstance(coeff_indices_path, str):
                continue

            src_indices = src_emb / coeff_indices_path
            if not src_indices.exists():
                continue

            dst_indices = dst_emb / coeff_indices_path
            dst_indices.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=src_indices, dst=dst_indices)

    dst_emb.mkdir(parents=True, exist_ok=True)
    dct3d_yaml_path = dst_emb / "dct3d.yaml"
    with dct3d_yaml_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"embeddings": {"per_signal_overrides": filtered_overrides}},
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info(
        "Staged task-used DCT3D artifacts from %s -> %s",
        src_emb,
        dst_emb,
    )

    return True


def _validate_inherited_embeddings_strict(  # NOSONAR - Ignore cognitive complexity
    per_signal_overrides: Mapping[str, Any],
    signal_names_by_role: Mapping[str, set[str]],
    signal_specs: SignalSpecRegistry,
    encoder_name: str = "dct3d",
    required_encoder_kwargs: tuple[str, ...] = (),
) -> None:
    """
    Strict validation: check signal-level rank-mode parameters for inherited signals.

    Validates the inherited DCT3D rank-mode artifact schema.

    For each inherited role/signal pair:
    1. Role must exist in overrides
    2. Signal must exist in overrides[role]
    3. If the current signal spec's ``encoder_name`` matches ``encoder_name``:
       - Override must have matching ``encoder_name``
       - Override must have ``encoder_kwargs`` with required keys:
         * selection_mode='rank'
         * coeff_indices_path
         * coeff_shape
         * num_coeffs
         * every entry in ``required_encoder_kwargs``

    Parameters
    ----------
    per_signal_overrides : Mapping[str, Any]
        Loaded overrides from source run's ``<encoder_name>.yaml``.
    signal_names_by_role : Mapping[str, set[str]]
        Signal names that should be inherited, keyed by role.
    signal_specs : SignalSpecRegistry
        Signal spec registry (built with default config before inheritance).
    encoder_name : str
        Rank-tuned encoder whose inherited artifacts are being validated.
        Optional. Default: ``"dct3d"``.
    required_encoder_kwargs : tuple[str, ...]
        Encoder-specific ``encoder_kwargs`` keys that must also be present. Optional. Default: ``()``.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If validation fails with detailed message about missing/invalid entries.
    TypeError
        If validation fails with detailed message about invalid types.

    """

    label = encoder_name.upper()
    artifact = f"{encoder_name}.yaml"

    for role, signal_names in signal_names_by_role.items():
        if not signal_names:
            # Nothing to inherit/validate for this role in the current task.
            continue

        # Check role exists
        if role not in per_signal_overrides:
            raise ValueError(
                f"{label} warmstart source policy requires source role {role!r}, "
                f"but source embeddings/{artifact} has no entries for this role. "
                f"Source model may not have used {label} rank-mode tuning for this role. "
                "Use finetune init=scratch, add manual per-signal overrides, or warmstart from a matching source run."
            )

        role_overrides = per_signal_overrides[role]

        # Check each rank-tuned signal in this role
        for sig_name in sorted(signal_names):
            spec = signal_specs.get(role, sig_name)
            if (spec is None) or (spec.encoder_name != encoder_name):
                continue  # Skip signals not using this encoder

            # Signal must exist in overrides
            if sig_name not in role_overrides:
                raise ValueError(
                    f"{label} warmstart source policy for role {role!r} is missing signal "
                    f"'{sig_name}' in source embeddings/{artifact}. Expected inherited {label} signals "
                    f"to have rank-mode overrides. Available signals in "
                    f"source: {list(role_overrides.keys())}."
                )

            sig_override = role_overrides[sig_name]

            # Validate structure
            if not isinstance(sig_override, dict):
                raise TypeError(
                    f"{label} warmstart source policy has invalid override for {role}:{sig_name}. "
                    f"Expected dict, got {type(sig_override).__name__}"
                )

            # Check encoder_name
            if sig_override.get("encoder_name") != encoder_name:
                raise TypeError(
                    f"{label} warmstart source policy: {role}:{sig_name} has "
                    f"encoder_name='{sig_override.get('encoder_name')}', expected '{encoder_name}'."
                )

            # Check encoder_kwargs
            kwargs = sig_override.get("encoder_kwargs")
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"{label} warmstart source policy: {role}:{sig_name} missing or invalid 'encoder_kwargs' "
                    "(expected dict)."
                )

            # Check required kwargs fields (shared rank fields + any encoder-specific ones)
            required_fields = [
                "selection_mode",
                "coeff_indices_path",
                "coeff_shape",
                "num_coeffs",
                *required_encoder_kwargs,
            ]
            missing = [f for f in required_fields if f not in kwargs]
            if missing:
                raise ValueError(
                    f"{label} warmstart source policy: {role}:{sig_name} encoder_kwargs missing required fields: "
                    f"{missing}."
                )

            # Check selection_mode is 'rank'
            if kwargs["selection_mode"] != "rank":
                raise ValueError(
                    f"{label} warmstart source policy: {role}:{sig_name} has selection_mode='{kwargs['selection_mode']}', "
                    f"expected 'rank' for inherited embeddings."
                )

    logger.info(
        "Strict validation passed: selected inherited %s signals have valid rank-mode overrides | signals=%s",
        label,
        {role: sorted(names) for role, names in signal_names_by_role.items()},
    )
