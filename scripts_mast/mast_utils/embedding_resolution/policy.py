"""
Embedding profile detection and encoder tune/source policy planning.

Determines whether a run uses rank-tuned DCT3D and classifies each task signal, per phase and finetune init mode, as
config-owned, source-inherited, or to-be-tuned. The encoder-specific role semantics live here rather than in the
shared resolve layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mmt.utils.config.schema import ExperimentConfig


def _uses_dct3d_tuning_policy(cfg_mmt: ExperimentConfig) -> bool:
    """
    Return whether the selected embedding profile participates in DCT3D tune/source policy.

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment configuration.

    Returns
    -------
    bool
        True if the selected embedding profile is DCT3D-derived and has a DCT3D config block.

    """

    profile = str(cfg_mmt.raw.get("embeddings_profile", "")).lower()
    return profile.startswith("dct3d") and isinstance(cfg_mmt.embeddings.get("dct3d"), Mapping)


def _resolve_dct3d_signal_policy(
    cfg_mmt: ExperimentConfig,
    dct3d_signals: Mapping[str, set[str]],
    source_overrides: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, set[str]]]:
    """
    Resolve the fixed DCT3D source/tune policy at signal granularity.

    Pretrain and scratch finetune tune every provided DCT3D signal. Warmstart finetune inherits existing input and
    actuator signals from the source run, tunes missing input and actuator signals, and always tunes output signals.
    Explicit profile ``per_signal_overrides`` are removed before this helper is called and remain config-owned.

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment configuration.
    dct3d_signals : Mapping[str, set[str]]
        Current task DCT3D signal names keyed by role.
    source_overrides : Mapping[str, Any] | None
        Source run ``per_signal_overrides`` used to classify warmstart signals as existing or missing.

    Returns
    -------
    dict[str, dict[str, set[str]]]
        Signal policy grouped as ``{action: {role: signal_names}}``.

    Raises
    ------
    ValueError
        If the phase/init combination is unsupported.

    """

    phase = str(cfg_mmt.raw.get("phase", ""))
    all_tune_by_role = {role: set(names) for role, names in dct3d_signals.items() if names}
    all_tune_policy = {"tune": all_tune_by_role} if all_tune_by_role else {}

    if phase == "pretrain":
        return all_tune_policy

    if phase != "finetune":
        raise ValueError(f"Cannot derive DCT3D signal policy for phase={phase!r}.")

    init_mode = str((cfg_mmt.raw.get("cli") or {}).get("init", "")).lower()
    if init_mode == "scratch":
        return all_tune_policy

    if init_mode != "warmstart":
        raise ValueError(f"Cannot derive DCT3D signal policy for finetune init={init_mode!r}.")

    source_overrides = source_overrides or {}
    signal_policy: dict[str, dict[str, set[str]]] = {}

    for role, signal_names in dct3d_signals.items():
        if role == "output":
            if signal_names:
                signal_policy.setdefault("tune", {})[role] = set(signal_names)
            continue

        role_source_overrides = source_overrides.get(role, {})
        if not isinstance(role_source_overrides, Mapping):
            role_source_overrides = {}

        existing = {name for name in signal_names if name in role_source_overrides}
        missing = signal_names - existing

        if existing:
            signal_policy.setdefault("source", {})[role] = existing
        if missing:
            signal_policy.setdefault("tune", {})[role] = missing

    return signal_policy
