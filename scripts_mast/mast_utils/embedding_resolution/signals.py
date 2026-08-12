"""
Signal selection, filtering, and set helpers for embedding resolution.

Low-level, encoder-agnostic utilities that operate on ``role -> signal-name`` mappings and per-signal override
dicts: selecting which task signals use DCT3D, filtering override blocks down to a
chosen signal set, and adding/subtracting signal-name sets by role. These are the primitives the policy and resolve
layers build on; they hold no phase or role semantics of their own.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from mmt.data.signal_spec import SignalSpecRegistry


DCT3D_ROLES = ("input", "actuator", "output")


def _signal_names_for_role(role_signals: Any) -> list[str]:
    """
    Return signal names from a task role entry.

    Parameters
    ----------
    role_signals : Any
        Role entry from ``signals_by_role``. It may be a mapping of signal name to metadata/modality or an iterable of
        signal names.

    Returns
    -------
    list[str]
        Signal names for the role.

    """

    if isinstance(role_signals, Mapping):
        return [str(name) for name in role_signals.keys()]
    return [str(name) for name in role_signals]


def _dct3d_signal_names_by_role(
    signal_specs: SignalSpecRegistry,
    signals_by_role: Mapping[str, Any],
) -> dict[str, set[str]]:
    """
    Return task-used DCT3D signal names keyed by role.

    Parameters
    ----------
    signal_specs : SignalSpecRegistry
        Signal spec registry built from the current embedding profile defaults.
    signals_by_role : Mapping[str, Any]
        Dict mapping role -> list of signal names.

    Returns
    -------
    dict[str, set[str]]
        DCT3D signal names keyed by role.

    """

    names_by_role: dict[str, set[str]] = {role: set() for role in DCT3D_ROLES}
    for role in DCT3D_ROLES:
        for sig_name in _signal_names_for_role(signals_by_role.get(role, [])):
            spec = signal_specs.get(role, sig_name)
            if (spec is not None) and (spec.encoder_name == "dct3d"):
                names_by_role[role].add(sig_name)
    return {role: names for role, names in names_by_role.items() if names}


def _filter_embedding_overrides_by_signal_names(
    overrides: Mapping[str, Any],
    signal_names_by_role: Mapping[str, set[str]],
) -> dict[str, Any]:
    """
    Return only per-signal overrides for the requested role/signal pairs.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Per-signal overrides keyed by role.
    signal_names_by_role : Mapping[str, set[str]]
        Signal names to keep, keyed by role.

    Returns
    -------
    dict[str, Any]
        Filtered per-signal overrides.

    """

    filtered: dict[str, Any] = {}
    for role, signal_names in signal_names_by_role.items():
        role_overrides = overrides.get(role)
        if not isinstance(role_overrides, Mapping):
            continue

        kept = {name: deepcopy(role_overrides[name]) for name in signal_names if name in role_overrides}
        if kept:
            filtered[role] = kept
    return filtered


def _signal_names_from_overrides(
    overrides: Mapping[str, Any],
    signal_names_by_role: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """
    Return selected signal names that have explicit per-signal overrides.

    Parameters
    ----------
    overrides : Mapping[str, Any]
        Per-signal overrides keyed by role.
    signal_names_by_role : Mapping[str, set[str]]
        Candidate signal names keyed by role.

    Returns
    -------
    dict[str, set[str]]
        Signal names present in both inputs, keyed by role.

    """

    selected: dict[str, set[str]] = {}
    for role, signal_names in signal_names_by_role.items():
        role_overrides = overrides.get(role)
        if not isinstance(role_overrides, Mapping):
            continue

        names = {name for name in signal_names if name in role_overrides}
        if names:
            selected[role] = names
    return selected


def _subtract_signal_names_by_role(
    signal_names_by_role: Mapping[str, set[str]],
    names_to_remove: Mapping[str, set[str]],
) -> dict[str, set[str]]:
    """
    Remove role/signal names from a signal-name mapping.

    Parameters
    ----------
    signal_names_by_role : Mapping[str, set[str]]
        Source signal names keyed by role.
    names_to_remove : Mapping[str, set[str]]
        Signal names to remove, keyed by role.

    Returns
    -------
    dict[str, set[str]]
        Remaining signal names keyed by role.

    """

    remaining: dict[str, set[str]] = {}
    for role, signal_names in signal_names_by_role.items():
        names = set(signal_names) - set(names_to_remove.get(role, set()))
        if names:
            remaining[role] = names
    return remaining


def _signal_policy_roles(signal_policy: Mapping[str, set[str]] | None) -> list[str]:
    """
    Return roles that have at least one signal in a compacted action policy.

    Parameters
    ----------
    signal_policy : Mapping[str, set[str]] | None
        Role -> signal-name mapping for one policy action.

    Returns
    -------
    list[str]
        Roles with selected signals.

    """

    if not signal_policy:
        return []
    return [role for role in DCT3D_ROLES if signal_policy.get(role)]
