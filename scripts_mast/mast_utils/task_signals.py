"""
MAST task → MMT signal mapping helpers.

This module is part of the MAST integration layer (scripts_mast/).

It converts a FAIR MAST-style task config (config_task_*.yaml / pretrain_task_*.yaml) into the minimal, dataset-agnostic
structure that the core MMT package expects:

    signals_by_role: dict[str, dict[str, str]]
        role -> { canonical_signal_name -> modality }

Where:
  - role is one of {"input", "actuator", "output"}
  - canonical_signal_name is "source-signal" (e.g., "pf_active-coil_current")
  - modality is inferred from metadata["values_shape"] via mmt.signal_spec.infer_modality

MMT core does NOT depend on FAIR MAST task YAML schema; only this adapter does.
"""

from __future__ import annotations

from typing import Any, Iterable
from collections.abc import Mapping

from mmt.data import infer_modality


# ----------------------------------------------------------------------------------------------------------------------

Role = str
CanonicalName = str
Modality = str

ROLE_TO_KEY = {
    "input": "input_name",
    "actuator": "actuator_name",
    "output": "output_name",
}


# ----------------------------------------------------------------------------------------------------------------------
def _pairs_to_names(pairs: Iterable[tuple[str, str]]) -> list[str]:
    """
    Convert [[source, signal], ...] pairs to canonical names 'source-signal'.

    Parameters
    ----------
    pairs : Iterable[tuple[str, str]]
        Input pairs to be converted to names.

    Returns
    -------
    list[str]
        Converted input `pairs` into names in list["<source>-<signal>"] format.

    """

    names: list[str] = []
    for src, sig in pairs:
        names.append(f"{src}-{sig}")
    return names


# ----------------------------------------------------------------------------------------------------------------------
def build_signals_by_role_from_task_definition(
    cfg_task: Mapping[str, Any],
    dict_metadata: Mapping[str, Any],
) -> dict[Role, dict[CanonicalName, Modality]]:
    """
    Build signals_by_role from a FAIR MAST-style task config.

    Parameters
    ----------
    cfg_task:
        Task configuration mapping (dict) containing:
            cfg_task["sources_and_signals"] with keys:
              - input_name:    [[source, signal], ...]
              - actuator_name: [[source, signal], ...]
              - output_name:   [[source, signal], ...]

    dict_metadata:
        benchmark metadata mapping (dict) with role-scoped entries:
            dict_metadata["input"][canonical_name]["values_shape"]
            dict_metadata["actuator"][canonical_name]["values_shape"]
            dict_metadata["output"][canonical_name]["values_shape"]

    Returns
    -------
    dict[Role, dict[CanonicalName, Modality]]
        Dictionary mapping each role to a nested dictionary {canonical_name -> modality}.

    Raises
    ------
    TypeError
        If `cfg_task['sources_and_signals']` is not a mapping (dict).
        If `cfg_task['sources_and_signals']['{task_key}']` is not a list.
        If `dict_metadata['{role}']` is not mapping (dict).
    KeyError
        If a given signal name is missing from `dict_metadata['{role}']`.

    """

    ss_cfg = cfg_task.get("sources_and_signals") or {}
    if not isinstance(ss_cfg, dict):
        raise TypeError("`cfg_task['sources_and_signals']` must be a mapping (dict).")

    signals_by_role: dict[Role, dict[CanonicalName, Modality]] = {}

    for role, task_key in ROLE_TO_KEY.items():
        pairs = ss_cfg.get(task_key) or []
        if not isinstance(pairs, list):
            raise TypeError(f"`cfg_task['sources_and_signals']['{task_key}']` must be a list.")

        role_meta = dict_metadata.get(role) or {}
        if not isinstance(role_meta, dict):
            raise TypeError(f"`dict_metadata['{role}']` must be a mapping (dict).")

        role_map: dict[CanonicalName, Modality] = {}
        for name in _pairs_to_names(pairs=pairs):
            meta = role_meta.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} missing from `dict_metadata['{role}']`.")

            values_shape = meta.get("values_shape", ())
            modality = infer_modality(values_shape=tuple(values_shape))
            role_map[name] = modality

        signals_by_role[role] = role_map

    return signals_by_role
