"""
MAST task → MMT signal mapping helpers.

This module is part of the MAST integration layer (scripts_mast/).

It converts a FAIRMAST-style task config (config_task_*.yaml / pretrain_task_*.yaml)
into the minimal, dataset-agnostic structure that the core MMT package expects:

    signals_by_role: Dict[str, Dict[str, str]]
        role -> { canonical_signal_name -> modality }

Where:
  - role is one of {"input", "actuator", "output"}
  - canonical_signal_name is "source-signal" (e.g. "pf_active-coil_current")
  - modality is inferred from metadata["values_shape"] via mmt.signal_spec.infer_modality

MMT core does NOT depend on FAIRMAST task YAML schema; only this adapter does.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

from mmt.data import infer_modality


Role = str
CanonicalName = str
Modality = str


def _pairs_to_names(pairs: Iterable[Tuple[str, str]]) -> list[str]:
    """Convert [[source, signal], ...] pairs to canonical names 'source-signal'."""
    names: list[str] = []
    for src, sig in pairs:
        names.append(f"{src}-{sig}")
    return names


def build_signals_by_role_from_task_config(
    cfg_task: Mapping[str, Any],
    dict_metadata: Mapping[str, Any],
) -> Dict[Role, Dict[CanonicalName, Modality]]:
    """
    Build signals_by_role from a FAIRMAST-style task config.

    Parameters
    ----------
    cfg_task:
        Task configuration dict containing:
            cfg_task["sources_and_signals"] with keys:
              - input_name:    [[source, signal], ...]
              - actuator_name: [[source, signal], ...]
              - output_name:   [[source, signal], ...]

    dict_metadata:
        benchmark metadata dict with role-scoped entries:
            dict_metadata["input"][canonical_name]["values_shape"]
            dict_metadata["actuator"][canonical_name]["values_shape"]
            dict_metadata["output"][canonical_name]["values_shape"]

    Returns
    -------
    signals_by_role:
        Dict mapping each role to a dict {canonical_name -> modality}.
    """
    ss_cfg = cfg_task.get("sources_and_signals") or {}
    if not isinstance(ss_cfg, dict):
        raise TypeError("cfg_task['sources_and_signals'] must be a mapping.")

    role_to_key = {
        "input": "input_name",
        "actuator": "actuator_name",
        "output": "output_name",
    }

    signals_by_role: Dict[Role, Dict[CanonicalName, Modality]] = {}

    for role, task_key in role_to_key.items():
        pairs = ss_cfg.get(task_key) or []
        if not isinstance(pairs, list):
            raise TypeError(
                f"cfg_task['sources_and_signals']['{task_key}'] must be a list."
            )

        role_meta = dict_metadata.get(role) or {}
        if not isinstance(role_meta, dict):
            raise TypeError(f"dict_metadata['{role}'] must be a mapping.")

        role_map: Dict[CanonicalName, Modality] = {}
        for name in _pairs_to_names(pairs):
            meta = role_meta.get(name)
            if meta is None:
                raise KeyError(f"Signal {name!r} missing from dict_metadata['{role}'].")

            values_shape = meta.get("values_shape", ())
            modality = infer_modality(tuple(values_shape))
            role_map[name] = modality

        signals_by_role[role] = role_map

    return signals_by_role
