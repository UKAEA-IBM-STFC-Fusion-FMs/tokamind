"""
Shared helpers for the `scripts_mast/run_*.py` entrypoints.

Purpose
-------
The open-source `mmt/` package is intentionally dataset-agnostic.
All FAIR/MAST integration (benchmark task_config loading, dataset creation,
metadata handling) lives under `scripts_mast/`.

As a consequence, the phase entrypoints (`run_pretrain.py`, `run_finetune.py`,
`run_eval.py`) would otherwise repeat the same boilerplate:
  - device selection + multiprocessing setup
  - standard shot -> windows transform chain
  - collate construction

This module centralises those shared blocks while keeping the run scripts thin
and easy to read.

Config conventions
------------------
Helpers assume the config layout:

    scripts_mast/configs/
    common/
        embeddings.yaml
        eval.yaml
        finetune.yaml
        pretrain.yaml

    tasks_overrides/
        <task>/
        finetune_overrides.yaml
        eval_overrides.yaml
        embeddings_overrides/
                <embedding_profile>.yaml
                vae.yaml

Notes
-----
- This module may import benchmark FAIRMAST utilities from `scripts/...`.
  Do NOT import it from the core `mmt/` package.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.multiprocessing as mp


from mmt.data import (
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    EmbedChunksTransform,
    BuildTokensTransform,
    FinalizeWindowTransform,
    ComposeTransforms,
    MMTCollate,
)


# ------------------------------------------------------------------
# Runtime / config bootstrap
# ------------------------------------------------------------------


def setup_device_and_mp() -> torch.device:
    """Pick the best available torch device and configure multiprocessing.

    Returns
    -------
    torch.device
        CUDA if available, else MPS, else CPU.

    Notes
    -----
    - We force the multiprocessing start method to ``spawn`` because some
      datasets/transforms are not fork-safe and PyTorch DataLoader workers
      often behave better with spawn.
    """

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Needed for DataLoader(num_workers>0) on some platforms.
    # force=True avoids "context has already been set" errors in notebooks/tests.
    mp.set_start_method("spawn", force=True)

    return device


# ------------------------------------------------------------------
# Extract signal stats
# ------------------------------------------------------------------


def extract_signal_stats(dict_metadata: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract simple (mean/std) statistics per signal from FAIRMAST metadata.

    This is primarily used by evaluation code (metrics/traces) to
    de-normalize outputs.

    Returns
    -------
    dict
        Mapping: ``signal_name -> {"mean": ..., "std": ...}``.
    """

    stats: Dict[str, Dict[str, Any]] = {}
    for role in ("input", "actuator", "output"):
        role_meta = dict_metadata.get(role, {})
        if not isinstance(role_meta, dict):
            continue
        for name, meta in role_meta.items():
            if isinstance(meta, dict) and ("mean" in meta) and ("std" in meta):
                stats[name] = meta
    return stats


# ------------------------------------------------------------------
# Transforms (shot → windows)
# ------------------------------------------------------------------


def build_default_transform(
    cfg_mmt: Any,
    *,
    dict_metadata: Mapping[str, Any],
    signal_specs: Any,
    codecs: Mapping[int, Any],
    keep_output_native: bool,
) -> Any:
    """Build the standard MMT transform chain used by pretrain/finetune/eval.

    The chain matches what is currently duplicated in run_pretrain/run_finetune/run_eval:

      ChunkWindowsTransform
      → SelectValidWindowsTransform
      → TrimChunksTransform
      → EmbedChunksTransform
      → BuildTokensTransform

    Parameters
    ----------
    cfg_mmt:
        Merged ExperimentConfig.
    dict_metadata:
        FAIRMAST metadata dict.
    signal_specs:
        SignalSpec registry.
    codecs:
        Codec mapping (signal_id -> codec).
    keep_output_native:
        If True, the final window dict will keep native output payloads
        (needed for eval metrics/traces). If False, native outputs are dropped
        by FinalizeWindowTransform to reduce RAM usage (especially when caching).

    Returns
    -------
    ComposeTransforms
        A callable transform mapping a shot object to an iterable of window dicts.
    """

    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    return ComposeTransforms(
        [
            ChunkWindowsTransform(
                dict_metadata=dict_metadata,
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks["stride"],
            ),
            SelectValidWindowsTransform(
                min_valid_inputs_actuators=cfg_valid_win["min_valid_inputs_actuators"],
                min_valid_chunks=cfg_valid_win["min_valid_chunks"],
                min_valid_outputs=cfg_valid_win["min_valid_outputs"],
                window_stride_sec=cfg_valid_win["window_stride_sec"],
            ),
            TrimChunksTransform(max_chunks=cfg_trim["max_chunks"]),
            EmbedChunksTransform(
                signal_specs=signal_specs,
                codecs=codecs,
            ),
            BuildTokensTransform(signal_specs=signal_specs),
            FinalizeWindowTransform(keep_output_native=keep_output_native),
        ]
    )


# ------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------


def make_collate_fn(
    *,
    signal_specs: Any,
    base_cfg: Optional[Mapping[str, Any]] = None,
    keep_output_native: bool,
    # force-drop lists (mostly used for eval ablations)
    drop_inputs: Optional[Sequence[str]] = None,
    drop_actuators: Optional[Sequence[str]] = None,
    drop_outputs: Optional[Sequence[str]] = None,
) -> MMTCollate:
    """Create an MMTCollate configured for train/eval.

    Important:
    ------------------------

    To keep YAML configs user-friendly, dropout overrides are still specified
    by *signal name* in config, but we convert them **once at startup** to
    id-based dicts (keyed by ``SignalSpec.signal_id``) before constructing the
    collate.

    Parameters
    ----------
    signal_specs:
        SignalSpec registry for the *current task*.
    base_cfg:
        Base collate configuration (usually cfg_mmt.collate for train).
        For eval you can pass None.
    keep_output_native:
        Whether to include native output payloads in batches.
    drop_inputs / drop_actuators / drop_outputs:
        If provided, these signals are *forced dropped* by setting per-signal
        dropout overrides to 1.0.
    drop_inputs:
        Optional list of input signal names to force-drop (p=1.0).
    drop_actuators:
        Optional list of actuator signal names to force-drop (p=1.0).
    drop_outputs:
        Optional list of output signal names to force-drop (p=1.0).

    Returns
    -------
    MMTCollate
        Ready to be used as DataLoader.collate_fn.
    """

    cfg: Dict[str, Any] = dict(base_cfg or {})
    cfg["keep_output_native"] = bool(keep_output_native)

    def _name_to_id(role: str, name: str) -> int:
        spec = signal_specs.get(role, name)
        if spec is None:
            raise KeyError(f"Unknown {role} signal name {name!r}")
        return int(spec.signal_id)

    def _convert_overrides(
        role: str, overrides: Any, *, label: str
    ) -> Dict[int, float]:
        if overrides is None:
            return {}
        if not isinstance(overrides, Mapping):
            raise TypeError(
                f"{label} must be a dict keyed by signal name (str) or signal_id (int); "
                f"got {type(overrides).__name__}."
            )

        out: Dict[int, float] = {}
        for k, v in overrides.items():
            if isinstance(k, int):
                sid = int(k)
            else:
                sid = _name_to_id(role, str(k))
            out[sid] = float(v)
        return out

    # Convert any configured overrides (name -> id). If user passes ids directly,
    # we keep them as-is (useful for programmatic callers).
    in_over = _convert_overrides(
        "input",
        cfg.get("p_drop_inputs_overrides", {}),
        label="p_drop_inputs_overrides",
    )
    act_over = _convert_overrides(
        "actuator",
        cfg.get("p_drop_actuators_overrides", {}),
        label="p_drop_actuators_overrides",
    )
    out_over = _convert_overrides(
        "output",
        cfg.get("p_drop_outputs_overrides", {}),
        label="p_drop_outputs_overrides",
    )

    # Force-drop lists (names) -> id overrides.
    if drop_inputs:
        for name in drop_inputs:
            in_over[_name_to_id("input", str(name))] = 1.0
    if drop_actuators:
        for name in drop_actuators:
            act_over[_name_to_id("actuator", str(name))] = 1.0
    if drop_outputs:
        for name in drop_outputs:
            out_over[_name_to_id("output", str(name))] = 1.0

    cfg["p_drop_inputs_overrides"] = in_over
    cfg["p_drop_actuators_overrides"] = act_over
    cfg["p_drop_outputs_overrides"] = out_over

    # For eval-only native output collation, collate needs id->name mapping once.
    if keep_output_native:
        cfg["output_id_to_name"] = {
            int(spec.signal_id): spec.name
            for spec in signal_specs.specs_for_role("output")
        }

    return MMTCollate(cfg_collate=cfg)
