"""
DCT3D rank-mode embedding tuning.

This module contains the dataset-agnostic orchestration for selecting compact rank-mode DCT3D embeddings. Dataset
integration layers such as ``scripts_mast`` are responsible for building an iterable of raw tokamind
windows; this module owns the shared part:

    raw windows
      -> ChunkWindowsTransform
      -> SelectValidWindowsTransform
      -> TrimChunksTransform
      -> TuneRankedDCT3DTransform
      -> dct3d_indices/*.npy + dct3d.yaml

The saved artifact format is the stable contract consumed by ``build_codecs``. Each tuned signal is written as a
per-signal override with ``encoder_name: dct3d`` and rank-mode ``encoder_kwargs`` pointing to the selected coefficient
indices.

This module intentionally does not know about benchmark-specific dataset APIs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

import numpy as np
import yaml

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.data.transforms.chunk_windows import ChunkWindowsTransform
from mmt.data.transforms.compose import ComposeTransforms
from mmt.data.transforms.select_valid_windows import SelectValidWindowsTransform
from mmt.data.transforms.trim_chunks import TrimChunksTransform
from mmt.data.transforms.tune_ranked_dct3d import TuneRankedDCT3DTransform


logger = logging.getLogger("mmt.TuneRankedDCT3D")


def build_dct3d_tuning_transform(
    *,
    signal_specs: SignalSpecRegistry,
    dict_metadata: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
    tuning_cfg: Mapping[str, Any],
    roles: Sequence[str] = ("input", "actuator", "output"),
) -> tuple[ComposeTransforms, TuneRankedDCT3DTransform]:
    """
    Build the standard DCT3D rank-tuning transform chain.

    Parameters
    ----------
    signal_specs : SignalSpecRegistry
        Signal spec registry.
    dict_metadata : Mapping[str, Any]
        Tokamind task metadata dictionary.
    preprocess_cfg : Mapping[str, Any]
        Preprocessing configuration containing canonical role-specific ``chunks`` and ``valid_windows`` sections,
        plus optional ``embed_chunks``. The legacy ``chunk`` and ``trim_chunks`` sections remain supported for saved
        configurations.
    tuning_cfg : Mapping[str, Any]
        DCT3D tuning configuration containing ``objective`` and optional ``guardrails`` sections.
    roles : Sequence[str]
        Roles to tune.
        Optional. Default: ("input", "actuator", "output").

    Returns
    -------
    tuple[ComposeTransforms, TuneRankedDCT3DTransform]
        Transform pipeline and the underlying tuner object.
    """

    objective_cfg = tuning_cfg.get("objective", {})
    valid_cfg = preprocess_cfg["valid_windows"]
    embed_cfg = preprocess_cfg.get("embed_chunks") or {}

    tuner = TuneRankedDCT3DTransform(
        signal_specs=signal_specs,
        thresholds=objective_cfg.get("thresholds", {}),
        max_budget=objective_cfg.get("max_budget", {}),
        roles=list(roles),
        guardrails=tuning_cfg.get("guardrails") or {},
        nan_imputation=embed_cfg.get("nan_imputation", "zero"),
    )

    chunks_cfg = preprocess_cfg.get("chunks")
    if chunks_cfg is not None:
        chunk_transform = ChunkWindowsTransform(
            dict_metadata=dict_metadata,
            chunks_cfg=chunks_cfg,
        )
        trim_transform = None
    else:
        # Support legacy saved configs. Current configs use ``preprocess.chunks``;
        # their role-aware chunk transform already applies the max-chunk limit.
        chunk_cfg = preprocess_cfg["chunk"]
        trim_cfg = preprocess_cfg["trim_chunks"]
        chunk_transform = ChunkWindowsTransform(
            dict_metadata=dict_metadata,
            chunk_length_sec=float(chunk_cfg["chunk_length"]),
            stride_sec=chunk_cfg.get("stride"),
        )
        trim_transform = TrimChunksTransform(max_chunks=int(trim_cfg["max_chunks"]))

    transforms: list[Any] = [
        chunk_transform,
        SelectValidWindowsTransform(
            min_valid_inputs_actuators=int(valid_cfg["min_valid_inputs_actuators"]),
            min_valid_chunks=int(valid_cfg["min_valid_chunks"]),
            min_valid_outputs=int(valid_cfg["min_valid_outputs"]),
            accept_nan_inputs_actuators=bool(valid_cfg.get("accept_nan_inputs_actuators", True)),
            accept_nan_outputs=bool(valid_cfg.get("accept_nan_outputs", True)),
            window_stride_sec=(
                float(valid_cfg["window_stride_sec"]) if valid_cfg.get("window_stride_sec") is not None else None
            ),
        ),
    ]
    if trim_transform is not None:
        transforms.append(trim_transform)
    transforms.append(tuner)
    transform = ComposeTransforms(transforms)

    return transform, tuner


def save_dct3d_rank_overrides(
    *,
    best: Mapping[str, Mapping[str, Mapping[str, Any]]],
    signal_specs: SignalSpecRegistry,
    run_dir: Path,
    allowed_signal_names: Mapping[str, set[str]] | None = None,
    merge_existing: bool = True,
) -> dict[str, Any]:
    """
    Save DCT3D rank-mode coefficient indices and per-signal overrides.

    Parameters
    ----------
    best : Mapping[str, Mapping[str, Mapping[str, Any]]]
        Result of ``TuneRankedDCT3DTransform.select_best()``.
    signal_specs : SignalSpecRegistry
        Signal spec registry used during tuning.
    run_dir : Path
        Run directory. Artifacts are written under ``run_dir/embeddings``.
    allowed_signal_names : Mapping[str, set[str]] | None
        Optional signal-name filter keyed by role.
    merge_existing : bool
        Whether to merge new overrides into an existing ``dct3d.yaml``.
        Optional. Default: True.

    Returns
    -------
    dict[str, Any]
        Newly written per-signal overrides. If ``merge_existing`` is true, the file may contain additional existing
        overrides.
    """

    run_dir = Path(run_dir)
    emb_dir = run_dir / "embeddings"
    indices_dir = emb_dir / "dct3d_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    per_signal_overrides: dict[str, Any] = {}
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            if (allowed_signal_names is not None) and (name not in allowed_signal_names.get(role, set())):
                continue

            spec = signal_specs.get(role, name)
            if (spec is None) or (spec.encoder_name != "dct3d"):
                continue

            filename = f"{role}_{name}.npy"
            np.save(indices_dir / filename, info["coeff_indices"])
            per_signal_overrides.setdefault(role, {})[name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": {
                    "selection_mode": "rank",
                    "coeff_indices_path": f"dct3d_indices/{filename}",
                    "coeff_shape": list(info["coeff_shape"]),
                    "num_coeffs": int(info["num_coeffs"]),
                    "explained_energy": float(info["explained_energy"]),
                    "dim_distribution": dict(info.get("dim_distribution", {})),
                    "tuning_info": {
                        "target_energy": float(info["target_energy"]),
                        "k_target": int(info.get("k_target") or info["num_coeffs"]),
                        "guardrail_min_k": int(info.get("guardrail_min_k", 0)),
                        "k_after_guardrails": int(info.get("k_after_guardrails") or info["num_coeffs"]),
                        "k_final": int(info["num_coeffs"]),
                        "n_windows": int(info["n_windows"]),
                        "max_budget": None if info.get("max_budget") is None else int(info["max_budget"]),
                        "flags": list(info.get("flags", [])),
                        "tuned_in_run_id": str(run_dir.name),
                    },
                },
            }

    dct3d_yaml_path = emb_dir / "dct3d.yaml"
    merged_overrides = per_signal_overrides
    if merge_existing and dct3d_yaml_path.exists():
        with dct3d_yaml_path.open(encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing_overrides = existing.get("embeddings", {}).get("per_signal_overrides", {}) or {}
        for role, sigs in per_signal_overrides.items():
            existing_overrides.setdefault(role, {}).update(sigs)
        merged_overrides = existing_overrides

    with dct3d_yaml_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(
            data={"embeddings": {"per_signal_overrides": merged_overrides}},
            stream=f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info("Saved tuned overrides -> %s", dct3d_yaml_path)
    logger.info("Saved coefficient indices -> %s", indices_dir)

    return per_signal_overrides


def load_dct3d_rank_overrides(run_dir: Path) -> dict[str, Any]:
    """
    Load DCT3D rank-mode overrides from a run directory.

    Parameters
    ----------
    run_dir : Path
        Run directory containing ``embeddings/dct3d.yaml``.

    Returns
    -------
    dict[str, Any]
        Per-signal overrides, or ``{}`` if no artifact is present.
    """

    dct3d_yaml = Path(run_dir) / "embeddings" / "dct3d.yaml"
    if not dct3d_yaml.exists():
        return {}

    with dct3d_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("embeddings", {}).get("per_signal_overrides", {}) or {}


def run_dct3d_tuning_from_windows(
    *,
    windows: Iterable[Any],
    signal_specs: SignalSpecRegistry,
    dict_metadata: Mapping[str, Any],
    preprocess_cfg: Mapping[str, Any],
    tuning_cfg: Mapping[str, Any],
    run_dir: Path,
    roles: Sequence[str] = ("input", "actuator", "output"),
    signal_names_by_role: Mapping[str, set[str]] | None = None,
    max_windows: int | None = None,
    merge_existing: bool = True,
) -> dict[str, Any]:
    """
    Tune DCT3D rank-mode embeddings from an iterable of raw tokamind windows.

    Parameters
    ----------
    windows : Iterable[Any]
        Iterable yielding raw tokamind window dictionaries.
    signal_specs : SignalSpecRegistry
        Signal spec registry.
    dict_metadata : Mapping[str, Any]
        Tokamind task metadata dictionary.
    preprocess_cfg : Mapping[str, Any]
        Preprocessing configuration.
    tuning_cfg : Mapping[str, Any]
        DCT3D tuning configuration.
    run_dir : Path
        Run directory where artifacts are written.
    roles : Sequence[str]
        Roles to tune.
        Optional. Default: ("input", "actuator", "output").
    signal_names_by_role : Mapping[str, set[str]] | None
        Optional signal-name filter keyed by role.
    max_windows : int | None
        Optional cap on processed windows.
    merge_existing : bool
        Whether to merge into existing ``dct3d.yaml``.
        Optional. Default: True.

    Returns
    -------
    dict[str, Any]
        Newly written per-signal DCT3D rank-mode overrides.
    """

    transform, tuner = build_dct3d_tuning_transform(
        signal_specs=signal_specs,
        dict_metadata=dict_metadata,
        preprocess_cfg=preprocess_cfg,
        tuning_cfg=tuning_cfg,
        roles=roles,
    )

    n_seen = 0
    for window in windows:
        out = transform(window)
        if out is not None:
            n_seen += 1
        if (max_windows is not None) and (n_seen >= int(max_windows)):
            break

    logger.info("Tuning: processed %d windows", n_seen)

    best = tuner.select_best()
    summary = tuner.summarize_selection(best=best)
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            logger.debug(
                "[%s:%s] shape=%s K_target=%s K_final=%d expl_energy=%.4f flags={guardrail_up:%s,budget_cap:%s}",
                role,
                name,
                info["coeff_shape"],
                info.get("k_target", "-"),
                info["num_coeffs"],
                info["explained_energy"],
                bool(info.get("guardrail_increased_k", False)),
                bool(info.get("budget_capped", False)),
            )
    logger.info(
        "DCT3D tuning summary: signals=%d guardrail_up=%d budget_capped=%d",
        summary["signals"],
        summary["guardrail_up"],
        summary["budget_capped"],
    )

    return save_dct3d_rank_overrides(
        best=best,
        signal_specs=signal_specs,
        run_dir=Path(run_dir),
        allowed_signal_names=signal_names_by_role,
        merge_existing=merge_existing,
    )
