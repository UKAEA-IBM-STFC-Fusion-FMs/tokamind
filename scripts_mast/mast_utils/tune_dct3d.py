"""
DCT3D embedding tuning orchestration for MAST.

This module is intentionally orchestration-first:

- dataset construction and streaming for a small tuning subset,
- transform wiring (`TuneRankedDCT3DTransform`) with role-specific objective config (thresholds, guardrails, budgets),
- persistence of run-local artifacts (`dct3d_indices/*.npy`, `dct3d.yaml`),
- loading helpers for downstream finetune/eval inheritance.

The transform owns selection policy and signal-level metadata computation.
This module consumes that output and writes stable runtime artifacts.

Main entrypoints
----------------
- `run_dct3d_tuning(...)`
  Runs tuning and writes:
  - `runs/<run_id>/embeddings/dct3d_indices/<role>_<signal>.npy`
  - `runs/<run_id>/embeddings/dct3d.yaml`
- `load_embeddings_overrides(...)`
  Reads `dct3d.yaml` and returns `embeddings.per_signal_overrides`.

Persisted rank metadata
-----------------------
Each tuned signal in `dct3d.yaml` stores rank-mode kwargs and tuning metadata:
- `coeff_shape`, `num_coeffs`, `explained_energy`
- `dim_distribution.{unique_h,unique_w,unique_t}`
- `tuning_info.{target,k_target,guardrail_min_k,k_after_guardrails,k_final, n_windows,max_budget,flags,tuned_in_run_id}`
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import numpy as np
import yaml

from mmt.data import (
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    TuneRankedDCT3DTransform,
    ComposeTransforms,
)
from mmt.data.signal_spec import SignalSpecRegistry
from mmt.utils.config.schema import ExperimentConfig

from .benchmark_imports import (
    initialize_MAST_dataset,
    initialize_TokaMark_dataset,
    get_train_test_val_shots,
)


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.TuneRankedDCT3D")


# ----------------------------------------------------------------------------------------------------------------------
def run_dct3d_tuning(  # NOSONAR - Ignore cognitive complexity
    cfg_mmt: ExperimentConfig,
    signal_specs: SignalSpecRegistry,
    cfg_task: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    run_dir: Path,
    roles: Sequence[str] = ("input", "actuator", "output"),
) -> dict[str, Any]:
    """
    Tune DCT3D rank-mode embeddings and save results to the run folder.

    Builds a small MAST dataset subsample, streams windows through TuneRankedDCT3DTransform to accumulate
    per-coefficient energies E[c_i²], then selects top-K coefficients meeting the explained-variance threshold for each
    signal.

    Outputs written to `run_dir/embeddings/`:
      - `dct3d_indices/<role>_<signal_name>.npy`   — selected coefficient indices
      - `dct3d.yaml`                               — per-signal rank-mode overrides

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config. Reads `embeddings.tune_embeddings` for tuning params, and `preprocess` for
        chunk/window settings.
    signal_specs : SignalSpecRegistry
        Signal spec registry (built from default spatial embeddings config).
    cfg_task : Mapping[str, Any]
        Benchmark task definition (dictionary from load_task_definition()).
    dict_task_metadata : Mapping[str, Any]
        Task metadata dictionary (dictionary loaded from get_task_metadata()).
    run_dir : Path
        Training run directory. Results saved to `run_dir/embeddings/`.
    roles : Sequence[str]
        Roles to tune.
        Optional. Default: ("input", "actuator", "output").

    Returns
    -------
    dict[str, Any]
        Per-signal overrides to merge into `cfg_mmt.raw["embeddings"]["per_signal_overrides"]`.
        Structure: `{role: {signal_name: {encoder_name, encoder_kwargs}}}`.

    Raises
    ------
    ValueError
        If window iterable returned None.

    """

    cfg_data = cfg_mmt.data
    cfg_prep = cfg_mmt.preprocess
    cfg_tune = cfg_mmt.embeddings.get("tune_embeddings", {})
    cfg_objective = cfg_tune.get("objective", {})

    n_shots = cfg_tune.get("n_shots", 100)
    max_windows = cfg_tune.get("max_windows", 15000)
    local_flag = cfg_data.get("local", True)
    local_path = cfg_data.get("local_path", None)
    max_budget_cfg = cfg_objective.get("max_budget", {})
    budget_summary = (
        {r: max_budget_cfg.get(r) for r in roles} if isinstance(max_budget_cfg, Mapping) else max_budget_cfg
    )
    guardrails_cfg = cfg_tune.get("guardrails") or {}
    guardrails_state = "enabled" if guardrails_cfg.get("enable") else "disabled"

    logger.info(
        "n_shots=%d | max_windows=%d | roles=%s | budgets=%s | guardrails=%s",
        n_shots,
        max_windows,
        ",".join(roles),
        budget_summary,
        guardrails_state,
    )

    # ..................................................................................................................
    # Dataset: subsample training shots for tuning
    # ..................................................................................................................

    train_shots, _, _ = get_train_test_val_shots(max_index=n_shots, shuffle=True, seed=cfg_mmt.seed)

    store_settings = {"base_local_zarr_path": local_path} if (local_flag and local_path) else None

    mast_dataset = initialize_MAST_dataset(
        config_task=cfg_task,
        shots_list=train_shots,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        store_manager_settings=store_settings,
        verbose=False,
    )

    # ..................................................................................................................
    # Transform pipeline
    # ..................................................................................................................

    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    tune_transform = TuneRankedDCT3DTransform(
        signal_specs=signal_specs,
        thresholds=cfg_objective.get("thresholds", {}),
        max_budget=max_budget_cfg,
        roles=roles,
        guardrails=guardrails_cfg,
    )

    transform_pipeline = ComposeTransforms(
        [
            ChunkWindowsTransform(
                dict_metadata=dict_task_metadata,
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks.get("stride"),
            ),
            SelectValidWindowsTransform(
                min_valid_inputs_actuators=cfg_valid_win["min_valid_inputs_actuators"],
                min_valid_chunks=cfg_valid_win["min_valid_chunks"],
                min_valid_outputs=cfg_valid_win["min_valid_outputs"],
                window_stride_sec=cfg_valid_win["window_stride_sec"],
            ),
            TrimChunksTransform(max_chunks=cfg_trim["max_chunks"]),
            tune_transform,
        ]
    )

    ds_windows = initialize_TokaMark_dataset(
        dataset=mast_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=cfg_task,
        custom_transform=transform_pipeline,
        test_mode=False,
        shuffle_windows=False,
        shuffle_buffer_size=512,
        verbose=False,
    )

    if ds_windows is None:
        raise ValueError("Window iterable returned None — cannot run DCT3D tuning without data.")

    # ..................................................................................................................
    # Stream windows to accumulate energies
    # ..................................................................................................................

    n_windows_total = 0
    for _ in ds_windows:
        n_windows_total += 1
        if (max_windows is not None) and (n_windows_total >= max_windows):
            break

    logger.info("Tuning: processed %d windows", n_windows_total)

    # ..................................................................................................................
    # Select best and log
    # ..................................................................................................................

    best = tune_transform.select_best()
    summary = tune_transform.summarize_selection(best=best)
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

    # ..................................................................................................................
    # Save indices and config
    # ..................................................................................................................

    emb_dir = run_dir / "embeddings"
    indices_dir = emb_dir / "dct3d_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    per_signal_overrides: dict = {}

    for role, by_sig in best.items():
        for name, info in by_sig.items():
            spec = signal_specs.get(role, name)
            if (spec is None) or (spec.encoder_name != "dct3d"):
                continue

            coeff_indices = info["coeff_indices"]
            filename = f"{role}_{name}.npy"
            np.save(indices_dir / filename, coeff_indices)

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

    # Merge with any existing overrides (e.g., inherited roles from source) so that a partial retune does not drop roles
    # that were not re-tuned this run.

    dct3d_yaml_path = emb_dir / "dct3d.yaml"
    if dct3d_yaml_path.exists():
        with dct3d_yaml_path.open(encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing_overrides = existing.get("embeddings", {}).get("per_signal_overrides", {})
        for role, sigs in per_signal_overrides.items():
            existing_overrides.setdefault(role, {}).update(sigs)
        merged_overrides = existing_overrides
    else:
        merged_overrides = per_signal_overrides

    with dct3d_yaml_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(
            data={"embeddings": {"per_signal_overrides": merged_overrides}},
            stream=f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info("Saved tuned overrides → %s", dct3d_yaml_path)
    logger.info("Saved coefficient indices → %s", indices_dir)

    return per_signal_overrides


# ----------------------------------------------------------------------------------------------------------------------
def load_embeddings_overrides(run_dir: Path) -> dict[str, Any]:
    """
    Load per-signal rank-mode overrides from a run's embeddings folder.

    Parameters
    ----------
    run_dir:
        Training run directory. Reads `run_dir/embeddings/dct3d.yaml`.

    Returns
    -------
    dict
        `{role: {signal_name: {encoder_name, encoder_kwargs}}}`.
        Returns `{}` if the file does not exist (e.g., run used spatial encoding and no tuning was performed).

    """

    dct3d_yaml = run_dir / "embeddings" / "dct3d.yaml"
    if not dct3d_yaml.exists():
        return {}

    with dct3d_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("embeddings", {}).get("per_signal_overrides", {})
