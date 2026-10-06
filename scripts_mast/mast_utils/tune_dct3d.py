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

from mmt.data import run_dct3d_tuning_from_windows, load_dct3d_rank_overrides
from mmt.data.signal_spec import SignalSpecRegistry
from mmt.utils.config.schema import ExperimentConfig

from .benchmark_imports import (
    initialize_MAST_dataset,
    initialize_TokaMark_dataset,
    get_train_test_val_shots,
)
from .tokamark_split import resolve_split_assets


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
    signal_names_by_role: Mapping[str, set[str]] | None = None,
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
        Merged experiment config. Reads `embeddings.tuning` for tuning params, and `preprocess` for
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
    signal_names_by_role : Mapping[str, set[str]] | None
        Optional signal-name filter keyed by role. When provided, tuning still accumulates role-level statistics through
        the transform, but only matching signals are saved into `dct3d.yaml`.

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
    cfg_tune = cfg_mmt.embeddings.get("tuning", {})

    n_shots = cfg_tune.get("n_shots", 100)
    max_windows = cfg_tune.get("max_windows", 15000)
    local_flag = cfg_data.get("local", True)
    local_path = cfg_data.get("local_path", None)
    split_assets = resolve_split_assets(split=cfg_data["split"])
    roles_to_tune = list(roles)
    cfg_objective = cfg_tune.get("objective", {})
    max_budget_cfg = cfg_objective.get("max_budget", {})
    budget_summary = (
        {r: max_budget_cfg.get(r) for r in roles_to_tune} if isinstance(max_budget_cfg, Mapping) else max_budget_cfg
    )
    guardrails_cfg = cfg_tune.get("guardrails") or {}
    guardrails_state = "enabled" if guardrails_cfg.get("enable") else "disabled"

    logger.info(
        "n_shots=%d | max_windows=%d | roles=%s | budgets=%s | guardrails=%s",
        n_shots,
        max_windows,
        ",".join(roles_to_tune),
        budget_summary,
        guardrails_state,
    )

    # ..................................................................................................................
    # Dataset: subsample training shots for tuning
    # ..................................................................................................................

    train_shots, _, _ = get_train_test_val_shots(
        max_index=n_shots,
        shuffle=True,
        seed=cfg_mmt.seed,
        data_splits_file_path=split_assets["data_splits_file_path"],
    )

    store_settings = {"base_local_zarr_path": local_path} if (local_flag and local_path) else None

    mast_dataset = initialize_MAST_dataset(
        config_task=cfg_task,
        shots_list=train_shots,
        local_flag=local_flag,
        use_std_scaling=True,
        stats_metadata_file_path=split_assets["stats_metadata_file_path"],
        remove_outliers=True,
        outlier_metadata_file=split_assets["outlier_metadata_file"],
        store_manager_settings=store_settings,
        verbose=False,
    )

    ds_windows = initialize_TokaMark_dataset(
        dataset=mast_dataset,
        task_metadata=dict_task_metadata,
        config_metadata=cfg_task,
        custom_transform=None,
        test_mode=False,
        shuffle_windows=False,
        shuffle_buffer_size=512,
        verbose=False,
    )

    if ds_windows is None:
        raise ValueError("Window iterable returned None — cannot run DCT3D tuning without data.")

    return run_dct3d_tuning_from_windows(
        windows=ds_windows,
        signal_specs=signal_specs,
        dict_metadata=dict_task_metadata,
        preprocess_cfg=cfg_prep,
        tuning_cfg=cfg_tune,
        run_dir=run_dir,
        roles=roles_to_tune,
        signal_names_by_role=signal_names_by_role,
        max_windows=max_windows,
        merge_existing=True,
    )


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

    return load_dct3d_rank_overrides(run_dir=run_dir)
