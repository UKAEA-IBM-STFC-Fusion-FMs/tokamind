"""
DCT3D tuning entrypoint for MMT - RANK MODE ONLY.

This script:
- Computes per-coefficient energy E[c_i²] over a sample of windows
- Selects top-K coefficients by explained variance for each signal
- Writes coefficient indices to .npy files and updated config

Output files:
  tasks_overrides/<task>/embeddings_overrides/dct3d.yaml
  tasks_overrides/<task>/embeddings_overrides/dct3d_indices/*.npy
  tasks_overrides/<task>/embeddings_overrides/history/dct3d_<timestamp>.yaml
  tasks_overrides/<task>/embeddings_overrides/history/dct3d_indices_<timestamp>/*.npy

For spatial mode (no tuning), users manually specify keep_h/w/t in config.

Notes
-----
- This script uses TuneRankedDCT3DTransform (variance-based selection)
- No grid search - much faster than old approach
- Outputs rank mode configs with coeff_indices_path references
"""

from __future__ import annotations

import argparse
import logging
import datetime
from pathlib import Path

import numpy as np
import yaml

from mast_utils.benchmark_imports import (
    initialize_MAST_dataset,
    initialize_model_dataset_iterable,
    get_task_metadata,
    get_train_test_val_shots,
)

from mast_utils import (
    load_experiment_config,
    load_task_definition,
    build_signals_by_role_from_task_definition,
    setup_device_and_mp,
)

from mmt.utils.config.validator import validate_config

from mmt.utils import set_seed, setup_logging
from mmt.data import (
    build_signal_specs,
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    TuneRankedDCT3DTransform,
    ComposeTransforms,
)


def parse_args_tune_dct3d() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DCT3D embedding parameters for a given task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",  # "pretrain_inputs_actuators_to_inputs_outputs",  # "_test",
        help="Task folder name under scripts_mast/configs/tasks_overrides/<task>/",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="input,actuator,output",
        help=(
            "Comma-separated roles to tune and write overrides for. "
            "Subset of: input, actuator, output. Default: input,actuator,output"
        ),
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    # ------------------------------------------------------------------
    # Device / multiprocessing
    # ------------------------------------------------------------------
    device = setup_device_and_mp()

    # ------------------------------------------------------------------
    # Load merged config (common + task + overrides)
    # ------------------------------------------------------------------
    args = parse_args_tune_dct3d()
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]

    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="tune_dct3d",
        embeddings_profile="dct3d",
    )
    validate_config(cfg_mmt)

    cfg_data = cfg_mmt.data
    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]
    cfg_tune = cfg_mmt.raw["tune_dct3d"]
    cfg_sampling = cfg_tune["sampling"]

    debug_mode = cfg_mmt.runtime["debug_logging"]
    local_flag = cfg_data.get("local", True)

    # benchmark task config (with overrides such as subset_of_shots/local)
    cfg_task = load_task_definition(args.task)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # For tune_dct3d, loader sets run_dir to: scripts_mast/configs/tasks_overrides/<task>/
    logger = setup_logging(
        cfg_mmt.paths["tune_dir"],
        logger_name="mmt",
        filename="tune_dct3d.log",
        console=True,
    )
    logger.setLevel("DEBUG" if debug_mode else "INFO")

    logger = logging.getLogger("mmt.TuneDCT3D")
    logger.info(
        "task=%s | phase=%s | device=%s | roles=%s",
        cfg_mmt.task,
        cfg_mmt.phase,
        device,
        ",".join(roles),
    )

    # -------------------------------------------------------------------
    # Initialize task-specific metadata
    # -------------------------------------------------------------------
    dict_task_metadata = get_task_metadata(
        cfg_task,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Initialize MAST datasets
    # ------------------------------------------------------------------

    # we randomly select n_shots from the train
    train_shots_, _, _ = get_train_test_val_shots(
        max_index=cfg_data["subset_of_shots"],
        shuffle=True,
        seed=cfg_mmt.seed,
    )

    mast_dataset_train = initialize_MAST_dataset(
        cfg_task,
        train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Signal specs
    # ------------------------------------------------------------------
    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task,
        dict_task_metadata,
    )

    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=cfg_chunks["chunk_length"],
    )

    # ------------------------------------------------------------------
    # Model-specific transform chain (shot -> windows)
    # ------------------------------------------------------------------
    # Extract guardrails config (may be None if not present)
    guardrails_cfg = cfg_tune.get("guardrails")
    
    tune_transform = TuneRankedDCT3DTransform(
        signal_specs=signal_specs,
        thresholds=cfg_tune["objective"]["thresholds"],
        max_budget=cfg_tune["objective"]["max_budget"],
        roles=roles,
        guardrails=guardrails_cfg,
    )

    mmt_transform_map = ComposeTransforms(
        [
            ChunkWindowsTransform(
                dict_metadata=dict_task_metadata,
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
            tune_transform,
        ]
    )

    # ------------------------------------------------------------------
    # Window-level iterable (benchmark wrapper yields windows directly)
    # ------------------------------------------------------------------
    ds_windows = initialize_model_dataset_iterable(
        mast_dataset_train,
        dict_task_metadata,
        cfg_task,
        model_specific_transform=mmt_transform_map,
        test_mode=False,
        shuffle_windows=False,  # we shuffle when loading using get_train_test_val_shots
        shuffle_buffer_size=512,
        verbose=False,
    )
    
    assert ds_windows is not None, "window iterable should not be None"

    # ------------------------------------------------------------------
    # Shot exploration
    # ------------------------------------------------------------------
    max_windows = cfg_sampling.get("max_windows", None)

    n_windows_total = 0
    for _window in ds_windows:
        n_windows_total += 1
        if max_windows is not None and n_windows_total >= max_windows:
            break

    logger.info("Processed windows: total=%d", n_windows_total)

    # ------------------------------------------------------------------
    # Select and print best configs
    # ------------------------------------------------------------------
    best = tune_transform.select_best()

    logger.info("Selected configurations:")
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            logger.info(
                "[%s:%s] shape=%s num_coeffs=%d expl_energy=%.4f",
                role,
                name,
                info["coeff_shape"],
                info["num_coeffs"],
                info["explained_energy"],
            )

    # ------------------------------------------------------------------
    # Write coefficient indices (.npy files) and config (rank mode)
    # ------------------------------------------------------------------
    base_dir = Path(cfg_mmt.paths["tune_dir"])
    base_dir.mkdir(parents=True, exist_ok=True)

    # Create indices directory
    indices_dir = base_dir / "dct3d_indices"
    indices_dir.mkdir(exist_ok=True)

    # Timestamp for history
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    hist_dir = base_dir / "history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    hist_indices_dir = hist_dir / f"dct3d_indices_{ts}"
    hist_indices_dir.mkdir(exist_ok=True)

    # Build config with rank mode
    overrides_out = {"embeddings": {"per_signal_overrides": {}}}

    for role, by_sig in best.items():
        for name, info in by_sig.items():
            spec = signal_specs.get(role, name)
            if spec is None or spec.encoder_name != "dct3d":
                continue

            # Save coefficient indices to .npy files
            coeff_indices = info["coeff_indices"]
            filename = f"{role}_{name}.npy"

            # Save to main location
            np.save(indices_dir / filename, coeff_indices)
            # Save to history
            np.save(hist_indices_dir / filename, coeff_indices)

            # Compute dimension distribution statistics
            h, w, t = info["coeff_shape"]
            indices_3d = np.unravel_index(coeff_indices, (h, w, t))
            unique_h = len(np.unique(indices_3d[0]))
            unique_w = len(np.unique(indices_3d[1]))
            unique_t = len(np.unique(indices_3d[2]))
            
            # Build config entry (rank mode)
            overrides_out["embeddings"]["per_signal_overrides"].setdefault(role, {})
            overrides_out["embeddings"]["per_signal_overrides"][role][name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": {
                    "selection_mode": "rank",
                    "coeff_indices_path": f"dct3d_indices/{filename}",
                    "coeff_shape": list(info["coeff_shape"]),
                    "num_coeffs": int(info["num_coeffs"]),
                    "explained_energy": float(info["explained_energy"]),
                    "dim_distribution": {
                        "unique_h": int(unique_h),
                        "unique_w": int(unique_w),
                        "unique_t": int(unique_t),
                    },
                },
            }

    # Write YAML configs
    out_path = base_dir / "dct3d.yaml"
    hist_path = hist_dir / f"dct3d_{ts}.yaml"

    for p in (out_path, hist_path):
        with p.open("w", encoding="utf-8") as f:
            yaml.safe_dump(overrides_out, f, sort_keys=False, default_flow_style=False)

    logger.info("Wrote tuned embedding overrides to %s", out_path)
    logger.info("Wrote coefficient indices to %s", indices_dir)
    logger.info("Archived to %s and %s", hist_path, hist_indices_dir)


if __name__ == "__main__":
    main()
