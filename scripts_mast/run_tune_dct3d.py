"""
DCT3D tuning entrypoint for MMT (per-task embedding hyperparameters).

This script:
- parses `--task` and `--roles`,
- loads and validates the merged config for phase="tune_dct3d",
- resolves the benchmark task_config and builds datasets/metadata,
- streams windows and evaluates candidate (keep_h, keep_w, keep_t) settings,
- selects the best per-signal configuration,
- writes tuned results to:

    tasks_overrides/<task>/embeddings_overrides/dct3d.yaml

This phase does not train the transformer; it tunes embedding parameters only.

Notes
-----
- This script is specific to the DCT3D embedding family and always writes into
  the "dct3d" embedding profile file.
- The tuner writes ONLY per-signal overrides; defaults remain in
  scripts_mast/configs/common/embeddings.yaml.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import yaml

from scripts_mast.mast_utils.benchmark_imports import (
    initialize_MAST_dataset,
    initialize_model_dataset,
    get_task_metadata,
    get_train_test_val_shots,
)

from scripts_mast.mast_utils import (
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
    TuneDCT3DTransform,
    ComposeTransforms,
    WindowStreamedDataset,
)


def parse_args_tune_dct3d() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DCT3D embedding parameters for a given task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Task folder name under scripts_mast/configs/tasks_overrides/<task>/",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default=["input", "actuator", "output"],
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
    roles = args.roles

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
    n_shots = cfg_sampling.get("n_shots")

    # benchmark task config (with overrides such as subset_of_shots/local)
    cfg_task = load_task_definition(args.task)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # For tune_dct3d, loader sets run_dir to: scripts_mast/configs/tasks_overrides/<task>/
    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename=None,
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
        max_index=n_shots,
        shuffle=True,
        seed=cfg_mmt.seed,
    )

    train_mast_dataset = initialize_MAST_dataset(
        cfg_task,
        train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
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
    tune_transform = TuneDCT3DTransform(
        signal_specs=signal_specs,
        keep_h=cfg_tune["search_space"]["keep_h"],
        keep_w=cfg_tune["search_space"]["keep_w"],
        keep_t=cfg_tune["search_space"]["keep_t"],
        thresholds=cfg_tune["objective"]["thresholds"],
        roles=roles,
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
    # Shot-level dataset (wrapped) for tuning
    # ------------------------------------------------------------------
    model_dataset = initialize_model_dataset(
        train_mast_dataset,
        dict_task_metadata,
        cfg_task,
        model_specific_transform=mmt_transform_map,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Window-level dataset (streaming)
    # ------------------------------------------------------------------
    ds_windows = WindowStreamedDataset(
        model_dataset,
        shuffle_shots=False,
        seed=cfg_mmt.seed,
    )

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
                "[%s:%s] shape=%s keep=(%s,%s,%s) expl_energy=%.4f eff_dim=%s",
                role,
                name,
                info["rep_shape"],
                info["eff_keep_h"],
                info["eff_keep_w"],
                info["eff_keep_t"],
                info["explained_energy_mean_windows"],
                info["effective_dim"],
            )

    # ------------------------------------------------------------------
    # Write tuned overrides to: tasks_overrides/<task>/embeddings_overrides/dct3d.yaml
    # (ONLY per-signal overrides; no defaults)
    # ------------------------------------------------------------------
    overrides_out = {"embeddings": {"per_signal_overrides": {}}}

    for role, by_sig in best.items():
        for name, info in by_sig.items():
            spec = signal_specs.get(role, name)
            if spec is None or spec.encoder_name != "dct3d":
                continue

            tuned_keep_h = int(info["eff_keep_h"])
            tuned_keep_w = int(info["eff_keep_w"])
            tuned_keep_t = int(info["eff_keep_t"])

            base_kwargs = dict(spec.encoder_kwargs or {})
            base_keep_h = int(base_kwargs.get("keep_h", tuned_keep_h))
            base_keep_w = int(base_kwargs.get("keep_w", tuned_keep_w))
            base_keep_t = int(base_kwargs.get("keep_t", tuned_keep_t))

            if (tuned_keep_h, tuned_keep_w, tuned_keep_t) == (
                base_keep_h,
                base_keep_w,
                base_keep_t,
            ):
                continue

            new_kwargs = copy.deepcopy(base_kwargs)
            new_kwargs.update(
                {"keep_h": tuned_keep_h, "keep_w": tuned_keep_w, "keep_t": tuned_keep_t}
            )

            overrides_out["embeddings"]["per_signal_overrides"].setdefault(role, {})
            overrides_out["embeddings"]["per_signal_overrides"][role][name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": new_kwargs,
            }

    out_path = Path(cfg_mmt.paths["run_dir"]) / "embeddings_overrides" / "dct3d.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(overrides_out, f, sort_keys=False, default_flow_style=False)

    logger.info("Wrote tuned embedding overrides to %s", out_path)


if __name__ == "__main__":
    main()
