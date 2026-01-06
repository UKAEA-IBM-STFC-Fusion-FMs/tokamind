from __future__ import annotations

import argparse
import torch
import torch.multiprocessing as mp

import copy
import yaml
from pathlib import Path

import numpy as np

from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from mmt.utils.config import (
    build_task_config,
    load_experiment_config,
)
from mmt.utils import (
    set_seed,
    setup_logging,
    initialize_mmt_datasets,
)

from mmt.data import (
    build_signal_role_modality_map,
    build_signal_specs,
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    TuneDCT3DTransform,
    ComposeTransforms,
    WindowStreamedDataset,
)


import logging

DEBUG_MODE = False


def parse_args_tune_dct3d() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune parse_args_tune_dct3d for a given task/phase config."
    )
    parser.add_argument(
        "--phase_config",
        type=str,
        default="mmt/configs/task_2-1/tune_dct3d.yaml",
        help=(
            "Path to the phase YAML config file "
            "(e.g. mmt/configs/task_2-1/tune_dct3d.yaml)"
        ),
    )
    args, _ = parser.parse_known_args()
    return args


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main() -> None:
    # ------------------------------------------------------------------
    # Device / multiprocessing
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Needed for DataLoader(num_workers>0) on some platforms
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------
    # Load MMT config (phase + experiment_base + embeddings + baseline)
    # ------------------------------------------------------------------
    args = parse_args_tune_dct3d()
    cfg_mmt = load_experiment_config(args.phase_config)

    if "tune_dct3d" not in cfg_mmt.raw:
        raise KeyError("Missing 'tune_dct3d' section in config.")

    # Small sub-configs for readability
    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    cfg_tune = cfg_mmt.raw["tune_dct3d"]

    # Baseline task config (with overrides such as subset_of_shots)
    cfg_task = build_task_config(cfg_mmt)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename=None,
        console=True,
    )
    logger.setLevel("DEBUG" if DEBUG_MODE else "INFO")

    logger = logging.getLogger("mmt.TuneDCT3D")
    logger.info(f"task = {cfg_mmt.task} | phase = {cfg_mmt.phase} | device = {device}")

    # ------------------------------------------------------------------
    # Baseline datasets + metadata
    # ------------------------------------------------------------------
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs
    signals_role_modality_map = build_signal_role_modality_map(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_role_modality_map,
        dict_metadata=dict_metadata,
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
    )

    mmt_transform_map = ComposeTransforms(
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
            TrimChunksTransform(
                max_chunks=cfg_trim["max_chunks"],
            ),
            tune_transform,
        ]
    )

    # ------------------------------------------------------------------
    # Model-level datasets (TaskModelTransformWrapper, shot-based)
    # ------------------------------------------------------------------
    datasets_mmt = initialize_mmt_datasets(
        datasets_train_val_test,
        dict_metadata,
        cfg_task,
        model_specific_transform=mmt_transform_map,
        verbose=True,
    )

    # random shot selection
    # Select the dataset and enforce n_shots
    cfg_sampling = cfg_tune["sampling"]
    ds_shots_full = datasets_mmt["train"]

    n_shots = cfg_sampling.get("n_shots")
    if n_shots is not None:
        rng = np.random.default_rng(cfg_mmt.seed)
        all_indices = np.arange(len(ds_shots_full))
        sel_indices = rng.choice(
            all_indices, size=min(n_shots, len(all_indices)), replace=False
        )
        # simple index-based subset
        ds_shots = [ds_shots_full[i] for i in sel_indices]
    else:
        ds_shots = ds_shots_full

    logger.info(
        "ds_shots_full=%d, requested n_shots=%s, selected=%d, sel_indices=%s",
        len(ds_shots_full),
        str(n_shots),
        len(ds_shots),
        np.array(sel_indices).tolist() if n_shots is not None else None,
    )

    # ------------------------------------------------------------------
    # Streaming dataset (window-level)
    # ------------------------------------------------------------------
    ds_windows = WindowStreamedDataset(
        ds_shots,
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

    logger.info(f"Processed windows: total={n_windows_total}")

    # ------------------------------------------------------------------
    # Select and print best configs
    # ------------------------------------------------------------------
    best = tune_transform.select_best()

    logger.info("Selected configurations:")
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            logger.info(
                f"[{role}:{name}] "
                f"shape={info['rep_shape']} "
                f"keep=({info['eff_keep_h']},{info['eff_keep_w']},{info['eff_keep_t']}) "
                f"rmse={info['rmse_mean_windows']:.4e} "
                f"eff_dim={info['effective_dim']}"
            )

    # ------------------------------------------------------------------
    # Create a tuned copy of embeddings config
    # ------------------------------------------------------------------
    embeddings_tuned = copy.deepcopy(cfg_mmt.embeddings)

    # Ensure per-signal override section exists
    embeddings_tuned.setdefault("per_signal_overrides", {})
    for role, by_sig in best.items():
        embeddings_tuned["per_signal_overrides"].setdefault(role, {})

        for name, info in by_sig.items():
            # Only override signals that are actually using DCT3D
            spec = signal_specs.get(role, name)
            if spec is None or spec.encoder_name != "dct3d":
                continue

            embeddings_tuned["per_signal_overrides"][role][name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": {
                    "keep_h": int(info["eff_keep_h"]),
                    "keep_w": int(info["eff_keep_w"]),
                    "keep_t": int(info["eff_keep_t"]),
                },
            }

    out_path = Path(cfg_mmt.paths["run_dir"]) / "embeddings_tuned.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(
            embeddings_tuned,
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info(f"Wrote tuned embeddings config to {out_path}")


if __name__ == "__main__":
    main()
