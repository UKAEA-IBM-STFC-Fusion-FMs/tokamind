from __future__ import annotations

import argparse
import torch
import torch.multiprocessing as mp


from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from mmt.utils.config import (
    build_baseline_task_config,
    load_experiment_config,
    validate_train_config,
)
from mmt.utils import (
    set_seed,
    setup_logging,
    initialize_mmt_datasets,
    initialize_mmt_dataloaders,
)

from mmt.data import (
    build_signal_role_modality_map,
    build_signal_specs,
    build_codecs,
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    ComposeTransforms,
    WindowStreamedDataset,
    MMTCollate,
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
    validate_train_config(cfg_mmt.raw)

    # Small sub-configs for readability
    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    cfg_data = cfg_mmt.data
    cfg_model = cfg_mmt.model
    cfg_loader = cfg_mmt.loader
    cfg_collate = cfg_mmt.collate
    cfg_train = cfg_mmt.train

    enable_cache = cfg_data["cache"].get("enable", False)
    num_workers_cache = cfg_data["cache"].get("num_workers", 0)
    keep_output_native = cfg_data.get("keep_output_native", False)

    cfg_backbone = cfg_model["backbone"]
    cfg_modality_heads = cfg_model["modality_heads"]
    cfg_output_adapters = cfg_model["output_adapters"]
    max_positions = cfg_trim["max_chunks"]

    # Baseline task config (with overrides such as subset_of_shots)
    cfg_task = build_baseline_task_config(cfg_mmt)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename="finetune.log",
        console=True,
    )
    logger.setLevel("DEBUG" if DEBUG_MODE else "INFO")

    logger = logging.getLogger("mmt.Task")
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
    codecs = build_codecs(signal_specs)

    # ------------------------------------------------------------------
    # Model-specific transform chain (shot -> windows)
    # ------------------------------------------------------------------
    mmt_transform_map = ComposeTransforms(
        [
            ChunkWindowsTransform(
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
                chunk_length_sec=cfg_chunks["chunk_length"],
                delta=cfg_task["task_window_segmenter"]["delta"],
                output_length=cfg_task["task_window_segmenter"]["output_length"],
                max_chunks=cfg_trim["max_chunks"],
            ),
            # TODO: we will add the new transform here
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

    if DEBUG_MODE:
        ds = datasets_mmt["train"]
        shot = ds[0]
        for _ in shot:
            # apply only the transforms you want to debug
            continue

    # ------------------------------------------------------------------
    # Streaming dataset (window-level)
    # ------------------------------------------------------------------
    # For tuning, streaming is the default to avoid caching full datasets.
    # Shuffle shots only if you want a random sample of windows early.
    shuffle_shots = bool(_get_nested(cfg_mmt.raw, "loader.shuffle", True))
    ds_windows = WindowStreamedDataset(
        ds_stream,
        shuffle_shots=shuffle_shots,
        seed=cfg_mmt.seed,
    )

    # ------------------------------------------------------------------
    # Dataloaders (always window-level batches)
    # ------------------------------------------------------------------
    collate_fn = MMTCollate({**cfg_collate, "keep_output_native": keep_output_native})

    dataloaders_mmt = initialize_mmt_dataloaders(
        datasets_for_loader,
        collate_fn,
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
        shuffle=cfg_loader["shuffle"],
        drop_last=cfg_loader["drop_last"],
        seed=cfg_mmt.seed,
    )


if __name__ == "__main__":
    main()
