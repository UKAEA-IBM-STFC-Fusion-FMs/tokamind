from __future__ import annotations

import argparse
import torch
import torch.multiprocessing as mp

from typing import Any

import time

from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from scripts.pipelines.utils.utils import (
    initialize_model_dataset,
)

from scripts_mast.mast_utils import (
    build_task_config,
    build_signals_by_role_from_task_config,
)


from mmt.utils.config import (
    load_experiment_config,
    validate_train_config,
)
from mmt.utils import (
    set_seed,
    setup_logging,
    initialize_mmt_dataloaders,
)

from mmt.data import (
    build_signal_specs,
    build_codecs,
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    EmbedChunksTransform,
    BuildTokensTransform,
    ComposeTransforms,
    WindowStreamedDataset,
    WindowCachedDataset,
    MMTCollate,
)

from mmt.models import MultiModalTransformer
from mmt.train.loop import train_finetune

import logging


DEBUG_MODE = False


def parse_args_finetune() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finetuning for a given task/phase config."
    )
    parser.add_argument(
        "--phase_config",
        type=str,
        default="scripts_mast/configs/task_2-1/finetune_default.yaml",
        help=(
            "Path to the phase YAML config file "
            "(e.g. mmt/configs/task_2-1/finetune_default.yaml)"
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
    args = parse_args_finetune()
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
    cfg_task = build_task_config(cfg_mmt)

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
    datasets_shots_raw, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs
    signals_by_role = build_signals_by_role_from_task_config(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
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
            EmbedChunksTransform(
                signal_specs=signal_specs,
                codecs=codecs,
                keep_output_native=keep_output_native,
            ),
            BuildTokensTransform(
                signal_specs=signal_specs,
            ),
        ]
    )

    # ------------------------------------------------------------------
    # Model-level datasets (TaskModelTransformWrapper, shot-based)
    # ------------------------------------------------------------------
    datasets_shots_wrapped = {
        "train": initialize_model_dataset(
            datasets_shots_raw.get("train"),
            dict_metadata,
            cfg_task,
            model_specific_transform=mmt_transform_map,
            verbose=True,
        ),
        "val": initialize_model_dataset(
            datasets_shots_raw.get("val"),
            dict_metadata,
            cfg_task,
            model_specific_transform=mmt_transform_map,
            verbose=False,
        ),
    }

    if DEBUG_MODE and datasets_shots_wrapped["train"] is not None:
        # Trigger one full-shot iteration to exercise the wrapper + transforms.
        ds = datasets_shots_wrapped["train"]
        shot = ds[0]
        for _ in shot:
            continue

    # ------------------------------------------------------------------
    # Optionally materialise streaming datasets to RAM (window-level)
    # ------------------------------------------------------------------
    datasets_windows: dict[str, Any] = {}

    for split, ds_stream in datasets_shots_wrapped.items():
        if ds_stream is None:
            datasets_windows[split] = None
            continue

        # Decide whether this split should be cached
        use_cache_for_split = enable_cache and split in ("train", "val")

        if use_cache_for_split:
            logger = logging.getLogger("mmt.Cache")
            logger.info("Starting caching split=%s", split)
            t0 = time.perf_counter()
            ds = WindowCachedDataset.from_streaming(
                ds_stream,
                num_workers_cache=num_workers_cache,
            )
            t1 = time.perf_counter()
            logger.info("Finished caching %s in %.3f seconds", split, t1 - t0)
        else:
            # Window-level streaming dataset (also used for test / non-cached splits)
            ds = WindowStreamedDataset(
                ds_stream,
                shuffle_shots=cfg_loader["shuffle"],
                seed=cfg_mmt.seed,
            )

        datasets_windows[split] = ds

    # ------------------------------------------------------------------
    # Dataloaders (always window-level batches)
    # ------------------------------------------------------------------
    collate_fn = MMTCollate({**cfg_collate, "keep_output_native": keep_output_native})

    dataloaders_mmt = initialize_mmt_dataloaders(
        datasets_windows,
        collate_fn,
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
        shuffle=cfg_loader["shuffle"],
        drop_last=cfg_loader["drop_last"],
        seed=cfg_mmt.seed,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = MultiModalTransformer(
        signal_specs=signal_specs,
        d_model=cfg_backbone["d_model"],
        n_layers=cfg_backbone["n_layers"],
        n_heads=cfg_backbone["n_heads"],
        dim_ff=cfg_backbone["dim_ff"],
        dropout=cfg_backbone["dropout"],
        backbone_activation=cfg_backbone["activation"],
        max_positions=max_positions,
        modality_heads_cfg=cfg_modality_heads,
        output_adapters_cfg=cfg_output_adapters,
        debug_tokens=False,
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Optional warm-start from previous run
    # ------------------------------------------------------------------
    model_init_cfg = cfg_mmt.raw.get("model_init", None)
    if model_init_cfg is not None:
        run_init = model_init_cfg.get("model_dir", None)
        load_parts = model_init_cfg.get("load_parts", None)

        if run_init is not None:
            from mmt.checkpoints import load_parts_from_run_dir

            load_parts_from_run_dir(
                model,
                run_init,
                load_parts=load_parts,
                map_location=device,
            )

    # ------------------------------------------------------------------
    # Run a short finetuning test (few epochs, config-driven)
    # ------------------------------------------------------------------
    logger = logging.getLogger("mmt.Train")
    logger.info("Starting finetuning test run...")
    history = train_finetune(
        model=model,
        train_loader=dataloaders_mmt["train"],
        val_loader=dataloaders_mmt["val"],
        run_dir=cfg_mmt.paths["run_dir"],
        train_cfg=cfg_train,
        loader_cfg=cfg_loader,
    )

    logger.info("%s", history)


if __name__ == "__main__":
    main()
