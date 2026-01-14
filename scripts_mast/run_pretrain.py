"""
Pretraining entrypoint for MMT using the convention-based config system.

This script:
- parses `--task` (a pretrain-* task folder),
- loads and validates the merged config for phase="pretrain",
- resolves the benchmark task_config and builds datasets/metadata,
- builds signal specs/codecs and the standard shot→windows transform pipeline,
- optionally warm-starts from `model_init.model_dir`,
- runs the training loop and writes outputs under cfg_mmt.paths["run_dir"].

Shared boilerplate (device/MP setup, default transforms, window datasets,
collate construction) lives in `scripts_mast.mast_utils.pipeline_helpers`.
"""

from __future__ import annotations

import argparse
import logging

from mast_utils.benchmark_imports import (
    initialize_MAST_dataset,
    initialize_model_dataset,
    get_task_metadata,
    get_train_test_val_shots,
)

from mast_utils import (
    load_experiment_config,
    load_task_definition,
    build_signals_by_role_from_task_definition,
    setup_device_and_mp,
    build_default_transform,
    build_window_datasets,
    make_collate_fn,
)

from mmt.utils.config.validator import validate_config

from mmt.utils import (
    set_seed,
    setup_logging,
    initialize_mmt_dataloaders,
)

from mmt.data import (
    build_signal_specs,
    build_codecs,
)

from mmt.models import MultiModalTransformer
from mmt.train.loop import train_finetune


def parse_args_pretrain() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pretraining for a given pretrain task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Pretrain task folder name under scripts_mast/configs/tasks_overrides/<task>/",
    )
    parser.add_argument(
        "--emb_profile",
        type=str,
        default="dct3d",
        help="embeddings_profile chosen for the task: "
        "scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml",
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
    args = parse_args_pretrain()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="pretrain",
        embeddings_profile=args.emb_profile,
    )
    validate_config(cfg_mmt)

    cfg_data = cfg_mmt.data
    cfg_model = cfg_mmt.model
    cfg_loader = cfg_mmt.loader
    cfg_collate = cfg_mmt.collate
    cfg_train = cfg_mmt.train

    enable_cache = cfg_data["cache"].get("enable", False)
    num_workers_cache = cfg_data["cache"].get("num_workers", 0)
    keep_output_native = cfg_data.get("keep_output_native", False)
    local_flag = cfg_data.get("local", True)
    debug_mode = cfg_mmt.runtime["debug_logging"]

    cfg_backbone = cfg_model["backbone"]
    cfg_modality_heads = cfg_model["modality_heads"]
    cfg_output_adapters = cfg_model["output_adapters"]
    max_positions = cfg_mmt.preprocess["trim_chunks"]["max_chunks"]

    # benchmark task config (with overrides such as subset_of_shots/local)
    cfg_task = load_task_definition(args.task)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename="pretrain.log",
        console=True,
    )
    logger.setLevel("DEBUG" if debug_mode else "INFO")

    logging.getLogger("mmt.Task").info(
        "task=%s | phase=%s | device=%s", cfg_mmt.task, cfg_mmt.phase, device
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

    train_shots_, _, val_shots_ = get_train_test_val_shots(
        max_index=cfg_data["subset_of_shots"],
    )

    train_mast_dataset = initialize_MAST_dataset(
        cfg_task,
        train_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
    )

    val_mast_dataset = initialize_MAST_dataset(
        cfg_task,
        val_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
    )

    # ------------------------------------------------------------------
    # Signal specs
    # ------------------------------------------------------------------

    # Build signals_by_role from benchmark config + metadata and signal specs/codecs
    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task,
        dict_task_metadata,
    )

    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
    )

    codecs = build_codecs(signal_specs)

    # ------------------------------------------------------------------
    # Model-specific transform chain (shot -> windows)
    # ------------------------------------------------------------------
    mmt_transform_map = build_default_transform(
        cfg_mmt,
        dict_metadata=dict_task_metadata,
        signal_specs=signal_specs,
        codecs=codecs,
        keep_output_native=keep_output_native,
    )

    # ------------------------------------------------------------------
    # Model-level datasets (shot-based wrappers)
    # ------------------------------------------------------------------
    model_datasets = {
        "train": initialize_model_dataset(
            train_mast_dataset,
            dict_task_metadata,
            cfg_task,
            model_specific_transform=mmt_transform_map,
            verbose=True,
        ),
        "val": initialize_model_dataset(
            val_mast_dataset,
            dict_task_metadata,
            cfg_task,
            model_specific_transform=mmt_transform_map,
            verbose=False,
        ),
    }

    # Optional debug: iterate a single shot to exercise wrapper + transforms
    if debug_mode and model_datasets["train"] is not None:
        ds = model_datasets["train"]
        shot = ds[0]
        for _ in shot:
            continue

    # ------------------------------------------------------------------
    # Window-level datasets (cached or streaming)
    # ------------------------------------------------------------------
    datasets_windows = build_window_datasets(
        model_datasets=model_datasets,
        enable_cache=enable_cache,
        num_workers_cache=num_workers_cache,
        seed=cfg_mmt.seed,
        shuffle_shots=cfg_loader["shuffle"],
        cache_splits=("train", "val"),
    )

    # ------------------------------------------------------------------
    # Dataloaders (always window-level batches)
    # ------------------------------------------------------------------
    collate_fn = make_collate_fn(
        base_cfg=cfg_collate,
        keep_output_native=keep_output_native,
    )

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
    model_source_cfg = cfg_mmt.raw.get("model_source") or None
    if model_source_cfg is not None:
        run_init = model_source_cfg.get("run_dir", None)
        load_parts = model_source_cfg.get("load_parts", None)

        if run_init is not None:
            from mmt.checkpoints import load_parts_from_run_dir

            load_parts_from_run_dir(
                model,
                run_init,
                load_parts=load_parts,
                map_location=device,
            )

    # ------------------------------------------------------------------
    # Pretrain (using the shared training loop)
    # ------------------------------------------------------------------
    logger = logging.getLogger("mmt.Train")
    logger.info("Starting pretraining...")
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
