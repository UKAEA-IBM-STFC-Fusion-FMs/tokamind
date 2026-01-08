"""
Finetuning entrypoint for MMT using the convention-based config system.

This script:
- parses `--task`,
- loads and validates the merged config for phase="finetune",
- resolves the baseline task_config and builds datasets/metadata,
- builds signal specs/codecs and the standard shot→windows transform pipeline,
- optionally warm-starts from `model_init.model_dir`,
- runs the finetuning loop and writes outputs under cfg_mmt.paths["run_dir"].

Shared boilerplate (device/MP setup, default transforms, window datasets,
collate construction) lives in `scripts_mast.mast_utils.pipeline_helpers`.
"""

from __future__ import annotations

import argparse
import logging

from scripts.pipeline_tools.initialize_model_dataset import (
    initialize_model_dataset,
)

from scripts.pipeline_tools.initialize_dataset_and_metadata import (
    initialize_datasets_and_metadata_for_task,
)

from scripts_mast.mast_utils import (
    build_task_config,
    build_signals_by_role_from_task_config,
    setup_device_and_mp,
    build_default_transform,
    build_window_datasets,
    make_collate_fn,
    DEFAULT_CONFIGS_ROOT,
)

from mmt.utils.config import (
    load_experiment_config,
    validate_config,
)
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

DEBUG_MODE = False


def parse_args_finetune() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finetuning for a given task (convention-based configs)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task_2-1",
        help="Task folder name under scripts_mast/configs/tasks/<task>/",
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
    args = parse_args_finetune()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="finetune",
        configs_root=DEFAULT_CONFIGS_ROOT,
    )
    validate_config(cfg_mmt.raw)

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
    max_positions = cfg_mmt.preprocess["trim_chunks"]["max_chunks"]

    # Baseline task config (with overrides such as subset_of_shots/local)
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

    logging.getLogger("mmt.Task").info(
        "task=%s | phase=%s | device=%s", cfg_mmt.task, cfg_mmt.phase, device
    )

    # ------------------------------------------------------------------
    # Baseline datasets + metadata
    # ------------------------------------------------------------------
    datasets_shots_raw, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs/codecs
    signals_by_role = build_signals_by_role_from_task_config(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_metadata,
        chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
    )
    codecs = build_codecs(signal_specs)

    # ------------------------------------------------------------------
    # Model-specific transform chain (shot -> windows)
    # ------------------------------------------------------------------
    mmt_transform_map = build_default_transform(
        cfg_mmt,
        dict_metadata=dict_metadata,
        signal_specs=signal_specs,
        codecs=codecs,
        keep_output_native=keep_output_native,
    )

    # ------------------------------------------------------------------
    # Model-level datasets (shot-based wrappers)
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
    # Window-level datasets (cached or streaming)
    # ------------------------------------------------------------------
    datasets_windows = build_window_datasets(
        datasets_shots_wrapped=datasets_shots_wrapped,
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
    # Finetune
    # ------------------------------------------------------------------
    logging.getLogger("mmt.Train").info("Starting finetuning...")
    history = train_finetune(
        model=model,
        train_loader=dataloaders_mmt["train"],
        val_loader=dataloaders_mmt["val"],
        run_dir=cfg_mmt.paths["run_dir"],
        train_cfg=cfg_train,
        loader_cfg=cfg_loader,
    )

    logging.getLogger("mmt.Train").info("%s", history)


if __name__ == "__main__":
    main()
