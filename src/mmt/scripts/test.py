from __future__ import annotations

import torch.multiprocessing as mp
import argparse

from scripts.pipelines.utils.utils import (
    ComposeTransforms,
    initialize_model_datasets,
    initialize_dataloaders,
)
from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from mmt.data.baseline_bridge import (
    build_baseline_task_config,
)

from mmt.utils.config_loader import load_experiment_config
from mmt.utils.seed import set_seed
from mmt.utils.logger import setup_logging

from mmt.data.embeddings.signal_spec import (
    build_signal_role_modality_map,
    build_signal_specs,
    build_codecs,
)

from mmt.data.transforms.chunk_windows import ChunkWindowsTransform
from mmt.data.transforms.drop_na import DropNaChunksTransform
from mmt.data.transforms.trim_chunks import TrimChunksTransform
from mmt.data.transforms.embed_chunks import EmbedChunksTransform
from mmt.data.transforms.build_tokens import BuildTokensTransform
from mmt.data.collate import MMTCollate


DEBUG_MODE = False


def parse_args_finetune() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finetuning for a given task/phase config."
    )
    parser.add_argument(
        "--phase_config",
        type=str,
        default="mmt/configs/task_2-1/finetune_default.yaml",
        help="Path to the phase YAML config file "
        "(e.g. mmt/configs/task_2-1/finetune_default.yaml)",
    )
    args, _ = parser.parse_known_args()
    return args


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main() -> None:
    mp.set_start_method("spawn", force=True)

    # MMT config (phase + experiment_base + embeddings + baseline path)
    args = parse_args_finetune()
    cfg_mmt = load_experiment_config(args.phase_config)
    cfg_chunks = cfg_mmt.preprocessing["chunk"]
    cfg_trim = cfg_mmt.preprocessing["trim_chunks"]
    cfg_drop_nan = cfg_mmt.preprocessing["drop_na"]
    cfg_loader = cfg_mmt.training["loader"]

    # Baseline: config_task con override (subset_of_shots, local)
    cfg_task = build_baseline_task_config(cfg_mmt)

    # setting the seed
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # setting the logger
    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename="finetune.log",
        console=True,
    )
    logger.setLevel("DEBUG" if DEBUG_MODE else "INFO")
    logger.info(f"task = {cfg_mmt.task} | phase = {cfg_mmt.phase}")

    # Baseline: datasets + metadata
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs
    signals_role_modality_map = build_signal_role_modality_map(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(cfg_mmt.embeddings, signals_role_modality_map)
    codecs = build_codecs(signal_specs)

    # Model-specific transform (per ora: identity chain)
    model_specific_transform = ComposeTransforms(
        [
            ChunkWindowsTransform(
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks["stride"],
            ),
            DropNaChunksTransform(
                min_valid_inputs_actuators=cfg_drop_nan["min_valid_inputs_actuators"],
                min_valid_chunks=cfg_drop_nan["min_valid_chunks"],
                min_valid_outputs=cfg_drop_nan["min_valid_outputs"],
            ),
            TrimChunksTransform(
                chunk_length_sec=cfg_chunks["chunk_length"],
                delta=cfg_task["task_window_segmenter"]["delta"],
                output_length=cfg_task["task_window_segmenter"]["output_length"],
                max_chunks=cfg_trim["max_chunks"],
            ),
            EmbedChunksTransform(
                signal_specs=signal_specs,
                codecs=codecs,
            ),
            BuildTokensTransform(
                chunk_length_sec=cfg_chunks["chunk_length"],
                delta=cfg_task["task_window_segmenter"]["delta"],
                output_length=cfg_task["task_window_segmenter"]["output_length"],
                signal_specs=signal_specs,
            ),
        ]
    )

    # Datasets per il modello (TaskModelTransformWrapper)
    datasets_mmt = initialize_model_datasets(
        datasets_train_val_test,
        dict_metadata,
        cfg_task,
        model_specific_transform=model_specific_transform,
        verbose=True,
    )

    # Dataloaders
    collate_fn = MMTCollate(cfg_mmt.collate)
    dataloaders_mmt = initialize_dataloaders(
        datasets_mmt,
        collate_fn,
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
        shuffle=cfg_loader["shuffle"],
        drop_last=cfg_loader["drop_last"],
        seed=cfg_mmt.seed,
    )

    # small debug printing
    print("\n[Debug] Shots per split (TaskModelTransformWrapper):")
    for split in ("train", "val", "test"):
        ds = datasets_mmt.get(split)
        if ds is None:
            print(f"  {split}: None")
        else:
            print(f"  {split}: {len(ds)} shots")

    print("\n[Debug] Inspect first few windows of first train shot:")
    ds_train = datasets_mmt["train"]
    gen = ds_train[3]  # generator over windows for shot 0

    for i, win in enumerate(gen):
        if i >= 100:
            break

    # ------------------------------------------------------------------
    # Debug: inspect one batch from the MMT train dataloader
    # ------------------------------------------------------------------
    train_loader = dataloaders_mmt["train"]

    print("\n[Debug] Fetching one batch from MMT train dataloader...")
    batch = next(iter(train_loader))

    print("[Debug] Batch keys and shapes:")
    for key, value in batch.items():
        if hasattr(value, "shape"):
            dtype = getattr(value, "dtype", None)
            print(f"  {key}: shape={tuple(value.shape)}, dtype={dtype}")
        else:
            print(f"  {key}: type={type(value)} -> {value}")

    # If masks are present, print some basic stats to see dropout/masking
    for mask_name in ("padding_mask", "input_mask", "actuator_mask", "output_mask"):
        if mask_name in batch:
            mask = batch[mask_name]
            if hasattr(mask, "numel"):
                total = mask.numel()
                kept = int(mask.sum().item())
                print(f"  {mask_name}: kept {kept}/{total} tokens ({kept / total:.1%})")


if __name__ == "__main__":
    main()
