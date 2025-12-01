from __future__ import annotations

import torch.multiprocessing as mp
import argparse

from scripts.pipelines.utils.utils import (
    ComposeTransforms,
    initialize_model_datasets,
    # initialize_dataloaders,
)
from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from mmt.data.baseline_bridge import (
    build_baseline_task_config,
)

from mmt.utils.config_loader import load_experiment_config
from mmt.utils.seed import set_seed
from mmt.utils.logging import setup_logging


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


DEBUG = True

# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------


def main() -> None:
    args = parse_args_finetune()
    mp.set_start_method("spawn", force=True)

    # MMT config (phase + experiment_base + embeddings + baseline path)
    cfg_mmt = load_experiment_config(args.phase_config)

    # Baseline: config_task con override (subset_of_shots, local)
    cfg_task = build_baseline_task_config(cfg_mmt)

    # setting the seed
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # setting the logger
    logger = setup_logging(cfg_mmt.paths["output_root"], logger_name="mmt.finetune")
    logger.info(f"Task: {cfg_mmt.task}, phase: {cfg_mmt.phase}")
    logger.info(f"Baseline config: {cfg_mmt.baseline_config_path}")

    # Baseline: datasets + metadata
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Model-specific transform (per ora: identity chain)
    model_specific_transform = ComposeTransforms(
        [
            # placeholder
        ]
    )

    # Datasets per il modello (TaskModelTransformWrapper)
    datasets_mmt = initialize_model_datasets(
        datasets_train_val_test,
        dict_metadata,
        cfg_task,
        model_specific_transform=model_specific_transform,
        verbose=DEBUG,
    )

    # Dataloaders (quando avremo collate + parametri nel config)
    # placeholder:
    # collate_fn = collate_finetune_mmt(cfg, dict_metadata)
    # dataloaders_mmt = initialize_dataloaders(
    #     datasets_mmt,
    #     collate_fn,
    #     batch_size=cfg.training["batch_size"],
    #     num_workers=cfg.training["num_workers"],
    #     shuffle=True,
    #     drop_last=False,
    #     seed=cfg.seed,
    # )

    # Piccolo debug come prima
    print("\n[Debug] Shots per split (TaskModelTransformWrapper):")
    for split in ("train", "val", "test"):
        ds = datasets_mmt.get(split)
        if ds is None:
            print(f"  {split}: None")
        else:
            print(f"  {split}: {len(ds)} shots")


if __name__ == "__main__":
    main()
