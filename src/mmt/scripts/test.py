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

from mmt.data.embeddings.signal_spec import (
    build_signal_role_modality_map,
    build_signal_specs,
)

from mmt.data.transforms.chunking import ChunkingTransform
from mmt.data.transforms.drop_na import DropNaChunksTransform


DEBUG_MODE = True


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
    args = parse_args_finetune()
    mp.set_start_method("spawn", force=True)

    # MMT config (phase + experiment_base + embeddings + baseline path)
    cfg_mmt = load_experiment_config(args.phase_config)
    cfg_chunks = cfg_mmt.preprocessing["chunking"]
    cfg_drop_nan = cfg_mmt.preprocessing["drop_nan"]

    # Baseline: config_task con override (subset_of_shots, local)
    cfg_task = build_baseline_task_config(cfg_mmt)

    # setting the seed
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # setting the logger
    logger = setup_logging(
        cfg_mmt.paths["output_root"],
        logger_name="mmt",
        filename="finetune.log",
        console=True,
    )
    logger.setLevel("DEBUG" if DEBUG_MODE else "INFO")

    # Baseline: datasets + metadata
    datasets_train_val_test, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs
    signals_by_role = build_signal_role_modality_map(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(cfg_mmt.embeddings, signals_by_role)

    # Model-specific transform (per ora: identity chain)
    model_specific_transform = ComposeTransforms(
        [
            ChunkingTransform(
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks["stride"],
            ),
            DropNaChunksTransform(
                min_valid_inputs_actuators=cfg_drop_nan["min_valid_inputs_actuators"],
                min_valid_chunks=cfg_drop_nan["min_valid_chunks"],
                min_valid_outputs=cfg_drop_nan["min_valid_outputs"],
            ),
        ]
    )

    # Datasets per il modello (TaskModelTransformWrapper)
    datasets_mmt = initialize_model_datasets(
        datasets_train_val_test,
        dict_metadata,
        cfg_task,
        model_specific_transform=model_specific_transform,
        verbose=False,
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
        chunks = win.get("chunks", {})
        # logger.info(
        #     f"[Test] Window {i}: after DropNa → signals: {list(win['chunks']['input'][0]['signals'].keys())}"
        # )
        # n_input = len(chunks.get("input", []))
        # n_act = len(chunks.get("actuator", []))
        # t_cut = win.get("t_cut", None)
        #
        # print(
        #     f"  window {i:02d}: t_cut={t_cut:.6f} → "
        #     f"{n_input} input chunks, {n_act} actuator chunks"
        # )
        if i >= 10:
            break


if __name__ == "__main__":
    main()
