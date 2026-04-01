"""
Entry script orchestration helpers for run_pretrain.py, run_finetune.py, run_eval.py.

This module consolidates shared boilerplate that was previously duplicated across the three main entry scripts.
By extracting these helpers, we:
  - Reduce code duplication by ~60% across entry scripts
  - Improve maintainability (single source of truth)
  - Make entry scripts more readable (focus on orchestration, not details)

Key helpers:
------------
init_run_context()                   — Device, seed, logging setup
build_mast_datasets()                — Task metadata + MAST dataset initialization
build_window_data()                  — Window iterables/datasets + dataloaders
build_model_and_optional_warmstart() — Model construction + warmstart

Design notes:
-------------
- These helpers are intentionally high-level orchestration functions
- Low-level primitives (transforms, collate, etc.) remain in pipeline_ops.py
- Embedding resolution logic lives in embedding_resolution.py / tune_dct3d.py
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Literal, Union

import torch

from mmt.data.signal_spec import SignalSpecRegistry
from mmt.utils.config.schema import ExperimentConfig
from mmt.utils import set_seed, setup_logging
from mmt.data import initialize_mmt_dataloader, WindowCachedDataset
from mmt.models import MultiModalTransformer
from mmt.checkpoints import load_parts_from_run_dir


from .benchmark_imports import (
    initialize_MAST_dataset,
    initialize_TokaMark_dataset,
    get_task_metadata,
    get_train_test_val_shots,
)

from .pipeline_ops import (
    setup_device_and_mp,
    build_default_transform,
    make_collate_fn,
)


# ======================================================================================================================
# Runtime Context Initialization
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def init_run_context(
    cfg_mmt: ExperimentConfig,
    phase: Literal["pretrain", "finetune", "eval"],
) -> tuple[torch.device, logging.Logger]:
    """
    Initialize runtime context: device, seed, logging.

    This helper consolidates the device/seed/logging setup that is duplicated across all three entry scripts (pretrain,
    finetune, eval).

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval", for logging purposes.

    Returns
    -------
    tuple[torch.device, logging.Logger]
        (device, logger) ready for use.

    """

    # Device + multiprocessing setup
    device = setup_device_and_mp()

    # Seed for reproducibility
    set_seed(seed=cfg_mmt.seed, deterministic=True, warn_only=True)

    # Logging setup
    debug_mode = cfg_mmt.runtime["debug_logging"]
    log_filename = f"{cfg_mmt.run_id}.log" if (phase != "eval") else f"{cfg_mmt.eval_id}.log"

    logger = setup_logging(
        run_dir=cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename=log_filename,
        console=True,
    )
    logger.setLevel("DEBUG" if debug_mode else "INFO")

    # Log context
    logging.getLogger("mmt.Task").info("task=%s | phase=%s | device=%s", cfg_mmt.task, cfg_mmt.phase, device)

    return device, logger


# ======================================================================================================================
# MAST Dataset Initialization
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_mast_datasets(
    cfg_task: Mapping[str, Any],
    cfg_data: Mapping[str, Any],
    phase: Literal["pretrain", "finetune", "eval"],
) -> tuple[dict[str, Any], Union[Any, None], Union[Any, None], Union[Any, None]]:
    """
    Build task metadata and MAST datasets for train/val/test.

    This helper consolidates the task metadata + MAST dataset initialization that is duplicated across all three entry
    scripts.

    Parameters
    ----------
    cfg_task : Mapping[str, Any]
        Benchmark task definition (dictionary from load_task_definition()).
    cfg_data : Mapping[str, Any]
        Data config section from cfg_mmt.data.
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".

    Returns
    -------
    tuple[dict[str, Any], Union[Any, None], Union[Any, None], Union[Any, None]]
        (dict_task_metadata, mast_train, mast_val, mast_test)
        For eval: (metadata, None, None, mast_test)
        For pretrain/finetune: (metadata, mast_train, mast_val, None)

    """

    # Task metadata
    dict_task_metadata = dict(get_task_metadata(config_task=cfg_task, verbose=False))

    # Shot splits
    train_shots, test_shots, val_shots = get_train_test_val_shots(max_index=cfg_data["subset_of_shots"])

    local_flag = cfg_data.get("local", True)
    local_path = cfg_data.get("local_path", None)

    # Build store_manager_settings only when running locally and a path is provided.
    store_settings = {"base_local_zarr_path": local_path} if (local_flag and local_path) else None

    # Initialize datasets based on phase
    if phase == "eval":
        mast_test = initialize_MAST_dataset(
            config_task=cfg_task,
            shots_list=test_shots,
            local_flag=local_flag,
            use_std_scaling=True,
            return_incomplete_shots=True,
            store_manager_settings=store_settings,
            verbose=False,
        )
        return dict_task_metadata, None, None, mast_test

    else:  # -> I.e., phase is "pretrain" or "finetune"
        mast_train = initialize_MAST_dataset(
            config_task=cfg_task,
            shots_list=train_shots,
            local_flag=local_flag,
            use_std_scaling=True,
            return_incomplete_shots=True,
            store_manager_settings=store_settings,
            verbose=False,
        )

        mast_val = initialize_MAST_dataset(
            config_task=cfg_task,
            shots_list=val_shots,
            local_flag=local_flag,
            use_std_scaling=True,
            return_incomplete_shots=True,
            store_manager_settings=store_settings,
            verbose=False,
        )

        return dict_task_metadata, mast_train, mast_val, None


# ======================================================================================================================
# Window Data Construction
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_window_data(  # NOSONAR - Ignore cognitive complexity
    cfg_mmt: ExperimentConfig,
    mast_datasets: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    cfg_task: Mapping[str, Any],
    signal_specs: SignalSpecRegistry,
    codecs: Mapping[int, Any],
    phase: Literal["pretrain", "finetune", "eval"],
) -> dict[str, dict[str, Any]]:
    """
    Build window iterables, datasets, and dataloaders.

    This helper consolidates the window data construction pipeline that is duplicated across all three entry scripts.
    It handles:
    - Transform pipeline construction
    - Window iterable creation
    - Optional caching to RAM
    - Collate function creation
    - DataLoader construction

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    mast_datasets : Mapping[str, Any]
        Dict with keys like 'train', 'val', 'test' mapping to MAST datasets.
        Example: {"train": mast_train, "val": mast_val}
    dict_task_metadata : Mapping[str, Any]
        Task metadata from get_task_metadata().
    cfg_task : Mapping[str, Any]
        Benchmark task definition.
    signal_specs : SignalSpecRegistry
        Signal spec registry.
    codecs : Mapping[int, Any]
        Codec mapping (signal_id -> codec).
    phase : Literal["pretrain", "finetune", "eval"]
        Phase name, either "pretrain", "finetune", or "eval".

    Returns
    -------
    dict[str, dict[str, Any]]
        Nested dict structure:
        {
            "train": {"iterable": ..., "dataset": ..., "loader": ...},
            "val": {"iterable": ..., "dataset": ..., "loader": ...},
            # or for eval:
            "test": {"iterable": ..., "dataset": ..., "loader": ...},
        }

    """

    cfg_data = cfg_mmt.data
    cfg_cache = cfg_data["cache"]
    cfg_loader = cfg_mmt.loader

    # Eval config has no collate block; default to empty mapping.
    cfg_collate = cfg_mmt.raw.get("collate", {})
    keep_output_native = cfg_data.get("keep_output_native", False)
    debug_mode = cfg_mmt.runtime["debug_logging"]

    # Build transform pipeline (shared across all splits)
    mmt_transform = build_default_transform(
        cfg_mmt=cfg_mmt,
        dict_metadata=dict_task_metadata,
        signal_specs=signal_specs,
        codecs=codecs,
        keep_output_native=keep_output_native,
    )

    # Caching config
    enable_cache = cfg_cache.get("enable", False)

    result = {}

    for split_name, mast_dataset in mast_datasets.items():
        if mast_dataset is None:
            continue

        # Determine shuffle behavior
        is_train = split_name == "train"
        shuffle_at_iterable_level = is_train and cfg_loader["shuffle_train"] and not enable_cache

        # Create window iterable
        window_iterable = initialize_TokaMark_dataset(
            dataset=mast_dataset,
            task_metadata=dict_task_metadata,
            config_metadata=cfg_task,
            custom_transform=mmt_transform,
            test_mode=(phase == "eval"),
            shuffle_windows=shuffle_at_iterable_level,
            shuffle_buffer_size=512,
            verbose=(split_name == "train"),
        )

        if window_iterable is None:
            continue

        # Optional debug: test iteration
        if debug_mode and (split_name == "train"):
            logger = logging.getLogger("mmt.WindowData")
            logger.info("Debug mode: testing window iteration...")
            for i, _ in enumerate(window_iterable):
                if i >= 2:
                    break
            logger.info("Debug iteration successful")

        # Cache or stream
        if enable_cache:
            max_windows_cfg = cfg_cache.get("max_windows") or {}
            window_dataset = WindowCachedDataset.from_streaming(
                streaming_dataset=window_iterable,
                max_windows=max_windows_cfg.get(split_name, None),
                num_workers_cache=cfg_cache.get("num_workers", 0),
                shuffle_shots=(is_train and cfg_loader["shuffle_train"]),
                seed=cfg_mmt.seed,
                dtype=cfg_cache.get("dtype", None),
                split_name=split_name,
            )
        else:
            window_dataset = window_iterable

        # Collate function
        if split_name == "train":
            collate_fn = make_collate_fn(
                signal_specs=signal_specs,
                base_cfg=cfg_collate,
                keep_output_native=keep_output_native,
            )
        else:
            # For val/test: no dropout, keep all signals.
            # For eval: may have forced drops from cfg_mmt.eval.drop.
            drop_cfg = {}
            if phase == "eval":
                cfg_drop = cfg_mmt.eval.get("drop", {}) or {}
                drop_cfg = {
                    "drop_inputs": cfg_drop.get("inputs", []),
                    "drop_actuators": cfg_drop.get("actuators", []),
                    "drop_outputs": cfg_drop.get("outputs", []),
                }

            collate_fn = make_collate_fn(
                signal_specs=signal_specs,
                keep_output_native=keep_output_native,
                **drop_cfg,
            )

        # DataLoader
        dataloader = initialize_mmt_dataloader(
            dataset=window_dataset,
            collate_fn=collate_fn,
            batch_size=cfg_loader["batch_size"],
            num_workers=cfg_loader["num_workers"],
            shuffle=(is_train and cfg_loader["shuffle_train"]),
            drop_last=cfg_loader["drop_last"],
            seed=cfg_mmt.seed,
        )

        result[split_name] = {
            "iterable": window_iterable,
            "dataset": window_dataset,
            "loader": dataloader,
        }

    return result


# ======================================================================================================================
# Model Construction
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def build_model_and_optional_warmstart(
    cfg_mmt: ExperimentConfig,
    signal_specs: SignalSpecRegistry,
    device: torch.device,
    skip_warmstart: bool = False,
) -> MultiModalTransformer:
    """
    Build MMT model and optionally warmstart from source.

    This helper consolidates the model construction + warmstart logic that is duplicated across all three entry scripts.

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    signal_specs : SignalSpecRegistry
        Signal spec registry.
    device : torch.device
        Device to move model to.
    skip_warmstart : bool
        If True, skip loading parts from model_source.run_dir even if configured. Useful for eval where full
        best-checkpoint loading happens separately.
        Optional. Default: False.

    Returns
    -------
    MultiModalTransformer
        Model ready for training/eval.

    """

    cfg_model = cfg_mmt.model
    cfg_backbone = cfg_model["backbone"]
    cfg_modality_heads = cfg_model["modality_heads"]
    cfg_output_adapters = cfg_model["output_adapters"]
    max_positions = cfg_mmt.preprocess["trim_chunks"]["max_chunks"]

    # Construct model
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

    # Optional warmstart
    model_source_cfg = cfg_mmt.raw.get("model_source")
    if (not skip_warmstart) and (model_source_cfg is not None):
        run_init = model_source_cfg.get("run_dir")
        load_parts = model_source_cfg.get("load_parts")

        if run_init is not None:
            load_parts_from_run_dir(
                model=model,
                run_dir=run_init,
                load_parts=load_parts,
                map_location=str(device),
            )

    return model
