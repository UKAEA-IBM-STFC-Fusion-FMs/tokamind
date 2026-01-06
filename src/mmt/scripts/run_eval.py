from __future__ import annotations

import argparse
import torch
import torch.multiprocessing as mp


import time

from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)

from scripts.pipelines.utils.utils import (
    initialize_model_dataset,
)

from mmt.utils.config import (
    build_task_config,
    load_experiment_config,
    validate_eval_config,
)
from mmt.utils import (
    set_seed,
    setup_logging,
    initialize_mmt_dataloaders,
)

from mmt.data import (
    build_signal_role_modality_map,
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
from mmt.eval import evaluate_metrics, save_traces_for_subset
from mmt.checkpoints import load_best_weights

import logging


DEBUG_MODE = False


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run eval for a given task/phase config."
    )
    parser.add_argument(
        "--phase_config",
        type=str,
        default="mmt/configs/task_2-1/eval_default.yaml",
        help=(
            "Path to the phase YAML config file "
            "(e.g. mmt/configs/task_2-1/eval_default.yaml)"
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
    args = parse_args_eval()
    cfg_mmt = load_experiment_config(args.phase_config)
    validate_eval_config(cfg_mmt.raw)

    # Small sub-configs for readability
    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    cfg_data = cfg_mmt.data
    cfg_model = cfg_mmt.model
    cfg_loader = cfg_mmt.loader
    # cfg_collate = cfg_mmt.collate
    cfg_eval = cfg_mmt.eval

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
        filename="eval.log",
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
    signal_stats = {}
    for role in ("input", "actuator", "output"):
        role_meta = dict_metadata.get(role, {})
        if not isinstance(role_meta, dict):
            continue
        for name, meta in role_meta.items():
            if isinstance(meta, dict) and ("mean" in meta) and ("std" in meta):
                signal_stats[name] = meta

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
        "test": initialize_model_dataset(
            datasets_shots_raw.get("test"),
            dict_metadata,
            cfg_task,
            model_specific_transform=mmt_transform_map,
            verbose=True,
        ),
    }

    if DEBUG_MODE and datasets_shots_wrapped["test"] is not None:
        # Trigger one full-shot iteration to exercise the wrapper + transforms.
        ds = datasets_shots_wrapped["test"]
        shot = ds[0]
        for _ in shot:
            continue

    # ------------------------------------------------------------------
    # Window-level dataset for EVAL (test only)
    # ------------------------------------------------------------------

    datasets_windows = {}

    if enable_cache:
        logger = logging.getLogger("mmt.Cache")
        logger.info("Starting caching test")
        t0 = time.perf_counter()
        datasets_windows["test"] = WindowCachedDataset.from_streaming(
            datasets_shots_wrapped["test"],
            num_workers_cache=num_workers_cache,
        )
        t1 = time.perf_counter()
        logger.info("Finished caching test in %.3f seconds", t1 - t0)

    else:
        # Always deterministic for eval
        datasets_windows["test"] = WindowStreamedDataset(
            datasets_shots_wrapped["test"],
            shuffle_shots=False,
            seed=cfg_mmt.seed,
        )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    cfg_drop = cfg_eval.get("drop", {}) or {}

    drop_inputs = cfg_drop.get("inputs", []) or []
    drop_actuators = cfg_drop.get("actuators", []) or []
    drop_outputs = cfg_drop.get("outputs", []) or []

    collate_cfg = {
        "keep_output_native": keep_output_native,  # whatever you already use
        # force-drop selected signals
        "p_drop_inputs_overrides": {k: 1.0 for k in drop_inputs},
        "p_drop_actuators_overrides": {k: 1.0 for k in drop_actuators},
        "p_drop_outputs_overrides": {k: 1.0 for k in drop_outputs},
    }
    collate_fn = MMTCollate(cfg_collate=collate_cfg)

    eval_loader = initialize_mmt_dataloaders(
        datasets_windows,
        collate_fn,
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
        shuffle=False,
        drop_last=cfg_loader["drop_last"],
        seed=cfg_mmt.seed,
    )["test"]

    # ------------------------------------------------------------------
    # 5. Build the MMT model
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
    )
    model.to(device)

    # ------------------------------------------------------------------
    # 6. Load best weights from training run
    # ------------------------------------------------------------------
    train_run_dir = cfg_mmt.model_init["model_dir"]
    epoch_best, best_val, metadata = load_best_weights(
        run_dir=train_run_dir, model=model, map_location=device
    )
    logger = logging.getLogger("mmt.Eval")
    logger.info(
        f"Loaded checkpoint from {train_run_dir} (epoch={epoch_best}, best_val={best_val})"
    )
    logger.info(
        "[Eval] Forced drops: inputs=%s actuators=%s outputs=%s",
        drop_inputs,
        drop_actuators,
        drop_outputs,
    )
    model.eval()

    # ------------------------------------------------------------------
    # 6) Evaluation: metrics + optional traces
    # ------------------------------------------------------------------
    run_dir = cfg_mmt.paths["run_dir"]
    cfg_metrics = cfg_eval.get("metrics", {})
    cfg_traces = cfg_eval.get("traces", {})

    # All output specs for this task
    drop_outputs_set = set(drop_outputs or [])
    output_specs = [
        spec
        for spec in signal_specs.specs_for_role("output")
        if spec.name not in drop_outputs_set
    ]
    id_to_name = {spec.signal_id: spec.name for spec in output_specs}

    # Map: output_name -> codec (using signal_id internally)
    output_codecs = {
        spec.name: codecs[spec.signal_id]
        for spec in output_specs
        if spec.signal_id in codecs
    }

    # Map: output_name -> stats (mean/std) filtered to task outputs
    output_stats = {
        spec.name: signal_stats[spec.name]
        for spec in output_specs
        if spec.name in signal_stats
    }

    # Metrics evaluation (always full window-level metrics)
    if cfg_metrics.get("enable", True):
        logger.info(f"Computing metrics in: {run_dir}/metrics/")
        summary_metrics = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=device,
            stats=output_stats,
            codecs=output_codecs,
            id_to_name=id_to_name,
            run_dir=run_dir,  # function will create metrics/ inside it
            debug=True,
        )
        logger.info(f"Summary metrics: {summary_metrics}")

    # Trace extraction (subset of shots)
    if cfg_traces.get("enable", False):
        logger.info(
            f"Saving traces (n_max={cfg_traces.get('n_max', 10)}) in: {run_dir}/traces/"
        )
        save_traces_for_subset(
            model=model,
            dataloader=eval_loader,
            device=device,
            stats=output_stats,
            run_dir=run_dir,
            codecs=output_codecs,
            id_to_name=id_to_name,
            traces_cfg=cfg_traces,
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
