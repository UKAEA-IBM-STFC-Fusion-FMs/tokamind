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
    extract_signal_stats,
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
from mmt.eval import evaluate_metrics, save_traces_for_subset
from mmt.checkpoints import load_best_weights


DEBUG_MODE = False


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for a given task (convention-based configs)."
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
    args = parse_args_eval()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="eval",
        configs_root=DEFAULT_CONFIGS_ROOT,
    )
    validate_config(cfg_mmt.raw)

    cfg_data = cfg_mmt.data
    cfg_model = cfg_mmt.model
    cfg_loader = cfg_mmt.loader
    cfg_eval = cfg_mmt.eval

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
        filename="eval.log",
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

    # Stats (mean/std) for de-normalizing outputs during metrics/traces
    signal_stats = extract_signal_stats(dict_metadata)

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
    # Model-level dataset (shot-based wrapper)
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

    # Optional debug: iterate a single shot to exercise wrapper + transforms
    if DEBUG_MODE and datasets_shots_wrapped["test"] is not None:
        ds = datasets_shots_wrapped["test"]
        shot = ds[0]
        for _ in shot:
            continue

    # ------------------------------------------------------------------
    # Window-level dataset for EVAL (test only)
    # ------------------------------------------------------------------
    datasets_windows = build_window_datasets(
        datasets_shots_wrapped=datasets_shots_wrapped,
        enable_cache=enable_cache,
        num_workers_cache=num_workers_cache,
        seed=cfg_mmt.seed,
        shuffle_shots=False,  # deterministic for eval
        cache_splits=("test",),  # eval uses test split only
    )

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    cfg_drop = cfg_eval.get("drop", {}) or {}
    drop_inputs = cfg_drop.get("inputs", []) or []
    drop_actuators = cfg_drop.get("actuators", []) or []
    drop_outputs = cfg_drop.get("outputs", []) or []

    collate_fn = make_collate_fn(
        keep_output_native=keep_output_native,
        drop_inputs=drop_inputs,
        drop_actuators=drop_actuators,
        drop_outputs=drop_outputs,
    )

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
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Load best weights from training run
    # ------------------------------------------------------------------
    train_run_dir = cfg_mmt.model_init["model_dir"]
    epoch_best, best_val, _metadata = load_best_weights(
        run_dir=train_run_dir, model=model, map_location=device
    )

    logger = logging.getLogger("mmt.Eval")
    logger.info(
        "Loaded checkpoint from %s (epoch=%s, best_val=%s)",
        train_run_dir,
        epoch_best,
        best_val,
    )
    logger.info(
        "[Eval] Forced drops: inputs=%s actuators=%s outputs=%s",
        drop_inputs,
        drop_actuators,
        drop_outputs,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Evaluation: metrics + optional traces
    # ------------------------------------------------------------------
    run_dir = cfg_mmt.paths["run_dir"]
    cfg_metrics = cfg_eval.get("metrics", {})
    cfg_traces = cfg_eval.get("traces", {})

    # All output specs for this task (excluding forced-dropped outputs)
    drop_outputs_set = set(drop_outputs or [])
    output_specs = [
        spec
        for spec in signal_specs.specs_for_role("output")
        if spec.name not in drop_outputs_set
    ]
    id_to_name = {spec.signal_id: spec.name for spec in output_specs}

    output_codecs = {
        spec.name: codecs[spec.signal_id]
        for spec in output_specs
        if spec.signal_id in codecs
    }

    output_stats = {
        spec.name: signal_stats[spec.name]
        for spec in output_specs
        if spec.name in signal_stats
    }

    if cfg_metrics.get("enable", True):
        logger.info("Computing metrics in: %s/metrics/", run_dir)
        summary_metrics = evaluate_metrics(
            model=model,
            dataloader=eval_loader,
            device=device,
            stats=output_stats,
            codecs=output_codecs,
            id_to_name=id_to_name,
            run_dir=run_dir,
            debug=True,
        )
        logger.info("Summary metrics: %s", summary_metrics)

    if cfg_traces.get("enable", False):
        logger.info(
            "Saving traces (n_max=%s) in: %s/traces/",
            cfg_traces.get("n_max", 10),
            run_dir,
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
