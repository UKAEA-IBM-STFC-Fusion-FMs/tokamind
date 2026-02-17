"""
Evaluation entrypoint for MMT using the convention-based config system.

This script is intentionally thin:
- parses `--task`,
- loads and validates the merged config for phase="eval",
- resolves the benchmark task_config and builds datasets/metadata,
- builds signal specs/codecs and the standard shot→windows transform pipeline,
- loads the best checkpoint from `model_init.model_dir`,
- runs a single-pass evaluation loop:
  - benchmark-aligned metrics (windows_metrics.csv + tasks_metrics.csv)
  - optional MMT-native diagnostics (per-timestamp CSV + traces),
- writes outputs under the eval run directory (cfg_mmt.paths["run_dir"]).

Shared boilerplate (device/MP setup, default transforms, window datasets,
collate construction, etc.) lives in `scripts_mast.mast_utils.pipeline_helpers`.
"""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from mast_utils.benchmark_imports import (
    initialize_MAST_dataset,
    initialize_model_dataset_iterable,
    get_task_metadata,
    get_train_test_val_shots,
)

from mast_utils import (
    load_experiment_config,
    load_task_definition,
    build_signals_by_role_from_task_definition,
    setup_device_and_mp,
    extract_signal_stats,
    build_default_transform,
    make_collate_fn,
)

from mast_utils.eval_benchmark import evaluate_benchmark_and_diagnostics

from mmt.utils.config.validator import validate_config

from mmt.utils import (
    set_seed,
    setup_logging,
    sdpa_math_only_ctx,
)

from mmt.data import (
    build_signal_specs,
    build_codecs,
    initialize_mmt_dataloader,
    WindowCachedDataset,
)
from mmt.models import MultiModalTransformer
from mmt.checkpoints import load_best_weights


def parse_args_eval() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation for a given task (convention-based configs)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Task folder name under scripts_mast/configs/tasks_overrides/<task>/",
    )
    parser.add_argument(
        "--model",
        type=str,
        # required=True,
        default="_test",
        help="Model to evaluate (run_id or path). Example: ft_task_1-1_from_base_v1",
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
        model=args.model,
    )
    validate_config(cfg_mmt)

    cfg_data = cfg_mmt.data
    cfg_cache = cfg_data["cache"]
    cfg_model = cfg_mmt.model
    cfg_loader = cfg_mmt.loader
    cfg_eval = cfg_mmt.eval

    keep_output_native = cfg_data.get("keep_output_native", False)
    local_flag = cfg_data.get("local", True)
    debug_mode = cfg_mmt.runtime["debug_logging"]
    amp_enabled = cfg_eval.get("amp", {}).get("enable", True)

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
        filename=f"{cfg_mmt.eval_id}.log",
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

    _, test_shots_, _ = get_train_test_val_shots(
        max_index=cfg_data["subset_of_shots"],
    )

    mast_dataset_test = initialize_MAST_dataset(
        cfg_task,
        test_shots_,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Signal specs
    # ------------------------------------------------------------------

    # Stats (mean/std) for de-normalizing outputs during metrics/traces
    signal_stats = extract_signal_stats(dict_task_metadata)

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

    # For rank mode DCT3D: pass embeddings_overrides dir to load .npy coefficient indices
    embeddings_dir = Path(cfg_mmt.paths["task_config_dir"]) / "embeddings_overrides"
    codecs = build_codecs(signal_specs, config_dir=embeddings_dir)

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
    # Window-level iterable (benchmark wrapper yields windows directly)
    # ------------------------------------------------------------------
    window_iterable_test = initialize_model_dataset_iterable(
        mast_dataset_test,
        dict_task_metadata,
        cfg_task,
        model_specific_transform=mmt_transform_map,
        test_mode=True,
        shuffle_windows=False,
        shuffle_buffer_size=512,
        verbose=True,
    )

    # Optional debug: iterate a few windows to exercise wrapper + transforms
    if debug_mode and window_iterable_test is not None:
        logger.info("Debug mode: testing window iteration...")
        for i, window in enumerate(window_iterable_test):
            if i >= 2:  # Just test first 2 windows
                break
        logger.info("Debug iteration successful")

    # ------------------------------------------------------------------
    # Window-level dataset for EVAL (test only)
    # ------------------------------------------------------------------
    enable_cache = cfg_cache.get("enable", False)

    if enable_cache:
        # Cache mode: materialize windows to RAM
        logger.info("Caching enabled - materializing windows to RAM")
        window_dataset_test = WindowCachedDataset.from_streaming(
            window_iterable_test,
            max_windows=None,  # No limit for eval
            num_workers_cache=cfg_cache.get("num_workers", 0),
            dtype=cfg_cache.get("dtype", None),
        )
    else:
        # Streaming mode: use iterable directly
        logger.info("Streaming mode - using window iterable directly")
        window_dataset_test = window_iterable_test

    # ------------------------------------------------------------------
    # DataLoader
    # ------------------------------------------------------------------
    cfg_drop = cfg_eval.get("drop", {}) or {}
    drop_inputs = cfg_drop.get("inputs", []) or []
    drop_actuators = cfg_drop.get("actuators", []) or []
    drop_outputs = cfg_drop.get("outputs", []) or []

    collate_fn = make_collate_fn(
        signal_specs=signal_specs,
        keep_output_native=keep_output_native,
        drop_inputs=drop_inputs,
        drop_actuators=drop_actuators,
        drop_outputs=drop_outputs,
    )

    eval_loader = initialize_mmt_dataloader(
        window_dataset_test,
        collate_fn,
        batch_size=cfg_loader["batch_size"],
        num_workers=cfg_loader["num_workers"],
        shuffle=False,
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
    )
    model.to(device)

    # ------------------------------------------------------------------
    # Load best weights from training run
    # ------------------------------------------------------------------
    train_run_dir = cfg_mmt.model_source["run_dir"]
    epoch_best, best_val, _metadata = load_best_weights(
        run_dir=train_run_dir, model=model, map_location=str(device)
    )

    logger = logging.getLogger("mmt.Eval")
    logger.info(
        "Loaded checkpoint from %s (epoch=%s, best_val=%s)",
        train_run_dir,
        epoch_best,
        best_val,
    )
    logger.info(
        "Forced drops: inputs=%s actuators=%s outputs=%s",
        drop_inputs,
        drop_actuators,
        drop_outputs,
    )
    model.eval()

    # ------------------------------------------------------------------
    # Evaluation: metrics + optional traces
    # ------------------------------------------------------------------
    run_dir = cfg_mmt.paths["run_dir"]
    cfg_compute_metrics = cfg_eval.get("compute_metrics", {})
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

    do_eval = bool(
        cfg_compute_metrics.get("per_task", False)
        or cfg_compute_metrics.get("per_window", False)
        or cfg_compute_metrics.get("per_timestamp", False)
        or cfg_traces.get("enable", False)
    )

    if do_eval:
        logger.info("Running evaluation in: %s", run_dir)
        with sdpa_math_only_ctx():
            result = evaluate_benchmark_and_diagnostics(
                model=model,
                dataloader=eval_loader,
                device=device,
                stats=output_stats,
                codecs=output_codecs,
                id_to_name=id_to_name,
                run_dir=run_dir,
                task_name=args.task,
                amp_enabled=amp_enabled,
                compute_metrics_cfg=cfg_compute_metrics,
                traces_cfg=cfg_traces,
            )

        if "task_metrics" in result:
            logger.info("Benchmark task metrics: %s", result["task_metrics"])
        logger.info("Benchmark outputs dir: %s", result.get("benchmark_dir"))
    else:
        logger.info(
            "[eval] compute_metrics: per_task, per_window, per_timestamp and traces are disabled; skipping evaluation."
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
