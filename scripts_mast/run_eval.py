"""
Evaluation entrypoint for MMT using the convention-based config system.

This script is intentionally thin:
- parses `--task`,
- loads and validates the merged config for phase="eval",
- resolves task metadata + datasets,
- resolves embeddings/codecs from the source training run,
- builds eval window data via shared helpers,
- loads the best checkpoint from model_source.run_dir,
- runs a single-pass evaluation loop:
  - benchmark-aligned metrics (windows_metrics.csv + tasks_metrics.csv)
  - optional MMT-native diagnostics (per-timestamp CSV + traces),
- writes outputs under the eval run directory (cfg_mmt.paths["run_dir"]).

Shared boilerplate lives in:
- `mast_utils.entry_script_helpers`
- `mast_utils.embedding_resolution`
"""

from __future__ import annotations

import argparse
import logging

from pathlib import Path

from mast_utils import (
    load_experiment_config,
    load_task_definition,
    build_signals_by_role_from_task_definition,
    extract_signal_stats,
    init_run_context,
    build_mast_datasets,
    build_window_data,
    build_model_and_optional_warmstart,
    resolve_eval_embeddings,
    evaluate_benchmark_and_diagnostics,
)

from mmt.utils import validate_config, sdpa_math_only_ctx
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
    # Load merged config (common + task + overrides)
    # ------------------------------------------------------------------
    args = parse_args_eval()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="eval",
        model=args.model,
    )
    validate_config(cfg_mmt)

    # ------------------------------------------------------------------
    # Runtime context (device, seed, logging)
    # ------------------------------------------------------------------
    device, _ = init_run_context(cfg_mmt, phase="eval")

    cfg_data = cfg_mmt.data
    cfg_eval = cfg_mmt.eval

    amp_enabled = cfg_eval.get("amp", {}).get("enable", True)

    # Benchmark task config (with overrides such as subset_of_shots/local)
    cfg_task = load_task_definition(args.task)

    # ------------------------------------------------------------------
    # Task metadata + MAST test dataset
    # ------------------------------------------------------------------
    dict_task_metadata, _mast_train, _mast_val, mast_dataset_test = build_mast_datasets(
        cfg_task,
        cfg_data=cfg_data,
        phase="eval",
    )

    # ------------------------------------------------------------------
    # Signal specs + embeddings
    # ------------------------------------------------------------------
    # Stats (mean/std) for de-normalizing outputs during metrics/traces
    signal_stats = extract_signal_stats(dict_task_metadata)

    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task,
        dict_task_metadata,
    )

    train_run_dir = Path(cfg_mmt.model_source["run_dir"])
    signal_specs, codecs = resolve_eval_embeddings(
        cfg_mmt=cfg_mmt,
        signals_by_role=signals_by_role,
        dict_task_metadata=dict_task_metadata,
        train_run_dir=train_run_dir,
    )

    # ------------------------------------------------------------------
    # Window data (test split only)
    # ------------------------------------------------------------------
    window_data = build_window_data(
        cfg_mmt=cfg_mmt,
        mast_datasets={"test": mast_dataset_test},
        dict_task_metadata=dict_task_metadata,
        cfg_task=cfg_task,
        signal_specs=signal_specs,
        codecs=codecs,
        phase="eval",
    )
    eval_loader = window_data["test"]["loader"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model_and_optional_warmstart(
        cfg_mmt=cfg_mmt,
        signal_specs=signal_specs,
        device=device,
        skip_warmstart=True,
    )

    # ------------------------------------------------------------------
    # Load best weights from training run
    # ------------------------------------------------------------------
    epoch_best, best_val, _metadata = load_best_weights(
        run_dir=str(train_run_dir), model=model, map_location=str(device)
    )

    cfg_drop = cfg_eval.get("drop", {}) or {}
    drop_inputs = cfg_drop.get("inputs", []) or []
    drop_actuators = cfg_drop.get("actuators", []) or []
    drop_outputs = cfg_drop.get("outputs", []) or []

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
