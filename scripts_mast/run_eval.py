"""
Evaluation entrypoint for MMT, with optional verified CGPO test-set provenance.

This script is intentionally thin:
- parses `--task`,
- loads and validates the merged config for phase="eval",
- resolves task metadata + datasets,
- resolves embeddings/codecs from the source training run,
- builds eval window data via shared helpers,
- loads the best checkpoint from model_source.run_dir,
- runs a single-pass evaluation loop:
  - benchmark-aligned metrics (windows_metrics.csv + shots_metrics.csv + task_metrics.csv)
  - optional MMT-native diagnostics (per-timestamp CSV + traces),
- writes outputs under the eval run directory (cfg_mmt.paths["run_dir"]).

Shared boilerplate lives in:
- `mast_utils.entry_helpers`
- `mast_utils.embedding_resolution`
"""

from __future__ import annotations

import argparse
import logging
import json

import yaml
from pathlib import Path

from mmt.utils import validate_config, sdpa_math_only_ctx
from mmt.checkpoints import load_best_weights
from mmt.data import build_decoders

from mast_utils import (
    load_experiment_config,
    validate_mast_config,
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


# ----------------------------------------------------------------------------------------------------------------------
def parse_args_eval() -> argparse.Namespace:
    """
    Parse arguments for evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed evaluation arguments in argparse.Namespace format.

    """

    parser = argparse.ArgumentParser(
        description="Run evaluation for a given task (convention-based configs).",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Task identifier. Task-specific eval config lives in scripts_mast/configs/<model_profile>/tasks/eval_tasks.yaml, "
        "where <model_profile> is inferred from --model_source (not selected on the CLI for eval).",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        # required=True,  # NOSONAR - Ignore weak warning
        default="_test",
        help="Model to evaluate (run_id or path). Example: ft_task_1-1_from_base_v1",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional eval tag. If provided, outputs are saved under eval_<tag>/ instead of eval/.",
    )
    parser.add_argument(
        "--cgpo_protocol", type=Path, help="Pair collection whose official test protocol must be verified."
    )
    args = parser.parse_args()

    return args


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    """
    Execute main evaluation pipeline.

    Returns
    -------
    None

    """

    # ..................................................................................................................
    # Load merged config (common + task + overrides)
    # ..................................................................................................................

    args = parse_args_eval()
    from mast_utils.gpo.protocol import (
        apply_source_contract,
        benchmark_manifest,
        digest,
        file_digest,
        source_fingerprint,
        validate_collection,
        verified_test_batches,
    )
    from mmt.utils.config.experiment.inheritance import load_source_run_config_yaml

    protocol_dir = args.cgpo_protocol
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="eval",
        model_source=args.model_source,
        tag=args.tag,
        integration_hook=(
            lambda merged, phase: apply_source_contract(merged, Path(__file__).parent / "configs", collection=True)
        )
        if protocol_dir
        else None,
        save_config=not bool(protocol_dir),
    )
    validate_config(cfg=cfg_mmt)
    source_split = cfg_mmt.data.pop("split", None) if protocol_dir else None
    validate_mast_config(cfg=cfg_mmt)
    if protocol_dir:
        cfg_mmt.data["split"] = source_split
        eval_dir = Path(cfg_mmt.paths["run_dir"])
        if eval_dir.exists() and any(eval_dir.iterdir()):
            raise FileExistsError(f"Verified evaluation requires a fresh tag: {eval_dir}")

    # ..................................................................................................................
    # Runtime context (device, seed, logging)
    # ..................................................................................................................

    device, _ = init_run_context(cfg_mmt=cfg_mmt, phase="eval")

    cfg_data = cfg_mmt.data
    cfg_eval = cfg_mmt.eval

    amp_enabled = cfg_eval.get("amp", {}).get("enable", True)

    # Benchmark task config (with overrides such as subset_size/local)
    cfg_task = load_task_definition(task_key=args.task)
    evaluation_manifest = None
    test_counts = {}
    if protocol_dir:
        cc = json.loads((protocol_dir / "collection_config.json").read_text())
        validate_collection(cc, cc["contract"], protocol_dir)
        contract = cc["contract"]
        model_dir = Path(cfg_mmt.model_source["run_dir"])
        source_cfg = load_source_run_config_yaml(model_run_dir=model_dir)
        checkpoint = source_fingerprint(model_dir)
        provenance = source_cfg.get("gpo_provenance")
        if provenance:
            if provenance.get("collection_digest") != digest(cc):
                raise ValueError("CGPO model was trained against a different collection.")
        elif checkpoint != contract["source"]:
            raise ValueError("Base checkpoint differs from the collected reference.")
        if (
            contract["task"] != args.task
            or contract["task_definition"] != digest(cfg_task)
            or benchmark_manifest(cfg_data) != contract["split_manifest"]
        ):
            raise ValueError("Evaluation task or official test split differs from collection.")
        if not cfg_eval.get("compute_metrics", {}).get("per_task"):
            raise ValueError("Verified CGPO evaluation requires per_task metrics.")
        evaluation_manifest = {
            "format": "cgpo-evaluation-v1",
            "task": args.task,
            "collection_digest": digest(cc),
            "split_manifest": contract["split_manifest"],
            "checkpoint": checkpoint,
            "source_config_digest": file_digest(model_dir / f"{model_dir.name}.yaml"),
            "test_window_counts": test_counts,
            "complete": False,
            "evaluation_contract": digest(
                {
                    "model": cfg_mmt.raw["model"],
                    "preprocess": cfg_mmt.raw["preprocess"],
                    "cache_dtype": cfg_data["cache"].get("dtype") if cfg_data["cache"]["enable"] else None,
                    "eval": cfg_eval,
                    "collate": cfg_mmt.raw.get("collate"),
                }
            ),
        }

    # ..................................................................................................................
    # Task metadata + MAST test dataset
    # ..................................................................................................................

    dict_task_metadata, _mast_train, _mast_val, mast_dataset_test = build_mast_datasets(
        cfg_task=cfg_task,
        cfg_data=cfg_data,
        phase="eval",
        cfg_model_source=cfg_mmt.model_source,
    )

    # ..................................................................................................................
    # Signal specs + embeddings
    # ..................................................................................................................

    # Stats (mean/std) for de-normalizing outputs during metrics/traces
    signal_stats = extract_signal_stats(dict_metadata=dict_task_metadata)

    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task=cfg_task,
        dict_metadata=dict_task_metadata,
    )

    train_run_dir = Path(str(cfg_mmt.model_source["run_dir"]))
    signal_specs, codecs = resolve_eval_embeddings(
        cfg_mmt=cfg_mmt,
        signals_by_role=signals_by_role,
        dict_task_metadata=dict_task_metadata,
        train_run_dir=train_run_dir,
    )

    if evaluation_manifest:
        evaluation_manifest["embedding_config_digest"] = digest(cfg_mmt.embeddings)

    # ..................................................................................................................
    # Window data (test split only)
    # ..................................................................................................................

    if cfg_data.get("cache", {}).get("enable", False):
        logging.getLogger("mmt").info("")

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
    if evaluation_manifest:
        eval_loader = verified_test_batches(eval_loader, contract["split_manifest"]["shots"]["test"], test_counts)

    # ..................................................................................................................
    # Model
    # ..................................................................................................................

    logging.getLogger("mmt").info("")
    model = build_model_and_optional_warmstart(
        cfg_mmt=cfg_mmt, signal_specs=signal_specs, device=device, skip_warmstart=True
    )
    logging.getLogger("mmt").info("")

    # ..................................................................................................................
    # Load best weights from training run
    # ..................................................................................................................

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

    # ..................................................................................................................
    # Evaluation: metrics + optional traces
    # ..................................................................................................................

    run_dir = Path(str(cfg_mmt.paths["run_dir"]))
    cfg_compute_metrics = cfg_eval.get("compute_metrics", {})
    cfg_traces = cfg_eval.get("traces", {})

    # All output specs for this task (excluding forced-dropped outputs)
    drop_outputs_set = set(drop_outputs or [])
    output_specs = [spec for spec in signal_specs.specs_for_role("output") if spec.name not in drop_outputs_set]
    id_to_name = {spec.signal_id: spec.name for spec in output_specs}

    _id_decoders = build_decoders(registry=signal_specs, codecs=codecs, role="output")
    output_decoders = {
        spec.name: _id_decoders[spec.signal_id] for spec in output_specs if spec.signal_id in _id_decoders
    }

    output_stats = {spec.name: signal_stats[spec.name] for spec in output_specs if spec.name in signal_stats}

    do_eval = bool(
        cfg_compute_metrics.get("per_task", False)
        or cfg_compute_metrics.get("per_shot", False)
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
                decoders=output_decoders,
                id_to_name=id_to_name,
                run_dir=run_dir,
                task_name=args.task,
                amp_enabled=amp_enabled,
                compute_metrics_cfg=cfg_compute_metrics,
                traces_cfg=cfg_traces,
            )

        if "task_metrics" in result:
            logger.info("Benchmark task metrics: %s", result["task_metrics"])
        logger.info("Task metrics dir: %s", result.get("metrics_task_dir"))
    else:
        logger.info(
            "[eval] compute_metrics: per_task, per_shot, per_window, per_timestamp, "
            "and traces are disabled; skipping evaluation."
        )

    if evaluation_manifest:
        expected = set(contract["split_manifest"]["shots"]["test"])
        if set(map(int, test_counts)) != expected:
            raise ValueError(
                f"Incomplete test coverage; missing shots: {sorted(expected - set(map(int, test_counts)))}"
            )
        evaluation_manifest["metrics_digest"] = file_digest(run_dir / "metrics" / args.task / "task_metrics.csv")
        snapshot = run_dir / "evaluation_config.yaml"
        with snapshot.open("x") as stream:
            yaml.safe_dump(cfg_mmt.raw, stream, sort_keys=False)
        evaluation_manifest["config_digest"] = file_digest(snapshot)
        evaluation_manifest["complete"] = True
        with (run_dir / "cgpo_evaluation.json").open("x") as stream:
            json.dump(evaluation_manifest, stream, indent=2)
    logger.info("Done.")


# ======================================================================================================================
if __name__ == "__main__":
    main()
