"""
Finetuning entrypoint for MMT using the convention-based config system.

This script:
- parses `--task`,
- loads and validates the merged config for phase="finetune",
- resolves task metadata and datasets,
- resolves embeddings/codecs using `embeddings.mode` (`source` or `config`),
- builds window data and model via shared helpers,
- runs the finetuning loop and writes outputs under cfg_mmt.paths["run_dir"].

Shared boilerplate lives in:
- `mast_utils.entry_helpers`
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
    init_run_context,
    build_mast_datasets,
    build_window_data,
    build_model_and_optional_warmstart,
    resolve_finetune_embeddings,
)

from mmt.utils import validate_config, sdpa_math_only_ctx
from mmt.train import train_finetune


def parse_args_finetune() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finetuning for a given task (convention-based configs)."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Task folder name under scripts_mast/configs/tasks_overrides/<task>/",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="warmstart",
        choices=("warmstart", "scratch"),
        help="Finetune initialization mode. "
        "'warmstart' requires --model, 'scratch' ignores --model.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="_test",
        help="Source model for warm-start (run_id or path). Example: tokamind_base_v1",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional experiment tag for versioning (e.g., 'lr1e-4', 'exp1'). "
        "Run ID format is ft-{task}-ws-{model}[-{tag}] or ft-{task}-scratch[-{tag}].",
    )
    parser.add_argument(
        "--emb_profile",
        type=str,
        default="dct3d",
        help="embeddings_profile chosen for the task: "
        "scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    # ------------------------------------------------------------------
    # Load merged config (common + task + overrides)
    # ------------------------------------------------------------------
    args = parse_args_finetune()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="finetune",
        embeddings_profile=args.emb_profile,
        model=args.model,
        tag=args.tag,
        finetune_init=args.init,
    )
    validate_config(cfg_mmt)

    # ------------------------------------------------------------------
    # Runtime context (device, seed, logging)
    # ------------------------------------------------------------------
    device, _ = init_run_context(cfg_mmt, phase="finetune")

    cfg_data = cfg_mmt.data
    cfg_loader = cfg_mmt.loader
    cfg_train = cfg_mmt.train

    # Benchmark task config (with overrides such as subset_of_shots/local)
    cfg_task = load_task_definition(args.task)

    # ------------------------------------------------------------------
    # Task metadata + MAST datasets
    # ------------------------------------------------------------------
    dict_task_metadata, mast_dataset_train, mast_dataset_val, _mast_test = (
        build_mast_datasets(
            cfg_task,
            cfg_data=cfg_data,
            phase="finetune",
        )
    )

    # ------------------------------------------------------------------
    # Signal specs + embeddings
    # ------------------------------------------------------------------
    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task,
        dict_task_metadata,
    )

    run_dir = Path(cfg_mmt.paths["run_dir"])
    signal_specs, codecs = resolve_finetune_embeddings(
        cfg_mmt=cfg_mmt,
        signals_by_role=signals_by_role,
        dict_task_metadata=dict_task_metadata,
        run_dir=run_dir,
        cfg_task=cfg_task,
    )

    # ------------------------------------------------------------------
    # Window data
    # ------------------------------------------------------------------
    logging.getLogger("mmt").info("")
    window_data = build_window_data(
        cfg_mmt=cfg_mmt,
        mast_datasets={"train": mast_dataset_train, "val": mast_dataset_val},
        dict_task_metadata=dict_task_metadata,
        cfg_task=cfg_task,
        signal_specs=signal_specs,
        codecs=codecs,
        phase="finetune",
    )

    dataloader_mmt_train = window_data["train"]["loader"]
    dataloader_mmt_val = window_data["val"]["loader"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    logging.getLogger("mmt").info("")
    model = build_model_and_optional_warmstart(
        cfg_mmt=cfg_mmt,
        signal_specs=signal_specs,
        device=device,
    )

    # ------------------------------------------------------------------
    # Finetune
    # ------------------------------------------------------------------
    logging.getLogger("mmt.Train").info("")
    with sdpa_math_only_ctx():
        train_finetune(
            model=model,
            train_loader=dataloader_mmt_train,
            val_loader=dataloader_mmt_val,
            run_dir=cfg_mmt.paths["run_dir"],
            train_cfg=cfg_train,
            loader_cfg=cfg_loader,
        )


if __name__ == "__main__":
    main()
