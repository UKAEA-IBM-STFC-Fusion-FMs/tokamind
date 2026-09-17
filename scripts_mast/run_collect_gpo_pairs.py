"""
Entrypoint for GPO preference-pair dataset collection.

This script mirrors run_eval.py in structure: it loads the same merged config,
resolves the same embeddings and model weights, and runs one inference pass
over the official train and validation shot splits. Instead of computing benchmark metrics it writes
(y_w_emb, y_l_emb) preference pairs to disk for downstream GPO fine-tuning.

Schema v3: pairs are stored in embedding (coefficient) space so the GPO
training loop can operate in the same space as the model's training loss.
Compact decoded native-space error diagnostics are recorded for model-error
pairs; full native arrays are not retained in the pair dataset.

Output layout
-------------
    runs/<model_source>/gpo_pairs/
        metadata.json
        collection_config.json
        <signal_name>__shard_000000.npz
        <signal_name>__shard_000001.npz
        ...

Each .npz shard contains arrays (schema v3):
    shot_id        int64   (B,)
    window_index   int64   (B,)
    y_w_emb        float32 (B, D)  — ground-truth embedding  (preferred)
    y_l_emb        float32 (B, D)  — model-prediction embedding (dispreferred)
    native_nrmse   float32 (B,)    — decoded native RMSE / signal std
    native_nmae    float32 (B,)    — decoded native MAE / signal std
    native_mse     float32 (B,)    — decoded native MSE
    embedding_gap  float32 (B,)    — mean squared embedding distance
    emb_dim        int64   — embedding dimension D

Usage
-----
    python scripts_mast/run_collect_gpo_pairs.py \\
        --task task_4-1 \\
        --model_source ft-task_4-1-scratch-mmt

Optional flags
--------------
    --split           both (official train and validation shots; test excluded)
    --val_fraction    must be 0; validation uses official MAST shots
    --train_fraction  random fraction of the train split to collect from (default: 1.0 = all)
    --shard_size      windows per .npz shard (default: 2048)
    --multi_signal    joint|independent — multi-output loss strategy (default: joint)
    --tag             appends a sub-folder suffix, e.g. gpo_pairs_<tag>/
    --overwrite       replace output directory if it already exists
"""

from __future__ import annotations

import argparse
import itertools
import logging
import math
import random
from pathlib import Path

from mast_utils import (
    build_mast_datasets,
    build_model_and_optional_warmstart,
    build_signals_by_role_from_task_definition,
    build_window_data,
    extract_signal_stats,
    init_run_context,
    load_experiment_config,
    load_task_definition,
    resolve_eval_embeddings,
    validate_mast_config,
)
from mast_utils.gpo import collect_gpo_pairs
from mast_utils.gpo.protocol import apply_source_contract, collection_contract

from mmt.checkpoints import load_best_weights
from mmt.data import build_decoders
from mmt.utils import sdpa_math_only_ctx, validate_config


# ----------------------------------------------------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect GPO preference pairs from a trained TokaMind model.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--task",
        type=str,
        default="_test",
        help="Task identifier (same as run_eval.py).",
    )
    parser.add_argument(
        "--model_source",
        type=str,
        default="_test",
        help="Run ID or path of the trained model to collect pairs from.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["both"],
        help=(
            "Collect official train and validation splits together.  "
            "The test split is reserved for evaluation and must not be used here."
        ),
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.0,
        help=("Must be zero: validation is defined by official MAST shots."),
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help=(
            "Random fraction of the train split to collect pairs from.  "
            "Must be in (0.0, 1.0].  Default: 1.0 (collect from all windows).  "
            "Values < 1.0 randomly subsample the dataloader so only that fraction "
            "of batches are processed — useful for large datasets where collecting "
            "from every window is unnecessary or too slow."
        ),
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2048,
        help="Target number of windows per .npz shard file.  Default: 2048.",
    )
    parser.add_argument(
        "--multi_signal",
        type=str,
        default="joint",
        choices=["joint", "independent"],
        help=(
            "Multi-output loss strategy for tasks with several output signals.  "
            "'joint' (default) uses a single combined GPO loss; "
            "'independent' trains a separate loss per signal."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag appended to the output folder name (gpo_pairs_<tag>/).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the output directory if it already exists (otherwise the run aborts).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    args = _parse_args()
    log = logging.getLogger("mmt.GPO")

    if args.val_fraction != 0.0:
        raise SystemExit("CGPO uses official MAST validation shots; --val_fraction must be 0.")
    if not (0.0 < args.train_fraction <= 1.0):
        raise SystemExit(f"--train_fraction must be in (0.0, 1.0], got {args.train_fraction!r}.")

    # ------------------------------------------------------------------------------------------------------------------
    # Config: reuse the eval phase config — same data/preprocess/loader settings.
    # ------------------------------------------------------------------------------------------------------------------
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="eval",
        model_source=args.model_source,
        tag=args.tag,
        integration_hook=lambda merged, phase: apply_source_contract(merged, "scripts_mast/configs", collection=True),
        save_config=False,
    )
    validate_config(cfg=cfg_mmt)
    source_split = cfg_mmt.data.pop("split")
    validate_mast_config(cfg=cfg_mmt)
    cfg_mmt.data["split"] = source_split

    device, _ = init_run_context(cfg_mmt=cfg_mmt, phase="eval")

    cfg_data = cfg_mmt.data
    cfg_eval = cfg_mmt.eval
    amp_enabled = cfg_eval.get("amp", {}).get("enable", True)

    cfg_task = load_task_definition(task_key=args.task)

    # ------------------------------------------------------------------------------------------------------------------
    # MAST datasets — disjoint train/validation shots; test is reserved for evaluation.
    # ------------------------------------------------------------------------------------------------------------------
    dict_task_metadata, mast_train, mast_val, _test = build_mast_datasets(
        cfg_task=cfg_task,
        cfg_data=cfg_data,
        phase="finetune",
        cfg_model_source=cfg_mmt.model_source,
    )
    mast_datasets = {"train": mast_train, "val": mast_val}
    protocol_contract = collection_contract(cfg_mmt.raw, cfg_task)

    # ------------------------------------------------------------------------------------------------------------------
    # Embeddings + signal specs
    # ------------------------------------------------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------------------------------------------------
    # Window data includes transient native targets for schema-v3 diagnostics;
    # only compact scalar diagnostics are persisted in the pair shards.
    # ------------------------------------------------------------------------------------------------------------------
    cfg_mmt.data["keep_output_native"] = True
    window_data = build_window_data(
        cfg_mmt=cfg_mmt,
        mast_datasets=mast_datasets,
        dict_task_metadata=dict_task_metadata,
        cfg_task=cfg_task,
        signal_specs=signal_specs,
        codecs=codecs,
        phase="eval",
        window_test_mode=False,
    )
    dataloader = window_data["train"]["loader"]

    # ------------------------------------------------------------------------------------------------------------------
    # Optional random subsampling of the train split.
    # ------------------------------------------------------------------------------------------------------------------
    # When --train_fraction < 1.0 we randomly select a subset of batches from
    # the cached dataloader.  We shuffle the batch indices with a fixed seed so
    # the subset is reproducible but not contiguous (not just the first N batches).
    # The resulting loader is an iterator, which collect_gpo_pairs accepts fine.
    if args.train_fraction < 1.0:
        try:
            n_batches_total = len(dataloader)
        except TypeError:
            raise SystemExit(
                "--train_fraction requires a cached (map-style) dataloader "
                "(data.cache.enable must be true).  "
                "For streaming datasets set --train_fraction 1.0 and use "
                "data.cache.max_windows.train in the config instead."
            )
        n_keep = max(1, math.ceil(n_batches_total * args.train_fraction))
        rng = random.Random(42)
        kept_indices = sorted(rng.sample(range(n_batches_total), n_keep))
        dataloader = itertools.compress(dataloader, (i in set(kept_indices) for i in range(n_batches_total)))
        log.info(
            "train_fraction=%.2f → keeping %d / %d batches (randomly sampled, seed=42)",
            args.train_fraction,
            n_keep,
            n_batches_total,
        )

    dataloader = itertools.chain(dataloader, window_data["val"]["loader"])

    # ------------------------------------------------------------------------------------------------------------------
    # Model + checkpoint
    # ------------------------------------------------------------------------------------------------------------------
    model = build_model_and_optional_warmstart(
        cfg_mmt=cfg_mmt, signal_specs=signal_specs, device=device, skip_warmstart=True
    )
    epoch_best, best_val, _ = load_best_weights(run_dir=str(train_run_dir), model=model, map_location=str(device))
    log.info("Loaded checkpoint from %s (epoch=%s, best_val=%s)", train_run_dir, epoch_best, best_val)

    model.eval()

    # ------------------------------------------------------------------------------------------------------------------
    # Output signal id→name map and native decoders for pair-quality diagnostics.
    # ------------------------------------------------------------------------------------------------------------------
    cfg_drop = cfg_eval.get("drop", {}) or {}
    drop_outputs_set = set(cfg_drop.get("outputs", []) or [])
    output_specs = [s for s in signal_specs.specs_for_role("output") if s.name not in drop_outputs_set]
    id_to_name = {spec.signal_id: spec.name for spec in output_specs}
    id_decoders = build_decoders(registry=signal_specs, codecs=codecs, role="output")
    all_signal_stats = extract_signal_stats(dict_metadata=dict_task_metadata)
    missing_native_support = [
        spec.name for spec in output_specs if spec.signal_id not in id_decoders or spec.name not in all_signal_stats
    ]
    if missing_native_support:
        raise SystemExit(
            "Schema-v3 collection requires a native decoder and statistics for every output signal. "
            f"Missing support for: {missing_native_support}"
        )
    native_decoders = {spec.name: id_decoders[spec.signal_id] for spec in output_specs}
    native_stats = {spec.name: all_signal_stats[spec.name] for spec in output_specs}

    # ------------------------------------------------------------------------------------------------------------------
    # Determine output directory (optionally tagged)
    # ------------------------------------------------------------------------------------------------------------------
    out_subdir = f"gpo_pairs_{args.tag}" if args.tag else "gpo_pairs"
    gpo_out_dir = train_run_dir / out_subdir

    log.info("Writing GPO pairs to: %s", gpo_out_dir)
    log.info(
        "Split: %s | val_fraction: %.2f | train_fraction: %.2f | multi_signal: %s",
        args.split,
        args.val_fraction,
        args.train_fraction,
        args.multi_signal,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Collection loop
    # ------------------------------------------------------------------------------------------------------------------
    with sdpa_math_only_ctx():
        result = collect_gpo_pairs(
            model=model,
            dataloader=dataloader,
            device=device,
            id_to_name=id_to_name,
            run_dir=train_run_dir,
            out_dir=gpo_out_dir,
            task_name=args.task,
            split=args.split,
            val_fraction=args.val_fraction,
            train_fraction=args.train_fraction,
            multi_signal=args.multi_signal,
            amp_enabled=amp_enabled,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
            native_decoders=native_decoders,
            native_stats=native_stats,
            protocol_contract=protocol_contract,
        )

    log.info("Done.  GPO dataset: %s", result)


# ======================================================================================================================
if __name__ == "__main__":
    main()
