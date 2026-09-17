# CGPO configuration and shot protocol (v1)

This protocol replaces the row-level holdout used by the historical CGPO runs.
It implements configuration/provenance checks and disjoint supervision. Metric
aggregation, strict checkpoint resume and cluster orchestration are separate
follow-up changes; this is not yet a declaration that the whole pipeline is validated.

## Configuration precedence

1. Load the finetune base, CGPO phase and the selected `GPO_TASKS_YAML` task recipe.
2. Inherit the **model, embedding configuration, full preprocessing, benchmark
   split and subset size** from the source checkpoint's saved configuration.
   CGPO task preprocessing entries no longer override the source window geometry.
3. Reapply `scripts_mast/configs/local_overrides.yaml`. Local data paths and
   execution settings are allowed. Changes to the source contract or run identity
   are rejected. Collation must be deterministic and `drop_last` must be false.

Collection uses the same source contract and training-mode window construction
as CGPO training, including for validation. Both default to cached float16 windows.
A different effective cache dtype requires matching collection and training
settings. Window-count truncation is rejected; `train_fraction` may subsample
collection batches from training only. It never subsamples validation.

## Persistent shot split and provenance

New collections use `protocol: mast-shot-split-v1` with `split: both` and
`val_fraction: 0`. The official MAST train, validation and test shot lists are
recorded in `collection_config.json`; all three must be nonempty and disjoint.
No test pairs are collected. Validation shots receive no training gradients in
CGPO, including from the MSE anchor. They were excluded from the source model's
training split, although its early stopping may already have used validation.

Training and validation have separate MAST datasets and window caches. Pair
membership is determined by shot before any filters, for all signals together.
Native-NRMSE percentile bounds and per-shot caps are fitted on training pairs
and reused unchanged on validation. Calibration likewise reads training shots
only. The MSE anchor still sees every valid window in its own split, including
windows without preference pairs.

The collection contract records task definition, source representation, official
split/statistics/outlier asset hashes, checkpoint and embedding file hashes,
and effective cache dtype. Shard hashes are written at collection finalization.
Training rejects changed artifacts and incompatible or legacy collections.
Training-fitted filter thresholds are saved in `gpo_filters.json`.

Each new run reserves `gpo_request.yaml` exclusively. Embedding resolution writes
`<run_id>.yaml` without overwriting an earlier resolved snapshot. An existing
run requires an identical request and resolved configuration to resume. This
protects configurations; the existing training-loop checkpoint-resume fallback
still needs the separate planned correction. Do not use resume until that fix.

## Starting a corrected run

Use a new collection tag and a new training tag. Old collections are preserved;
there is no automatic migration because their provenance cannot be verified.
From the repository root, with the project environment installed:

```bash
BASE=ft-task_1-1-scratch-mmt-dct3d-embed-mse
python scripts_mast/run_collect_gpo_pairs.py \
  --task task_1-1 --model_source "$BASE" --split both --tag shots-v1
python scripts_mast/validate_gpo_pairs.py "runs/$BASE/gpo_pairs_shots-v1"
python scripts_mast/calibrate_gpo_task.py "runs/$BASE/gpo_pairs_shots-v1" \
  --dry_run --no_plots
python scripts_mast/run_gpo_finetune.py \
  --task task_1-1 --model_source "$BASE" \
  --gpo_dir "runs/$BASE/gpo_pairs_shots-v1" --tag cgpo-shots-v1
```

Calibration dry-run inspects training statistics without changing the shared
recipe. Applying calibration still uses the existing `--gpo_tasks_yaml` option;
the dedicated calibration-artifact workflow is a later change. Review the
selected beta/blacklist before launching training.

External cluster launchers that pass `--split train --val_fraction 0.1` are
incompatible with this protocol. Use the direct commands above until those
launchers are updated to `--split both --val_fraction 0` and their orchestration
bugs are fixed. Historical commands in the older runbooks describe legacy runs.

## Verification

```bash
PYTHONPATH=src:scripts_mast python -m pytest -q tests/test_cgpo_protocol.py
```

Tests cover real configuration resolution for all 14 benchmark tasks, local
precedence, immutable snapshots, artifact tampering, train-only filter fitting
and calibration, and collection-to-training entrypoint wiring with synthetic
MAST loaders and a small CPU model. They do not replace a full MAST/GPU pilot.
