# CGPO configuration and shot protocol (v1)

This protocol replaces the row-level holdout used by the historical CGPO runs.
It implements configuration/provenance checks, disjoint supervision, sample-counted
metrics and strict epoch-boundary resume. Cluster orchestration remains a separate
follow-up change; this is not yet a declaration that the whole pipeline is validated.

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
run requires an identical request and resolved configuration to resume.

## Metrics and checkpoint selection

`train.checkpoint_metric: mse` is the CGPO default. It selects the best checkpoint
and drives early stopping using embedding-space MSE on **all valid independent
validation windows**, regardless of pair availability or active loss terms. Each
signal first averages its per-window coefficient MSE; `mse` is the unweighted
mean of the observed signal means. This is not native-space NRMSE. A validation
metric with no observations or a non-finite value stops training with an error.

MSE, preference loss, SFT and diagnostics have separate per-signal sums/counts.
For example, window MSEs 1 and 100 give 50.5 even if only the first has a pair.
History includes `mse/count/<signal>`, `mse/sum/<signal>`, `mse/<signal>`,
`<term>/count/<signal>` and `<term>/<diagnostic>/<signal>`. Diagnostic counts
and sums are also logged under `<term>/count/<diagnostic>/<signal>` and
`<term>/sum/<diagnostic>/<signal>`. Signal identifiers are runtime numeric IDs.
`pair_window_coverage` counts windows with at least one valid signal pair;
`pair_batch_coverage` counts batches with at least one such window. Their
numerators and denominators are logged separately, as is pair coverage per signal.

`train.checkpoint_metric: objective` explicitly selects the epoch objective
instead. Each term is first reduced per signal over its own observations, then
signal and term weights are applied. CGPO remains diagnostic under the default
`mse` selection. Epoch metrics are invariant to batch partitioning up to floating
point rounding; batch coverage and the optimization trajectory need not be.
The old pair-batch denominator and `pair_hit_rate` are no longer used. INFO logs
summarize only MSE, objective, term means and window/batch pair coverage; full
per-signal sums, counts and diagnostics remain in history.

## Strict resume

`GPO_RESUME=1` resumes the same tag from the last **completed epoch**. Work after
that checkpoint is replayed. Before MAST dataset construction, the entrypoint
checks the immutable request, original source/reference and pair hashes, training
signature and a versioned SHA-256 checkpoint manifest. Missing, corrupt, partial
or legacy checkpoints fail; there is no fallback to random policy weights or a
fresh optimizer. The manifest also checks that the best checkpoint is intact and
consistent with the latest save. An interruption during checkpoint writing can
therefore require restoring an intact checkpoint backup.

Resume restores model, optimizer, scheduler, AMP scaler, Python/NumPy/Torch RNGs,
DataLoader/sampler generators, stage/epoch/step counters, early-stop state and
history. The frozen reference always comes from the original, hash-verified
source checkpoint, never from the resumed policy. A completed/early-stopped run
performs no additional epochs. Changing loss, beta, filters, reference, selection
metric, batch settings, schedule or protocol requires a **new run tag**.

Exact resume supports map-style cached datasets with `persistent_workers=false`;
streaming datasets and persistent-worker RNG state are not resumable. The first
checkpoint save warns about each unsupported loader, once per training invocation,
while ordinary training continues. The CUDA
RNG state requires the original device topology. CPU deterministic tests establish
exact equality; bitwise reproducibility across different hardware/software or
nondeterministic GPU kernels is not claimed.

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
PYTHONPATH=src:scripts_mast python -m pytest -q tests/test_cgpo_protocol.py tests/test_cgpo_metrics_resume.py
```

Tests cover real configuration resolution for all 14 benchmark tasks, local
precedence, immutable snapshots, artifact tampering, train-only filter fitting
and calibration, and collection-to-training entrypoint wiring with synthetic
MAST loaders and a small CPU model. Metrics tests cover masked signals, sparse
pairs, batch partitioning and MSE-only ablations. Resume tests compare continuous
and interrupted training (including stage boundaries), checkpoint selection,
early-stop state, missing/corrupt files and changed objectives/protocols. They do
not replace a full MAST/GPU pilot.
