# Evaluation

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [Training](training.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md)

Evaluation loads a trained run, runs one pass on the test split, and writes metrics/traces under an eval directory.

## Run Command
```bash
python scripts_mast/run_eval.py \
  --task <task> \
  --model_source <run_id_or_path>
```

## Source Model Resolution
`--model_source` accepts:
- run id under `runs/`
- path to a run directory

The loader resolves `model_source.run_dir` and imports:
- `model`
- `embeddings`
- `preprocess.chunk`
- `preprocess.trim_chunks`

from the source run snapshot.

## Output Paths
Eval writes to:

```text
runs/<model_id>/eval/
  eval.yaml
  metrics/
  traces/
```

Use a fresh `--tag <tag>` to keep a new evaluation separate from earlier
results. Its directory and config snapshot are named `eval_<tag>/eval_<tag>.yaml`.

## What Is Evaluated
- best checkpoint from source run (fallback to latest if needed)
- same model spec as source run
- same embedding spec as source run
- same chunking/trim behavior as source run
- same data split as source run (`data.split` is inherited automatically; do not set it in eval config)

## Forced Drop Ablations
Configure deterministic drops in `eval.drop`:

```yaml
eval:
  drop:
    inputs: ["summary-ip"]
    actuators: ["pf_active-coil_voltage"]
    outputs: ["pf_active-coil_current"]
```

Behavior:
- dropped inputs/actuators: tokens are omitted
- dropped outputs: excluded from metrics/traces

## Metrics Configuration
```yaml
eval:
  compute_metrics:
    per_task: true
    per_shot: false
    per_window: false
    per_timestamp: false
```

Outputs:
- benchmark-level files in `metrics/<task>/`:
  - `task_metrics.csv` (if `per_task: true`)
  - `shots_metrics.csv` (if `per_shot: true`)
  - `windows_metrics.csv` (if `per_window: true`)
- optional per-timestamp csv in `metrics/<task>/timestamps_metrics.csv` (if `per_timestamp: true`)

## GS diagnostics

The GS task options live under `eval.compute_metrics` in
`scripts_mast/configs/mmt/tasks/eval_tasks.yaml`. They run on an existing
checkpoint; no retraining is needed. Outputs are in `metrics/<task>/`.

- `gs_metrics` reports ψ-operator error and, when current is predicted, GS
  self-consistency and RHS current error. `smoothness` compares predicted and
  reference roughness. Their default (unsuffixed) columns use the **one-cell-eroded limiter**.
- `regional_mse` reports pointwise prediction error separately for ψ and, when
  predicted, toroidal current. Its regions are the full grid, the **non-eroded**
  limiter (boundary cells included), and the rest of the grid. The limiter is
  fixed device geometry, not the plasma boundary.
- `psi_error_map` and `jtor_error_map` write full-grid NPZ maps of per-cell
  MSE, RMSE, MAE, signed bias and valid counts for the predicted field.
- `gs_topology` is independent and requires local equilibrium reference data.

The regional CSVs (`regional_mse_shots.csv`, `regional_mse_task.csv`) distinguish
`mse` (average window MSE within each shot, then average shots equally) from
`mse_pooled` (sum of squared errors divided by the total valid-cell count).
TokaMark NRMSE divides each shot's root MSE by the signal standard deviation,
then averages shots equally. It uses the same TokaMark default (random-split)
standard deviation for all three regions, regardless of evaluation split.
Inside and outside NRMSE do **not** add to full-grid NRMSE.
See the generated `regional_mse_protocol.json` for units, masks and coverage.

## What Is a Trace
A trace is a per-shot diagnostic record that aligns:
- model predictions
- reference targets
- time axis for selected windows/signals

Traces are used for qualitative inspection (shape, lag, spikes, drop effects), not only aggregate scoring.

## Trace Configuration
```yaml
eval:
  traces:
    enable: true
    n_max: 5
    signals: null
    times_indexes: null
```

Outputs:
- trace artifacts under `traces/` (each `.npz` holds `true`, `pred`, `window_index`)
- each artifact is limited by `n_max` and optional signal/time filters

Filter behavior:
- `signals: null`: include all output signals
- `times_indexes: null`: include all available timestamps
- explicit lists: keep only selected outputs/time indexes

## Required Eval Data Setting
`data.keep_output_native` is **auto-derived** by the config validator and does not need to be set manually.
For eval it is always `true`. For training it is `true` only when the loss requires native-space targets
(e.g. `native_sparse_mse`).

## Sparse Evaluation
The TokaMark evaluator computes metrics using `nanmean`, which correctly ignores NaN values in ground truth. This enables benchmark-comparable sparse evaluation even when signals have missing timesteps or channels.

For this to work correctly, the pipeline preserves NaN values in `window["output"]` through to the evaluator — they are never overwritten. The imputation applied in `EmbedChunksTransform` operates only on a temporary local copy used for encoding and does not affect the ground truth values used for scoring.

Eval inherits `preprocess.embed_chunks` from the source training run so the token representation, including the NaN/inf imputation policy, matches training. Eval-specific sparse-window policy remains controlled by `preprocess.valid_windows`.

The supplied MAST training and eval phase configs allow partially missing outputs
(`preprocess.valid_windows.accept_nan_outputs: true`). Missing reference values
are not treated as zero error.

## Loss Choice and NaN Behavior

See [Training — Loss Configuration](training.md#loss-configuration) and [Training — NaN Handling](training.md#nan-handling-in-training) for the full description of how loss choice interacts with NaN positions in output signals.
