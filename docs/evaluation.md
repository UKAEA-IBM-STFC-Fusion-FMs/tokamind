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

## CRPS (probabilistic) Diagnostic
The Continuous Ranked Probability Score scores the model's predictive distribution in native physical units,
alongside the standard point metrics (which are left unchanged). It is enabled per run:

```yaml
eval:
  compute_metrics:
    crps:
      enable: true
      n_samples: 50
```

`crps.enable` is the master on/off (plus `n_samples`); the **granularity follows the same `per_*` flags as the point
metrics**, writing parallel `crps_*` files.

Behavior:
- **Gaussian head** (`model.output_adapters.type: gaussian`): CRPS is estimated from `n_samples` (`>= 2`)
  reparameterized samples decoded to native space, scored per position with NaN masking.
- **Deterministic head**: a point forecast's CRPS is exactly its MAE, so CRPS reduces to `|pred − target|`
  (no sampling, `n_samples` ignored) with a warning. This keeps probabilistic and deterministic models on one
  comparable CRPS axis.

Both `CRPS` (native units, comparable to `RMSE`/`MAE`) and `NCRPS = CRPS / signal_std` (comparable to `NRMSE`/`NMAE`
— same per-signal std normalizer) are reported. CRPS/NCRPS are **linear**, so they aggregate by **plain mean** over
the same shot-weighted hierarchy NRMSE uses (window → shot → signal → task), *not* RMSE's square/sqrt path.

Outputs (each gated by its `per_*` flag, written only when `crps.enable: true`):
- `per_task` → `crps_task_metrics.csv` — per-signal `CRPS_mean/CRPS_std_pop/NCRPS_mean/NCRPS_std_pop` (mean ±
  population-std across shots) plus a task-level row (equal-weight over signals).
- `per_shot` → `crps_shot_metrics.csv` — per `(feature_name, shot_id)` mean CRPS/NCRPS over that shot's windows.
- `per_window` → `crps_window_metrics.csv` — per `(shot_id, window_index, feature_name)` mean CRPS/NCRPS over the
  window's valid positions.
- `per_timestamp` → `crps_timestamps_metrics.csv` — per `(shot_id, window_index, time_id, feature_name)` raw
  CRPS/NCRPS (the source for plotting).

A task-level `CRPS_mean` / `NCRPS_mean` (shot-weighted) is computed whenever task/shot aggregation is enabled
(`per_task` or `per_shot`), logged as `Additional task metrics` after the benchmark metrics line and returned in
`result["crps_metrics"]`.

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
- when `compute_metrics.crps.enable: true` and the model has a Gaussian head, each artifact also carries
  `pred_samples` of shape `(n_windows, n_samples, *native)` for confidence-band plotting
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

During training, `accept_nan_outputs=False` in `SelectValidWindowsTransform` ensures windows with partial-NaN outputs are dropped before reaching the loss. During eval, `accept_nan_outputs=True` allows those windows through so the evaluator can score on the non-NaN positions using `nanmean`.

## Loss Choice and NaN Behavior

See [Training — Loss Configuration](training.md#loss-configuration) and [Training — NaN Handling](training.md#nan-handling-in-training) for the full description of how loss choice interacts with NaN positions in output signals.
