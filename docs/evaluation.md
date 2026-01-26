# Evaluation (metrics, traces, and ablations)

Evaluation is designed to be:

- **reproducible** (strict checkpoint loading),
- **config-driven** (no ad-hoc flags),
- **flexible** (evaluate with missing inputs/outputs via masking, without changing model weights).

This document describes what evaluation does, where outputs are written, and how to run common ablations.

---

## Selecting which model to evaluate

Evaluation needs a trained run directory that contains checkpoints.

Recommended config key:

```yaml
model_source:
  run_dir: "<training_run_id>"
```

> Legacy naming: some configs may use `model_init.model_dir`. The meaning is the same:
> it is the training run directory whose checkpoint weights will be loaded.

Evaluation loads the **best** checkpoint if present, otherwise it falls back to the latest checkpoint.

To keep evaluation consistent across models, the config loader rebuilds the **model spec** from the source run's saved merged config:

- `model`
- `embeddings`
- `preprocess.chunk` and `preprocess.trim_chunks`

It reads them from:

```
runs/<training_run_id>/<training_run_id>.yaml
```

So your `eval.yaml` / `eval_overrides.yaml` should focus on evaluation knobs (drop lists, metrics, traces, etc.), not architecture.


---

## Output location

Eval outputs are written under the training run directory:

```
runs/<training_run_id>/<eval_id>/
  <eval_id>.yaml
  metrics/
  traces/
```

- `eval_id` can be set in `tasks_overrides/<task>/eval_overrides.yaml`.
- If omitted, the loader typically uses a timestamped default like `eval__YYYYMMDD_HHMMSS`.

This keeps all evaluations for a run grouped next to the run.

---

## Keep the model spec fixed

A key design choice is:

> Evaluation does not rebuild a “smaller model” for subsets of signals.
> It evaluates the same trained model and uses **masking** to simulate missing inputs/outputs.

This avoids:
- changing signal ids,
- resizing heads/encoders,
- accidental checkpoint mismatches.

---

## Forcing missing inputs/actuators/outputs (ablations)

Eval supports deterministic ablations via config:

```yaml
eval:
  drop:
    inputs:    ["summary-ip"]
    actuators: ["pf_active-coil_voltage"]
    outputs:   ["pf_active-coil_current"]
```

Semantics:

- Dropped **inputs/actuators**: no tokens are emitted for those signals.
  The backbone sees less context; model weights are unchanged.
- Dropped **outputs**: the output mask is set false so they:
  - contribute no loss,
  - are excluded from metrics,
  - are excluded from traces.

Implementation detail:

- The collate function is configured with force-drop overrides:

```yaml
collate:
  p_drop_inputs_overrides:    { "<name>": 1.0, ... }
  p_drop_actuators_overrides: { "<name>": 1.0, ... }
  p_drop_outputs_overrides:   { "<name>": 1.0, ... }
```

The `eval.drop.*` convenience lists are typically translated into these overrides internally.

---

## Metrics

Metrics are controlled by `eval.compute_metrics`.

Metrics are computed only for outputs that remain active (`output_mask=True`).

```yaml
eval:
  compute_metrics:
    summary: true
    per_window: true
    per_timestamp: false
```

What each flag does:

- `summary`: writes `<task_name>_metrics_summary.csv` (per-output averages) and returns/logs a summary dict.
- `per_window`: writes `<task_name>_metrics_per_window.csv` (per-shot, per-window, per-output).
- `per_timestamp`: writes `<task_name>_metrics_per_timestamp.csv` (per-shot, per-window, per-time, per-output).

Outputs are written under:

```
<eval_run_dir>/metrics/
```

To disable metrics entirely, set all flags to `false`:

```yaml
eval:
  compute_metrics:
    summary: false
    per_window: false
    per_timestamp: false
```

---

## Traces (qualitative inspection)

Enable/disable:

```yaml
eval:
  traces:
    enable: true
    n_max: 8
    signals: ["pf_active-coil_current"]   # null → all outputs
    times_indexes: [0, 1, 2]              # null → full horizon
```

Traces are intended for “quick look” diagnostics:

- `n_max`: max number of windows to save traces for
- `signals`: optional subset of output signals to save
- `times_indexes`: optional subset of time indices

Outputs are generally saved under:

```
<eval_run_dir>/traces/
```

### `keep_output_native`

Traces (and some metrics) often need native outputs.
For that reason, evaluation typically sets:

```yaml
data:
  keep_output_native: true
```

If you set this to `false`, you may still be able to compute metrics from `output_emb`,
but you will likely lose rich trace information.

---

## Cached vs streaming evaluation

Evaluation can use either window-level dataset:

- **Cached** windows (recommended): faster, deterministic epoch sizing.
- **Streaming** windows: lower memory, but less direct control over “number of windows”.

If you evaluate on streaming windows, ensure your configuration/runner defines an appropriate
number of batches/windows to iterate over (depending on your evaluation loop).

---

## Common evaluation workflows

### Evaluate a trained run (default)
1. Set `model_source.run_dir` to the training run you want to evaluate.
2. Run `run_eval.py --task <task>`.

### Evaluate under missing inputs
Set:

```yaml
eval:
  drop:
    inputs: ["some_input_signal"]
```

### Evaluate only a subset of outputs
Set:

```yaml
eval:
  drop:
    outputs: ["output_to_disable"]
```

This keeps the model spec fixed and simply disables supervision/metrics for those outputs.
