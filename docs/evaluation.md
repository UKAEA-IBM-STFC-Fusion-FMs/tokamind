# Evaluation

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md)

Evaluation loads a trained run, runs one pass on the test split, and writes metrics/traces under an eval directory.

## Run Command
```bash
python scripts_mast/run_eval.py \
  --task <task> \
  --model <run_id_or_path>
```

## Source Model Resolution
`--model` accepts:
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
- trace artifacts under `traces/`
- each artifact is limited by `n_max` and optional signal/time filters

Filter behavior:
- `signals: null`: include all output signals
- `times_indexes: null`: include all available timestamps
- explicit lists: keep only selected outputs/time indexes

## Required Eval Data Setting
Eval requires:
```yaml
data:
  keep_output_native: true
```

This is needed to decode and score outputs in native space.
