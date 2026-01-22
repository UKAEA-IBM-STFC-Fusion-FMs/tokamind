# Configuration Guide

This repository uses a **convention-based** configuration system.

- The core Python package (`src/mmt/`) is dataset-agnostic.
- Dataset/task integration lives under `scripts_mast/`.
- Runs are configured by selecting a **task** and a **phase**; the loader finds and merges YAML files by convention.

The goal is: **easy reuse of shared defaults** + **small per-task overrides** + **predictable output locations**.

---

## 1) Directory layout

All experiment YAML files live under:

```text
scripts_mast/configs/
  common/
    core.yaml
    embeddings.yaml
    finetune.yaml
    pretrain.yaml
    eval.yaml
    tune_dct3d.yaml

  tasks_overrides/
    <task>/
      core_overrides.yaml      # optional (task-wide overrides)
      finetune_overrides.yaml        # optional
      pretrain_overrides.yaml        # optional
      eval_overrides.yaml            # optional
      tune_dct3d_overrides.yaml      # optional (rare)
      embeddings_overrides/              # task-level embedding overrides (selected by profile)
        <profile>.yaml                  # required for pretrain/finetune/eval (can be empty)
                                        # (DCT3D tuning writes to: dct3d.yaml)
```

A **task** is simply a folder under `scripts_mast/configs/tasks_overrides/<task>/`.

---

## 2) Phases

Supported phases:

- `pretrain` — train from scratch or warm-start, using pretraining task definitions
- `finetune` — train from scratch or warm-start, using downstream task definitions
- `eval` — evaluate a trained run (metrics + optional traces)
- `tune_dct3d` — tune DCT3D embedding parameters and write `embeddings_overrides/<profile>.yaml` (default: `dct3d`)

Each phase has a corresponding runner:

```bash
python scripts_mast/run_pretrain.py --task <task>
python scripts_mast/run_finetune.py --task <task>
python scripts_mast/run_eval.py --task <task>
python scripts_mast/run_tune_dct3d.py --task <task>
```

---

## 3) Merge order (the most important rule)

The loader deep-merges configs in the following order (**later wins**):

1. `common/core.yaml`
2. `common/embeddings.yaml`
3. `common/<phase>.yaml`
4. `tasks_overrides/<task>/core_overrides.yaml`
5. `tasks_overrides/<task>/<phase>_overrides.yaml` *(optional)*
6. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` *(required for pretrain/finetune/eval; not merged during tune_dct3d)*

### Deep merge semantics
- Dictionaries merge recursively.
- Scalars replace.
- Lists replace (they are **not** concatenated).

---

## 4) What goes in each config file

### `common/core.yaml`
Put **stable, task-agnostic defaults** here:

- `seed`
- `runtime` info: if `debug_logging` is true, the logger saves more diagnostics
- global `data` defaults
  - `local`: if `true` the CSD3 version of the MAST data is loaded -- you need permission to access it 
  - `subset_of_shots`: number of analyzed shots (set to `null` to include all of them)
- preprocessing that should not drift across phases (e.g., chunking, trimming)
- model architecture defaults (`model.backbone`, `model.modality_heads`, adapter defaults)

Do **not** put:
- `task` / `task_config`
- phase-specific window-selection thresholds (`preprocess.valid_windows`)
- run-specific settings (`run_id`, `eval_id`, model weight sources)

### `common/embeddings.yaml`
Defines default embedding/codec settings:

- `embeddings.defaults` by `(role, modality)`
- keep `embeddings.per_signal_overrides` empty in common (use `{}`), unless you truly want a global override

Task-specific tuned overrides belong in `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`.

### `common/finetune.yaml` and `common/pretrain.yaml`
Training-phase defaults:

- `train.*` (stages, LR/WD, scheduler settings)
- `loader.*` (batch size, num workers, shuffling)
- `collate.*` (dropout/masking defaults)
- `data.cache.*`
- `preprocess.valid_windows` *(allowed to differ by phase)*

Any task-specific changes should go in:
- `tasks_overrides/<task>/finetune_overrides.yaml` or
- `tasks_overrides/<task>/pretrain_overrides.yaml`

### `common/eval.yaml`
Evaluation defaults:

- `data.keep_output_native: true` (required for metrics/traces)
- `loader.*` for eval
- `eval.metrics.*` and `eval.traces.*`
- `preprocess.valid_windows` *(allowed to differ by phase)*

The training run to evaluate must be provided in `tasks_overrides/<task>/eval_overrides.yaml`.

### `common/tune_dct3d.yaml`
Tuning defaults:

- `tune_dct3d.sampling.*` (how many shots/windows)
- `tune_dct3d.objective.*` (thresholds and max budget)
- `tune_dct3d.search_space.*` (keep_h/keep_w/keep_t)
- `preprocess.valid_windows`

Task-specific tuning tweaks (rare) go in `tasks_overrides/<task>/tune_dct3d_overrides.yaml`.

---

## 5) Task configs

### `tasks_overrides/<task>/core_overrides.yaml`
This file defines optional task-wide overrides that should apply to *all* phases.

Minimal example:

```yaml
task: "task_2-1"

# Optional: task-wide model/data overrides (apply to ALL phases)
model:
  output_adapters:
    hidden_dim:
      default: 0
      bucketed:
        enable: true
        rules:
          - {max_out_dim: 64, hidden: 0}
          - {max_out_dim: 512, hidden: 32}
          - {max_out_dim: 4096, hidden: 64}
          - {max_out_dim: null, hidden: d_model}
      manual:
        equilibrium-psi: 64

data:
  subset_of_shots: 2
  local: false
```

**Rule of thumb:**
- Put **task-wide** overrides here (apply to pretrain/finetune/eval/tune).
- Put **phase-specific** and **run-specific** overrides in `<phase>_overrides.yaml`.

### Important
The benchmark-style task definition is inferred **only** from `task`.
Resolution order:

1) **Benchmark task (preferred)**  
   If the benchmark package knows the task name, we load it via the benchmark API:

- `MAST_benchmark.tasks.get_task_config(task_name)`

2) **Local task (registry)**  
   If benchmark does not know the task name (raises `KeyError`), we treat it as a **local** task and load it via a local registry map:

- `LOCAL_TASK_CONFIGS_MAP[task_name]` → path under `scripts_mast/configs/`

The local registry lives in:

- `scripts_mast/mast_utils/task_config.py`

## Adding a NEW task

### A) Adding a benchmark task (already in benchmark)

1. (Optional) Create folder: `scripts_mast/configs/tasks_overrides/<task_name>/`
2. (Optional) Create `core_overrides.yaml` containing:

```yaml
task: <task_name>
```

3. Run:

```bash
python scripts_mast/run_finetune.py --task <task_name>
```

No additional YAML paths are required; the benchmark benchmark owns the mapping from task name → YAML.


### B) Adding a local task (not in benchmark)

1. Add a benchmark-style YAML under:

- `scripts_mast/configs/local_tasks_def/<task_name>.yaml`

2. Register the task in `LOCAL_TASK_CONFIGS_MAP` in:

- `scripts_mast/mast_utils/task_config.py`

Example:

```python
LOCAL_TASK_CONFIGS_MAP["my_local_task"] = "local_tasks_def/my_local_task.yaml"
```

3. (Optional) Create `scripts_mast/configs/tasks_overrides/my_local_task/core_overrides.yaml`:

```yaml
task: my_local_task
```
---

## 6) Run-specific overrides

### `run_id` (training)
To force a deterministic training output folder name, add to the relevant overrides file:

- `tasks_overrides/<task>/finetune_overrides.yaml`
- `tasks_overrides/<task>/pretrain_overrides.yaml`

Example:

```yaml
run_id: "task_2-1__finetune__v1"
```

If `run_id` is omitted, the loader creates a timestamped one:

```
<task>__<phase>__YYYYMMDD_HHMMSS
```

### `eval_id` (evaluation)
To name the eval output folder, set in `tasks_overrides/<task>/eval_overrides.yaml`:

```yaml
eval_id: "eval_test"
```

If omitted, eval defaults to `eval__<timestamp>`.

---

## 7) Selecting which weights to load: 

There are two distinct behaviors:

- **Warm-start** (training phases): start a **new** run directory, optionally initializing from another run.
- **Resume** (training phases): continue the **same** run directory, including optimizer/scheduler/scaler state.

### Warm-start (pretrain / finetune)
Use `model_source.run_dir` (recommended naming) in the phase overrides:

```yaml
model_source:
  run_dir: "some_previous_run"
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: true
```

Warm-start uses **overlap loading**: only parameters with matching *(key, shape)* are copied; the rest remain freshly initialized.

### Resume (pretrain / finetune)
Use:

```yaml
train:
  resume: true
```

Resume is mutually exclusive with warm-start.

---

## 8) Output locations

### pretrain / finetune
Outputs go to:

```text
<repo_root>/runs/<run_id>/
  <run_id>.yaml
  checkpoints/
  logs/
  ...
```

### eval
Eval outputs go next to the training run:

```text
<repo_root>/runs/<training_run_id>/<eval_id>/
  <eval_id>.yaml
  metrics/
  traces/
  eval.log
```

Note: during **evaluation**, the loader rebuilds the model using the saved training config at
`<repo_root>/runs/<training_run_id>/<training_run_id>.yaml` so you do not need to duplicate
`model` / `embeddings` settings in eval YAMLs.

### tune_dct3d
Tuning writes its main artifact directly into the task folder:

```text
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/dct3d.yaml
```

---

## 9) Embeddings and tuning

### Defaults vs overrides
- `common/embeddings.yaml` defines defaults per `(role, modality)`.
- `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` contains **only per-signal overrides**.

This file is **required** for `pretrain`, `finetune`, and `eval` for the selected embedding profile.
If you do not want task-specific embedding overrides yet, create an empty YAML file at that path.
(The profile is selected by the phase runners via `--emb_profile`; default: `dct3d`.)

The tuning script `run_tune_dct3d.py` writes `embeddings_overrides/dct3d.yaml` automatically.

### `embeddings_overrides/<profile>.yaml` format
It should contain only:

```yaml
embeddings:
  per_signal_overrides:
    input:
      some_signal_name:
        encoder_name: "dct3d"
        encoder_kwargs:
          keep_h: 1
          keep_w: 1
          keep_t: 20
```

Avoid duplicating defaults in this file.

---

## 10) Validation rules (what the validator enforces)

### Training (`pretrain`, `finetune`)
The validator enforces:

- Required global fields in `train.*` (early stopping, scheduler warmup fraction, stages, etc.).
- Each stage must define epochs, grad accumulation, lr/wd per block, and freeze flags.
- `null` lr/wd values inherit from `backbone`.
- If a block is frozen, lr/wd are forced to 0 (with a warning).
- If `data.cache.enable == false` (streaming windows), you must set `loader.batches_per_epoch`.

### Evaluation (`eval`)
The validator enforces:

- `data.keep_output_native` must be `true` (required for metrics/traces).

(Loading the trained run directory itself is validated by the loader when it computes paths.)

---

## 11) Task definition resolution (inferred from `task`)

MMT does **not** store a `task_config:` pointer in YAML anymore.

The benchmark-style task definition is resolved **only** from the task name (`task`)
by `scripts_mast/mast_utils/task_definition.py`:

1) **Benchmark task (preferred)**  
   Load by name via the benchmark API:
   `MAST_benchmark.tasks.get_task_config(task_name)`

2) **Local task (registry)**  
   If the benchmark does not know the task name (`KeyError`), load by name via a local registry map:
   `LOCAL_TASK_DEFS_MAP[task_name]` → a YAML path under `scripts_mast/configs/`

Local task definitions typically live under:

- `scripts_mast/configs/local_tasks_def/<task_name>.yaml`

---

## 12) Common gotchas

1) **Streaming without batches_per_epoch**
If `data.cache.enable: false`, you must set `loader.batches_per_epoch` for training.

2) **Empty dict vs null in YAML**
Prefer explicit empty dicts:

```yaml
p_drop_inputs_overrides: {}
```

If you leave a mapping key blank, YAML may parse it as `null`.

3) **Eval must point to a training run**
Set `model_source.run_dir` in `tasks_overrides/<task>/eval_overrides.yaml`.

4) **Keep output native for eval**
`data.keep_output_native` must be `true` in eval.

5) **Tune writes to the task folder**
`tune_dct3d` writes `embeddings_overrides/dct3d.yaml` to `tasks_overrides/<task>/embeddings_overrides/`.

---

## 13) Minimal examples

### A) Add a new downstream task
1. Create `scripts_mast/configs/tasks_overrides/<task_name>/core_overrides.yaml` with `task` and `task_config`.
2. Optionally add `finetune_overrides.yaml` / `eval_overrides.yaml`.
3. Run:

```bash
python scripts_mast/run_finetune.py --task <task_name>
python scripts_mast/run_eval.py --task <task_name>
```

### B) Tune embeddings for a task
```bash
python scripts_mast/run_tune_dct3d.py --task <task_name>
```
This writes `tasks_overrides/<task>/embeddings_overrides/dct3d.yaml`.

---

## 14) Toy example (benchmark-free)

For a benchmark-free smoke run, see the repo’s toy example (synthetic data) under `examples/`.

