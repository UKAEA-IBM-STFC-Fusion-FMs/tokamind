# Configuration Guide

This repository uses a **convention-based** configuration system.

- The core Python package (`src/mmt/`) is dataset-agnostic.
- Dataset/task integration lives under `scripts_mast/`.
- Runs are configured by selecting a **task** and a **phase**; the loader finds and merges YAML files by convention.

The goals are:

- **Explicit phase configs** (no hidden globals)
- **Small per-task overrides**
- **Stable model/eval behavior** (no architecture drift at eval time)
- **Predictable output locations**

---

## 0) Design overview: run context vs model identity

This project intentionally separates **run context** from **model identity**, so that finetuning and evaluation stay consistent across runs.

### A) Run context (explicit in every phase config)

These settings describe *how* to run (and can differ between your laptop and an HPC/GPU node):

- `seed`
- `runtime.*`
- `data.local`
- `data.subset_of_shots` *(may be `null`, but should be present)*

### B) Model identity (anchored to a training run)

These settings define *what* was trained and must not drift between training and evaluation:

- `model` (architecture)
- `preprocess.chunk` and `preprocess.trim_chunks` (token history shape)
- `embeddings` (default codec settings + per-signal overrides)

### Key rules

- **pretrain** defines the base `model` and `preprocess.{chunk, trim_chunks}` and writes a merged config snapshot to:

  `runs/<run_id>/<run_id>.yaml`

- **finetune** requires `model_source.run_id` (or `model_source.model_path`) and **inherits `model` + `preprocess.{chunk, trim_chunks}`** from the source run config snapshot.
  Finetune-side YAML should focus on *training settings* (stages, LR/WD, freezing, window selection, etc.).

- **eval** requires `model_source.run_id` (or `model_source.model_path`) and **rebuilds `model`, `embeddings`, and `preprocess.{chunk, trim_chunks}`** from the source run config snapshot.
  Eval-side YAML should focus on *metrics/traces* and *evaluation window selection*.

This design prevents “config drift” (e.g., changing adapter sizing in finetune but forgetting to mirror it in eval).

---

## 1) Directory layout

All experiment YAML files live under:

```text
scripts_mast/configs/
  common/
    embeddings.yaml
    pretrain.yaml
    finetune.yaml
    eval.yaml
    tune_dct3d.yaml

  tasks_overrides/
    <task>/
      pretrain_overrides.yaml        # optional
      finetune_overrides.yaml        # optional
      eval_overrides.yaml            # optional
      tune_dct3d_overrides.yaml      # optional (rare)
      embeddings_overrides/          # task-level embedding overrides (selected by profile)
        <profile>.yaml               # required for pretrain/finetune/eval (can be empty)
                                     # (DCT3D tuning writes to: dct3d.yaml by default)
```

A **task** is simply a folder under `scripts_mast/configs/tasks_overrides/<task>/`.

---

## 2) Phases

Supported phases:

- `pretrain` — train from scratch (or warm-start), using pretraining task definitions
- `finetune` — warm-start from a trained run, using downstream task definitions
- `eval` — evaluate a trained run (metrics + optional traces)
- `tune_dct3d` — tune DCT3D embedding parameters and write `embeddings_overrides/<profile>.yaml` (default profile: `dct3d`)

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

1. `common/embeddings.yaml`
2. `common/<phase>.yaml`
3. `tasks_overrides/<task>/<phase>_overrides.yaml` *(optional)*
4. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` *(required for pretrain/finetune/eval; not merged during tune_dct3d)*

### Deep merge semantics

- Dictionaries merge recursively.
- Scalars replace.
- Lists replace (they are **not** concatenated).

### Additional identity-inheritance step (finetune / eval)

After the YAML merge above, **finetune** and **eval** load the source run’s saved snapshot:

- expected file: `runs/<source_run_id>/<source_run_id>.yaml`
- if missing: **raise an error** (eval/finetune must not guess model shapes)

Then:

- **finetune**: inherits `model` and `preprocess.{chunk, trim_chunks}` from the source snapshot
- **eval**: inherits `model`, `embeddings`, and `preprocess.{chunk, trim_chunks}` from the source snapshot

---

## 4) What goes in each config file

### `common/embeddings.yaml`

Defines default embedding/codec settings:

- `embeddings.defaults` by `(role, modality)`
- keep `embeddings.per_signal_overrides` empty in common (use `{}`), unless you truly want a global override

Task-specific tuned overrides belong in:

- `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

### `common/pretrain.yaml`

Pretrain defines the *base model identity*:

- **Run context**: `seed`, `runtime.*`, `data.local`, `data.subset_of_shots`
- **Model identity**: `model`, `preprocess.chunk`, `preprocess.trim_chunks`
- Training defaults: `train.*`, `loader.*`, `collate.*`, `data.cache.*`, `preprocess.valid_windows`

### `common/finetune.yaml`

Finetune defines *how to adapt a pretrained model*:

- **Run context**: `seed`, `runtime.*`, `data.local`, `data.subset_of_shots`
- Training defaults: `train.*`, `loader.*`, `collate.*`, `data.cache.*`, `preprocess.valid_windows`
- Finetune-side model knobs that are intended to be captured in the finetune snapshot (e.g., output adapter policy)

**Do not rely on finetune.yaml to define `model` or `preprocess.{chunk, trim_chunks}`**.
Those are inherited from `model_source.run_id` (or `model_source.model_path`).

### `common/eval.yaml`

Evaluation defaults:

- **Run context**: `seed`, `runtime.*`, `data.local`, `data.subset_of_shots`
- `data.keep_output_native: true` (required for metrics/traces)
- `loader.*` for eval
- `eval.metrics.*` and `eval.traces.*`
- `preprocess.valid_windows` *(allowed to differ by phase)*

The training run to evaluate must be provided in:

- `tasks_overrides/<task>/eval_overrides.yaml`

### `common/tune_dct3d.yaml`

Tuning defaults:

- `tune_dct3d.sampling.*` (how many shots/windows)
- `tune_dct3d.objective.*` (thresholds and max budget)
- `tune_dct3d.search_space.*` (keep_h/keep_w/keep_t)
- `preprocess.valid_windows`

Task-specific tuning tweaks (rare) go in:

- `tasks_overrides/<task>/tune_dct3d_overrides.yaml`

---

## 5) Task overrides

A task can override phase defaults by adding any of the following (all optional):

- `tasks_overrides/<task>/pretrain_overrides.yaml`
- `tasks_overrides/<task>/finetune_overrides.yaml`
- `tasks_overrides/<task>/eval_overrides.yaml`

Typical contents:

- `run_id` / `eval_id`
- `model_source.run_id` / `model_source.model_path` *(required for finetune and eval)*
- phase-specific `preprocess.valid_windows.window_stride_sec`
- any task-specific `collate` drop probabilities

If you want a task-wide change to apply to multiple phases, copy it into each relevant `<phase>_overrides.yaml`.

---

## 6) Run ids and output locations

### `run_id` (training phases)

Set in either:

- `tasks_overrides/<task>/pretrain_overrides.yaml`
- `tasks_overrides/<task>/finetune_overrides.yaml`

Example:

```yaml
run_id: "task_2-1__finetune__v1"
```

If `run_id` is omitted, the loader creates a timestamped one:

```
<task>__<phase>__YYYYMMDD_HHMMSS
```

### `eval_id` (evaluation)

Set in `tasks_overrides/<task>/eval_overrides.yaml`:

```yaml
eval_id: "eval_test"
```

If omitted, eval defaults to a timestamped id.

### Output folders

#### pretrain / finetune

```text
<repo_root>/runs/<run_id>/
  <run_id>.yaml
  checkpoints/
  logs/
  ...
```

#### eval

```text
<repo_root>/runs/<training_run_id>/<eval_id>/
  <eval_id>.yaml
  metrics/
  traces/
  eval.log
```

#### tune_dct3d

Tuning writes its main artifact directly into the task folder:

```text
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml
```

---

## 7) Selecting which weights to load (warm-start vs resume)

There are two distinct behaviors:

- **Warm-start** (training phases): start a **new** run directory, optionally initializing from another run.
- **Resume** (training phases): continue the **same** run directory, including optimizer/scheduler/scaler state.

### Warm-start (pretrain / finetune)

Use `model_source.run_id` (run id under `runs/`) in the phase overrides:

```yaml
model_source:
  run_id: \"some_previous_run\"
  model_path: null
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

## 8) Embeddings and tuning

### Defaults vs overrides

- `common/embeddings.yaml` defines defaults per `(role, modality)`.
- `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` contains **only per-signal overrides**.

This file is **required** for `pretrain`, `finetune`, and `eval` for the selected embedding profile.
If you do not want task-specific embedding overrides yet, create an empty YAML file at that path.
(The profile is selected by the phase runners via `--emb_profile`; default: `dct3d`.)

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

## 9) Common gotchas

1) **Missing task-level embedding overrides file**

If you see an error like “Missing required task-level embedding overrides file”, create:

- `scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

It can be empty.

2) **Finetune/eval run id must be a run id, not a path**

Use:

```yaml
model_source:
  run_id: "tokamind_base_v1"   # ✅
  model_path: null
```

Not:

```yaml
model_source:
  run_id: "runs/tokamind_base_v1"   # ❌
  model_path: null
```

If checkpoints live outside `runs/`, set `model_source.model_path` to that external run folder; it overrides `run_id` for checkpoint loading.


3) **Eval/finetune requires the source run snapshot YAML**

If `runs/<source_run_id>/<source_run_id>.yaml` is missing, evaluation/finetune should fail.
Copying model knobs into eval config is intentionally avoided.

4) **Streaming windows without `loader.batches_per_epoch`**

If `data.cache.enable: false` (streaming), training must define `loader.batches_per_epoch`.

