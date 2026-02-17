# Configuration Guide

This repository uses a **convention-based configuration system with CLI-based model selection**.

- The core Python package (`src/mmt/`) is dataset-agnostic.
- Dataset/task integration lives under `scripts_mast/`.
- Runs are configured by selecting a **task** and a **phase**; the loader finds and merges YAML files by convention.
- **Model sources are specified via CLI arguments** (not in YAML configs).

The goals are:

- **Explicit phase configs** (no hidden globals)
- **Small per-task overrides**
- **CLI-based model selection** (no manual config editing)
- **Auto-generated run IDs** (consistent naming conventions)
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

- **finetune** requires `--model <run_id_or_path>` CLI argument and **inherits `model` + `preprocess.{chunk, trim_chunks}`** from the source run config snapshot.
  - Auto-generates run_id as: `ft-{task}-{tag}-{model_id}` (or `ft-{task}-{model_id}` if no `--tag`)
  - Finetune-side YAML should focus on *training settings* (stages, LR/WD, freezing, window selection, etc.)
  - No need to edit YAML configs for model selection

- **eval** requires `--model <run_id_or_path>` CLI argument and **rebuilds `model`, `embeddings`, and `preprocess.{chunk, trim_chunks}`** from the source run config snapshot.
  - Auto-generates eval_id and saves results in `runs/{model}/eval/`
  - Eval-side YAML should focus on *metrics/traces* and *evaluation window selection*
  - No need to edit YAML configs for model selection

This design prevents "config drift" (e.g., changing adapter sizing in finetune but forgetting to mirror it in eval) and eliminates manual config editing for model selection.

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
    dct3d_indices/                   # shared coefficient indices (input/actuator)
      input_*.npy                    # referenced via @common/ prefix
      actuator_*.npy

  tasks_overrides/
    <task>/
      pretrain_overrides.yaml        # optional
      finetune_overrides.yaml        # optional
      eval_overrides.yaml            # optional
      tune_dct3d_overrides.yaml      # optional (rare)
      embeddings_overrides/          # task-level embedding overrides (selected by profile)
        <profile>.yaml               # required for pretrain/finetune/eval (can be empty)
                                     # (DCT3D tuning writes to: dct3d.yaml by default)
        dct3d_indices/               # task-specific coefficient indices (output)
          output_*.npy               # referenced via relative paths
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
# Pretrain (supports --run-id or --tag for naming)
python scripts_mast/run_pretrain.py --task <task> [--run-id <name>] [--tag <version>]

# Finetune (requires --model, optional --tag)
python scripts_mast/run_finetune.py --task <task> --model <source_model> [--tag <experiment_tag>]

# Eval (requires --model)
python scripts_mast/run_eval.py --task <task> --model <model_to_evaluate>

# Tune DCT3D embeddings
python scripts_mast/run_tune_dct3d.py --task <task>
```

**Examples:**
```bash
# Pretrain foundation model with explicit name
python scripts_mast/run_pretrain.py --task pretrain_inputs_actuators_to_inputs_outputs --run-id tokamind_v1

# Pretrain task-specific model with tag (creates: task_1-1_small_v2)
python scripts_mast/run_pretrain.py --task task_1-1 --tag small_v2

# Pretrain with default naming (uses task name)
python scripts_mast/run_pretrain.py --task task_1-1

# Finetune from pretrained model
python scripts_mast/run_finetune.py --task task_2-1 --model tokamind_v1

# Finetune with experiment tag
python scripts_mast/run_finetune.py --task task_2-1 --model tokamind_v1 --tag lr1e-4

# Evaluate a finetuned model
python scripts_mast/run_eval.py --task task_2-1 --model ft-task_2-1-lr1e-4-tokamind_v1
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
Those are inherited from the source model specified via `--model` CLI argument.

**Do not set `model_source` or `run_id` in finetune.yaml**.
These are auto-generated from CLI arguments.

### `common/eval.yaml`

Evaluation defaults:

- **Run context**: `seed`, `runtime.*`, `data.local`, `data.subset_of_shots`
- `data.keep_output_native: true` (required for metrics/traces)
- `loader.*` for eval
- `eval.metrics.*` and `eval.traces.*`
- `preprocess.valid_windows` *(allowed to differ by phase)*

**The model to evaluate is specified via `--model` CLI argument**, not in YAML configs.

**Do not set `model_source` or `eval_id` in eval.yaml**.
These are auto-generated from CLI arguments.

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

- Phase-specific `preprocess.valid_windows.window_stride_sec`
- Task-specific `collate` drop probabilities
- Task-specific training settings (learning rates, batch sizes, etc.)

**What NOT to include:**
- ❌ `run_id` / `eval_id` (auto-generated from CLI)
- ❌ `model_source.run_id` / `model_source.model_path` (specified via `--model` CLI argument)

If you want a task-wide change to apply to multiple phases, copy it into each relevant `<phase>_overrides.yaml`.

---

## 6) Run IDs and output locations

### Auto-generated run IDs

**Pretrain:**
- Specified via CLI arguments with flexible naming:
  - `--run-id <name>`: Explicit name (e.g., `tokamind_v1`)
  - `--tag <version>`: Generates `{task}_{tag}` (e.g., `task_1-1_small_v2`)
  - No arguments: Uses task name as run_id (e.g., `task_1-1`)
- **Priority:** `--run-id` > `--tag` > task name

**Finetune:**
- Auto-generated from CLI arguments: `ft-{task}-{tag}-{model_id}`
- If no `--tag`: `ft-{task}-{model_id}`
- Examples:
  - `ft-task_2-1-experiment1-tokamind_v1`
  - `ft-task_2-1-tokamind_v1`

**Eval:**
- Auto-generated based on model being evaluated
- Saves in `runs/{model}/eval/` directory

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

**For finetune**, use the `--model` CLI argument:

```bash
python scripts_mast/run_finetune.py --task task_2-1 --model pretrain_base_v1
```

**For pretrain** (optional warm-start from another pretrain run), set in `pretrain_overrides.yaml`:

```yaml
model_source:
  run_id: "some_previous_pretrain_run"
  model_path: null  # or path to external checkpoint directory
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

If you see an error like "Missing required task-level embedding overrides file", create:

- `scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

It can be empty.

2) **Forgot to specify --model for finetune/eval**

Finetune and eval require the `--model` CLI argument:

```bash
# ✅ Correct
python scripts_mast/run_finetune.py --task task_2-1 --model pretrain_base_v1

# ❌ Wrong - missing --model
python scripts_mast/run_finetune.py --task task_2-1
```

3) **Model argument is a run_id, not a path**

Use the run_id directly:

```bash
# ✅ Correct
--model tokamind_base_v1

# ❌ Wrong - don't include runs/ prefix
--model runs/tokamind_base_v1
```

If checkpoints live outside `runs/`, you can pass the full path:

```bash
--model /path/to/external/checkpoint/dir
```

4) **Eval/finetune requires the source run snapshot YAML**

If `runs/<source_run_id>/<source_run_id>.yaml` is missing, evaluation/finetune will fail.
The system intentionally avoids copying model knobs into eval config to prevent drift.

5) **Streaming windows without `loader.batches_per_epoch`**

If `data.cache.enable: false` (streaming), training must define `loader.batches_per_epoch`.

6) **Don't set model_source or run_id in YAML configs**

These are now specified via CLI arguments. Remove them from your task override files:

```yaml
# ❌ Don't do this anymore (finetune/eval)
model_source:
  run_id: "some_model"

# ❌ Don't do this anymore (pretrain/finetune)
run_id: "my_custom_id"
```

**Instead, use CLI arguments:**
```bash
# Pretrain: use --run-id or --tag
python run_pretrain.py --task task_1-1 --run-id my_model_v1

# Finetune/Eval: use --model
python run_finetune.py --task task_1-1 --model source_model
```

