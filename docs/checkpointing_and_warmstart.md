# Checkpointing, resume, and warm-start

MMT is designed so that you can safely:

- resume an interrupted run (**strict resume**),
- initialize a new run from a previous one (**warm-start / overlap load**),
- change the set of inputs/outputs between runs (within the limits described below).

This is possible because model parameters are organized into **four independent blocks** and because
warm-start loading is **key+shape based**.

---

## Model blocks saved to disk

The model is treated as four separately saved blocks:

1. **TokenEncoder** (`token_encoder.pt`)
2. **Backbone** (`backbone.pt`)
3. **Modality heads** (`modality_heads.pt`)
4. **Output adapters** (`output_adapters.pt`)

This aligns with the model architecture described in `docs/model_architecture.md`.

---

## Checkpoint directory layout

All checkpoints live under a run directory:

```
runs/<run_id>/
  checkpoints/
    latest/
    best/
```

### `checkpoints/latest/` (strict resume point)

`latest/` is a complete resume snapshot. It typically contains:

- `token_encoder.pt`
- `backbone.pt`
- `modality_heads.pt`
- `output_adapters.pt`
- `optimizer.pt`
- `scheduler.pt`
- `scaler.pt` (AMP)
- `rng.pt` (random states)
- `meta.json` (epoch, global_step, best_val_so_far, etc.)

### `checkpoints/best/` (best validation snapshot)

`best/` is the best model snapshot (for evaluation / warm-start). It contains:

- `token_encoder.pt`
- `backbone.pt`
- `modality_heads.pt`
- `output_adapters.pt`
- `meta.json` (epoch_best, best_val, etc.)

When both exist, evaluation prefers `best/` and falls back to `latest/`.

---

## Strict resume (continue the *same* run)

**Use strict resume when:**
- a run was interrupted and you want to continue it,
- you want to keep optimizer/scheduler/scaler/RNG states.

**What strict resume restores**
- all 4 model blocks
- optimizer state
- scheduler state
- AMP scaler state (if enabled)
- RNG state
- epoch/global_step bookkeeping

**Config**
Strict resume is usually controlled by:

```yaml
train:
  resume: true
```

**Important**
- Resume assumes you are continuing the **same run_dir** (same `run_id`).
- Resume is not compatible with warm-starting from another run.

---

## Warm-start (initialize a *new* run from an old run)

Warm-start is used to initialize a new run from an existing run directory, **without**
restoring optimizer/scheduler/RNG.

This is the recommended workflow for:
- pretrain → finetune
- finetune on task A → finetune on task B
- adding/removing signals between runs

### Config key: `model_source` (recommended)

**For finetune and eval**, the source model is specified via CLI:

```bash
# Finetune from a pretrained model
python scripts_mast/run_finetune.py --task task_2-1 --model pretrain_base_v1

# Evaluate a model
python scripts_mast/run_eval.py --task task_2-1 --model ft-task_2-1-exp1-pretrain_base_v1
```

The `--model` argument accepts either:
- A run_id (folder name under `runs/`)
- An absolute path to an external checkpoint directory

**For pretrain** (optional warm-start), set in `pretrain_overrides.yaml`:

```yaml
model_source:
  run_id: "some_previous_run"
  model_path: null   # if set, overrides run_id
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: true
```

### What warm-start does
- picks `run_dir/checkpoints/best/` if it exists, else `run_dir/checkpoints/latest/`
- loads selected blocks (`load_parts`)
- **overlap-loads** tensors whose *(key, shape)* match between checkpoint and current model
- leaves everything else at its normal initialization

Warm-start does **not** restore:
- optimizer
- scheduler
- AMP scaler
- RNG
- epoch counters

---

## Overlap loading rules (key + shape)

Warm-start is intentionally conservative:

- If a parameter key is missing in the current model → ignored
- If a parameter exists but the tensor **shape differs** → ignored
- Only parameters with identical key and shape are loaded

This makes the system robust to:

- adding new inputs (new TokenEncoder projections)
- removing inputs
- adding new outputs (new output adapters)
- removing outputs
- changing encoder embedding dims for specific signals (reinitializes the affected projection/head)

You still **cannot** warm-start across structural changes that alter the core shapes, such as:
- changing `d_model`
- changing transformer depth/heads in a way that changes backbone parameter shapes

---

## Practical recipes

### 1) Finetune requires a source model
Finetune always requires the `--model` CLI argument to specify the source model.
If you want to train a model from scratch, use `pretrain`.

```bash
# Finetune from a pretrained model
python scripts_mast/run_finetune.py --task task_2-1 --model pretrain_base_v1

# With experiment tag for versioning
python scripts_mast/run_finetune.py --task task_2-1 --model pretrain_base_v1 --tag experiment1
```

### 2) Pretrain with warm-start (optional)
For pretrain, you can optionally warm-start from another pretrain run by setting in `pretrain_overrides.yaml`:

```yaml
model_source:
  run_id: "pretrain_some_task"
  model_path: null   # if set, overrides run_id
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: true
```

### 3) Warm-start backbone only (re-learn token projections / heads)
```yaml
model_source:
  run_id: "pretrain_some_task"
  model_path: null
  load_parts:
    token_encoder: false
    backbone: true
    modality_heads: false
    output_adapters: false
```

---

## Where to set warm-start in configs

By convention:

- **Finetune**: Specify source model via `--model` CLI argument (not in YAML)
- **Pretrain** (optional warm-start): Set `model_source` in `tasks_overrides/<task>/pretrain_overrides.yaml`
- Put warm-start defaults in `common/pretrain.yaml` (as `null`)

This keeps `common/*` task-agnostic and makes runs reproducible.

---

## Troubleshooting

### “Missing token_encoder.pt / backbone.pt / ...”
Warm-start expects a run directory containing `checkpoints/best/` or `checkpoints/latest/`
with the relevant `*.pt` files. Verify the source run completed at least one checkpoint save.

### “Warm-start loads fewer params than expected”
This is usually correct: it means keys or shapes don’t match (e.g., you changed an embedding dim,
added/removed signals, or changed model hyperparameters).

Check logs: warm-start reports counts per block and often per-signal details.

### “Resume and warm-start conflict”
Use exactly one:
- `train.resume=true` to continue the same run
- `model_source.run_id=...` to initialize a new run
