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

In the config system, the source run is specified by:

```yaml
model_source:
  run_dir: "runs/some_previous_run"
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: true
```

> Legacy naming: some configs may still call this `model_init.model_dir`.  
> The semantics are the same: it is the **source run directory** for loading weights.

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

### 1) Finetune from scratch (no warm-start)
Set the source run to null / omit it:

```yaml
model_source:
  run_dir: null
```

(or delete the block entirely if your loader supports that).

### 2) Finetune from a pretrained run (common)
```yaml
model_source:
  run_dir: "runs/pretrain_some_task"
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: true
```

### 3) Warm-start backbone only (re-learn token projections / heads)
```yaml
model_source:
  run_dir: "runs/pretrain_some_task"
  load_parts:
    token_encoder: false
    backbone: true
    modality_heads: false
    output_adapters: false
```

---

## Where to set warm-start in configs

By convention:

- Put warm-start defaults in `common/finetune.yaml` or `common/pretrain.yaml` (as `null`)
- Set task/run-specific warm-start sources in:
  - `tasks/<task>/finetune_overrides.yaml`
  - `tasks/<task>/pretrain_overrides.yaml`

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
- `model_source.run_dir=...` to initialize a new run
