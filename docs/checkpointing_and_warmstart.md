# Checkpointing and Warmstart

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [Model Flexibility](model_flexibility.md) | [Evaluation](evaluation.md)

This document covers checkpoint layout, strict resume, and warmstart loading.

## Checkpoint Layout
Each training run writes checkpoints under:

```text
runs/<run_id>/
  checkpoints/
    latest/
    best/
```

### `checkpoints/latest`
Contains full training state for strict resume:
- model blocks (`token_encoder.pt`, `backbone.pt`, `modality_heads.pt`, `output_adapters.pt`)
- optimizer/scheduler/scaler/rng state
- metadata (`meta.json`)

`meta.json` tracks run progress (for example: epoch/step and best metric state).

### `checkpoints/best`
Contains best validation model state:
- model blocks
- metadata (`meta.json`)

This is the default candidate for eval loading.

## Strict Resume
Use strict resume to continue the same run directory.

Config:
```yaml
train:
  resume: true
```

Behavior:
- restores model blocks
- restores optimizer/scheduler/scaler/rng state
- continues epoch and global step from checkpoint metadata

Constraint:
- only valid for the same run directory.

## Warmstart
Use warmstart to initialize weights from another run while starting a separate run directory.

### Source selection
Finetune and eval source selection is CLI-based:

```bash
python scripts_mast/run_finetune.py --task <task> --init warmstart --model <run_id_or_path>
python scripts_mast/run_eval.py --task <task> --model <run_id_or_path>
```

The loader resolves source path into `model_source.run_dir`.

Finetune scratch mode does not use a source model:

```bash
python scripts_mast/run_finetune.py --task <task> --init scratch
```

### Warmstart matching rule
Warmstart loads tensors by:
- state key
- tensor shape

Matching entries are loaded. Non-matching entries keep initialization from current config.

This avoids brittle manual remapping when signal sets differ between runs.

### Block-level loading control
Optional in finetune config:

```yaml
model_source:
  load_parts:
    token_encoder: true
    backbone: true
    modality_heads: true
    output_adapters: false
```

## Resume vs Warmstart
- Resume: continue same run, includes optimizer/scheduler/scaler/rng.
- Warmstart: initialize model blocks from another run, excludes optimizer/scheduler/scaler/rng.
- Resume keeps training history continuity; warmstart starts a fresh optimization history.
- Scratch finetune: initialize model blocks from finetune config without source weights.

## Typical Sequences
### Pretrain -> Finetune
1. Run pretrain.
2. Run finetune:
   - warmstart: `--init warmstart --model <pretrain_run_id>`
   - scratch: `--init scratch`

### Finetune -> Eval
1. Run eval with `--model <finetune_run_id>`.
2. Eval loads checkpoint from source run and writes outputs under `runs/<model_id>/eval/`.

## Safe Model Changes with Warmstart
Warmstart supports evolving task specs between runs:
- adding/removing signals
- changing per-signal embedding dims
- changing output adapter set

Effects:
- matching parameters are reused
- unmatched parameters remain initialized

See [Model Flexibility](model_flexibility.md) for details.
