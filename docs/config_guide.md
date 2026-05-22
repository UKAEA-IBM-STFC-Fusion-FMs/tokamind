# Configuration Guide

Related documentation: [Project README](../README.md) | [Configuration Reference](config_reference.md) | [DCT3D Tuning](tuning_dct3d.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md)

This project uses convention-based configuration for three phases:
- `pretrain`
- `finetune`
- `eval`

## Design Rules
- Keep base defaults in `scripts_mast/configs/common/`.
- Keep task-specific changes in `scripts_mast/configs/tasks_overrides/<task>/`.
- Select finetune init mode from CLI (`--init warmstart|scratch`).
- Select source model from CLI (`--model`) for eval and for finetune warmstart.
- Store run-local tuned embedding artifacts in `runs/<run_id>/embeddings/`.
- Keep task profile files minimal: only task/profile-specific differences.

## Directory Layout
```text
scripts_mast/configs/
  common/
    embeddings.yaml
    pretrain.yaml
    finetune.yaml              # shared finetune data/preprocess/collate/loader defaults
    finetune_warmstart.yaml    # warmstart embeddings/train/model recipe + model_source
    finetune_scratch.yaml      # scratch embeddings/train/model recipe
    eval.yaml

  tasks_overrides/
    <task>/
      pretrain_overrides.yaml                 # optional
      finetune_overrides.yaml                 # optional
      eval_overrides.yaml                     # optional
      embeddings_overrides/
        dct3d.yaml                            # profile-specific overrides (can be comment-only)
        vae.yaml                              # optional alternative profile
```

Loader modules:
- `scripts_mast/mast_utils/config/loader.py` (orchestration)
- `scripts_mast/mast_utils/config/merge.py` (YAML load + deep merge)
- `scripts_mast/mast_utils/config/cli_overrides.py` (CLI injection)
- `scripts_mast/mast_utils/config/inheritance.py` (source inheritance + finetune model semantics)
- `scripts_mast/mast_utils/config/finalize.py` (path resolution + snapshot write)
- `scripts_mast/mast_utils/config/ids.py` (run-id/model-id naming)

## Entry Scripts
```bash
# Pretrain
python scripts_mast/run_pretrain.py \
  --task <task> \
  --emb_profile dct3d \
  [--run-id <run_id>] [--tag <tag>]

# Finetune
python scripts_mast/run_finetune.py \
  --task <task> \
  --init <warmstart|scratch> \
  --emb_profile dct3d \
  [--model <run_id_or_path>] \
  [--tag <tag>]

# Eval
python scripts_mast/run_eval.py \
  --task <task> \
  --model <run_id_or_path>
```

## Merge Order
For `pretrain`, merge order is:
1. `common/embeddings.yaml` (shared embedding defaults + tuning objective)
2. `common/pretrain.yaml` (pretrain runtime defaults)
3. `tasks_overrides/<task>/pretrain_overrides.yaml` (task-specific edits, optional)
4. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` (task/profile embedding edits, required)

For `finetune`, merge order is:
1. `common/embeddings.yaml` (shared embedding defaults + tuning objective)
2. `common/finetune.yaml` (shared finetune data/preprocess/collate/loader defaults)
3. `common/finetune_warmstart.yaml` **or** `common/finetune_scratch.yaml` (init-mode-specific embeddings/train/model recipe, selected automatically from `--init`)
4. `tasks_overrides/<task>/finetune_overrides.yaml` (task-specific edits, optional)
5. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` (task/profile embedding edits, required)

For `eval`, merge order is:
1. `common/embeddings.yaml` (base embedding defaults)
2. `common/eval.yaml` (eval runtime defaults)
3. `tasks_overrides/<task>/eval_overrides.yaml` (task eval edits, optional)

Then the loader applies phase-specific source/model rules based on `phase` and finetune `--init`.

## Source Model Resolution
For `eval`, `--model` can be:
- a run id under `runs/`
- an absolute/relative path to an external run directory

For `finetune`:
- `--init warmstart` requires `--model` (run id or path)
- `--init scratch` does not use a source model

When a source model is used, the loader resolves and stores:
- `model_source.run_id` (when applicable)
- `model_source.model_path` (when applicable)
- `model_source.run_dir` (resolved path)

## Inheritance Rules by Phase
### Pretrain
- Uses `model` and `preprocess` directly from merged config.
- Builds embeddings according to profile + runtime tuning settings.
- `data.split` sets the split strategy (`random` or `temporal`, default `random`).

### Finetune
Embedding/train/model configuration is owned by the init-mode files:
- `embeddings.role_mode` and finetune `embeddings.tuning` — defined in `finetune_scratch.yaml` or `finetune_warmstart.yaml`
- `model_scratch` — defined in `finetune_scratch.yaml`: complete scratch model architecture
- `model_overrides` — defined in `finetune_warmstart.yaml`: overrides applied on top of the source model
- `train` — defined in `finetune_warmstart.yaml` or `finetune_scratch.yaml`: optimizer, loss, AMP, early stop, and stage schedule

If `--init scratch`:
- `model = model_scratch`
- `preprocess.chunk` and `preprocess.trim_chunks` are used from merged finetune config.
- `data.split` is used directly from config.
- Stage schedule: single `ft_scratch` stage, all blocks trainable.

If `--init warmstart`:
- `model = deep_merge(source_model, model_overrides)`
- `preprocess.chunk` and `preprocess.trim_chunks` are used from the current merged finetune config, not inherited from the source run.
- `data.split` is inherited from source run. If the finetune config specifies a different split, a warning is logged and the source split is enforced.
- Stage schedule: `ft_heads` (backbone + token_encoder frozen) → `ft_full` (all blocks trainable).
- Embeddings are resolved by `embeddings.role_mode` per role:
  - `tune`: tune DCT3D coefficients in the current run
  - `source`: stage task-used source DCT3D artifacts and inherit that role
  - `config`: ignore source artifacts for that role and use merged config/profile defaults

### Eval
- Inherits `model`, `embeddings`, `preprocess.chunk`, `preprocess.trim_chunks`, and `data.split` from source run config.
- `data.split` must not be set in the eval config — it is always source-authoritative.
- Uses source run embedding artifacts for codec construction.
- Applies eval-only controls from merged `eval.*` settings (drops, metrics, traces, amp).

## Embedding Profiles
Use `--emb_profile <profile>` to select:
- `scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

Typical profiles:
- `dct3d`
- `vae`

See [DCT3D Tuning](tuning_dct3d.md) for role-based tuning behavior.

## Run IDs and Output Paths
### Pretrain / Finetune
- Output root: `runs/<run_id>/`
- Config snapshot: `runs/<run_id>/<run_id>.yaml`

`run_id` generation:
- Pretrain:
  - `--run-id` if provided
  - else `<task>_<tag>` if `--tag` provided
  - else `<task>`
- Finetune:
  - warmstart: `ft-<task>-ws-<model_id>-<tag>` if tag provided, else `ft-<task>-ws-<model_id>`
  - scratch: `ft-<task>-scratch-<tag>` if tag provided, else `ft-<task>-scratch`

### Eval
- Output root: `runs/<model_id>/eval/`
- Config snapshot: `runs/<model_id>/eval/eval.yaml`

## Required Task Files
For each task used in pretrain/finetune, keep:
- `scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`

Optional task files:
- `<phase>_overrides.yaml`

## Practical Checklist
1. Define `common/*.yaml` defaults.
2. Add task folder under `tasks_overrides/`.
3. Add `embeddings_overrides/<profile>.yaml`.
4. Run pretrain.
5. Run finetune:
   - warmstart: `--init warmstart --model <pretrain_run_id>`
   - scratch: `--init scratch`
6. Run eval with `--model <finetune_run_id>`.

For parameter-level details, use [Configuration Reference](config_reference.md).

## Snapshot Rule
Each run writes the fully merged config snapshot used at runtime.
- training: `runs/<run_id>/<run_id>.yaml`
- eval: `runs/<model_id>/eval/eval.yaml`

Use these snapshots for exact reproducibility and debugging.
