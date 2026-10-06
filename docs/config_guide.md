# Configuration Guide

Related documentation: [Project README](../README.md) | [Configuration Reference](config_reference.md) | [DCT3D Tuning](tuning_dct3d.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md)

This project uses convention-based configuration for three phases:
- `pretrain`
- `finetune`
- `eval`

## Design Rules
- Each model profile is a self-contained folder `scripts_mast/configs/<model_profile>/` holding `phases/`, `tasks/`, and `embeddings/`.
- Keep phase defaults in `scripts_mast/configs/<model_profile>/phases/`.
- Keep task-specific phase changes in `scripts_mast/configs/<model_profile>/tasks/<phase>_tasks.yaml`. `tasks/` is **optional** — the loader treats a missing or empty `<phase>_tasks.yaml` as "no overrides".
- Keep representation choices in `scripts_mast/configs/<model_profile>/embeddings/`.
- Select model architecture from CLI for pretrain/finetune (`--model_profile mmt`).
- Eval inherits model architecture from the source run selected by `--model_source`.
- Select finetune init mode from CLI (`--init warmstart|scratch`).
- Select source model from CLI (`--model_source`) for eval and finetune warmstart.
- Store run-local tuned embedding artifacts in `runs/<run_id>/embeddings/`.

## Directory Layout
```text
scripts_mast/configs/
  mmt/                             # one self-contained folder per model profile
    phases/
      pretrain.yaml
      finetune_warmstart.yaml
      finetune_scratch.yaml
      eval.yaml
    tasks/
      pretrain_tasks.yaml          # tasks: {<task>: {...}}
      finetune_tasks.yaml          # tasks: {<task>: {...}}
      eval_tasks.yaml              # tasks: {<task>: {...}}
    embeddings/
      dct3d/
        _default.yaml
      vae/
        task_1-1.yaml
        task_1-2.yaml

  local_overrides.yaml             # gitignored; applied last
  local_tasks_def/                 # task definitions (model-agnostic)
```

## Entry Scripts
```bash
# Pretrain
python scripts_mast/run_pretrain.py \
  --task <task> \
  --model_profile mmt \
  --emb_profile dct3d \
  [--run-id <run_id>] [--tag <tag>]

# Finetune
python scripts_mast/run_finetune.py \
  --task <task> \
  --init <warmstart|scratch> \
  --model_profile mmt \
  --emb_profile dct3d \
  [--model_source <run_id_or_path>] \
  [--tag <tag>]

# Eval
python scripts_mast/run_eval.py \
  --task <task> \
  --model_source <run_id_or_path>
```

## Merge Order
For `pretrain` and `finetune`, merge order is:
1. `<model_profile>/embeddings/<profile>/_default.yaml` if present, otherwise `<model_profile>/embeddings/dct3d/_default.yaml`
2. `<model_profile>/phases/<phase>.yaml`, or `<model_profile>/phases/finetune_warmstart.yaml` / `<model_profile>/phases/finetune_scratch.yaml`
3. `<model_profile>/tasks/<phase>_tasks.yaml["tasks"][<task>]`
4. `<model_profile>/embeddings/<profile>/<task>.yaml` if present
5. `local_overrides.yaml` if present

Profiles without their own `_default.yaml`, such as task-only VAE profiles, fall back to `<model_profile>/embeddings/dct3d/_default.yaml` for baseline encoder defaults.

For `eval`, the model architecture is inherited from the source run selected by `--model_source`. Merge order is:
1. `<model_profile>/embeddings/<profile>/_default.yaml` if present, otherwise `<model_profile>/embeddings/dct3d/_default.yaml`
2. `<model_profile>/phases/eval.yaml`
3. `<model_profile>/tasks/eval_tasks.yaml["tasks"][<task>]`
4. source-run inheritance
5. `local_overrides.yaml` if present

Then the loader applies CLI overrides and phase-specific source/model rules.

## Embedding Profiles
Use `--emb_profile <profile>` to select `<model_profile>/embeddings/<profile>/`.

Typical profiles:
- `dct3d`: manual DCT3D per-signal overrides and tuning settings
- `vae`: task-specific VAE encoder assignments
- `identity`: **virtual profile** (no files on disk) — every signal uses the identity codec at full/raw dimension, with no tuning. It works without a per-model `embeddings/identity/` folder. Use it as a no-transform baseline or the "don't tune anything" mode. Per-signal overrides still apply on top.
  - Caveat: identity **output** signals skip the embedding step (no `output_emb`), so they must be supervised by a native loss (`native_sparse_mse`), not `embed_mse`. This is enforced at training setup: `resolve_loss_output_filters` raises a clear error if an identity output is included in an embedding-space term, or if any output ends up supervised by no capable term.

VAE task files are present only for tasks that actually have a VAE profile. DCT3D task files are optional; most tasks use `dct3d/_default.yaml` unchanged.
DCT3D role/modality bootstrap defaults are internal to the loader. User-facing DCT3D profile YAML contains only
manual per-signal overrides and tuning settings. Tuning settings merge `embeddings.dct3d.tuning.common` with
`embeddings.dct3d.tuning.<phase>`.

## DCT3D Policy
DCT3D policy is fixed by phase and init mode:

```text
pretrain            input=tune    actuator=tune    output=tune
finetune scratch    input=tune    actuator=tune    output=tune
finetune warmstart  input/actuator existing=source, missing=tune; output=tune
eval                inherit source run
```

DCT3D warmstart policy is signal-aware. A signal is `existing` when the source run has an entry for it in
`embeddings/dct3d.yaml`; otherwise it is `missing`. Explicit manual per-signal overrides are excluded from tune/source
policy and use the configured value directly:

```yaml
embeddings:
  dct3d:
    per_signal_overrides: {}
```

VAE profiles do not use DCT3D tune/source policy.

## Source Model Resolution
For `eval`, `--model_source` can be:
- a run id under `runs/`
- an absolute/relative path to an external run directory

For `finetune`:
- `--init warmstart` requires `--model_source` (run id or path)
- `--init scratch` does not use a source model

When a source model is used, the loader resolves and stores:
- `model_source.run_id` when applicable
- `model_source.model_path` when applicable
- `model_source.run_dir` as a resolved path

## Inheritance Rules
### Pretrain
- Uses `model`, `preprocess`, and selected embedding profile directly from merged config.
- DCT3D profiles tune all DCT3D signals except explicit manual per-signal overrides.
- `data.split` sets the split strategy (`random` or `temporal`, default `random`).

### Finetune Scratch
- `model = model_scratch`
- Uses current finetune preprocess settings.
- DCT3D profiles tune all DCT3D signals except explicit manual per-signal overrides.
- Stage schedule: single `ft_scratch` stage, all blocks trainable.

### Finetune Warmstart
- `model = deep_merge(source_model, model_overrides)`
- Uses current finetune `preprocess.chunk` and `preprocess.trim_chunks`, not source values.
- Inherits `data.split` from source run.
- DCT3D profiles inherit existing source input/actuator artifacts, tune missing input/actuator signals, and tune
  outputs. Explicit manual per-signal overrides use the configured value directly.
- Stage schedule: `ft_heads` then `ft_full`.

### Eval
- Inherits `model`, `embeddings`, `preprocess.chunk`, `preprocess.trim_chunks`, and `data.split` from source run config.
- Uses source run embedding artifacts for codec construction.
- Applies eval-only controls from merged `eval.*` settings.

## Run IDs and Output Paths
### Pretrain / Finetune
- Output root: `runs/<run_id>/`
- Config snapshot: `runs/<run_id>/<run_id>.yaml`

`run_id` generation:
- Pretrain: `--run-id`, else `<task>_<tag>`, else `<task>`
- Finetune warmstart: `ft-<task>-ws-<model_id>[-<tag>]`
- Finetune scratch: `ft-<task>-scratch[-<tag>]`

### Eval
- Output root: `runs/<model_id>/eval/`
- Config snapshot: `runs/<model_id>/eval/eval.yaml`

## Practical Checklist
1. Define phase defaults in `<model_profile>/phases/*.yaml`.
2. Add task-level runtime deltas under `<model_profile>/tasks/<phase>_tasks.yaml` only when needed.
3. Add `<model_profile>/embeddings/<profile>/<task>.yaml` only when that task has real profile-specific settings.
4. Run pretrain.
5. Run finetune:
   - warmstart: `--init warmstart --model_source <pretrain_run_id>`
   - scratch: `--init scratch`
6. Run eval with `--model_source <finetune_run_id>`.

## Snapshot Rule
Each run writes the fully merged config snapshot used at runtime.
- training: `runs/<run_id>/<run_id>.yaml`
- eval: `runs/<model_id>/eval/eval.yaml`

Use these snapshots for exact reproducibility and debugging.
