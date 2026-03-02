# Configuration Reference

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [DCT3D Tuning](tuning_dct3d.md) | [Evaluation](evaluation.md)

This page documents active configuration keys used by the entry scripts.

## Core Keys
### `seed`
- Type: `int`
- Used in: pretrain, finetune, eval
- Description: global random seed for deterministic setup.

### `task`
- Type: `str`
- Used in: pretrain, finetune, eval
- Description: task folder name under `scripts_mast/configs/tasks_overrides/`.
- Source: CLI `--task`.

### `phase`
- Type: `str`
- Values: `pretrain`, `finetune`, `eval`
- Description: execution phase selected by the loader.

### `runtime.debug_logging`
- Type: `bool`
- Description: enables verbose logs in entry scripts.

## Data
### `data.local`
- Type: `bool`
- Description: selects local benchmark data mode.

### `data.subset_of_shots`
- Type: `int | null`
- Description: limits number of shots for faster runs.

### `data.keep_output_native`
- Type: `bool`
- Description: keeps native output payload for metrics/traces.
- Requirement: should be `true` for eval.

### `data.cache.enable`
- Type: `bool`
- Description: enables RAM materialization of window dataset.

### `data.cache.dtype`
- Type: `"float16" | "float32" | null`
- Description: optional dtype cast for cached tensors.

### `data.cache.num_workers`
- Type: `int`
- Description: worker count for cache build.

### `data.cache.max_windows.train`
- Type: `int | null`
- Description: optional cap for train cached windows.

### `data.cache.max_windows.val`
- Type: `int | null`
- Description: optional cap for val cached windows.

## Preprocess
### `preprocess.chunk.chunk_length`
- Type: `float` (seconds)
- Description: chunk duration used for input/actuator history.

### `preprocess.chunk.stride`
- Type: `float | null` (seconds)
- Description: chunk step; `null` means chunk-length step in current flow.

### `preprocess.trim_chunks.max_chunks`
- Type: `int`
- Description: maximum number of history chunks kept per window.

### `preprocess.valid_windows.min_valid_inputs_actuators`
- Type: `int`
- Description: minimum valid input/actuator signals required.

### `preprocess.valid_windows.min_valid_outputs`
- Type: `int`
- Description: minimum valid output signals required.

### `preprocess.valid_windows.min_valid_chunks`
- Type: `int`
- Description: minimum valid chunks required in history.

### `preprocess.valid_windows.window_stride_sec`
- Type: `float | null`
- Description: optional temporal subsampling stride for windows.

## Embeddings
Top-level location: `embeddings:`

### `embeddings.defaults`
- Type: mapping
- Description: default encoder config by role and modality.

Example:
```yaml
embeddings:
  defaults:
    input:
      timeseries:
        encoder_name: dct3d
        encoder_kwargs: { keep_h: 1, keep_w: 1, keep_t: 10 }
```

### `embeddings.per_signal_overrides`
- Type: mapping
- Description: per-signal encoder overrides merged at runtime.
- Typical source: run-local tuned rank overrides.

### `embeddings.mode`
- Type: `"source" | "config"`
- Used in: finetune
- Description:
  - `source`: stage only task-used source DCT3D artifacts into the current run and optionally retune selected roles.
  - `config`: ignore source artifacts and use merged config directly.

### `embeddings.tune_embeddings.roles.input`
- Type: `bool`
- Description: tune/retune input role.

### `embeddings.tune_embeddings.roles.actuator`
- Type: `bool`
- Description: tune/retune actuator role.

### `embeddings.tune_embeddings.roles.output`
- Type: `bool`
- Description: tune/retune output role.

### `embeddings.tune_embeddings.n_shots`
- Type: `int`
- Description: shot sample size for DCT3D tuning.

### `embeddings.tune_embeddings.max_windows`
- Type: `int | null`
- Description: max streamed windows for tuning.

### `embeddings.tune_embeddings.objective.thresholds.{input,actuator,output}`
- Type: `float` in `(0, 1]`
- Description: explained-energy targets by role.

### `embeddings.tune_embeddings.objective.max_budget.{input,actuator,output}`
- Type: `int`
- Description: maximum selected coefficients per role (hard final cap).

### `embeddings.tune_embeddings.guardrails`
- Type: mapping
- Description: optional minimum-dimension coverage constraints.

### `embeddings.tune_embeddings.guardrails.enable`
- Type: `bool`
- Description: enable/disable guardrail lifting during rank tuning.

### `embeddings.tune_embeddings.guardrails.timeseries.min_unique_t`
- Type: `int`
- Description: minimum unique T indices required for timeseries signals.

### `embeddings.tune_embeddings.guardrails.profile.min_unique_h`
- Type: `int`
- Description: minimum unique H indices required for profile signals.

### `embeddings.tune_embeddings.guardrails.profile.min_unique_t`
- Type: `int`
- Description: minimum unique T indices required for profile signals.

### `embeddings.tune_embeddings.guardrails.video.min_unique_h`
- Type: `int`
- Description: minimum unique H indices required for video signals.

### `embeddings.tune_embeddings.guardrails.video.min_unique_w`
- Type: `int`
- Description: minimum unique W indices required for video signals.

### `embeddings.tune_embeddings.guardrails.video.min_unique_t`
- Type: `int`
- Description: minimum unique T indices required for video signals.

## Collate
Top-level location: `collate:`

### `collate.p_drop_inputs`
- Type: `float` in `[0, 1]`
- Description: base probability of dropping each input signal token.

### `collate.p_drop_actuators`
- Type: `float` in `[0, 1]`
- Description: base probability of dropping each actuator signal token.

### `collate.p_drop_outputs`
- Type: `float` in `[0, 1]`
- Description: base probability of dropping each output signal token.

### `collate.p_drop_inputs_chunks`
- Type: `float` in `[0, 1]`
- Description: probability of dropping full input history chunks.

### `collate.p_drop_actuators_chunks`
- Type: `float` in `[0, 1]`
- Description: probability of dropping full actuator history chunks.

### `collate.p_drop_inputs_overrides`
- Type: `mapping[str, float]`
- Description: per-input signal drop probabilities overriding base value.

### `collate.p_drop_actuators_overrides`
- Type: `mapping[str, float]`
- Description: per-actuator signal drop probabilities overriding base value.

### `collate.p_drop_outputs_overrides`
- Type: `mapping[str, float]`
- Description: per-output signal drop probabilities overriding base value.

Description: collate drop settings are used for regularization and controlled eval ablations.

## Loader
Top-level location: `loader:`

### `loader.batch_size`
- Type: `int`
- Description: number of windows per batch.

### `loader.num_workers`
- Type: `int`
- Description: DataLoader worker count.

### `loader.shuffle_train`
- Type: `bool`
- Description: enables train-shuffle behavior.

### `loader.drop_last`
- Type: `bool`
- Description: drops incomplete last batch when `true`.

### `loader.batches_per_epoch`
- Type: `int | null`
- Description: optional cap for streaming-epoch batch count.

## Finetune Model Configuration
Top-level locations in `common/finetune.yaml`:
- `model_scratch`
- `finetune_model_overrides`
- `warmstart.model_overrides`

### `model_scratch`
- Type: mapping
- Description: scratch-only base model architecture.

### `finetune_model_overrides`
- Type: mapping
- Description: model overrides applied in both finetune modes.

### `warmstart.model_overrides`
- Type: mapping
- Description: warmstart-only model overrides applied on top of source model.

Finetune model materialization:
- `--init scratch`: `model = deep_merge(model_scratch, finetune_model_overrides)`
- `--init warmstart`: `model = deep_merge(source_model, finetune_model_overrides, warmstart.model_overrides)`

## Runtime Model
Top-level location in runtime config snapshot: `model:`

### `model.backbone.d_model`
- Type: `int`
- Description: shared token hidden dimension.

### `model.backbone.n_layers`
- Type: `int`
- Description: number of transformer layers.

### `model.backbone.n_heads`
- Type: `int`
- Description: attention heads per layer.

### `model.backbone.dim_ff`
- Type: `int`
- Description: feed-forward hidden size.

### `model.backbone.dropout`
- Type: `float`
- Description: dropout probability in backbone blocks.

### `model.backbone.activation`
- Type: `str`
- Description: feed-forward activation name.

### `model.modality_heads`
- Type: mapping
- Description: per-modality intermediate head sizes.

### `model.output_adapters.hidden_dim.default`
- Type: `int | "d_model"`
- Description: default hidden size for output adapters.

### `model.output_adapters.hidden_dim.bucketed.enable`
- Type: `bool`
- Description: enables bucket-based hidden size selection.

### `model.output_adapters.hidden_dim.bucketed.rules`
- Type: list
- Description: bucket rules by output dimension threshold.

### `model.output_adapters.hidden_dim.manual`
- Type: mapping
- Description: explicit per-output hidden sizes; overrides bucket/default.

## Training
Top-level location: `train:`

### `train.resume`
- Type: `bool`
- Description: strict resume of same run directory from `checkpoints/latest`.

### `train.early_stop.patience`
- Type: `int`
- Description: number of non-improving validations before stop.

### `train.early_stop.delta`
- Type: `float`
- Description: minimum improvement threshold.

### `train.amp.enable`
- Type: `bool`
- Description: toggles autocast mixed precision.

### `train.loss.output_weights`
- Type: mapping
- Description: per-output loss weighting.

### `train.optimizer.use_adamw`
- Type: `bool`
- Description: optimizer selector.

### `train.stages[]`
Each stage configures one training segment.

Stage keys:
- `name`: stage label for logs and history.
- `epochs`: epochs in this stage.
- `scheduler.grad_accum_steps`: gradient accumulation factor.
- `scheduler.warmup_steps_fraction`: warmup fraction (optional).
- `optimizer.lr.*`: learning rates per block.
- `optimizer.wd.*`: weight decay per block.
- `freeze.*`: block freeze flags.

## Model Source
Top-level location: `model_source:`

### `model_source.run_id`
- Type: `str | null`
- Description: source run identifier when selected by run id.

### `model_source.model_path`
- Type: `str | null`
- Description: source run directory path when selected by path.

### `model_source.run_dir`
- Type: `str`
- Description: resolved absolute source run directory.

Mode notes:
- finetune warmstart: `model_source` is set from CLI `--model`.
- finetune scratch: `model_source` is `null`.
- eval: `model_source` is required.

### `model_source.load_parts.*`
- Type: `bool`
- Keys: `token_encoder`, `backbone`, `modality_heads`, `output_adapters`
- Description: block-level warmstart load filter.
- Used in: finetune warmstart.

## Evaluation
Top-level location: `eval:`

### `eval.amp.enable`
- Type: `bool`
- Description: toggles mixed precision during eval forward.

### `eval.drop.inputs`
- Type: `list[str] | null`
- Description: input signals force-dropped during eval.

### `eval.drop.actuators`
- Type: `list[str] | null`
- Description: actuator signals force-dropped during eval.

### `eval.drop.outputs`
- Type: `list[str] | null`
- Description: output signals excluded from scoring/traces.

### `eval.compute_metrics.per_task`
- Type: `bool`
- Description: writes task-level benchmark metrics.

### `eval.compute_metrics.per_window`
- Type: `bool`
- Description: writes per-window benchmark metrics.

### `eval.compute_metrics.per_timestamp`
- Type: `bool`
- Description: writes per-timestamp diagnostic CSV.

### `eval.traces.enable`
- Type: `bool`
- Description: enables trace export.

### `eval.traces.n_max`
- Type: `int`
- Description: maximum number of shots traced.

### `eval.traces.signals`
- Type: `list[str] | null`
- Description: output-signal whitelist for traces.

### `eval.traces.times_indexes`
- Type: `list[int] | null`
- Description: time-index subset for trace export.

## Paths Written by Loader
### Pretrain and Finetune
- `paths.run_dir = runs/<run_id>`
- Config snapshot: `runs/<run_id>/<run_id>.yaml`

### Eval
- `paths.model_run_dir = runs/<model_id>` (or external path)
- `paths.run_dir = <model_run_dir>/eval`
- Config snapshot: `<model_run_dir>/eval/eval.yaml`

## Task Files
Per task under `scripts_mast/configs/tasks_overrides/<task>/`:
- Optional: `pretrain_overrides.yaml`, `finetune_overrides.yaml`, `eval_overrides.yaml`
- Required for pretrain/finetune: `embeddings_overrides/<profile>.yaml`

## Quick Validation Checklist
1. Phase is one of `pretrain`, `finetune`, `eval`.
2. Task embedding profile file exists for pretrain/finetune.
3. Finetune/eval include `--model` on CLI.
4. Eval keeps `data.keep_output_native: true`.
5. Finetune with `embeddings.mode=config` sets all `tune_embeddings.roles` to `false`.
