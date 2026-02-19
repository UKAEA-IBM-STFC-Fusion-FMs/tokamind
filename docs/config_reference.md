# Configuration Reference

This document provides a comprehensive reference for all configuration parameters used in the MMT training system. Parameters are organized by category for easy navigation.

For information about the configuration system architecture and merge order, see [`config_guide.md`](config_guide.md).

---

## Table of Contents

1. [Core Parameters](#core-parameters)
2. [Runtime Configuration](#runtime-configuration)
3. [Data Configuration](#data-configuration)
4. [Preprocessing Configuration](#preprocessing-configuration)
5. [Model Configuration](#model-configuration)
6. [Training Configuration](#training-configuration)
7. [Evaluation Configuration](#evaluation-configuration)
8. [Embedding Configuration](#embedding-configuration)
9. [Collate Configuration](#collate-configuration)
10. [Loader Configuration](#loader-configuration)
11. [Model Source Configuration](#model-source-configuration)
12. [DCT3D Tuning Configuration](#dct3d-tuning-configuration)
13. [Path Configuration](#path-configuration)

---

## Core Parameters

These parameters are required in every phase configuration and define the basic execution context.

### `seed`

**Type:** `int`  
**Required:** Yes (all phases)  
**Default:** `54`

Random seed for reproducibility. Controls PyTorch, NumPy, and Python random number generators.

**Example:**
```yaml
seed: 42
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `task`

**Type:** `str`  
**Required:** Yes (set automatically by loader)  
**Default:** None

Task identifier corresponding to a folder under [`scripts_mast/configs/tasks_overrides/<task>/`](../scripts_mast/configs/tasks_overrides/).

**Example:**
```yaml
task: "task_1-1"
```

**Note:** This is automatically set by the config loader based on the `--task` CLI argument and should not be manually specified in YAML files.

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `phase`

**Type:** `str`  
**Required:** Yes (set automatically by loader)  
**Default:** None  
**Valid values:** `pretrain`, `finetune`, `eval`, `tune_dct3d`

Execution phase identifier.

**Example:**
```yaml
phase: "pretrain"
```

**Note:** This is automatically set by the config loader and should not be manually specified in YAML files.

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `run_id`

**Type:** `str`
**Required:** No (auto-generated from CLI)
**CLI Arguments:** `--run-id <name>` or `--tag <version>` (pretrain only)

Identifier for the training run. Determines the output directory under [`runs/<run_id>/`](../runs/).

**Pretrain:** Specified via CLI with flexible naming:
- `--run-id <name>`: Explicit name (e.g., `tokamind_v1`)
- `--tag <version>`: Generates `{task}_{tag}` (e.g., `task_1-1_small_v2`)
- No arguments: Uses task name (e.g., `task_1-1`)

**Finetune:** Auto-generated as `ft-{task}-{tag}-{model_id}` (or `ft-{task}-{model_id}` if no `--tag`)

**Examples:**
```bash
# Pretrain with explicit name
python run_pretrain.py --task pretrain_inputs_actuators_to_inputs_outputs --run-id tokamind_v1

# Pretrain with tag
python run_pretrain.py --task task_1-1 --tag small_v2  # Creates: task_1-1_small_v2

# Finetune (auto-generated)
python run_finetune.py --task task_1-1 --model tokamind_v1 --tag exp1  # Creates: ft-task_1-1-exp1-tokamind_v1
```

**Used in:** pretrain, finetune

---

### `eval_id`

**Type:** `str | null`  
**Required:** No  
**Default:** `null` (auto-generated as `<task>__eval__<timestamp>`)

Identifier for the evaluation run. Determines the output directory under [`runs/<model_run_id>/<eval_id>/`](../runs/).

**Example:**
```yaml
eval_id: "eval_baseline"
```

**Auto-generated format:** `task_1-1__eval__20260213_102030`

**Used in:** eval

---

### `embeddings_profile`

**Type:** `str`  
**Required:** No  
**Default:** `"dct3d"`

Embedding profile name used to select task-level embedding overrides from [`tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`](../scripts_mast/configs/tasks_overrides/).

**Example:**
```yaml
embeddings_profile: "vae"
```

**Common profiles:**
- `dct3d` - DCT3D compression (default)
- `vae` - VAE-based compression

**Used in:** pretrain, finetune, eval

---

## Runtime Configuration

Runtime settings control logging, debugging, and execution behavior.

### `runtime.debug_logging`

**Type:** `bool`  
**Required:** Yes  
**Default:** `false`

Enable verbose debug logging for troubleshooting.

**Example:**
```yaml
runtime:
  debug_logging: true
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

## Data Configuration

Data configuration controls dataset loading, caching, and subset selection.

### `data.local`

**Type:** `bool`  
**Required:** Yes (all phases)  
**Default:** `true`

Whether to use local dataset paths or remote/HPC paths.

**Example:**
```yaml
data:
  local: true  # Use local dataset
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `data.subset_of_shots`

**Type:** `int | null`  
**Required:** Yes (all phases)  
**Default:** `null`

Limit the number of shots to load. Use `null` to load all available shots.

**Example:**
```yaml
data:
  subset_of_shots: 100  # Use only 100 shots
```

```yaml
data:
  subset_of_shots: null  # Use all shots
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `data.keep_output_native`

**Type:** `bool`  
**Required:** No  
**Default:** `false` (pretrain/finetune), `true` (eval)

Whether to keep output signals in their native (non-embedded) format. Required for evaluation metrics and trace generation.

**Example:**
```yaml
data:
  keep_output_native: true
```

**Used in:** pretrain, finetune, eval

---

### `data.cache.enable`

**Type:** `bool`  
**Required:** No  
**Default:** `true` (pretrain/finetune), `false` (eval)

Enable in-memory caching of preprocessed windows. Significantly speeds up training but requires more memory.

**Example:**
```yaml
data:
  cache:
    enable: true
```

**Used in:** pretrain, finetune, eval

---

### `data.cache.dtype`

**Type:** `str`  
**Required:** No  
**Default:** `"float16"`  
**Valid values:** `"float16"`, `"float32"`

Data type for cached tensors. `float16` reduces memory usage.

**Example:**
```yaml
data:
  cache:
    dtype: float16
```

**Used in:** pretrain, finetune, eval

---

### `data.cache.num_workers`

**Type:** `int`  
**Required:** No  
**Default:** `32`

Number of parallel workers for cache building.

**Example:**
```yaml
data:
  cache:
    num_workers: 16
```

**Used in:** pretrain, finetune, eval

---

### `data.cache.max_windows.train`

**Type:** `int | null`  
**Required:** No  
**Default:** `null`

Maximum number of training windows to cache. Use `null` for no limit.

**Example:**
```yaml
data:
  cache:
    max_windows:
      train: 10000
```

**Used in:** pretrain, finetune

---

### `data.cache.max_windows.val`

**Type:** `int | null`  
**Required:** No  
**Default:** `null`

Maximum number of validation windows to cache. Use `null` for no limit.

**Example:**
```yaml
data:
  cache:
    max_windows:
      val: 2000
```

**Used in:** pretrain, finetune

---

## Preprocessing Configuration

Preprocessing parameters control how raw signals are chunked, trimmed, and validated.

### `preprocess.chunk.chunk_length`

**Type:** `float`  
**Required:** Yes (pretrain, tune_dct3d)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `0.005` (5 milliseconds)

Duration of each chunk in seconds. This is a fundamental parameter that defines the temporal resolution of the model.

**Example:**
```yaml
preprocess:
  chunk:
    chunk_length: 0.005  # 5 ms chunks
```

**Note:** This parameter is part of the model identity and must remain consistent between pretrain, finetune, and eval. See [`config_guide.md#model-identity`](config_guide.md#b-model-identity-anchored-to-a-training-run).

**Used in:** pretrain, finetune (inherited), eval (inherited), tune_dct3d

---

### `preprocess.chunk.stride`

**Type:** `float | null`  
**Required:** No  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `null`

Stride between consecutive chunks in seconds. Use `null` for non-overlapping chunks (stride = chunk_length).

**Example:**
```yaml
preprocess:
  chunk:
    stride: 0.0025  # 2.5 ms stride (50% overlap)
```

```yaml
preprocess:
  chunk:
    stride: null  # Non-overlapping chunks
```

**Used in:** pretrain, finetune (inherited), eval (inherited), tune_dct3d

---

### `preprocess.trim_chunks.max_chunks`

**Type:** `int`  
**Required:** Yes (pretrain, tune_dct3d)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `50`

Maximum number of chunks to retain in the history for each window. Older chunks are discarded.

**Example:**
```yaml
preprocess:
  trim_chunks:
    max_chunks: 100
```

**Note:** This parameter affects the model's input shape and is part of the model identity.

**Used in:** pretrain, finetune (inherited), eval (inherited), tune_dct3d

---

### `preprocess.valid_windows.min_valid_inputs_actuators`

**Type:** `int`  
**Required:** Yes  
**Default:** `1`

Minimum number of valid (non-NaN) input or actuator signals required for a window to be considered valid.

**Example:**
```yaml
preprocess:
  valid_windows:
    min_valid_inputs_actuators: 2
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `preprocess.valid_windows.min_valid_outputs`

**Type:** `int`  
**Required:** Yes  
**Default:** `1`

Minimum number of valid (non-NaN) output signals required for a window to be considered valid.

**Example:**
```yaml
preprocess:
  valid_windows:
    min_valid_outputs: 1
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `preprocess.valid_windows.min_valid_chunks`

**Type:** `int`  
**Required:** Yes  
**Default:** `1`

Minimum number of valid chunks required in the history for a window to be considered valid.

**Example:**
```yaml
preprocess:
  valid_windows:
    min_valid_chunks: 10
```

**Used in:** pretrain, finetune, eval, tune_dct3d

---

### `preprocess.valid_windows.window_stride_sec`

**Type:** `float | null`  
**Required:** Yes  
**Default:** `0.01` (pretrain/finetune), `null` (eval)

Stride between consecutive windows in seconds. Controls window sampling density.

**Example:**
```yaml
preprocess:
  valid_windows:
    window_stride_sec: 0.02  # Sample every 20 ms
```

```yaml
preprocess:
  valid_windows:
    window_stride_sec: null  # Evaluate on all possible windows
```

**Note:** This parameter can differ between phases. Use smaller strides for denser sampling during training, and `null` during evaluation to test on all windows.

**Used in:** pretrain, finetune, eval, tune_dct3d

---

## Model Configuration

Model configuration defines the neural network architecture.

### `model.backbone.d_model`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Hidden dimension size for the transformer backbone.

**Example:**
```yaml
model:
  backbone:
    d_model: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.backbone.n_layers`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `4`

Number of transformer layers in the backbone.

**Example:**
```yaml
model:
  backbone:
    n_layers: 6
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.backbone.n_heads`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `6`

Number of attention heads in each transformer layer.

**Example:**
```yaml
model:
  backbone:
    n_heads: 8
```

**Note:** `d_model` must be divisible by `n_heads`.

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.backbone.dim_ff`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `768`

Dimension of the feedforward network in each transformer layer.

**Example:**
```yaml
model:
  backbone:
    dim_ff: 1024
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.backbone.dropout`

**Type:** `float`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `0.05`

Dropout probability for regularization.

**Example:**
```yaml
model:
  backbone:
    dropout: 0.1
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.backbone.activation`

**Type:** `str`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `"relu"`  
**Valid values:** `"relu"`, `"gelu"`, `"silu"`

Activation function for the feedforward network.

**Example:**
```yaml
model:
  backbone:
    activation: gelu
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.timeseries.hidden`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Hidden dimension for timeseries modality head.

**Example:**
```yaml
model:
  modality_heads:
    timeseries:
      hidden: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.timeseries.out_dim`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Output dimension for timeseries modality head.

**Example:**
```yaml
model:
  modality_heads:
    timeseries:
      out_dim: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.profile.hidden`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Hidden dimension for profile modality head.

**Example:**
```yaml
model:
  modality_heads:
    profile:
      hidden: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.profile.out_dim`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Output dimension for profile modality head.

**Example:**
```yaml
model:
  modality_heads:
    profile:
      out_dim: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.video.hidden`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Hidden dimension for video modality head.

**Example:**
```yaml
model:
  modality_heads:
    video:
      hidden: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.modality_heads.video.out_dim`

**Type:** `int`  
**Required:** Yes (pretrain)  
**Inherited:** Yes (finetune, eval inherit from source run)  
**Default:** `192`

Output dimension for video modality head.

**Example:**
```yaml
model:
  modality_heads:
    video:
      out_dim: 256
```

**Used in:** pretrain, finetune (inherited), eval (inherited)

---

### `model.output_adapters.hidden_dim.default`

**Type:** `int`  
**Required:** Yes  
**Default:** `0` (pretrain), varies (finetune)

Default hidden dimension for output adapters. Use `0` for no hidden layer.

**Example:**
```yaml
model:
  output_adapters:
    hidden_dim:
      default: 128
```

**Used in:** pretrain, finetune (can override), eval (inherited)

---

### `model.output_adapters.hidden_dim.bucketed.enable`

**Type:** `bool`  
**Required:** No  
**Default:** `false` (pretrain), `true` (finetune)

Enable bucketed hidden dimensions based on output signal size.

**Example:**
```yaml
model:
  output_adapters:
    hidden_dim:
      bucketed:
        enable: true
```

**Used in:** pretrain, finetune

---

### `model.output_adapters.hidden_dim.bucketed.rules`

**Type:** `list[dict]`  
**Required:** If `bucketed.enable: true`  
**Default:** See example below

Rules for bucketed hidden dimensions. Each rule specifies a maximum output dimension and corresponding hidden dimension.

**Example:**
```yaml
model:
  output_adapters:
    hidden_dim:
      bucketed:
        enable: true
        rules:
          - {max_out_dim: 64,   hidden: 0}
          - {max_out_dim: 512,  hidden: 32}
          - {max_out_dim: 4096, hidden: 64}
          - {max_out_dim: 8192, hidden: 128}
          - {max_out_dim: null, hidden: d_model}  # fallback
```

**Note:** Rules are evaluated in order. The special value `d_model` uses the backbone's hidden dimension.

**Used in:** finetune

---

### `model.output_adapters.hidden_dim.manual`

**Type:** `dict[str, int]`  
**Required:** No  
**Default:** `{}`

Manual overrides for specific output signals. Takes precedence over default and bucketed rules.

**Example:**
```yaml
model:
  output_adapters:
    hidden_dim:
      manual:
        equilibrium-pis: 128
        summary-ip: 64
```

**Used in:** pretrain, finetune

---

## Training Configuration

Training configuration controls the optimization process, learning rates, and training stages.

### `train.resume`

**Type:** `bool`  
**Required:** Yes  
**Default:** `false`

Resume training from the latest checkpoint in the current run directory. Restores model weights, optimizer state, scheduler state, scaler state, RNG state, and epoch counters.

**Example:**
```yaml
train:
  resume: true
```

**Note:** This is different from warm-start ([`model_source`](#model-source-configuration)), which initializes from a different run.

**Used in:** pretrain, finetune

---

### `train.early_stop.patience`

**Type:** `int`  
**Required:** Yes  
**Default:** `10`

Number of epochs without validation improvement before stopping training.

**Example:**
```yaml
train:
  early_stop:
    patience: 15
```

**Used in:** pretrain, finetune

---

### `train.early_stop.delta`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.0`

Minimum change in validation loss to qualify as an improvement.

**Example:**
```yaml
train:
  early_stop:
    delta: 0.001
```

**Used in:** pretrain, finetune

---

### `train.amp.enable`

**Type:** `bool`  
**Required:** Yes  
**Default:** `true`

Enable Automatic Mixed Precision (AMP) training for faster computation and reduced memory usage.

**Example:**
```yaml
train:
  amp:
    enable: true
```

**Used in:** pretrain, finetune

---

### `train.loss.output_weights`

**Type:** `dict[str, float]`  
**Required:** No  
**Default:** `{}`

Per-output signal loss weights. Signals not specified use weight 1.0.

**Example:**
```yaml
train:
  loss:
    output_weights:
      summary-ip: 1.5
      equilibrium-pis: 0.8
```

**Used in:** pretrain, finetune

---

### `train.optimizer.use_adamw`

**Type:** `bool`  
**Required:** Yes  
**Default:** `true`

Use AdamW optimizer. If `false`, uses Adam.

**Example:**
```yaml
train:
  optimizer:
    use_adamw: true
```

**Used in:** pretrain, finetune

---

### `train.stages`

**Type:** `list[dict]`
**Required:** Yes
**Default:** See examples in phase configs

List of training stages. Each stage defines epochs, learning rates, weight decay, gradient accumulation, warmup, and freeze settings.

**Example:**
```yaml
train:
  stages:
    - name: pretrain_main
      epochs: 50
      
      scheduler:
        grad_accum_steps: 1
        warmup_steps_fraction: 0.1
      
      optimizer:
        lr:
          token_encoder: 5e-3
          backbone: 1e-3
          modality_heads: 5e-3
          output_adapters: 5e-3
        wd:
          token_encoder: 0.01
          backbone: 0.01
          modality_heads: 0.01
          output_adapters: 0.01
      
      freeze:
        token_encoder: false
        backbone: false
        modality_heads: false
        output_adapters: false
```

**Stage parameters:**
- `name` (str): Stage identifier
- `epochs` (int): Number of epochs for this stage
- `scheduler.grad_accum_steps` (int): Gradient accumulation steps
- `scheduler.warmup_steps_fraction` (float, optional): Fraction of total steps for warmup (default: 0.1)
  - Linear warmup from ~0 to 1.0× initial LR
  - Must be in range [0.0, 1.0)
  - Example: 0.1 means 10% of steps are warmup, 90% are cosine decay
- `optimizer.lr` (dict): Learning rates per model component
- `optimizer.wd` (dict): Weight decay per model component
- `freeze` (dict): Freeze flags per model component

**Scheduler behavior (step-based):**
- **Total steps** = ceil(batches_per_epoch / grad_accum_steps) × epochs
- **Warmup phase:** Linear ramp from ~0 to 1.0× initial LR over warmup_steps
- **Warmup steps** = round(warmup_steps_fraction × total_steps)
- **Decay phase:** Cosine annealing from 1.0× to 0× initial LR over remaining steps
- **Stepping:** `scheduler.step()` is called after each optimizer step (not per epoch)

**Note:** Use `lr: 0.0` or `freeze: true` to disable training for a component.

**Used in:** pretrain, finetune

---

## Evaluation Configuration

Evaluation configuration controls metrics computation and trace generation.

### `eval.amp.enable`

**Type:** `bool`  
**Required:** Yes  
**Default:** `true`

Enable Automatic Mixed Precision (AMP) during evaluation.

**Example:**
```yaml
eval:
  amp:
    enable: true
```

**Used in:** eval

---

### `eval.drop.inputs`

**Type:** `list[str] | null`  
**Required:** No  
**Default:** `null`

List of input signals to drop during evaluation. Use `null` to evaluate with all inputs.

**Example:**
```yaml
eval:
  drop:
    inputs: ["pf_active-coil_current"]
```

**Used in:** eval

---

### `eval.drop.actuators`

**Type:** `list[str] | null`  
**Required:** No  
**Default:** `null`

List of actuator signals to drop during evaluation. Use `null` to evaluate with all actuators.

**Example:**
```yaml
eval:
  drop:
    actuators: ["gas_injection-total_injected"]
```

**Used in:** eval

---

### `eval.drop.outputs`

**Type:** `list[str] | null`  
**Required:** No  
**Default:** `null`

List of output signals to drop during evaluation. Use `null` to evaluate all outputs.

**Example:**
```yaml
eval:
  drop:
    outputs: ["summary-ip"]
```

**Used in:** eval

---

### `eval.compute_metrics.per_task`

**Type:** `bool`  
**Required:** Yes  
**Default:** `true`

Compute aggregated metrics per task (shot).

**Example:**
```yaml
eval:
  compute_metrics:
    per_task: true
```

**Used in:** eval

---

### `eval.compute_metrics.per_window`

**Type:** `bool`  
**Required:** Yes  
**Default:** `false`

Compute metrics per window.

**Example:**
```yaml
eval:
  compute_metrics:
    per_window: true
```

**Used in:** eval

---

### `eval.compute_metrics.per_timestamp`

**Type:** `bool`  
**Required:** Yes  
**Default:** `false`

Compute metrics per timestamp within each window.

**Example:**
```yaml
eval:
  compute_metrics:
    per_timestamp: true
```

**Used in:** eval

---

### `eval.traces.enable`

**Type:** `bool`  
**Required:** Yes  
**Default:** `true`

Enable saving prediction traces for visualization.

**Example:**
```yaml
eval:
  traces:
    enable: true
```

**Used in:** eval

---

### `eval.traces.n_max`

**Type:** `int`  
**Required:** If `traces.enable: true`  
**Default:** `2`

Maximum number of traces to save per output signal.

**Example:**
```yaml
eval:
  traces:
    n_max: 5
```

**Used in:** eval

---

### `eval.traces.signals`

**Type:** `list[str] | null`  
**Required:** No  
**Default:** `null`

List of output signals to save traces for. Use `null` to save traces for all outputs.

**Example:**
```yaml
eval:
  traces:
    signals: ["summary-ip", "equilibrium-pis"]
```

**Used in:** eval

---

### `eval.traces.times_indexes`

**Type:** `list[int] | null`  
**Required:** No  
**Default:** `null`

List of time indices to save in traces. Use `null` to save the full prediction horizon.

**Example:**
```yaml
eval:
  traces:
    times_indexes: [0, 1, 5, 10]
```

**Used in:** eval

---

## Embedding Configuration

Embedding configuration defines how signals are encoded/compressed before being fed to the model.

### `embeddings.defaults.<role>.<modality>.encoder_name`

**Type:** `str`  
**Required:** Yes  
**Valid values:** `"dct3d"`, `"identity"`, `"vae"`

Default encoder type for signals with the specified role and modality.

**Example:**
```yaml
embeddings:
  defaults:
    input:
      timeseries:
        encoder_name: "dct3d"
```

**Roles:** `input`, `actuator`, `output`  
**Modalities:** `timeseries`, `profile`, `video`

**Used in:** pretrain, finetune, eval (inherited)

---

### `embeddings.defaults.<role>.<modality>.encoder_kwargs`

**Type:** `dict`  
**Required:** Yes (depends on encoder)

Encoder-specific parameters. For DCT3D, specifies compression dimensions.

**Example (DCT3D):**
```yaml
embeddings:
  defaults:
    input:
      timeseries:
        encoder_name: "dct3d"
        encoder_kwargs:
          keep_h: 1
          keep_w: 1
          keep_t: 10
```

**DCT3D parameters:**
- `keep_h` (int): Number of DCT coefficients to keep in height dimension
- `keep_w` (int): Number of DCT coefficients to keep in width dimension
- `keep_t` (int): Number of DCT coefficients to keep in time dimension

**Used in:** pretrain, finetune, eval (inherited)

---

### `embeddings.per_signal_overrides.<role>.<signal_name>`

**Type:** `dict`  
**Required:** No  
**Default:** `{}`

Per-signal encoder overrides. Takes precedence over defaults.

**Example:**
```yaml
embeddings:
  per_signal_overrides:
    input:
      summary-ip:
        encoder_name: "dct3d"
        encoder_kwargs:
          keep_h: 1
          keep_w: 1
          keep_t: 4
      magnetics-flux_loop_flux:
        encoder_name: "dct3d"
        encoder_kwargs:
          keep_h: 15
          keep_w: 1
          keep_t: 1
```

**Note:** Task-specific overrides should be placed in [`tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`](../scripts_mast/configs/tasks_overrides/), not in [`common/embeddings.yaml`](../scripts_mast/configs/common/embeddings.yaml).

**Used in:** pretrain, finetune, eval (inherited)

---

## Collate Configuration

Collate configuration controls data augmentation through random signal and chunk dropping.

### `collate.p_drop_inputs`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.1` (pretrain), `0.05` (finetune)

Probability of dropping each input signal during training (data augmentation).

**Example:**
```yaml
collate:
  p_drop_inputs: 0.15
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_inputs_overrides`

**Type:** `dict[str, float]`  
**Required:** No  
**Default:** `{}`

Per-signal drop probability overrides for input signals.

**Example:**
```yaml
collate:
  p_drop_inputs_overrides:
    pf_active-solenoid_current: 0.2
    summary-ip: 0.05
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_outputs`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.0`

Probability of dropping each output signal during training.

**Example:**
```yaml
collate:
  p_drop_outputs: 0.1
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_outputs_overrides`

**Type:** `dict[str, float]`  
**Required:** No  
**Default:** `{}`

Per-signal drop probability overrides for output signals.

**Example:**
```yaml
collate:
  p_drop_outputs_overrides:
    equilibrium-pis: 0.15
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_actuators`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.0`

Probability of dropping each actuator signal during training.

**Example:**
```yaml
collate:
  p_drop_actuators: 0.05
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_actuators_overrides`

**Type:** `dict[str, float]`  
**Required:** No  
**Default:** `{}`

Per-signal drop probability overrides for actuator signals.

**Example:**
```yaml
collate:
  p_drop_actuators_overrides:
    gas_injection-total_injected: 0.1
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_inputs_chunks`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.1` (pretrain), `0.05` (finetune)

Probability of dropping individual chunks from input signal histories.

**Example:**
```yaml
collate:
  p_drop_inputs_chunks: 0.15
```

**Used in:** pretrain, finetune

---

### `collate.p_drop_actuators_chunks`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.0`

Probability of dropping individual chunks from actuator signal histories.

**Example:**
```yaml
collate:
  p_drop_actuators_chunks: 0.05
```

**Used in:** pretrain, finetune

---

## Loader Configuration

Loader configuration controls the PyTorch DataLoader behavior.

### `loader.batch_size`

**Type:** `int`  
**Required:** Yes  
**Default:** `512`

Number of windows per batch.

**Example:**
```yaml
loader:
  batch_size: 256
```

**Used in:** pretrain, finetune, eval

---

### `loader.num_workers`

**Type:** `int`  
**Required:** Yes  
**Default:** `0` (pretrain/finetune), `32` (eval)

Number of worker processes for data loading.

**Example:**
```yaml
loader:
  num_workers: 4
```

**Note:** For cached datasets, use `num_workers: 0` or `1` since multi-worker loading rarely helps with precomputed data.

**Used in:** pretrain, finetune, eval

---

### `loader.shuffle_train`

**Type:** `bool`  
**Required:** Yes (training phases)  
**Default:** `true`

Shuffle training data each epoch.

**Example:**
```yaml
loader:
  shuffle_train: true
```

**Used in:** pretrain, finetune

---

### `loader.batches_per_epoch`

**Type:** `int`
**Required:** Only for streaming datasets during training
**Default:** `null` (inferred from len(dataloader) for cached datasets)

Number of batches to process per training epoch for streaming datasets.

**Example:**
```yaml
loader:
  batches_per_epoch: 2000
```

**Notes:**
- **Cached datasets**: This parameter is optional and ignored. Epoch length is automatically determined from `len(dataloader)`.
- **Streaming datasets**: This parameter is **required** for training. The training loop will raise an error if missing.
- **Validation**: Always exhausts the full validation dataloader regardless of this setting.
- Used to compute scheduler steps: `steps_per_epoch = ceil(batches_per_epoch / grad_accum_steps)`

**Used in:** pretrain, finetune (training only)

---

### `loader.drop_last`

**Type:** `bool`
**Required:** Yes
**Default:** `false`

Drop the last incomplete batch if the dataset size is not divisible by batch size.

**Example:**
```yaml
loader:
  drop_last: true
```

**Used in:** pretrain, finetune, eval

---

## CLI Arguments

**IMPORTANT:** Model sources and run IDs are now specified via CLI arguments, not YAML configs.

### Pretrain CLI Arguments

**`--run-id <name>`** (optional)

Explicit run identifier. Takes precedence over `--tag` and task name.

**`--tag <version>`** (optional)

Version tag for the run. Generates run_id as `{task}_{tag}`.

**Examples:**
```bash
# Foundation model with explicit name
python run_pretrain.py --task pretrain_inputs_actuators_to_inputs_outputs --run-id tokamind_v1

# Task-specific with tag
python run_pretrain.py --task task_1-1 --tag small_v2  # Creates: task_1-1_small_v2

# Default (uses task name)
python run_pretrain.py --task task_1-1  # Creates: task_1-1
```

**Priority:** `--run-id` > `--tag` > task name

### Finetune & Eval CLI Arguments

**`--model <run_id_or_path>`** (required)

Specifies the source model to load. Accepts either:
- A run_id (folder name under `runs/`)
- An absolute path to an external checkpoint directory

**`--tag <experiment_tag>`** (optional for finetune)

Adds an experiment tag to the auto-generated run_id for versioning multiple experiments.

**Examples:**
```bash
# Finetune from a pretrained model
python run_finetune.py --task task_2-1 --model tokamind_v1

# Finetune with experiment tag
python run_finetune.py --task task_2-1 --model tokamind_v1 --tag experiment1

# Evaluate a model
python run_eval.py --task task_2-1 --model ft-task_2-1-experiment1-tokamind_v1
```

---

### `model_source.run_id` (YAML - Pretrain Only)

**Type:** `str | null`
**Required:** No (optional for pretrain warm-start)
**Default:** `null`

For **pretrain phase only**, you can optionally set this in `pretrain_overrides.yaml` to warm-start from another pretrain run.

**Example:**
```yaml
model_source:
  run_id: "pretrain_base_v1"
```

**Note:**
- For **finetune and eval**, use the `--model` CLI argument instead
- Provide only the run id (folder name), not a path like `"runs/pretrain_base_v1"`

**Used in:** pretrain (optional warm-start)

---

### `model_source.model_path`

**Type:** `str | null`  
**Required:** No  
**Default:** `null`

Absolute or relative path to an external model directory. Overrides `run_id` for checkpoint loading.

**Example:**
```yaml
model_source:
  model_path: "/external/models/pretrained_v2"
```

**Used in:** finetune, eval

---

### `model_source.load_parts.token_encoder`

**Type:** `bool`  
**Required:** No  
**Default:** `true`

Load token encoder weights from source model.

**Example:**
```yaml
model_source:
  load_parts:
    token_encoder: true
```

**Used in:** finetune

---

### `model_source.load_parts.backbone`

**Type:** `bool`  
**Required:** No  
**Default:** `true`

Load backbone weights from source model.

**Example:**
```yaml
model_source:
  load_parts:
    backbone: true
```

**Used in:** finetune

---

### `model_source.load_parts.modality_heads`

**Type:** `bool`  
**Required:** No  
**Default:** `true`

Load modality head weights from source model.

**Example:**
```yaml
model_source:
  load_parts:
    modality_heads: true
```

**Used in:** finetune

---

### `model_source.load_parts.output_adapters`

**Type:** `bool`  
**Required:** No  
**Default:** `false` (finetune)

Load output adapter weights from source model.

**Example:**
```yaml
model_source:
  load_parts:
    output_adapters: true
```

**Note:** Typically set to `false` for finetune since output adapters are often task-specific.

**Used in:** finetune

---

## DCT3D Tuning Configuration

DCT3D tuning configuration controls the automatic hyperparameter search for DCT3D compression.

### `tune_dct3d.sampling.max_windows`

**Type:** `int`  
**Required:** Yes  
**Default:** `15000`

Maximum number of windows to sample for tuning.

**Example:**
```yaml
tune_dct3d:
  sampling:
    max_windows: 20000
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.thresholds.input`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.999`

Target reconstruction quality (R² score) for input signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    thresholds:
      input: 0.9995
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.thresholds.actuator`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.999`

Target reconstruction quality (R² score) for actuator signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    thresholds:
      actuator: 0.9995
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.thresholds.output`

**Type:** `float`  
**Required:** Yes  
**Default:** `0.995`

Target reconstruction quality (R² score) for output signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    thresholds:
      output: 0.998
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.max_budget.input`

**Type:** `int`  
**Required:** Yes  
**Default:** `4096`

Maximum embedding dimension budget for input signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    max_budget:
      input: 8192
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.max_budget.actuator`

**Type:** `int`  
**Required:** Yes  
**Default:** `4096`

Maximum embedding dimension budget for actuator signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    max_budget:
      actuator: 8192
```

**Used in:** tune_dct3d

---

### `tune_dct3d.objective.max_budget.output`

**Type:** `int`
**Required:** Yes
**Default:** `4096`

Maximum embedding dimension budget for output signals.

**Example:**
```yaml
tune_dct3d:
  objective:
    max_budget:
      output: 8192
```

**Used in:** tune_dct3d

---

### `tune_dct3d.guardrails.enabled`

**Type:** `bool`
**Required:** No
**Default:** `false`

Enable guardrails to ensure minimum dimension coverage in coefficient selection.

**Example:**
```yaml
tune_dct3d:
  guardrails:
    enabled: true
```

**Used in:** tune_dct3d
**See also:** [Tuning DCT3D - Guardrails](tuning_dct3d.md#guardrails-optional-dimension-coverage)

---

### `tune_dct3d.guardrails.timeseries.min_unique_t`

**Type:** `int`
**Required:** No
**Default:** `5`

Minimum unique time indices required for timeseries signals (shape: T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    timeseries:
      min_unique_t: 10
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.guardrails.profile.min_unique_h`

**Type:** `int`
**Required:** No
**Default:** `10`

Minimum unique channel indices required for profile signals (shape: C, T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    profile:
      min_unique_h: 15
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.guardrails.profile.min_unique_t`

**Type:** `int`
**Required:** No
**Default:** `5`

Minimum unique time indices required for profile signals (shape: C, T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    profile:
      min_unique_t: 8
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.guardrails.video.min_unique_h`

**Type:** `int`
**Required:** No
**Default:** `10`

Minimum unique height indices required for video signals (shape: H, W, T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    video:
      min_unique_h: 15
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.guardrails.video.min_unique_w`

**Type:** `int`
**Required:** No
**Default:** `10`

Minimum unique width indices required for video signals (shape: H, W, T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    video:
      min_unique_w: 15
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.guardrails.video.min_unique_t`

**Type:** `int`
**Required:** No
**Default:** `5`

Minimum unique time indices required for video signals (shape: H, W, T).

**Example:**
```yaml
tune_dct3d:
  guardrails:
    video:
      min_unique_t: 8
```

**Used in:** tune_dct3d (when guardrails.enabled=true)

---

### `tune_dct3d.search_space.keep_h`

**Type:** `list[int]`  
**Required:** Yes  
**Default:** `[1, 4, 8, 16, 32, 64, 96, 128]`

Search space for height dimension DCT coefficients.

**Example:**
```yaml
tune_dct3d:
  search_space:
    keep_h: [1, 2, 4, 8, 16, 32, 64]
```

**Used in:** tune_dct3d

---

### `tune_dct3d.search_space.keep_w`

**Type:** `list[int]`  
**Required:** Yes  
**Default:** `[1, 4, 8, 16, 24, 32, 65]`

Search space for width dimension DCT coefficients.

**Example:**
```yaml
tune_dct3d:
  search_space:
    keep_w: [1, 2, 4, 8, 16, 32]
```

**Used in:** tune_dct3d

---

### `tune_dct3d.search_space.keep_t`

**Type:** `list[int]`  
**Required:** Yes  
**Default:** `[1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 224, 256, 512, 768, 1024, 1280, 2048]`

Search space for time dimension DCT coefficients.

**Example:**
```yaml
tune_dct3d:
  search_space:
    keep_t: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
```

**Used in:** tune_dct3d

---

## Path Configuration

Path configuration is automatically computed by the config loader and should not be manually specified in YAML files.

### `paths.repo_root`

**Type:** `str`  
**Computed:** Yes  
**Description:** Absolute path to the repository root.

**Used in:** All phases

---

### `paths.configs_root`

**Type:** `str`  
**Computed:** Yes  
**Description:** Absolute path to the configs directory.

**Used in:** All phases

---

### `paths.run_dir`

**Type:** `str`
**Computed:** Yes
**Description:** Absolute path to the run output directory.

**Format:**
- Pretrain/Finetune: `<repo_root>/runs/<run_id>/`
- Eval: `<repo_root>/runs/<model_id>/eval/`
- Tune DCT3D: `<repo_root>/scripts_mast/configs/tasks_overrides/<task>/`

**Used in:** All phases

---

### `paths.task`

**Type:** `str`
**Computed:** Yes
**Description:** Task identifier.

**Used in:** pretrain, finetune, eval

---

### `paths.phase`

**Type:** `str`
**Computed:** Yes
**Description:** Phase identifier (pretrain, finetune, eval, or tune_dct3d).

**Used in:** All phases

---

### `paths.run_id`

**Type:** `str`
**Computed:** Yes
**Description:** Run identifier for training phases. Auto-generated for finetune as `ft-{task}-{tag}-{model_id}` (or `ft-{task}-{model_id}` if no tag).

**Used in:** pretrain, finetune

---

### `paths.eval_id`

**Type:** `str`
**Computed:** Yes
**Description:** Evaluation identifier (always "eval").

**Used in:** eval

---

### `paths.model_run_dir`

**Type:** `str`
**Computed:** Yes
**Description:** Absolute path to the source model's run directory (`<repo_root>/runs/<model_id>/`).

**Used in:** eval

---

### `paths.config_dir`

**Type:** `str`
**Computed:** Yes
**Description:** Absolute path to the task configuration directory.

**Format:** `<repo_root>/scripts_mast/configs/tasks_overrides/<task>/`

**Used in:** tune_dct3d

---

### `paths.tune_dir`

**Type:** `str`
**Computed:** Yes
**Description:** Absolute path to the embeddings tuning directory.

**Format:** `<repo_root>/scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/`

**Used in:** tune_dct3d

---

### `paths.tune_dir`

**Type:** `str`  
**Computed:** Yes  
**Description:** Absolute path to the embedding tuning output directory.

**Used in:** tune_dct3d

---

## See Also

- [Configuration Guide](config_guide.md) - System architecture and merge order
- [Model Architecture](model_architecture.md) - Model structure and components
- [Checkpointing and Warm-start](checkpointing_and_warmstart.md) - Model loading strategies
- [Tuning Embeddings](tuning_embeddings.md) - DCT3D hyperparameter tuning
- [Evaluation](evaluation.md) - Metrics and trace generation