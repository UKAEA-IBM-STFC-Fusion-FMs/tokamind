# Datasets

Related documentation: [Project README](../README.md) | [Transforms](transforms.md) | [Training](training.md) | [Configuration Guide](config_guide.md)

This document describes shot-level and window-level data handling in the project.

## Terminology
### Shot
A shot is a long sequence record from the benchmark source (single discharge/time series session).

### Window
A window is one model sample derived from a shot. It includes:
- input and actuator history chunks
- output targets
- metadata (`shot_id`, `t_cut`, `window_index`)

`t_cut` is the prediction reference time used to align history and targets.

## Data Flow
1. MAST integration produces shot iterables.
2. Window transforms build model-ready window dicts.
3. `MMTCollate` batches tokenized windows for training/eval.

Split selection (`train` / `val` / `test`) is defined by task setup before the transform chain starts.

## Data Split Strategies
Two split strategies are supported, selected via `data.split` in pretrain and finetune configs:

| Split | Description |
|---|---|
| `random` (default) | Shots randomly partitioned into train/val/test. |
| `temporal` | Shots partitioned by campaign/discharge time (later campaigns are test). |

Each split has its own set of three consistent artifacts:
- shot assignment CSV
- signal normalization stats YAML (mean/std per signal)
- outlier metadata YAML

All three are resolved together by `tokamark_split.py` to ensure no cross-split contamination of statistics.

### Split inheritance
- **Pretrain / finetune scratch**: `data.split` is read directly from config.
- **Finetune warmstart**: split is inherited from the source run. If `data.split` in the finetune config differs from the source run, a warning is logged and the source split is enforced.
- **Eval**: `data.split` must not be set in eval config. Split is always inherited from the source run config automatically.

## NaN Handling
MAST signals can contain NaN values (missing channels, partial dropouts). The pipeline is designed to handle them without dropping windows unnecessarily:

- **At window selection**: `SelectValidWindowsTransform` has two independent NaN-tolerance flags:
  - `accept_nan_inputs_actuators` (default `True`): partial-NaN input/actuator signals pass through with NaN intact for downstream imputation.
  - `accept_nan_outputs` (default `True`): partial-NaN output signals pass through. Set to `False` in finetune/pretrain configs to drop windows with partial-NaN outputs from training, preventing corrupted targets from entering the loss.
  - In all cases, entirely-NaN or empty signals are always masked as invalid regardless of these flags.
- **At encoding**: `EmbedChunksTransform` imputes NaN values on a local copy before `codec.encode()`,
  controlled by `preprocess.embed_chunks.nan_imputation` (default `"zero"`):
  - `"zero"`: literal zero-fill. If data are standardized, zero corresponds to the signal mean. Fast,
    but creates hard discontinuities at NaN/valid boundaries that can contaminate DCT3D low-frequency coefficients.
  - `"interpolate"`: temporal then spatial linear interpolation with zero fallback. Avoids step edges
    that contaminate DCT3D coefficients — preferred for signals with structured boundary NaN.
  - `null`: no imputation; allowed only when all registered codecs can handle non-finite inputs natively.
  - Original values in `window["output"]` are never modified — NaN locations are preserved for eval
    metrics and for native-space loss targets. See
    [Training — NaN Handling](training.md#nan-handling-in-training) for the full interaction with loss choice.

## Dataset Types
### Streaming window iterable
Produced by MAST integration wrapper.

Characteristics:
- builds windows on the fly
- lower RAM usage
- shuffling behavior depends on iterable order and worker scheduling
- startup is fast because nothing is pre-materialized

### Cached window dataset (`WindowCachedDataset`)
Materializes windows in RAM.

Characteristics:
- fastest step throughput
- map-style indexing
- true window-level shuffle via DataLoader
- optional dtype cast during caching
- longer startup due to one-time cache build

## Caching Configuration
```yaml
data:
  cache:
    enable: true
    dtype: float16
    num_workers: 64
    max_windows:
      train: null
      val: null
```

## Loader Interaction
`loader` settings apply after dataset preparation:

```yaml
loader:
  batch_size: 512
  num_workers: 0
  shuffle_train: true
  drop_last: false
```

Practical note:
- with cached mode, we suggest to use `loader.num_workers=0` 
- with streaming mode, higher `loader.num_workers` may improve throughput

## Practical Guidance
- Prefer cached mode for training when RAM is available.
- Prefer streaming mode when RAM is limited or when rapid iteration on transforms is needed.
- Keep eval deterministic by disabling training-only stochastic drops.
