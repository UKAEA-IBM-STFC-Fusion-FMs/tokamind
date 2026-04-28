# Datasets

Related documentation: [Project README](../README.md) | [Transforms](transforms.md) | [Configuration Guide](config_guide.md)

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

- **At window selection**: `SelectValidWindowsTransform` uses `accept_nan=True` (default). Only signals that are entirely NaN or empty are masked as invalid. Windows with partially-NaN signals pass through.
- **At encoding**: `EmbedChunksTransform` applies local zero-fill imputation immediately before `codec.encode()`. Zero equals the signal mean in standardized space, making this the least-biased imputation. For output signals, imputation is applied to a temporary local copy only — the original NaN values in `window["output"]` are preserved for benchmark-comparable evaluation metrics.

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
