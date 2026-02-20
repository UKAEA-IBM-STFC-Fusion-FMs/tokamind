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
