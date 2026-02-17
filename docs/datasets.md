# Datasets and data loading

This project separates **shot-level datasets** (one item = one shot / sequence that yields many windows)
from **window-level datasets** (one item = one model training/eval window).

The MMT model always trains/evaluates on **window dicts** produced by the transforms pipeline
(see `docs/transforms.md`). Window dicts are then batched by `MMTCollate`.

---

## Concepts

### Shot
A *shot* is a long raw recording/trajectory (e.g. one experiment run). In the MAST integration, it is
identified by `shot_id`.

### Window
A *window* is a single model sample cut from a shot. It contains:

- context history as **chunks** (inputs + actuators),
- targets (outputs),
- metadata such as `shot_id`, `t_cut`, and a `window_index`.

After preprocessing + embedding + token building, a window dict contains token fields like:

- `emb_chunks`, `id`, `role`, `mod`, `pos`, `signal_name`
- (optionally) native output payload under `output` when `keep_output_native=True`

---

## Architecture Overview

The MMT data pipeline integrates with the MAST integration's window-level API:

1. **MAST Integration**: `initialize_model_dataset_iterable()` returns `TaskModelTransformWrapperIterable`
   - An IterableDataset that yields windows on-the-fly from shots
   - Applies model transforms (chunking, embedding, tokenization) per window
   - Handles shot-level shuffling and window filtering

2. **Window Caching**: `WindowCachedDataset.from_streaming()` materializes windows to RAM
   - Converts the IterableDataset into a map-style Dataset
   - Enables true window-level shuffling via `DataLoader(shuffle=True)`
   - Provides fastest training throughput after initial caching

---

## Window-level dataset: `WindowCachedDataset`

`WindowCachedDataset` stores a fully materialized **list of tokenized windows in RAM**.

Key properties:

- **Fastest per-step**: no per-window preprocessing in the training loop after caching.
- **Map-style Dataset**: `len(dataset) == num_windows` (true window count).
- `DataLoader(shuffle=True)` performs **window-level shuffling**.
- Easy to reason about epochs: one epoch = one pass over cached windows.

When to use:

- Default choice for train/val splits (recommended)
- When you can afford the RAM footprint
- When you care about window-level shuffling / deterministic epoch lengths
- When you want fastest training throughput

---

## How caching works

Caching is performed by:

```python
WindowCachedDataset.from_streaming(
    streaming_dataset,  # IterableDataset (e.g., TaskModelTransformWrapperIterable)
    num_workers=32,
    max_windows=None,
    dtype="float16",
    seed=42
)
```

The caching process:

1. **Parallel materialization**: Uses PyTorch DataLoader with multiple workers to consume the IterableDataset
2. **Type casting**: If `dtype` is set (e.g. float16), token embeddings and output embeddings are cast before storing
3. **Memory efficiency**: `prefetch_factor=1` reduces RAM spikes during caching
4. **Optional limits**: `max_windows` can cap the number of windows cached (useful for debugging)

Requirements for the input `streaming_dataset`:

- Must be an IterableDataset (e.g., `TaskModelTransformWrapperIterable` from MAST integration)
- Yields window dicts one at a time
- Handles shot-level shuffling internally if configured

---

## Shuffling semantics

### Cached path (recommended)
- Shuffling is **window-level**, handled by `DataLoader(shuffle=True)`
- Windows from different shots are mixed uniformly in each batch
- Deterministic when using a fixed seed with the DataLoader's generator

### Streaming path (MAST integration IterableDataset)
- Shuffling is **shot-level**, handled by the MAST integration's `TaskModelTransformWrapperIterable`
- Within each shot, windows are yielded in temporal order
- Batch composition depends on:
  - how many windows each shot yields,
  - how `DataLoader` worker prefetching interleaves the stream

Practical implication:

- **Cached mode** gives true window-level shuffling (recommended for training)
- **Streaming mode** is useful for evaluation or when RAM is limited

---

## Configuration

Caching is controlled via `data.cache` in the config:

```yaml
data:
  cache:
    enable: true          # Enable window caching
    dtype: float16        # Cast embeddings to float16 to save RAM
    num_workers: 32       # Parallel workers for caching
    max_windows:
      train: null         # null = cache all windows
      val: null
```

When `data.cache.enable=false`, the MAST integration's IterableDataset is used directly (streaming mode).

---

## Recommended defaults

For most users:

- **Always use cached windows** for train/val (set `data.cache.enable: true`)
- Use `dtype: float16` to reduce RAM usage
- Set `num_workers: 32` for fast caching (adjust based on your system)
- For evaluation, caching is optional but recommended for consistent metrics

---

## Troubleshooting

### "Caching uses too much RAM"
Reduce:
- `data.cache.max_windows.train` / `data.cache.max_windows.val` (for debugging)
- chunk counts (`preprocess.trim_chunks.max_chunks`)
- embedding sizes (via `embeddings.yaml` / tuned overrides)
- Use `dtype: float16` instead of `float32`

### "Caching is slow"
Increase:
- `data.cache.num_workers` (more parallel workers)
- Consider using faster storage (SSD vs HDD) for the MAST dataset

### "I want to use streaming mode"
Set `data.cache.enable: false` in your config. Note:
- Training will be slower (per-window preprocessing overhead)
- Shuffling is shot-level only
- Epoch length may vary if window counts change
