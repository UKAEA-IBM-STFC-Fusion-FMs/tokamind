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
- `output_emb`, `output_shapes`, `output_names`
- (optionally) native output payload under `output` when `keep_output_native=True`

---

## Window-level dataset options

MMT provides two ways to expose windows to a PyTorch `DataLoader`:

1) **Stream windows** directly from the shot dataset (low memory)  
2) **Cache windows** in RAM (fastest training, true window-level shuffling)

### Option 1 — `WindowStreamedDataset` (IterableDataset)

`WindowStreamedDataset` flattens a shot dataset into an **iterator of windows**.

Key properties:

- **Low memory**: windows are produced on the fly.
- **IterableDataset**: `DataLoader(shuffle=True)` is *not* applied by PyTorch for iterable datasets.
- Shuffling happens at the **shot index** level (`shuffle_shots=True`) inside the dataset.
- In the MMT entrypoints, `shuffle_shots` is enabled for the **train** split and disabled for **val/test** to keep evaluation deterministic.
- Multi-worker support: each worker gets a slice of shot indices; shuffling is per-worker and seeded.

Important caveat:

- `__len__` returns **number of shots**, not number of windows.
  If you need a strict notion of “batches per epoch”, use `loader.batches_per_epoch`
  (the training loop / validator may require this when streaming).

When to use:

- datasets too large to cache in memory,
- quick experiments where perfect window-level shuffling is not required,
- situations where window count is expensive to compute.

### Option 2 — `WindowCachedDataset` (map-style Dataset)

`WindowCachedDataset` stores a fully materialized **list of tokenized windows in RAM**.

Key properties:

- **Fastest per-step**: no per-window preprocessing in the training loop after caching.
- **Map-style Dataset**: `len(dataset) == num_windows` (true window count).
- `DataLoader(shuffle=True)` performs **window-level shuffling** (typically enabled for the **train** split only; controlled by `loader.shuffle_train`).
- Easy to reason about epochs: one epoch = one pass over cached windows.

When to use:

- you can afford the RAM footprint,
- you care about window-level shuffling / deterministic epoch lengths,
- you want fastest training throughput.

---

## How caching works

Caching is performed by:

- `materialize_tokenized_split_to_ram(streaming_dataset, num_workers_cache=..., max_windows=...)`

Requirements for the input `streaming_dataset`:

- It must be sequence-like (`__len__` and `__getitem__`).
- `__getitem__(i)` may return:
  - a single window dict,
  - an iterable of window dicts (list/tuple/generator),
  - `None` (if the shot yields no valid windows).

Parallel caching:

- PyTorch DataLoader workers cannot pickle generators.
- To support `num_workers_cache > 0`, caching wraps the shot dataset with
  `FlattenedStreamingDataset`, which converts each `__getitem__` output into a **plain list**
  of window dicts.

---

## Shuffling semantics

### Cached path
- Shuffling is **window-level**, handled by `DataLoader(shuffle=True)` for the **train** split (controlled by `loader.train_shuffle`).
- Validation/test loaders are **not shuffled** (deterministic).

### Streaming path
- Shuffling is **shot-level**, handled by `WindowStreamedDataset(shuffle_shots=True)` for the **train** split.
- Validation/test use `shuffle_shots=False` (deterministic).
- Within each shot, windows are yielded in the order produced by the upstream window generator.

Practical implication:

- If you want “mix windows from different shots in the same batch”, caching gives this by default.
- Streaming can still mix shots across a batch, but the batch composition depends on:
  - how many windows each shot yields,
  - how `DataLoader` worker prefetching interleaves the stream.

---

## Recommended defaults

For most users:

- Start with **cached windows** for train/val because it’s simpler and faster.
- Keep `loader.shuffle_train: true` for cached **training** (val/test remain deterministic).
- Use **streaming windows** only when you hit RAM limits.


---

## Troubleshooting

### “My epochs are weird / too short” in streaming mode
Remember that streaming datasets don’t know the number of windows up front.
Use `loader.batches_per_epoch` (and/or cache windows).

### “Caching uses too much RAM”
Reduce:
- `max_windows_per_split` (or split-specific caps like `max_windows_train` / `max_windows_val`),
- chunk counts (`preprocess.trim.max_chunks`),
- embedding sizes (via `embeddings.yaml` / tuned overrides).
