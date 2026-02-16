# Preprocessing and transforms pipeline

MMT uses a **transform chain** to convert raw per-window arrays into tokenized samples that can be fed
to the model.

All transforms follow the same simple contract:

- Input: a single `window: dict`
- Output: either a (possibly modified) `window: dict`, or `None` to drop the window

Transforms are typically composed with `ComposeTransforms`, which stops early if any stage returns `None`.

---

## Window dict lifecycle (high level)

A window dict goes through these stages:

1. **Raw window** (from a shot dataset adapter)
2. **Chunked window** (fixed-length chunk slots)
3. **Validated / subsampled window** (drop invalid or too-dense windows)
4. **Trimmed window** (keep last `max_chunks` and assign relative positions)
5. **Embedded window** (codec embeddings + caching)
6. **Tokenized window** (per-token arrays + output embeddings)

---

## Standard transform chain (train / eval)

The default chain used by pretrain/finetune/eval is:

1. `ChunkWindowsTransform`
2. `SelectValidWindowsTransform`
3. `TrimChunksTransform`
4. `EmbedChunksTransform`
5. `BuildTokensTransform`

This chain is intentionally **index-based** to make caching and determinism robust.

---

## Chunk identity: local vs global

Chunking introduces two integer indices:

### `chunk_index_in_window`
- Local index inside the current window and role (0..N-1)
- Ordered oldest → newest

### `chunk_index_global`
- A stable slot id on the stride grid
- Computed from `(t0_sec, stride_sec)` + local index
- Used as part of the embedding cache key

This is the key design that enables robust deduplication across overlapping windows.

---

## Transform-by-transform reference

Below: what each transform **expects**, what it **adds**, and what it may **drop**.

### 1) `ChunkWindowsTransform`

**Purpose**
- Convert raw signals into fixed-length chunk slots on a shared stride grid (in seconds)
- Support per-signal `dt` (different samples per chunk) while sharing the same chunk slots

**Reads**
- `window["input"]` and `window["actuator"]` payloads (raw arrays + time context)
- `window["t_cut"]` and/or window timing metadata
- global chunking params: `chunk_length_sec`, `stride_sec`

**Writes**
- `window["chunks"] = {"input": [...], "actuator": [...]}`

Each chunk dict contains (v0):

```python
{
  "role": "input" | "actuator",
  "chunk_index_in_window": int,
  "chunk_index_global": int,
  "signals": { <signal_name>: np.ndarray[..., T_chunk], ... }
}
```

Notes:
- Signals within a role share the **same number of chunks**, but each chunk can have a different
  sample length per signal depending on that signal’s `dt`.

---

### 2) `SelectValidWindowsTransform`

**Purpose**
- Drop unusable windows and optionally subsample windows in time (per shot)

**Reads**
- `window["chunks"][role][i]["signals"][name]`
- `window["output"][name]["values"]`
- `window["shot_id"]`, `window["t_cut"]`

**Writes / modifies**
- Masks invalid chunk signals by setting them to `None`
- May mask invalid outputs similarly
- Returns `None` to drop the window if minimum validity requirements are not met

Key config (phase-dependent):

```yaml
preprocess:
  valid_windows:
    min_valid_inputs_actuators: ...
    min_valid_chunks: ...
    min_valid_outputs: ...
    window_stride_sec: ...
```

Semantics:
- First apply validity checks.
- Then apply optional subsampling via `window_stride_sec` using per-shot `t_cut` state
  (only kept windows update the “last kept” timestamp).

---

### 3) `TrimChunksTransform`

**Purpose**
- Keep only the most recent chunk history (last `max_chunks`)
- Compute relative positions `pos` without using timestamps

Index-based position definition for a role with N chunks:

```text
oldest chunk_index_in_window = 0
newest chunk_index_in_window = N-1

pos = (N - 1 - chunk_index_in_window) + 1
# pos=1 is closest to output (newest)
```

**Reads**
- `window["chunks"]` (from chunking/validation)

**Writes / modifies**
- Trims each role list to at most `max_chunks`
- Adds `chunk["pos"]` to each chunk

---

### 4) `EmbedChunksTransform`

**Purpose**
- Convert chunk signals and outputs into embedding vectors using configured codecs
- Use stable caching to deduplicate embeddings across overlapping windows

**Reads**
- `window["chunks"][role][i]["signals"][name]` (raw chunk arrays)
- `window["output"][name]["values"]` (raw outputs)
- `window["shot_id"]`
- `SignalSpecRegistry` (maps `(role,name)` → `signal_id`, modality, embedding_dim)
- `codecs[signal_id]`

**Writes**
- Per chunk:
  - `chunk["embeddings"][signal_id] = np.ndarray[D]`
  - `chunk["orig_shapes"][signal_id] = tuple` (native shape used by decoders/metrics)

- Per window outputs:
  - `window["embedded_output"]`
  - `window["embedded_output_shapes"]`

- Memory reduction:
  - Drops `chunk["signals"]` on the returned copy
  - Optionally drops output native `values` if `keep_output_native=False`

**Caching key (v0)**
```text
(shot_id, role, signal_id, chunk_index_global)
```

This is why `chunk_index_global` is required.

---

### 5) `BuildTokensTransform`

**Purpose**
- Flatten embedded chunks into the per-token representation consumed by the model and `MMTCollate`

**Reads**
- Chunk embeddings + `pos` and indexes
- `SignalSpecRegistry` (for signal_id and modality mapping)
- `window["embedded_output"]` (output embeddings)

**Writes**
Token fields (must match collate):

```text
window["emb_chunks"]   : list of embeddings (ragged, one per token)
window["id"]           : int32 array (signal_id)
window["role"]         : int8 array (ROLE_CONTEXT / ROLE_ACTUATOR)
window["mod"]          : int16 array (modality id)
window["pos"]          : int32 array (relative position)
```

Output fields:

```text
window["output_emb"]
```

Note: per-token signal names and per-window output name/shape metadata are not stored; name-based dropout overrides are converted to id-based once at startup.

**Deterministic ordering**
- Chunks are ordered by `(pos, chunk_index_in_window)` (closest-to-output first)
- Signals within a chunk are sorted by `signal_id`

---

## Tuning pipeline: `TuneRankedDCT3DTransform`

During DCT3D tuning, the chain is typically:

`ChunkWindows → SelectValidWindows → TrimChunks → TuneRankedDCT3D → (optional EmbedChunks)`

`TuneRankedDCT3DTransform`:

- Computes full DCT for each chunk
- Accumulates per-coefficient energy `E[c_i²]` across all windows
- After streaming, selects top-K coefficients by variance for each signal
- Writes coefficient indices to `.npy` files
- Generates config with `selection_mode: rank`
- Can be configured to tune only a subset of roles via `roles` (any of: `input`, `actuator`, `output`)

**Key differences from old grid search approach:**
- **Rank mode** (current): Selects coefficients by variance, regardless of spatial position
  - 50-200x faster (no grid search)
  - Better compression: coefficients selected by actual importance
  - Adapts to each signal's spectral characteristics
- **Spatial mode** (legacy): Selects top-left-front block `(keep_h, keep_w, keep_t)`
  - Grid search over candidate dimensions
  - Assumes low-frequency components are most important

It is intentionally a **pass-through transform** so it can be inserted without changing the rest of the pipeline.

The `scripts_mast/run_tune_dct3d.py` runner exposes this as a CLI flag:

```bash
python scripts_mast/run_tune_dct3d.py --task <task> --roles output
```

See [Tuning DCT3D](tuning_dct3d.md) for detailed usage and configuration.

---

## Extending the pipeline

Common extension points:

- Add a new window filter transform (return `None` to drop).
- Add a new codec (implement encode/decode) and reference it via `embeddings.yaml`.
- Add an augmentation transform *before* embedding to avoid cache collisions.

Rules of thumb:

- Keep transforms single-purpose.
- Prefer index-based fields (`chunk_index_*`, `pos`) over floating timestamps for determinism.
- Any transform that changes raw chunk arrays should run **before** `EmbedChunksTransform`
  (otherwise caching may reuse stale embeddings).
