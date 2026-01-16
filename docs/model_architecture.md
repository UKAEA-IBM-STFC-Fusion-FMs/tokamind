# Model Architecture (MMT v0)

This document describes the **core model architecture** and the **data-to-model pipeline** used by the open-source Multi‑Modal Transformer (MMT v0).

The design is intentionally *spec-driven* and *modular*: the dataset integration layer provides **signal specs** + **window dictionaries**, and MMT provides a deterministic pipeline that turns those windows into tokens and predictions.

---

## High-level overview

MMT training/evaluation follows this path:

```mermaid
flowchart LR
  Raw[Raw window dict\n(inputs/actuators/outputs + metadata)] --> Chunk[ChunkWindowsTransform]
  Chunk --> Valid[SelectValidWindowsTransform]
  Valid --> Trim[TrimChunksTransform]
  Trim --> Embed[EmbedChunksTransform\n(+ caching)]
  Embed --> Tokens[BuildTokensTransform]
  Tokens --> Collate[MMTCollate]
  Collate --> Model[MultiModalTransformer\n(TokenEncoder → Backbone → ModalityHeads → OutputAdapters)]
  Model --> Pred[pred: Dict[signal_id → embedding]]
```

Key points:
- **All heavy preprocessing happens outside the model.**  
  The model sees only token embeddings + lightweight metadata tensors.
- **SignalSpecRegistry drives everything** (signals, modalities, encoder dims, output dims).
- **Checkpointing is block-wise**: token encoder, backbone, modality heads, output adapters.

---

## Core concepts

### Signals, roles, modalities

In MMT, each *role-specific* signal is described by a `SignalSpec`:

- `role`: `"input" | "actuator" | "output"`
- `name`: canonical signal name (string)
- `modality`: e.g. `"timeseries"`, `"profile"`, `"video"`
- `encoder_name` + `encoder_kwargs`: how chunks are embedded
- `embedding_dim`: size of the codec output embedding vector
- `signal_id`: integer ID used internally for fast routing

A single physical signal can appear in multiple roles (e.g., as input and output).  
In that case it gets **separate SignalSpecs** (and separate `signal_id`s).

### Canonical keys

To make checkpointing and warm-start stable, learnable per-signal modules are keyed by a **canonical string**:

```
canonical_key = f"{role}:{name}"
```

Examples:
- `input:pf_active-coil_current`
- `actuator:summary-power_nbi`
- `output:equilibrium-psi`

This prevents fragile coupling to numeric `signal_id` ordering.

---

## Data pipeline (windows → tokens)

The dataset integration layer must yield **window dictionaries**. Conceptually:

```python
window = {
  "input":    {name: {"time": ..., "values": np.ndarray[..., T]} , ...},
  "actuator": {name: {"time": ..., "values": np.ndarray[..., T]} , ...},
  "output":   {name: {"time": ..., "values": np.ndarray[..., T]} , ...},

  "t_cut": float,
  "shot_id": Any,
  "window_index": int,
}
```

The MMT transforms then produce token-ready fields.

### 1) `ChunkWindowsTransform` (index-based chunking)

Purpose: convert raw arrays into **chunk slots** on a shared time grid.

Output:
- `window["chunks"]["input"]` and `window["chunks"]["actuator"]`
- each chunk has:
  - `chunk_index_in_window`: `0..N-1` (oldest → newest within that role span)
  - `chunk_index_global`: stable slot id on the stride grid (**used for caching**)
  - `signals`: `{name → np.ndarray[..., T_chunk]}` (right‑padded with NaNs as needed)

This stage uses:
- `chunk_length_sec`
- `stride_sec` (often equal to chunk length)
- per-signal `dt` metadata to map time slots → sample indices

### 2) `SelectValidWindowsTransform` (mask/drop)

Purpose: drop unusable windows and optionally subsample windows by time.

Typical behavior:
- masks chunk signals that are empty / all‑NaN / non‑finite
- enforces thresholds like:
  - `min_valid_inputs_actuators`
  - `min_valid_chunks`
  - `min_valid_outputs`
- optional window subsampling using `window_stride_sec` and `t_cut`

### 3) `TrimChunksTransform` (history trimming + positions)

Purpose:
- keep only the most recent history (`max_chunks`)
- compute index-based positions `pos` (“distance to output”)

For a role with `N` chunks (oldest → newest, index `0..N-1`):

```
pos = (N - 1 - chunk_index_in_window) + 1
# newest chunk → pos=1
# older chunks → pos=2,3,...
```

### 4) `EmbedChunksTransform` (codecs + caching)

Purpose:
- encode chunk signals to embeddings using the configured codec per signal
- encode window-level outputs to embeddings
- optionally keep or drop native output arrays

Chunk outputs:
- `chunk["embeddings"][signal_id] = embedding_vector`
- `chunk["orig_shapes"][signal_id] = original_shape`
- `chunk["signals"]` is dropped on the returned copy (to reduce memory)

Caching (v0):
- **cache key** = `(shot_id, role, signal_id, chunk_index_global)`

This gives cache hits for overlapping windows when stride alignment matches.

### 5) `BuildTokensTransform` (deterministic token layout)

Purpose: flatten embedded chunks into the token fields consumed by collation/model.

It writes:
- `window["emb_chunks"]`: list of per-token embedding vectors (ragged, variable dims)
- `window["pos"]`: int positions
- `window["id"]`: int `signal_id`s
- `window["mod"]`: modality IDs (small ints)
- `window["role"]`: role IDs (`ROLE_CONTEXT`, `ROLE_ACTUATOR`)
- `window["output_emb"]`: `{signal_id → output_embedding}`

Notes:
- We intentionally do **not** carry per-token signal names or per-window output name/shape tables.
  Configs remain name-based (human-friendly), but name→id resolution happens once at startup.

Deterministic ordering:
- chunks sorted by `(pos, chunk_index_in_window)` → closest-to-output first
- signals within chunk sorted by `signal_id`

### 6) `MMTCollate` (batching + dropout masks)

Purpose: turn a list of windows into a padded, model-ready batch.

Key features:
- pads variable-length token sequences (L differs per window)
- keeps token embeddings ragged (projection happens in the model)
- applies:
  - per-token dropout (inputs / actuators)
  - per-chunk dropout (coarse time masking)
  - optional per-output dropout
- emits explicit padding semantics so PAD is never confused with a real signal

Outputs include:
- token tensors: `pos`, `id`, `mod`, `role`, `padding_mask`
- packed embeddings: `emb[sid]` and `emb_index[sid]` (ragged by signal_id)
- `output_emb` tensors stacked by `signal_id`
- optional `output_native` (eval) when enabled

---

## Model blocks (tokens → predictions)

The core model is `MultiModalTransformer`, composed of four blocks:

```mermaid
flowchart LR
  Tokens[Tokens + metadata] --> TE[TokenEncoder]
  TE --> BB[Transformer Backbone]
  BB --> CLS[CLS embedding]
  CLS --> MH[Modality heads\n(one MLP per modality)]
  MH --> OA[Output adapters\n(one per output signal)]
```

### 1) TokenEncoder

**What it does**
- projects per-token embeddings into the common `d_model`
- adds metadata embeddings: position, signal ID, modality, role
- prepends a learned **CLS token**

**Per-signal projections**
- built eagerly in `__init__`
- one `nn.Linear(embedding_dim → d_model)` per **canonical key**
- keyed by `role:name`, not numeric IDs

**Why this matters**
- strict resume works (all modules exist in state_dict)
- warm-start can match layers by stable keys

### 2) Backbone

A standard `nn.TransformerEncoder` over:

```
[CLS] + [token_1, token_2, ..., token_L]
```

Properties:
- **sequence-length agnostic** (L changes per window)
- independent of signal dimensionality (handled by TokenEncoder projections)

### 3) Modality heads

A small MLP per modality:

```
G_mod = head_mod(h_cls)
```

This creates a modality-specific latent subspace that is shared across signals of the same modality.

### 4) Output adapters

One adapter per output signal (role=`"output"`):

```
pred[signal_id] = adapter_{role:name}( G_modality )
```

Notes:
- adapter output dimension `K_t` equals the output signal `embedding_dim`
- adapters are stored in a ModuleDict keyed by canonical key
- the *returned* predictions dict is keyed by numeric `signal_id` for efficiency

---

## Checkpoints and model blocks

MMT checkpoints are split by block:

- `token_encoder.pt`
- `backbone.pt`
- `modality_heads.pt`
- `output_adapters.pt`

Typical run layout:

```
runs/<run_id>/
  checkpoints/
    latest/   # strict resume state (model + optimizer + rng, etc.)
    best/     # strict best weights (model blocks + meta)
```

Two workflows:
- **Strict resume** (same run): load all four blocks exactly (strict=True).
- **Warm-start** (new run): overlap-load by **key + tensor shape**, block-by-block, leaving unmatched parameters initialized.

(See `mmt.checkpoints` for APIs like `resume_from_latest`, `load_best_weights`, and `load_parts_from_run_dir`.)

---

## Tuning: DCT3D truncation search

The tuning pipeline uses `TuneDCT3DTransform` to evaluate and select DCT3D truncation parameters per `(role, signal)`.

Properties:
- operates on the same chunk arrays that will later be embedded
- computes errors on finite entries (ignores NaN padding)
- selects the smallest effective dimension under a threshold (configurable)

The tuned parameters are meant to be written back as *per-signal overrides* in configuration, so future runs can reuse them.

---

## Where to look in code

Core model:
- `mmt/mmt.py` — `MultiModalTransformer`
- `mmt/token_encoder.py` — `TokenEncoder`
- `mmt/backbone.py` — Transformer backbone wrapper
- `mmt/modality_heads.py` — modality MLP heads
- `mmt/output_adapters.py` — per-output adapters

Data pipeline:
- `mmt/data/transforms/*` — chunk/valid/trim/embed/build tokens + tuning
- `mmt/data/collate.py` — `MMTCollate`
- `mmt/data/signal_spec.py` — `SignalSpec`, `SignalSpecRegistry`

Checkpointing:
- `mmt/checkpoints/*`

---

## Design invariants (TL;DR)

- Everything is driven by **SignalSpecRegistry**.
- Learnable per-signal modules are keyed by **canonical strings** (`role:name`), not by numeric IDs.
- Chunk identity and caching are **index-based**, not time-based.
- The model is four clean blocks that can be saved/loaded independently.
