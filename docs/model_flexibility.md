# Model Flexibility (MMT v0)

This document describes how MMT v0 supports **flexible finetuning and evaluation** without requiring code changes.

MMT is designed so that you can:
- finetune on *subsets* of signals,
- add or remove input/actuator signals,
- add, remove, or reshape output heads,
- change chunking parameters (sequence length) safely,
- warm-start from another run with overlap loading,
- run evaluation ablations (drop signals) in a principled way.

---

## The design principle: “specs drive everything”

The central contract is:

> **Given a SignalSpecRegistry, the pipeline and the model instantiate exactly the layers required by those specs.**

A `SignalSpec` defines (role, name, modality, encoder, embedding_dim, signal_id).  
This enables the following:

- **Tokenization** (`BuildTokensTransform`) emits tokens only for signals present in the registry.
- **TokenEncoder** allocates one projection layer per **canonical key** (`role:name`) during `__init__`.
- **OutputAdapters** allocate one head per output signal (role=`output`), again keyed by canonical key.
- **The backbone is independent of signal count and signal dimensionality.**

---

## Warm-start vs strict resume

MMT supports two very different workflows.

### 1) Strict resume (continue the same run)

Use this when training was interrupted and you want to continue **exactly** the same experiment.

Characteristics:
- strict load of all four blocks
- restores optimizer / scheduler / scaler / RNG states
- requires the model structure to match exactly

### 2) Warm-start (initialize a new run from a previous run)

Use this when you want to start a **new run** (possibly with different signals, different heads, different chunking) from an existing checkpoint.

Warm-start is **overlap-based**:
- parameters are loaded only when **state_dict key + tensor shape** match
- everything else remains initialized

MMT implements warm-start per block:
- token encoder
- backbone
- modality heads
- output adapters

This is what makes cross-task finetuning robust.

---

## What you can change safely

### A) Change the set of input/actuator signals

**Supported.** If a signal is missing at finetune time:
- the pipeline simply produces **no tokens** for it,
- TokenEncoder never routes through that projection,
- the backbone receives fewer tokens and continues normally.

If you *add* a new input/actuator signal:
- the SignalSpecRegistry includes it,
- TokenEncoder allocates a new projection for it,
- warm-start will not load weights for that projection (it didn’t exist), so it starts random.

### B) Change the set of outputs

**Supported.**
- If you add a new output, a new OutputAdapter is created and starts random.
- If you remove an output, the adapter simply does not exist in the new model (and any old weights are ignored during warm-start overlap loading).

### C) Change encoder or embedding dimension for an existing signal

**Supported via partial loading.**
- If an input/actuator signal’s `embedding_dim` changes, its TokenEncoder projection layer shape changes → warm-start skips it → it is reinitialized.
- If an output signal’s `embedding_dim` changes, the OutputAdapter output dimension changes → warm-start skips it → it is reinitialized.

The rest of the model (backbone, other projections, other adapters) can still be reused.

### D) Change chunking / sequence length

**Supported, with one important constraint: positional capacity.**

Chunking parameters affect:
- how many chunks exist per window,
- how many tokens the collate produces,
- how long the transformer sequence is.

They **do not** fundamentally change the backbone weights.

However:
- Token positions are embedded via a learned table sized by `model.max_positions`.
- Your `preprocess.trim_chunks.max_chunks` must satisfy:

```
preprocess.trim_chunks.max_chunks <= model.max_positions
```

If you increase `max_chunks`, you must also increase `model.max_positions`.  
This changes the TokenEncoder `pos_embed` shape, so warm-start may reinitialize that table (shape mismatch), which is usually fine for finetuning.

---

## Finetuning flexibility: freezing and staged training

MMT supports stage-based finetuning where you can freeze or train each model block independently:

- `token_encoder`
- `backbone`
- `modality_heads`
- `output_adapters`

A common pattern:
1. **Stage 1**: freeze backbone, train token encoder + adapters (fast adaptation)
2. **Stage 2**: unfreeze backbone with smaller LR (full finetune)

MMT validates “freeze means lr=0”:
- if `freeze.<block>=True`, the validator forces LR/WD to 0 for that block to avoid silent misconfiguration.

---

## Evaluation flexibility

Evaluation can be made flexible in two complementary ways.

### 1) Evaluate with subsets of signals (via registry)

You can build a SignalSpecRegistry that contains only the signals you want to evaluate.  
This is the most “structural” ablation: the pipeline will not emit tokens for excluded signals.

Tradeoff:
- if you remove signals from the registry, you are also changing the model structure (TokenEncoder projections), which can complicate strict checkpoint loading.

### 2) Evaluate ablations without changing the model (recommended)

A more practical approach is:
- keep the same registry/model as training,
- **drop tokens at collate time** using per-signal dropout overrides.

This is especially useful to test robustness to missing sensors.

Conceptually:
- set input/actuator drop probability for a specific signal name to `1.0`
- the collate produces masks so dropped tokens contribute nothing
- the model sees “missing” tokens without you rebuilding the model

This keeps evaluation comparable and avoids shape churn.

### Output masking

Outputs are tracked with `output_mask` at batch time.
- If an output is absent or dropped, losses/metrics should ignore it.
- This lets you evaluate only a subset of outputs even if the model has more heads.

---

## Practical recipes

### Pretrain → finetune on a new task (some overlap, some new signals)

1. Keep the same `d_model`, backbone config, modality head config.
2. Provide a new SignalSpecRegistry for the finetune task.
3. Warm-start from the pretrain run and load the blocks you want (commonly backbone + heads; sometimes token encoder too).
4. In early stages, freeze the backbone so new projections / new heads stabilize quickly.

### “Re-head” outputs (same backbone, new output dimensionality)

1. Change the output codec / `embedding_dim` for one output.
2. Warm-start: backbone + other heads load; the changed adapter reinitializes.
3. Train adapters (and optionally heads/backbone) to adapt.

### Robustness evaluation: drop one input signal

Keep the model as-is but in evaluation config:
- set a per-signal input drop override to `1.0` for that signal name.

This simulates missing data with *no* code changes and keeps the run comparable.

---

## What this buys you

Because the system is factorized into:

**(Signal specs → deterministic tokenization) + (modular model blocks) + (mask-aware training/eval)**

…you can iterate on tasks and experiments without “ID-based weirdness”, and without constantly rewriting code to handle new signal sets.

---

## Summary checklist

When changing configs between runs:

- ✅ You can freely add/remove signals (inputs, actuators, outputs).
- ✅ You can change encoder types and embedding dims (affected layers reinit).
- ✅ You can change chunking (ensure `max_positions` covers `max_chunks`).
- ✅ You can warm-start overlap-load (key + shape match).
- ✅ You can do eval ablations via collate drop overrides.

The architecture is intentionally built so these changes are “configuration-only” operations.
