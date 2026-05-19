# Training

Related documentation: [Project README](../README.md) | [Configuration Reference](config_reference.md) | [Datasets](datasets.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md) | [Evaluation](evaluation.md)

This document covers the training loop configuration: stages, loss, NaN handling, AMP, early stopping, and resume behavior.

## Entry Scripts

| Script | Purpose |
|---|---|
| `run_pretrain.py` | Train a model from scratch on all task signals. |
| `run_finetune.py` | Finetune from a pretrained checkpoint (`--init warmstart`) or from scratch (`--init scratch`). |

Both scripts share the same training loop (`train_finetune`) and configuration structure. The difference is in model initialization and embedding resolution — see [Checkpointing and Warmstart](checkpointing_and_warmstart.md).

## Stages

Training is split into one or more sequential stages, each with independent learning rates, weight decay, and freeze settings. Stages run in order; epoch counters continue across stages.

The stage schedule depends on the finetune init mode and lives in the init-mode-specific config file, not in `finetune.yaml` itself:

**`--init warmstart`** (`finetune_warmstart.yaml`) — two stages:
```yaml
train:
  stages:
    - name: ft_heads        # freeze backbone + token_encoder; adapt output side only
      epochs: 5
      optimizer:
        lr:
          backbone: 0.0
          token_encoder: 0.0
          modality_heads: 1e-3
          output_adapters: 5e-3
      freeze:
        backbone: true
        token_encoder: true
        modality_heads: false
        output_adapters: false

    - name: ft_full         # unfreeze everything; joint fine-tuning
      epochs: 15
      optimizer:
        lr:
          backbone: 5e-4
          token_encoder: 5e-4
          modality_heads: 1e-3
          output_adapters: 5e-3
      freeze:
        backbone: false
        token_encoder: false
        modality_heads: false
        output_adapters: false
```

**`--init scratch`** (`finetune_scratch.yaml`) — single stage, all blocks trainable from epoch 0:
```yaml
train:
  stages:
    - name: ft_scratch
      epochs: 20
      optimizer:
        lr:
          backbone: 1e-3
          token_encoder: 5e-3
          modality_heads: 5e-3
          output_adapters: 5e-3
      freeze:
        backbone: false
        token_encoder: false
        modality_heads: false
        output_adapters: false
```

### LR/WD inheritance
If `lr.<block>` or `wd.<block>` is `null`, it inherits the `backbone` value. Use this to set a single rate that applies uniformly across all unfrozen blocks.

### Freeze rules
Setting `freeze.<block>: true` forces `lr` and `wd` to zero for that block regardless of what is specified. A warning is logged if non-zero values are overridden.

### Gradient accumulation
`scheduler.grad_accum_steps` accumulates gradients over N batches before each optimizer step. The loss is divided by N before backprop so the effective gradient scale matches a single step. Useful when RAM limits batch size.

### Warmup
`scheduler.warmup_steps_fraction` (optional, default `0.1`) sets the fraction of total steps in the stage used for linear LR warmup. Setting `0.0` disables warmup.

## Loss Configuration

```yaml
train:
  loss:
    terms:
      - type: embed_mse
        weight: 1.0
      - type: native_sparse_mse   # optional second term
        weight: 0.5
    output_weights:               # optional per-output scaling
      summary-ip: 1.2
```

### Term types

| Type | Space | NaN handling | Cost |
|---|---|---|---|
| `embed_mse` | Embedding (coefficient) | Implicit: NaN positions are zero-filled before encoding; the loss trains against the embedding of the zero-imputed signal with no explicit NaN awareness. | Cheapest — no decoding. |
| `native_sparse_mse` | Native (standardized) | Explicit: model predictions are decoded back to native space, then NaN positions from `output_native` are masked out before the mean. Only observed positions contribute. | Requires decoder forward pass per batch. |

When multiple terms are present, the total loss is a normalized weighted sum: `total = Σ(w_i · L_i) / Σ(w_i)`.

`output_weights` scales individual outputs within each term independently (not applied across terms).

### When to use which

- **`embed_mse`** is the default. It is fast and works well when output signals are reliably observed or when zero-imputed coefficients are an acceptable training proxy for the signal structure.
- **`native_sparse_mse`** is preferable when output signals have systematic NaN gaps (partial channels, timestep dropouts) and you want the loss to ignore those positions explicitly rather than train through imputed zeros.
- Both terms can be combined with independent weights for a mixed objective.

### Identity-encoded outputs

When an output signal uses `encoder_name: identity` (e.g. in a `dct3d_native_outputs` embedding profile),
`EmbedChunksTransform` **skips the embedding step** for that signal and does not store an `output_emb` entry.
This avoids holding two copies of the same data in memory — particularly important for large spatial outputs
(Thomson scattering profiles, bolometry) where caching the embedded and native arrays separately would cause
significant RAM overhead.

Consequences:
- `native_sparse_mse` works as normal — it reads from `output_native`, which is always present.
- `embed_mse` silently ignores identity-encoded outputs (no entry in `output_emb` → not supervised by that term).
- Using `embed_mse` as the sole loss term with identity outputs is a no-op for those outputs; combine with `native_sparse_mse` instead.

### `keep_output_native` auto-derivation
`data.keep_output_native` is derived automatically by the config validator — do not set it manually:
- eval phase → always `true`
- train phase → `true` iff any term in `train.loss.terms` is `native_sparse_mse` (or any future native-space loss)

This controls whether `FinalizeWindowTransform` retains native output arrays in the window dict so the loss (and eval scoring) can access them.

## NaN Handling in Training

Two independent knobs control how NaN values are handled during training:

### 1. Window selection: `accept_nan_outputs`
`preprocess.valid_windows.accept_nan_outputs` (default `true`) determines whether windows with partial-NaN output signals are kept or dropped before reaching the loss.

Typical training setting:
```yaml
preprocess:
  valid_windows:
    accept_nan_inputs_actuators: true   # keep partial-NaN inputs; zeros imputed before encoding
    accept_nan_outputs: false           # drop windows with any NaN in output targets
```

Setting `accept_nan_outputs: false` prevents corrupted or partially missing targets from entering the loss. Setting it to `true` (as in eval) allows those windows through — correct when the loss can handle NaN positions explicitly (e.g. `native_sparse_mse`) or when the evaluator uses `nanmean`.

### 2. Encoding: `embeddings.impute_na`
`embeddings.impute_na` (default `true`) controls whether NaN values are zero-filled before `codec.encode()`.

```yaml
embeddings:
  impute_na: true
```

- `true` (default): NaN → zero (signal mean in standardized space) on a local copy immediately before encoding. The original `window["output"][name]["values"]` is **never modified**, preserving NaN locations for eval metrics and for `native_sparse_mse` targets.
- `false`: no imputation. Only valid when all codecs can handle NaN inputs natively. An error is raised at pipeline construction if any codec has `requires_finite_input=True` (all current codecs do).

### Combined behavior

| `accept_nan_outputs` | `impute_na` | Loss term | Outcome |
|---|---|---|---|
| `false` | `true` | `embed_mse` | Windows with NaN outputs are dropped; remaining windows are fully observed. Cleanest training signal. |
| `true` | `true` | `embed_mse` | NaN positions become zeros in the target embedding. Loss trains through imputed positions with no masking. |
| `true` | `true` | `native_sparse_mse` | NaN positions imputed before encoding, but `output_native` preserves original NaNs. Loss masks imputed positions out at the native level. |
| `true` | `true` | `native_sparse_mse` + identity output | Embedding step is skipped entirely for identity outputs — no `output_emb` stored, no RAM duplication. `output_native` is the only copy. Loss masks NaN positions as normal. |

## AMP (Mixed Precision)

```yaml
train:
  amp:
    enable: true
```

When enabled, forward passes run under `torch.autocast` using `bfloat16` (or `float16` on older hardware). Loss computation is kept in `float32` for numerical stability. Gradient scaling is applied automatically via `GradScaler`.

AMP is enabled by default and recommended for all GPU training. Disable for CPU or MPS runs where autocast support is limited.

## Early Stopping

```yaml
train:
  early_stop:
    patience: 10
    delta: 0.0
```

Training stops early if validation loss does not improve by more than `delta` for `patience` consecutive validation epochs. The best checkpoint is always preserved independently of early stopping.

## Resume vs Warmstart

- **Resume** (`train.resume: true`): continues the same run directory, restoring model weights, optimizer, scheduler, scaler, and RNG state. Use to recover from interruptions.
- **Warmstart** (`--init warmstart --model <run_id>`): loads model weights from another run and starts a fresh optimizer/scheduler/RNG. Use for finetune from pretrain.
- **Scratch** (`--init scratch`): initializes from config only. No source weights loaded.

Resume and warmstart are mutually exclusive. See [Checkpointing and Warmstart](checkpointing_and_warmstart.md) for details.
