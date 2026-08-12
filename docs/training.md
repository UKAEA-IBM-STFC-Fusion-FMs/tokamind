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

The full finetune recipe depends on the init mode and lives in the init-mode-specific config file:

The `mmt` profile uses the `token_encoder`, `backbone`, `modality_heads`, and `output_adapters` blocks.

**`--init warmstart`** (`mmt/phases/finetune_warmstart.yaml`) — two stages:
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
If `lr.<block>` or `wd.<block>` is `null`, it inherits the `backbone` value. Use this to set a single rate that applies
uniformly across all unfrozen blocks.

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
| `embed_mse` | Embedding (coefficient) | Implicit: NaN positions are imputed before encoding according to `preprocess.embed_chunks.nan_imputation`; the loss trains against the embedding of the imputed signal with no explicit NaN awareness. | Cheapest — no decoding. |
| `native_sparse_mse` | Native (standardized) | Explicit: model predictions are decoded back to native space, then NaN positions from `output_native` are masked out before the mean. Only observed positions contribute. | Requires decoder forward pass per batch. |
| `embed_gaussian_crps` | Embedding (coefficient) | Same implicit handling as `embed_mse`. | Cheap — closed-form, no sampling/decoding. **Needs a Gaussian head.** |
| `native_sparse_sample_crps` | Native (standardized) | Same explicit NaN masking as `native_sparse_mse`. | `n_samples` × decoder forward passes per batch. **Needs a Gaussian head.** |

The two CRPS terms are *probabilistic*: they score the predictive distribution (mean **and** calibrated spread), not just the point prediction, and require `model.output_adapters.type: gaussian` (see [Probabilistic (Gaussian) head](#probabilistic-gaussian-head)). `embed_gaussian_crps` is closed-form; `native_sparse_sample_crps` takes an `n_samples` field (`>= 2`, default `16`) controlling how many reparameterized samples are decoded and scored.

When multiple terms are present, the total loss is a normalized weighted sum: `total = Σ(w_i · L_i) / Σ(w_i)`.

`output_weights` scales individual outputs within each term independently (not applied across terms).

Each term can optionally select outputs by name:

```yaml
train:
  loss:
    terms:
      - type: embed_mse
        weight: 1.0
        outputs:
          include:
            - soft_x_rays-horizontal_cam_lower

      - type: native_sparse_mse
        weight: 1.0
        outputs:
          exclude:
            - soft_x_rays-horizontal_cam_lower
```

Use either `outputs.include` or `outputs.exclude`, not both. Empty lists and unknown output names are startup errors.
When no `outputs` block is provided, the term applies to every output it can supervise.

### When to use which

- **`embed_mse`** is the default. It is fast and works well when output signals are reliably observed or when the configured imputed coefficients are an acceptable training proxy for the signal structure.
- **`native_sparse_mse`** is preferable when output signals have systematic NaN gaps (partial channels, timestep dropouts) and you want the loss to ignore those positions explicitly rather than train through imputed zeros.
- **`embed_gaussian_crps` / `native_sparse_sample_crps`** train calibrated predictive uncertainty (a Gaussian head) rather than only a point estimate. Use `embed_gaussian_crps` for cheap coeff-space CRPS; use `native_sparse_sample_crps` to score CRPS in native physical space (the only option when the decoder is nonlinear, e.g. VAE).
- Terms can be combined with independent weights for a mixed objective.

### Probabilistic (Gaussian) head

The CRPS terms require a probabilistic output head:

```yaml
model:
  output_adapters:
    type: gaussian            # deterministic (default) | gaussian
    gaussian:
      raw_scale_bias_init: -1.0   # initial sigma; lower = start near-deterministic
train:
  loss:
    terms:
      - type: embed_gaussian_crps
        weight: 1.0
      # - type: native_sparse_sample_crps
      #   weight: 1.0
      #   n_samples: 16
```

A Gaussian head emits a per-coordinate mean and scale; `pred[sid]` stays the mean (so MSE terms and eval keep
working), and the distribution is exposed as `pred_dist` for the CRPS terms. The mean head shares the deterministic
adapter's layout, so a deterministic checkpoint warm-starts the Gaussian mean head (only the scale head starts
fresh). The config validator rejects a CRPS term against a `deterministic` head with an error naming both the term
and the knob to flip.

Note: a Gaussian head trained with a point loss (`embed_mse` / `native_sparse_mse`) is equivalent to the
deterministic head — those losses ignore the scale. The scale is only trained by a probabilistic objective.

### Identity-encoded outputs

When an output signal uses `encoder_name: identity` in an embedding profile,
`EmbedChunksTransform` **skips the embedding step** for that signal and does not store an `output_emb` entry.
This avoids holding two copies of the same data in memory — particularly important for large spatial outputs
where caching the embedded and native arrays separately would cause
significant RAM overhead.

Consequences:
- Native-space terms (`native_sparse_mse`, `native_sparse_sample_crps`, `grad_shafranov_residual`) work as normal — they read from `output_native`, which is always present.
- Embedding-space terms (`embed_mse`, `embed_gaussian_crps`) cannot supervise identity-encoded outputs because they have no `output_emb` entry.
- Startup validation raises if an identity output is explicitly included in an embedding-space term, or if any output is not
  supervised by at least one capable loss term.

### `keep_output_native` auto-derivation
`data.keep_output_native` is derived automatically by the config validator — do not set it manually:
- eval phase → always `true`
- train phase → `true` iff any term in `train.loss.terms` is a native-space loss (`native_sparse_mse`,
  `native_sparse_sample_crps`, `grad_shafranov_residual`)

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

### 2. Encoding: `preprocess.embed_chunks.nan_imputation`
`preprocess.embed_chunks.nan_imputation` (default `"zero"`) controls how NaN/inf values are handled before DCT3D rank tuning and before runtime `codec.encode()`.

```yaml
preprocess:
  embed_chunks:
    nan_imputation: zero  # zero | interpolate | null
```

- `"zero"` (default): NaN → zero on a local copy immediately before encoding. If data are standardized, zero corresponds to the signal mean; otherwise this is a literal zero-fill. Fast but creates hard step discontinuities at NaN/valid boundaries, which DCT3D can amplify.
- `"interpolate"`: fills NaN via temporal then spatial linear interpolation with zero fallback. Avoids step edges — preferred when signals have structured boundary NaN (transitions from observed values to non-finite at a boundary).
- `null`: no imputation; the array is passed to the codec as-is. Allowed only when all registered codecs can handle non-finite inputs natively; current finite-only codecs raise at construction time.

In all runtime embedding cases, the original `window["output"][name]["values"]` is **never modified**, preserving NaN locations for eval metrics and for `native_sparse_mse` targets. DCT3D tuning uses the same imputation strategy only to select coefficient indices from finite arrays.

### Combined behavior

| `accept_nan_outputs` | `nan_imputation` | Loss term | Outcome |
|---|---|---|---|
| `false` | any | `embed_mse` | Windows with NaN outputs are dropped; remaining windows are fully observed. Cleanest training signal. |
| `true` | `"zero"` | `embed_mse` | NaN positions become zeros in the target embedding. In standardized data this is mean imputation. Loss trains through imputed positions with no masking. Can corrupt DCT3D for signals with structured boundary NaN. |
| `true` | `"interpolate"` | `embed_mse` | NaN positions filled by interpolation before encoding. Avoids step discontinuities — better DCT3D representation for structured boundary NaN. |
| `true` | any | `native_sparse_mse` | NaN positions imputed before encoding, but `output_native` preserves original NaNs. Loss masks imputed positions out at the native level. |

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
- **Warmstart** (`--init warmstart --model_source <run_id>`): loads model weights from another run and starts a fresh optimizer/scheduler/RNG. Use for finetune from pretrain.
- **Scratch** (`--init scratch`): initializes from config only. No source weights loaded.

Resume and warmstart are mutually exclusive. See [Checkpointing and Warmstart](checkpointing_and_warmstart.md) for details.
