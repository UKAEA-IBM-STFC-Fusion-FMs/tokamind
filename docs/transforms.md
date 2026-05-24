# Transforms

Related documentation: [Project README](../README.md) | [Datasets](datasets.md) | [Model Architecture](model_architecture.md) | [DCT3D Tuning](tuning_dct3d.md)

This document summarizes the preprocessing chain from raw windows to tokenized model inputs.

## Transform Contract
Each transform receives one `window: dict` and returns:
- modified `window: dict`, or
- `None` to drop the sample.

`ComposeTransforms` applies stages in order and stops when a stage returns `None`.

Ordering matters: later stages assume fields produced by earlier stages.

## Standard Chain
The shared entry helpers use:
1. `ChunkWindowsTransform`
2. `SelectValidWindowsTransform`
3. `TrimChunksTransform`
4. `EmbedChunksTransform`
5. `BuildTokensTransform`
6. `FinalizeWindowTransform`


## Stage Summary
### 1) ChunkWindowsTransform
- builds fixed chunk slots for input/actuator history
- records `chunk_index_in_window` and `chunk_index_global`

### 2) SelectValidWindowsTransform
- filters invalid windows by configured thresholds
- supports temporal subsampling via `window_stride_sec`
- `accept_nan_inputs_actuators` (default `True`): controls NaN tolerance for input/actuator signals; `True` passes partial-NaN signals through for downstream imputation, `False` masks them as invalid
- `accept_nan_outputs` (default `True`): same control for output signals; set to `False` in finetune/pretrain to drop windows with partial-NaN outputs from the training loss
- typical training config: `accept_nan_inputs_actuators=True`, `accept_nan_outputs=False` — keeps NaN-input windows so the model trains on the imputed-input distribution seen at eval, while excluding corrupted output targets from the loss

### 3) TrimChunksTransform
- keeps at most `max_chunks`
- derives position indices used by token encoding

### 4) EmbedChunksTransform
- applies per-signal codecs from `signal_specs`
- uses codec map built by `build_codecs`
- outputs fixed-width embedding vectors per signal/chunk
- NaN imputation is controlled by `preprocess.embed_chunks.nan_imputation` (default `"zero"`):
  - `"zero"`: zero-fill on a local copy before encoding; zero equals the signal mean only when data are standardized
  - `"interpolate"`: temporal then spatial linear interpolation with zero fallback; avoids hard step
    discontinuities at NaN boundaries that would contaminate DCT3D low-frequency coefficients — preferred
    for signals with structured boundary NaN (transitions from observed values to non-finite at a boundary)
  - `null`: no imputation; allowed only when all registered codecs can handle non-finite inputs natively
- for output signals: imputation is applied to a local copy only; original `window["output"][name]["values"]`
  is never modified, preserving NaN locations for benchmark-comparable eval metrics
- **identity-encoded outputs are not embedded**: if an output signal uses `encoder_name: identity`, the
  embedding step is skipped entirely for that signal — `window["output_emb"]` will not contain an entry for
  it. This avoids duplicating large output arrays in memory (the native values kept by `FinalizeWindowTransform`
  are sufficient for `native_sparse_mse`). As a consequence, `embed_mse` silently ignores identity-encoded
  outputs; use `native_sparse_mse` for those signals.

### 5) BuildTokensTransform
- converts embedded chunks into token fields
- emits role/modality/signal-id/position metadata
- prepares the tensor layout expected by `TokenEncoder`

### 6) FinalizeWindowTransform
- keeps or drops native output payload based on `keep_output_native`
- `keep_output_native` is **auto-derived** by the config validator — never set manually:
  - eval phase: always `true`
  - train phase: `true` iff any `train.loss.terms` entry is a native-space loss (e.g. `native_sparse_mse`)


## Configuration Keys
Main transform-related keys:

```yaml
preprocess:
  chunk:
    chunk_length: 0.005
    stride: null
  trim_chunks:
    max_chunks: 50
  valid_windows:
    min_valid_inputs_actuators: 1
    min_valid_outputs: 1
    min_valid_chunks: 1
    window_stride_sec: 0.01
```

## Tuning-Only Transform
`TuneRankedDCT3DTransform` is used only during DCT3D rank tuning

- role: collects pooled coefficient energies and selects rank-mode coefficients
- policy order: threshold target -> guardrail lift -> hard budget cap
- output: selected indices plus tuning metadata consumed by
  `runs/<run_id>/embeddings/dct3d.yaml`
- implementation: `src/mmt/data/transforms/tune_ranked_dct3d.py`
- detailed behavior: see [DCT3D Tuning](tuning_dct3d.md)

