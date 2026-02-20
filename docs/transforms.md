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

### 3) TrimChunksTransform
- keeps at most `max_chunks`
- derives position indices used by token encoding

### 4) EmbedChunksTransform
- applies per-signal codecs from `signal_specs`
- uses codec map built by `build_codecs`
- outputs fixed-width embedding vectors per signal/chunk

### 5) BuildTokensTransform
- converts embedded chunks into token fields
- emits role/modality/signal-id/position metadata
- prepares the tensor layout expected by `TokenEncoder`

### 6) FinalizeWindowTransform
- keeps or drops native output payload based on `keep_output_native`
- this controls whether eval can score/trace in native units

## Codec Modes Relevant to Transforms
For DCT3D signals:
- spatial mode uses configured `keep_h/keep_w/keep_t`
- rank mode reads `coeff_indices_path` from run-local embedding artifacts

Rank mode is deterministic once index files are fixed for a run.

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

## Extension Points
- Add a custom window filter transform before embedding.
- Add a codec implementing `encode`/`decode` and register it in codec factory logic.
- Add extra metadata fields in token-building stage if downstream code uses them.

When extending the chain, keep input/output field contracts explicit to avoid silent sample drops.
