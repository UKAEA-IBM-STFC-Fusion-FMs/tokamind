# Model Architecture

Related documentation: [Project README](../README.md) | [Transforms](transforms.md) | [Model Flexibility](model_flexibility.md)

This document describes the core model stack in `src/mmt/models`.

## End-to-End Path
1. Raw windows are transformed into tokenized batches.
2. Token features are projected by `TokenEncoder`.
3. `Backbone` processes token sequences.
4. `ModalityHeads` map backbone features by modality.
5. `OutputAdapters` produce per-output predictions.
6. Codec decode step maps predictions back to native output space for metrics/traces.

## Core Blocks
### TokenEncoder
Responsibilities:
- project per-signal embeddings to shared `d_model`
- add role/modality/position embeddings
- route by canonical signal key: `"<role>:<name>"`
- output one hidden vector per token for the backbone

### Backbone
Transformer encoder stack over the full token sequence.
- mixes information across signals and chunk positions
- preserves sequence length and hidden size (`d_model`)

### ModalityHeads
Optional modality-specific intermediate projection after backbone output.
- lets different modalities use separate post-backbone parameterization

### OutputAdapters
Per-output heads keyed by canonical output signal key.
- each head maps shared hidden features to the target embedding/output dimension

## Signal-Driven Construction
Model construction depends on `SignalSpecRegistry`:
- each role-specific signal has stable `signal_id`
- each signal includes `encoder_name`, `encoder_kwargs`, and `embedding_dim`
- output adapters are instantiated only for output-role specs

## Canonical Keying
Learnable signal-specific modules use:

```text
canonical_key = "<role>:<name>"
```

This keeps checkpoint matching stable across task variants.

## Token Payload
Tokenized batches include a feature tensor plus compact metadata:
- role id (input/actuator/output)
- modality id (timeseries/image/scalar, task-dependent)
- signal id (stable per signal key)
- position indices (chunk/time position)

These fields let the model share one backbone while preserving signal identity.

## Shape Expectations
- Input batch contains token tensors and metadata fields.
- Model output is a mapping from output signal ids to predicted embeddings/targets, depending on decode stage.

Typical batch axis order:
- `batch`
- `token`
- `feature`

## Checkpoint Blocks
Training checkpoints store four model blocks:
- token encoder
- backbone
- modality heads
- output adapters

See [Checkpointing and Warmstart](checkpointing_and_warmstart.md) for loading behavior.
