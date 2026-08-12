# Model Architecture

Related documentation: [Project README](../README.md) | [Transforms](transforms.md) | [Model Flexibility](model_flexibility.md)

This document describes the `mmt` model in `src/mmt/models`.

## End-to-End Path

1. Raw windows are transformed into tokenized batches.
2. `TokenEncoder` projects signal embeddings to the shared model dimension and adds role, modality, and position information.
3. `Backbone` processes the complete token sequence.
4. `ModalityHeads` produce modality-specific output features.
5. `OutputAdapters` map those features to each requested output embedding.
6. Codec decoding maps predictions back to native output space for losses, metrics, and traces.

## Core Blocks

### TokenEncoder

- projects per-signal embeddings to `d_model`;
- adds role/modality/position embeddings;
- keys signal-specific modules by `"<role>:<name>"`.

### Backbone

The Transformer encoder mixes information across signals and chunk positions while preserving the sequence length and
hidden dimension.

### ModalityHeads

These modality-specific intermediate projections provide separate post-backbone parameterisations for each output
modality.

### OutputAdapters

Per-output heads keyed by canonical output signal name map shared hidden features to the target embedding dimension.
`model.output_adapters.type` selects a deterministic or diagonal-Gaussian output head.

## Checkpoint Blocks

The stable config/checkpoint blocks are `token_encoder`, `backbone`, `modality_heads`, and `output_adapters`. Each is
saved as `<block_name>.pt`; the same names are used by stage optimiser, freeze, and warm-start configuration.
