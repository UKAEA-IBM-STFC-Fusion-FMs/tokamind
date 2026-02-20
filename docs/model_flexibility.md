# Model Flexibility

Related documentation: [Project README](../README.md) | [Checkpointing and Warmstart](checkpointing_and_warmstart.md) | [Model Architecture](model_architecture.md)

The model and pipeline are spec-driven. Changes in signal specs are handled through overlap-based warmstart and dynamic module construction.

## What Can Change Between Runs
### Input/Actuator signal set
Supported.
- Missing signals: no tokens are emitted for those keys.
- Added signals: corresponding encoder projections are initialized from config.

### Output signal set
Supported.
- Added outputs: corresponding output adapters are initialized from config.
- Removed outputs: adapters are absent in target run and ignored during loading.

### Per-signal embedding dimension
Supported.
- Affected projection/adapters may have shape mismatch and stay initialized.
- Matching tensors in other blocks are still reused.

### Chunk history length
Supported if compatible with model positional capacity.
- `preprocess.trim_chunks.max_chunks` must fit the positional embedding design.

## Warmstart Behavior
Warmstart loads by state key and shape.

Result:
- matching tensors are reused
- non-matching tensors stay initialized

This enables task-specific signal coverage changes without manual checkpoint surgery.

In practice, reuse quality is proportional to overlap between source and target specs.

## Resume Behavior
Strict resume continues the same run directory and restores optimizer/scheduler/scaler/rng state.

Resume is continuity; warmstart is transfer.

## Finetune Strategy
A common pattern is staged optimization:
1. train adapters and heads with conservative backbone updates
2. unfreeze broader blocks for full adaptation

Use `train.stages[*].freeze.*`, `optimizer.lr.*`, and `optimizer.wd.*` for each stage.

## Practical Limits
- Finetune and eval require a valid source run selected by `--model`.
- Eval expects source run artifacts and config snapshot to be present.
- Rank-mode codecs require corresponding `embeddings/dct3d_indices/*.npy` files in run directory.

## What Must Stay Stable
For safe weight reuse, keep these identifiers stable when possible:
- canonical signal keys (`<role>:<name>`)
- block names in checkpoint state dicts
- major backbone dimensions (`d_model`, heads/layers) when expecting high overlap
