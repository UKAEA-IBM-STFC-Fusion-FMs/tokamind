# DCT3D Tuning

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [Configuration Reference](config_reference.md)

DCT3D tuning selects rank-mode coefficients from data and writes run-local embedding artifacts.

## Core Idea
The codec supports two selection modes:
- `spatial`: fixed low-frequency block by `keep_h/keep_w/keep_t`
- `rank`: top coefficients by explained energy

Tuning computes rank selections and stores them per run. The objective is role-specific: each role can target different explained-energy thresholds and budgets.

## Where Tuning Runs
DCT3D tuning policy is fixed by phase and init mode:

```text
pretrain            input=tune    actuator=tune    output=tune
finetune scratch    input=tune    actuator=tune    output=tune
finetune warmstart  input/actuator existing=source, missing=tune; output=tune
eval                inherit source run
```

DCT3D warmstart policy is signal-aware. A signal is `existing` when the source run has an entry for it in
`embeddings/dct3d.yaml`; otherwise it is `missing`. Warmstart inherits existing input/actuator artifacts, tunes new
input/actuator signals, and tunes outputs.

Explicit manual `per_signal_overrides` are excluded from tune/source policy and use the configured value directly.
VAE profiles do not use this policy. If a DCT3D-derived profile marks an output as `identity`, the tuner skips that
signal because its final encoder is not `dct3d`.

## Runtime Artifacts
When rank tuning is used, a run writes:

```text
runs/<run_id>/embeddings/
  dct3d.yaml
  dct3d_indices/
    <role>_<signal>.npy
```

`dct3d.yaml` stores `embeddings.per_signal_overrides` with rank metadata. Each `.npy` file stores 1D coefficient indices consumed by rank-mode codecs at runtime.

Each tuned signal entry includes:
- rank payload: `coeff_shape`, `num_coeffs`, `explained_energy`
- coverage stats: `dim_distribution.unique_h/unique_w/unique_t`
- policy trace: `target_energy`, `k_target`, `guardrail_min_k`, `k_after_guardrails`, `k_final`, `max_budget`, `n_windows`, `flags`, `tuned_in_run_id`

## Key Config Block
Base tuning settings live in `scripts_mast/configs/<model_profile>/embeddings/dct3d/_default.yaml`:

```yaml
embeddings:
  dct3d:
    per_signal_overrides: {}

    tuning:
      common:
        n_shots: 100
        max_windows: 15000
        guardrails:
          enable: true

      pretrain:
        objective:
          thresholds:
            input: 0.999
            actuator: 0.999
            output: 0.995
          max_budget:
            input: 4096
            actuator: 4096
            output: 4096

      finetune:
        objective:
          thresholds:
            input: 0.999
            actuator: 0.999
            output: 0.999
          max_budget:
            input: 4096
            actuator: 4096
            output: 8192
```

The loader materializes the effective runtime block as `embeddings.tuning` for the tuner by merging
`embeddings.dct3d.tuning.common` with `embeddings.dct3d.tuning.<phase>`.

Parameter intent:
- `preprocess.embed_chunks.nan_imputation`: NaN/inf policy used before full-DCT energy accumulation
- `n_shots`: number of shots sampled for tuning statistics
- `max_windows`: upper bound on analyzed windows
- `thresholds`: minimum explained energy target by role
- `max_budget`: hard cap on selected coefficients by role
- `guardrails`: optional sanity checks to avoid under-dimensioned selections

## Selection Policy
`TuneRankedDCT3DTransform` applies selection in this order for each DCT3D signal:
1. Compute `K_target` from explained-energy threshold.
2. If guardrails are enabled, lift K to satisfy modality-specific minimum coverage.
3. Apply role budget as a hard cap.

Budget always caps final dimensionality. When budget prevents full guardrail satisfaction, tuning keeps the capped K and emits a warning.

## Guardrails by Modality
Guardrails operate on canonical DCT dimensions `(H, W, T)`:
- timeseries: usually enforces `min_unique_t`
- profile: usually enforces `min_unique_h` and `min_unique_t`
- video: usually enforces `min_unique_h`, `min_unique_w`, and `min_unique_t`

## Profile Files
Task profile files under `<model_profile>/embeddings/<profile>/<task>.yaml` should keep only authored profile choices.

Do not store:
- `coeff_indices` arrays
- committed per-run rank artifacts

Run-local artifacts belong in `runs/<run_id>/embeddings/`.

## Troubleshooting
### Missing inherited source signal in finetune
- Cause: warmstart can only inherit input/actuator signals that have entries in the source run's
  `embeddings/dct3d.yaml`.
- Fix: use `--init scratch`, add a manual per-signal override, or warmstart from a run with matching DCT3D artifacts.

### Rank mode cannot load indices
- Cause: missing `dct3d_indices/*.npy` for a rank override.
- Fix: ensure the run has matching artifacts in `runs/<run_id>/embeddings/`.

### No tuning executed
- Cause: the selected profile does not participate in DCT3D tuning, or all task signals use non-DCT3D encoders.
- Fix: use a DCT3D-derived profile when rank tuning is intended.
