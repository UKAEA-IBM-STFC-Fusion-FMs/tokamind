# DCT3D Tuning

Related documentation: [Project README](../README.md) | [Configuration Guide](config_guide.md) | [Configuration Reference](config_reference.md)

DCT3D tuning selects rank-mode coefficients from data and writes run-local embedding artifacts.

## Core Idea
The codec supports two selection modes:
- `spatial`: fixed low-frequency block by `keep_h/keep_w/keep_t`
- `rank`: top coefficients by explained energy

Tuning computes rank selections and stores them per run.
The objective is role-specific: each role can target different explained-energy thresholds and budgets.

## Where Tuning Runs
Tuning is integrated in training scripts and controlled by config:
- pretrain: role selection from `embeddings.tune_embeddings.roles`
- finetune: role selection from `embeddings.tune_embeddings.roles`, combined with `embeddings.mode`

There is no separate tuning phase in the open-source flow.

## Runtime Artifacts
When rank tuning is used, a run writes:

```text
runs/<run_id>/embeddings/
  dct3d.yaml
  dct3d_indices/
    <role>_<signal>.npy
```

`dct3d.yaml` stores `embeddings.per_signal_overrides` with rank metadata.
Each `.npy` file stores 1D coefficient indices consumed by rank-mode codecs at runtime.

## Key Config Block
Base tuning settings live in `scripts_mast/configs/common/embeddings.yaml`:

```yaml
embeddings:
  tune_embeddings:
    n_shots: 100
    max_windows: 15000
    objective:
      thresholds:
        input: 0.999
        actuator: 0.999
        output: 0.995
      max_budget:
        input: 4096
        actuator: 4096
        output: 4096
    guardrails:
      enable: true
```

Parameter intent:
- `n_shots`: number of shots sampled for tuning statistics
- `max_windows`: upper bound on analyzed windows
- `thresholds`: minimum explained energy target by role
- `max_budget`: hard cap on selected coefficients by role
- `guardrails`: optional sanity checks to avoid under-dimensioned selections

## Pretrain Behavior
`pretrain.yaml` controls which roles tune:

```yaml
embeddings:
  tune_embeddings:
    roles:
      input: false
      actuator: false
      output: false
```

Set any role to `true` to tune it during pretrain.

## Finetune Behavior
Finetune uses `embeddings.mode`:

```yaml
embeddings:
  mode: source   # source | config
  tune_embeddings:
    roles:
      input: false
      actuator: false
      output: false
```

### `mode: source`
- copies source run `embeddings/` artifacts
- inherited roles read source rank overrides
- roles set to `true` are retuned in current run
- inherited source roles are validated strictly

### `mode: config`
- ignores source run embedding artifacts
- uses merged profile config directly
- all roles must stay `false`

## Example Patterns
### Inherit all roles during finetune
```yaml
embeddings:
  mode: source
  tune_embeddings:
    roles:
      input: false
      actuator: false
      output: false
```

### Retune only output during finetune
```yaml
embeddings:
  mode: source
  tune_embeddings:
    roles:
      input: false
      actuator: false
      output: true
```

### Use profile config only
```yaml
embeddings:
  mode: config
  tune_embeddings:
    roles:
      input: false
      actuator: false
      output: false
```

## Profile Override Files
Task profile files under `embeddings_overrides/<profile>.yaml` should keep only config overrides.

Do not store:
- `coeff_indices` arrays
- committed per-run rank artifacts

Run-local artifacts belong in `runs/<run_id>/embeddings/`.

## Troubleshooting
### Missing inherited source role in finetune
- Cause: source run has no required role entries in `embeddings/dct3d.yaml`.
- Fix: set that role to retune, or switch to `mode: config`.

### Rank mode cannot load indices
- Cause: missing `dct3d_indices/*.npy` for a rank override.
- Fix: ensure the run has matching artifacts in `runs/<run_id>/embeddings/`.

### No tuning executed
- Cause: all role flags are `false`.
- Fix: set desired role(s) to `true`.
