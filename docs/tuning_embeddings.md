# Tuning embeddings (DCT3D)

Many MAST signals are encoded with a lightweight **DCT3D** codec before being projected into the model.
The codec has truncation parameters:

- `keep_h`, `keep_w`, `keep_t`

which control how many DCT coefficients are kept along each axis.

This document explains:

- what the tuning script optimizes,
- how the tuning pipeline works,
- where the tuned results are written,
- how tuned overrides are consumed by training and evaluation.

---

## Why tune DCT3D?

DCT3D truncation trades off:

- **smaller / cheaper embeddings** (fewer coefficients),
- vs **explained energy** percentage (higher percentage).

Different signals often have very different spectral content, so a single global truncation setting can
be overly conservative for some signals and too aggressive for others.

Tuning selects per-signal truncation settings that meet an error threshold.

---

## Where tuning fits in the pipeline

Tuning uses the same upstream preprocessing as training:

`ChunkWindows → SelectValidWindows → TrimChunks`

Then it applies a dedicated transform:

`TuneDCT3DTransform`

which *observes* the chunk arrays and computes explained energy for candidate truncations.

Tuning is typically placed:

- after trimming (so you tune the same chunk history the model will see),
- before embedding (so you tune on raw signals, not cached embeddings).

---

## Objective (what is optimized)

For each window and candidate truncation config:

1. Apply DCT3D → truncate to `(keep_h, keep_w, keep_t)` → inverse transform.
2. Compute RMSE on **finite** entries only (NaNs from padding are ignored).
3. Aggregate over chunks and windows.

Selection rule (per `(role, signal)`):

- Choose the *smallest effective config* whose score is below a threshold for that role.
- If no config meets the threshold, choose the config with the minimum score.

Effective size:
- If a signal has shape smaller than `(keep_h, keep_w, keep_t)`, the effective kept dims are clipped,
  and the effective dimension is computed with those clipped values.

---

## Tuning configuration

A typical tuning config exposes three concepts:

### 1) Sampling
How much data to use for tuning:

```yaml
tune_dct3d:
  sampling:
    n_shots: 32
    max_windows: 5000
```

- `n_shots`: how many shots to sample from the dataset split
- `max_windows`: cap on total windows processed (useful for fast iteration)

### 2) Thresholds
Per-role target error:

```yaml
tune_dct3d:
  objective:
    thresholds:
      input: 0.01
      actuator: 0.01
      output: 0.01
```

Thresholds are compared against the aggregated RMSE score per signal.

### 3) Search space
Candidate truncation parameters:

```yaml
tune_dct3d:
  search_space:
    keep_h: [4, 8, 12, 16]
    keep_w: [4, 8, 12, 16]
    keep_t: [2, 4, 6, 8, 10]
```

---

## Output: embeddings overrides

Tuning produces a task-local YAML file containing **only** per-signal overrides, for example:

```yaml
embeddings:
  per_signal_overrides:
    input:
      pf_active-coil_current:
        encoder_name: "dct3d"
        encoder_kwargs: { keep_h: 16, keep_w: 1, keep_t: 10 }
    output:
      equilibrium-psi:
        encoder_name: "dct3d"
        encoder_kwargs: { keep_h: 16, keep_w: 16, keep_t: 5 }
```

By convention this file lives at:

```
scripts_mast/configs/tasks_overrides/<task>/embeddings_overrides/dct3d.yaml
```

This is the `dct3d` embedding profile. Training/eval phases select the embedding profile via
`--emb_profile` (default: `dct3d`) and will merge:
`tasks_overrides/<task>/embeddings_overrides/<profile>.yaml`.

It is treated as an **auto-generated artifact**.

---

## How tuned overrides are used

The config loader typically merges tuned overrides **last**, so they win over defaults:

1. `common/core.yaml`
2. `common/embeddings.yaml`
3. `common/<phase>.yaml`
4. `tasks_overrides/<task>/core_overrides.yaml`
5. `tasks_overrides/<task>/<phase>_overrides.yaml` *(optional)*
6. `tasks_overrides/<task>/embeddings_overrides/<profile>.yaml` *(required for pretrain/finetune/eval; not merged during tune_dct3d)*

Note: during `tune_dct3d`, the config loader intentionally does **not** merge the task-level
`embeddings_overrides/<profile>.yaml` file. This allows the tuner to write deltas relative to
`common/embeddings.yaml` for the selected profile.

This means:

- future pretrain/finetune/eval runs for that task automatically pick up tuned DCT3D settings
- common defaults remain clean and task-agnostic

---

## Practical workflow

1) Start from conservative global defaults in `common/embeddings.yaml`.  
2) Run tuning:

```bash
python scripts_mast/run_tune_dct3d.py --task <task>
```

Optional: restrict tuning to a subset of roles via `--roles` (comma-separated), for example:

```bash
python scripts_mast/run_tune_dct3d.py --task <task> --roles output
```

3) Review the generated `embeddings_overrides/dct3d.yaml`.  
4) Commit it only if you want to ship it as the “official tuned preset” for that task.  
5) Run finetune/pretrain/eval as usual; they will automatically use tuned overrides.

---

## Tips

- Tune on the same split you care about most (often train or a representative subset).
- Keep `n_shots` and `max_windows` small while iterating; increase once stable.
- If many signals fail to meet the threshold, loosen thresholds or expand the search space.
- If tuning is too slow, reduce the search space or limit to a subset of signals.
