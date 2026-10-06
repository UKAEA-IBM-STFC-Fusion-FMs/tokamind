# GPO Experiment Design
## Continuous Generalized Preference Optimization for TokaMind / MAST

> **Source:** `docs/gpo_implementation_log.md` §§1–2, `docs/gpo_runbook.md`,
> `scripts_mast/configs/mmt/phases/gpo.yaml`, `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`

---

## 1. Goal

Improve NRMSE over the base fine-tune checkpoint for all TokaMind benchmark tasks by
applying Generalized Preference Optimization (GPO) in embedding space.  Standard supervised
fine-tuning minimises MSE against ground truth.  GPO adds a preference signal: for each
training window where the model was wrong, push the new prediction closer to the ground
truth `y_w` (preferred) and farther from the model's own prior prediction `y_l` (dispreferred).

### Theoretical foundation — Parseval's theorem

The model operates in **DCT3D / VAE embedding space** — not token probability space — so
log-probabilities are unavailable.  The proxy for preference strength is squared Euclidean
distance in embedding space.

The DCT3D codec uses `norm='ortho'` (orthonormal basis).  By Parseval's theorem:

```
‖x_pred − x_true‖²_native  ==  ‖z_pred − z_true‖²_DCT   (exact equality)
```

This means equal-weight MSE over DCT coefficients is already NRMSE-aligned.  No frequency
weighting is needed or correct — it would move the loss minimum away from the NRMSE minimum.

*Source: `docs/gpo_implementation_log.md` lines 19–33; `src/mmt/data/embeddings/dct3d.py`*

---

## 2. Task Taxonomy

The benchmark spans four task groups.  The pipeline is designed to accommodate all of them;
GPO configs are added group-by-group as base fine-tune runs complete and pair collections
are validated.

| Group | Tasks | Output modality | Encoding |
|-------|-------|-----------------|----------|
| 1 | task_1-1, task_1-2, task_1-3 | Equilibrium scalars, LCFS profiles, ψ-reconstruction | Identity (D=1), 1-D VAE, 2-D VAE |
| 2 | task_2-1, task_2-2, task_2-3 | Multi-output equilibrium + actuators | 1-D VAE + conditioning |
| 3 | task_3-1, task_3-2, task_3-3 | *(GPO config pending; base fine-tune included)* | TBD |
| 4 | task_4-1, task_4-2, task_4-3, task_4-4, task_4-5 | SXR lower/upper, EQ, plasma current, magnetics | 1-D VAE, identity |

**GPO-configured tasks (v8):** Groups 1, 2, and 4 (11 tasks as of v8).  The pipeline
accommodates additional tasks: group 3 tasks undergo base fine-tuning via the same
`ccc_finetune_all.sh` launcher and will receive GPO configs once their base checkpoints
are evaluated and pairs collected.  New groups can be added by writing entries in
`gpo_tasks.yaml` and re-running the calibration stage.

*Source: `docs/gpo_implementation_log.md` lines 4–8, §11; `docs/gpo_runbook.md` Stage 0*

---

## 3. Model Architecture

All tasks use the **MMT** (Multi-Modal Transformer) architecture.

| Component | Config |
|-----------|--------|
| Model profile | `mmt` |
| Embedding profile | `dct3d` |
| Backbone | Transformer (`d_model`, `n_layers`, `n_heads`, `dim_ff` per task YAML) |
| Modality heads | timeseries, profile, video heads |
| Output adapters | residual + Gaussian output (`gaussian: true`) |
| Checkpoint format | `backbone.pt`, `modality_heads.pt`, `output_adapters.pt`, `token_encoder.pt` |

*Source: task fine-tune YAML (e.g. `runs/<run_id>/<run_id>.yaml`); `docs/model_architecture.md`*

---

## 4. Pipeline Stages

```
[Source dataset (MAST)]
      │
      ▼  Stage 0 (optional): Base supervised fine-tuning  — ccc_finetune_all.sh
         Covers all task groups including group 3.
         Run ID: ft-<task>-scratch-mmt-<BASE_TAG>
[Base fine-tune checkpoints]
      │
      ▼  Stage 1: GPO preference-pair collection  — ccc_collect_gpo_all.sh
         Model forward pass on train split; record (y_w_emb, y_l_emb, native diagnostics)
         Written to: runs/ft-.../gpo_pairs_<GPO_TAG>/   (.npz shards, schema v3)
[Preference-pair datasets]
      │
      ▼  Stage 2: β calibration  — gpo_compare_all.sh  (CPU, login node)
         calibrate_gpo_task.py:
           - validates schema v3 (10 checks)
           - computes β_opt = 1/p50_MSE per signal, geometric mean across signals
           - determines shot blacklist (shots > 10× dataset mean MSE)
           - patches gpo_tasks.yaml in-place
           - saves 4-figure diagnostic PNGs to gpo_plots/<task>/
[gpo_tasks.yaml updated with calibrated β + blacklists]
      │
      ▼  Stage 3: GPO fine-tuning + evaluation  — ccc_gpo_all.sh  (SUBMIT_EVAL=1)
         Chained cluster jobs: GPO training → eval
         Run ID: ft-<task>-ws-ft-<task>-scratch-mmt-<BASE_TAG>-mmt-<GPO_TAG>
[GPO model checkpoints + eval metrics]
      │
      ▼  Stage 4: Comparison  — gpo_compare_all.sh --compare  (CPU, login node)
         compare_gpo_eval.py: Δ% table (NRMSE, NMAE, RMSE, MAE, n_shots)
[Comparison table]
      │
      ▼  (Optional) Iterative re-collection
         Re-run Stages 1–4 with GPO model as new source (y_l = ŷ_vN — harder anchor)
```

*Source: `docs/gpo_runbook.md` §Pipeline overview; `cluster/gpo_pipeline_all.sh`*

---

## 5. GPO Training Configuration (Phase-level defaults)

Source: `scripts_mast/configs/mmt/phases/gpo.yaml`

| Parameter | Value | Notes |
|-----------|-------|-------|
| `use_reference_model` | `true` | DPO-style KL constraint via frozen base checkpoint |
| `early_stop.patience` | `10` | Epochs without improvement before stopping |
| `amp.enable` | `true` | Mixed-precision training |
| `embed_mse weight` | `0.2` | Global reconstruction anchor on all batches (task override) |
| `continuous_gpo weight` | `0.8` (phase) / `1.0` (tasks) | Preference loss; task configs override to 1.0 |
| `sft_weight` | `0.3` (phase default) / `0.0` (all task configs) | SFT co-loss; overridden to 0.0 since v4 |
| `max epochs` | `200` | Early stopping typically fires much earlier |
| `warmup_steps_fraction` | `0.05` | 5% of steps for LR warm-up |
| `grad_accum_steps` | `1` | No gradient accumulation by default |
| `LR — backbone` | `1e-4` | Backbone frozen in all current configs |
| `LR — modality_heads` | `1e-4` (task override) | Phase default was 5e-4; reduced 5× in v7 |
| `LR — output_adapters` | `1e-4` (task override) | Same |
| `Freeze backbone` | `true` | Only output adapters + modality heads updated |
| `p_drop_inputs` | `0.0` | No stochastic augmentation during GPO |
| `data.cache.enable` | `true` | Pair dataset cached to RAM for fast iteration |
| `data.cache.dtype` | `float16` | Half-precision cache to reduce memory |
| `seed` | `54` | |

---

## 6. Loss Function Architecture

Source: `src/mmt/train/losses/continuous_gpo.py`; `docs/gpo_implementation_log.md §§3, 6–7`

```
LossAggregator
  ├── ContinuousGPOLoss (weight=1.0 per task)
  │     ├── Compute: reference-anchored margin
  │     │     margin = [d(ŷ, y_l) − d(ŷ_ref, y_l)] − [d(ŷ, y_w) − d(ŷ_ref, y_w)]
  │     ├── Objective variants:
  │     │     dpo   : −log σ(β × margin)
  │     │     ipo   : (margin − 1/(2β))²        ← preferred; restorative gradient
  │     │     slic  : max(0, 1 − β × margin)
  │     │     hinge : max(0, β_h − margin)
  │     ├── sft_weight: 0.0 (v4+); co-loss fires on pair windows only
  │     └── Optional: mse_gap_clip, log_mse_gap  (fat-tail tasks)
  └── EmbedMSELoss (weight=0.2; fires on ALL batches, including pair-absent)
```

**Why IPO is preferred (v7 lesson):** DPO's sigmoid `σ(−β × margin)` saturates as margin
grows, allowing the policy to drift away from `y_w` once it beats `y_l` by a wide margin
(reward hacking, confirmed empirically in v6 for tasks 4-1 and 4-2).  IPO's gradient
`2·(margin − 1/(2β))` is restorative: it penalises overshooting the target margin `1/(2β)`
and pulls the policy back if it drifts too far.

*Source: `docs/gpo_implementation_log.md` lines 184–193, `gpo_tasks.yaml` comments*

---

## 7. Pair Collection Schema (v3)

Source: `scripts_mast/mast_utils/gpo/writer.py`; `scripts_mast/mast_utils/gpo/collect.py`

Each `.npz` shard contains per-window records:
- `y_w_emb` — preferred response in DCT3D / VAE embedding space
- `y_l_emb` — dispreferred response (model prediction at collection time)
- Scalar diagnostics: `native_nrmse`, `mse_gap`, `shot_id`, `window_index`

**Schema version:** v3 (embedding-space pairs + scalar native diagnostics).
v3 is 10–100× smaller than v1 (which stored decoded native arrays).

**Pair source:** `model_error` — pairs are (ground truth, model prediction) for windows
where the model made a measurable error.

**Collection split:** Train split of MAST dataset.

*Source: `docs/gpo_implementation_log.md §9`*

---

## 8. Pair Filtering (Active filters in v8)

Source: `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`; `docs/gpo_implementation_log.md §11`

All tasks use the following filters applied at training time via `GpoPairDataset`:

| Filter | Value | Purpose |
|--------|-------|---------|
| `native_nrmse_percentiles` | `[25.0, 95.0]` | Remove trivially-solved (bottom 25%) and extreme outlier (top 5%) pairs in decoded native space |
| `max_pairs_per_shot_percentile` | `90.0` | Cap over-represented shots at p90 of per-shot window-count distribution |
| `shot_blacklist` | task-specific | Remove catastrophic outlier shots (see `gpo_beta_calibration_table.md`) |
| `min_window_index` | task-specific | Skip early ramp-up windows with pathological amplitude spikes |
| `mse_gap_clip` | task-specific | Hard-clip collection-time gap to bound gradient from outlier pairs |
| `log_mse_gap` | task-specific | `log(1 + MSE)` tail compression for fat-tailed pair distributions |

*Source: `gpo_tasks.yaml` lines 83–347 (groups 4), 389–691 (groups 1–2)*
