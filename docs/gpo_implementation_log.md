> Historical implementation notes. Statements below about shared-YAML calibration, reference KL,
> IPO gradients, old splits or cluster behavior are superseded by [the current runbook](gpo_runbook.md)
> and [the shot/resume protocol](cgpo_shot_protocol.md). Historical results are not revalidated here.

# GPO Implementation Log
## Continuous Generalized Preference Optimization for TokaMind / MAST

**Branch:** `gpo`
**Tasks (GPO-configured, v8):** `task_1-1` – `task_1-3`, `task_2-1` – `task_2-3`, `task_4-1` – `task_4-5` (11 tasks)
**Tasks (base fine-tune pipeline):** all 14 tasks — groups 1–4 including `task_3-1` – `task_3-3`
**Current config version:** v8
**Pair schema:** v3 (embedding-space pairs with native diagnostics)

---

## 1. Goal

Improve NRMSE over the base fine-tune checkpoint for all TokaMind benchmark tasks by
applying Generalized Preference Optimization (GPO) in embedding space. Standard fine-tuning
minimises MSE against ground truth. GPO adds a preference signal: for each training window
where the model was wrong, push the new prediction closer to the ground truth `y_w`
(preferred) and farther from the model's own prior prediction `y_l` (dispreferred).

The model operates in **DCT3D / VAE embedding space** — not token probability space — so
log-probabilities are unavailable. The proxy for preference strength is squared Euclidean
distance in that space.

**Key invariant — Parseval's theorem:**
DCT3D uses `norm='ortho'` (orthonormal basis). By Parseval's theorem:

```
‖x_pred − x_true‖²_native == ‖z_pred − z_true‖²_DCT   (exact equality)
```

This means equal-weight MSE over DCT coefficients is already NRMSE-aligned. No
frequency weighting is needed or correct — it would move the loss minimum away from
the NRMSE minimum.

---

## 2. Architecture Overview

```
Pair collection (run_collect_gpo_pairs.py)
  └─ Model forward pass on train split
  └─ Write (y_w_emb, y_l_emb, native diagnostics) → .npz shards (schema v3)
       └─ GpoPairWriter: atomic staging → promotion on finalize()

GPO fine-tuning (run_gpo_finetune.py)
  └─ _gpo_config_hook: merges gpo.yaml + gpo_tasks.yaml on top of finetune config
  └─ _GpoBatchInjector (wraps MAST DataLoader)
       ├─ Builds (shot_id, window_index) → {signal_id: (y_w, y_l)} index at init
       ├─ Optional: frozen reference model forward → batch["ref_preds"]
       └─ Injects y_w_emb / y_l_emb / gpo_pair_mask into every batch
  └─ train_finetune() [loop.py]
       └─ run_one_epoch() [loop_utils.py]  — pair-only val-loss averaging
            └─ LossAggregator → ContinuousGPOLoss.compute()
                 ├─ Margin computation (reference-anchored or raw)
                 ├─ Objective: dpo | ipo | slic | hinge
                 └─ Optional SFT co-loss (sft_weight; currently 0.0)
            └─ EmbedMSELoss.compute()   — global anchor on all batches (weight=0.2)
```

---

## 3. Version History and Root-Cause Analysis

### v1 — First implementation (β = 0.1)

**Change:** Basic `ContinuousGPOLoss` with unanchored margin, β=0.1 across all tasks.

**Outcome:** No measurable effect on any task.

**Root cause:** β=0.1 → DPO sigmoid gradient = β·σ(−β·margin) ≈ 0.05 everywhere.
The preference signal was present mathematically but orders of magnitude too small to
move any parameter. GPO was effectively disabled.

---

### v2 — Per-task β calibration

**Change:** Measured median MSE gap per task from collected pair shards. Set β so
that β × p50_MSE ≈ 0.7–0.9 (ideal DPO operating point). No reference model.

**β values chosen:**

| Task | Median MSE | β chosen | β×median |
|------|-----------|---------|---------|
| 4-1  | 0.0789    | 10.0    | 0.79    |
| 4-2  | 0.0786    | 10.0    | 0.79    |
| 4-3  | 0.0028    | 50.0    | 0.14    |
| 4-4  | 0.1095    | 8.0     | 0.88    |
| 4-5  | 0.0223    | 5.0     | 0.11    |

**Outcome:** All 5 tasks degraded — NRMSE increased by +58–136%.

**Root cause:** No reference model means the policy drifts freely. GPO pushes `ŷ`
away from `y_l` but without a KL anchor it also moves away from `y_w` for hard
windows, particularly during the later training epochs where gradient comes only
from the hardest, most unusual shots.

---

### v3 — DPO-style reference model (KL constraint)

**Change:** Added a frozen copy of the base finetune checkpoint as a reference model.
`_GpoBatchInjector` runs a `no_grad` forward pass on every batch, injecting
`batch["ref_preds"]`. `ContinuousGPOLoss` now computes the reference-anchored margin:

```
margin = [d(ŷ, y_l) − d(ŷ_ref, y_l)] − [d(ŷ, y_w) − d(ŷ_ref, y_w)]
```

This only rewards improvement *relative to the reference*, preventing policy drift.
Also added per-task dataset filters (shot blacklists, `min_margin_mse`, tail controls)
informed by the visualisation panels. Kept `sft_weight=0.3`.

Two GPU bugs fixed during v3:
- `94ad59c` — reference model was not moved to GPU before first forward pass
- `7365f4d` — batch was not moved to device before reference model forward

**Outcome:** Still degraded — +40–73% for tasks 4-1 to 4-4. Task 4-5 near-neutral (+0.1%).

**Root cause (discovered after analysis):** At t=0 with a reference model, margin = 0
for every pair because `ŷ ≡ ŷ_ref` at initialisation. The DPO gradient at margin=0
is β × 0.5 — non-zero but *identical for all pairs regardless of their MSE gap*. The
only gradient that differentiated easy pairs from hard ones was `sft_weight=0.3 ×
MSE(ŷ, y_w)`. Pairs come disproportionately from high-MSE shots (disruptions, unusual
discharges — the same shots that the blacklists were meant to suppress). The SFT
co-loss effectively fine-tuned output adapters on the most pathological tail of the
data, causing biased adaptation and NRMSE regression.

---

### v4 — Pure GPO ablation (sft_weight = 0.0)

**Change (`c439db9`):** Set `sft_weight: 0.0` on all five tasks. The DPO reference
model already provides a KL anchor — SFT is redundant and was shown to be harmful.

**Also fixed — four training infrastructure bugs** (commit `0bb0c4a`):
- **Bug A** — Device mismatch: `y_w_emb`/`y_l_emb` on CPU, `output_mask` on CUDA.
  Fix: extended `move_batch_to_device` to also move GPO dict tensors.
- **Bug B** — Embedding shape mismatch: output embeddings were re-tuned for the GPO
  run → different `encoded_dim` → `load_best_weights` failed with shape error.
  Fix: `gpo_inherit_all_embeddings=True` flag routes all roles through source branch.
- **Bug C** — Double cache materialisation: `build_window_data` was called with both
  `"train"` and `"val"` keys pointing to the same MAST dataset → iterated twice.
  Fix: pass only `{"train": mast_context}`; build val loader from cached dataset.
- **Bug D** — Wrong stages list: `deep_merge` appended the `"gpo"` stage to the
  finetune stages `["ft_heads", "ft_full"]` → 20 extra epochs before GPO.
  Fix: force-replace `train.stages` after merging `gpo.yaml`.

**Outcome:** Val-loss dilution bug discovered — 17–37% artificial deflation causing premature
early stopping. Fixing sft_weight necessary but insufficient on its own.

---

### v5 — Val-loss dilution fix + IPO for weak-margin tasks

**Changes:** Two commits:
1. `3230453` — val-loss dilution fix (see Section 5)
2. `1c2d00b` — IPO loss type for tasks 4-3 and 4-5 (see Section 6)

**Root cause of remaining degradation identified:** 46–85% of batches contained no GPO
pairs (windows with no matching shard row). With `sft_weight=0.0` these batches
produced zero gradient. The model's output adapters drifted on the non-pair majority,
undoing the preference signal from the paired minority.

---

### v6 — embed_mse reconstruction anchor on all batches

**Change:** Added `embed_mse` (weight=0.1) as a second loss term in all tasks.
Unlike `ContinuousGPOLoss` which returns zero for pair-absent batches, `embed_mse`
fires on every batch — providing a global reconstruction anchor that prevents drift.
Also added SLiC and Hinge loss variants for future experimentation.

**Outcome (confirmed from v6 training logs):** Reward hacking identified in DPO tasks
4-1 and 4-2 — `d_w` (distance to ground truth) rising monotonically while `pref_acc`
rises, indicating the policy is moving away from `y_w` while still beating `y_l`. The
DPO sigmoid saturates once the margin grows: `σ(−β·margin) → 0`, eliminating the
restorative pull toward `y_w`.

**Also identified:** LR of 5e-4 on `modality_heads` and `output_adapters` is too high
for tasks 4-1, 4-2, and 4-4 — training dynamics show overshooting.

---

### v7 — IPO for reward-hacking tasks, LR reduction, embed_mse weight increase

**Changes:**
- Tasks 4-1 and 4-2 switched from `loss_type: dpo` to `loss_type: ipo`.
  IPO gradient `2·(margin − 1/(2β))` is restorative: it penalises overshooting the
  target margin `1/(2β)` and pulls the policy back toward `y_w` if it drifts too far.
- LR reduced 5× on `modality_heads` and `output_adapters` (5e-4 → 1e-4) for all tasks,
  applying the v6 lesson globally.
- `embed_mse` weight increased from 0.1 to 0.2.

---

### v8 — Groups 1-x and 2-x onboarded, β re-calibration per signal (current)

**Changes:**
- All 11 TokaMind benchmark tasks now have GPO configs: tasks 1-1, 1-2, 1-3, 2-1,
  2-2, 2-3 added to `gpo_tasks.yaml` with per-signal β calibration.
- β set per-signal using measured p50 MSE gaps from `visualize_gpo_stats.py` panel 29
  (`β_opt = 1/p50_MSE`). IPO is used for all new tasks (D=1 identity encoders for
  task_1-1 have no sigmoid structure; 2D VAE tasks have near-zero β×p50).
- Pair filtering upgraded from `min_margin_mse` (legacy) to:
  - `native_nrmse_percentiles: [25.0, 95.0]` — removes trivially-solved pairs and
    extreme outliers measured in decoded native space (not just embedding gap).
  - `max_pairs_per_shot_percentile: 90.0` — caps over-represented shots at the
    p90 of the per-shot window-count distribution.
- Collection artefacts in `gpo_v8/` confirm pairs collected for tasks 1-2, 1-3, 2-1.

**Per-task β calibration (v8):**

| Task | Signal | p50 MSE | β | β×p50 | Loss | Note |
|------|--------|---------|---|-------|------|------|
| 4-1  | SXR    | 0.0789  | 10 | 0.79 | IPO  | v7: DPO→IPO (reward hacking) |
| 4-2  | SXR    | 0.0786  | 10 | 0.79 | IPO  | same |
| 4-3  | EQ     | 0.0028  | 50 | 0.14 | IPO  | β×p50≪0.5 |
| 4-4  | Ip     | 0.1095  |  8 | 0.88 | DPO  | β×p50 in target range |
| 4-5  | mag    | 0.0223  |  5 | 0.11 | IPO  | fat tail; clip+log applied |
| 1-1  | eq scalars | 0.003–0.010 | 100 | ≈0.5 | IPO | D=1; 15 signals; geomean β_opt≈170 |
| 1-2  | lcfs   | 0.0132/0.0067 | 50 | 0.50 | IPO | geomean p50≈0.0094 |
| 1-3  | psi    | 0.2608  |  2 | 0.52 | IPO  | IPO target 0.25 ≈ p50 |
| 2-1  | multi eq | 0.0182 | 25 | 0.46 | IPO  | multi-output; geomean |
| 2-2  | lcfs+NBI | 0.0156–0.0367 | 40 | 0.50 | IPO | geomean β_opt≈40 |
| 2-3  | psi+actuators | 0.3055 | 2 | 0.61 | IPO | v7: β=50 was oversaturated (σ=0.955) |

---

## 4. Training Infrastructure Bugs Fixed

All blocking issues discovered when first running GPO training end-to-end.
Fixed in commits `0bb0c4a`, `94ad59c`, `7365f4d`.

### Bug A — Device mismatch (task 4-1)
`ContinuousGPOLoss` received `y_w_emb`/`y_l_emb` tensors on CPU while `output_mask`
was on CUDA. Boolean-mask indexing `y_pred[mask]` raised a device error.

**Fix:** Extended `move_batch_to_device` in `loop_utils.py` to also move
`batch["y_w_emb"]` and `batch["y_l_emb"]` (both `dict[int, Tensor]`) to the target
device alongside the standard batch fields.

### Bug B — Embedding shape mismatch at checkpoint load (tasks 4-2, 4-3)
`resolve_finetune_embeddings` was re-tuning the output DCT3D embeddings for the GPO
run. This produced a different `encoded_dim` from the source finetune run, causing
`load_best_weights` to fail with a shape mismatch.

**Fix:** Added `gpo_inherit_all_embeddings = True` flag set inside `_gpo_config_hook`
in `run_gpo_finetune.py`. The embedding resolution policy (`policy.py`) checks this
flag and routes all roles — including output — through the `source` branch instead of
the `tune` branch, guaranteeing the loaded checkpoint's embedding dimensions are
preserved exactly.

### Bug C — Double cache materialisation (tasks 4-4, 4-5)
`build_window_data` was called with both `"train"` and `"val"` keys pointing to the
same MAST dataset object. `WindowCachedDataset.from_streaming` iterated the full
dataset twice (once per key), doubling RAM usage and wall-clock time.

**Fix:** Pass only `{"train": mast_context}` to `build_window_data`. The val
DataLoader is built later from the already-cached `WindowCachedDataset`, reusing the
materialised cache without re-streaming.

### Bug D — Wrong stages list (all tasks)
`deep_merge` in the config hook merged the stage lists by name, appending the GPO
`"gpo"` stage to the finetune stages `["ft_heads", "ft_full"]`. The merged config
ran all three stages — 20 epochs of standard fine-tuning before GPO even started.

**Fix:** After merging `gpo.yaml`, explicitly force-replace `train.stages` with only
the stages from `gpo.yaml`:

```python
gpo_stages = (gpo_phase.get("train") or {}).get("stages")
if gpo_stages is not None:
    merged.setdefault("train", {})["stages"] = gpo_stages
```

### Bug E — Reference model not on GPU (tasks 4-1, 4-3)
The frozen reference model was instantiated and weights loaded, but `.to(device)` was
not called. The first reference forward pass raised a device mismatch.

**Fix:** Added explicit `ref_model.to(device)` after weight loading in `run_gpo_finetune.py`.

---

## 5. Val-Loss Dilution Fix (`3230453`)

### Problem
`ContinuousGPOLoss.compute()` returns exactly `0.0` for batches that contain no GPO
pairs (windows with no matching shard row). `run_one_epoch` accumulated all batch
losses including these zero-loss batches and divided by `n_batches` (total count).

This diluted the reported validation loss by the fraction of empty batches:
- 17% dilution for tasks 4-1 and 4-2
- ~20% dilution for task 4-4
- 30–37% dilution for tasks 4-3 and 4-5 (after `min_margin_mse` / `min_window_index`
  filters remove a large fraction of pairs)

Early stopping compares the val loss against the best seen so far. A val loss that is
30% lower than the true GPO loss fires patience 30% too early — the best checkpoint is
selected at an epoch where the model has received far fewer real gradient updates than
it should.

### Fix
Added parallel accumulators in `run_one_epoch`:

```python
running_gpo_loss = 0.0
n_gpo_batches    = 0

# After each batch:
_y_l_batch = batch.get("y_l_emb")
_has_pairs = (
    bool(_y_l_batch)
    and isinstance(_y_l_batch, dict)
    and any(t.numel() > 0 for t in _y_l_batch.values())
)
if _has_pairs:
    running_gpo_loss += loss_val
    n_gpo_batches    += 1
```

When `n_gpo_batches > 0`, the returned `avg_loss` (used for early stopping and
checkpoint selection) is `running_gpo_loss / n_gpo_batches`. The diluted average is
preserved as `gpo_diluted_avg` in `avg_term_logs` for diagnostic logging.

Non-GPO stages (`n_gpo_batches == 0`) fall back to the original `running_loss /
n_batches` — fully backward-compatible.

---

## 6. IPO Loss for Weak-Margin Tasks (`1c2d00b`)

### Problem
Tasks 4-3 (β×p50 ≈ 0.14) and 4-5 (β×p50 ≈ 0.11) have near-zero starting margins.
At initialisation `ŷ ≡ ŷ_ref`, so all reference-anchored margins are zero. The DPO
gradient at margin=0 is β·σ(0) = β/2 — not zero, but *uniform across all pairs*
regardless of their MSE gap. The sigmoid is flat near 0.5, so DPO provides no signal
about which pairs are more or less informative.

More precisely: `σ(−β·margin)` when margin=0 equals 0.5 for every pair. This means
every pair receives the same gradient magnitude, regardless of whether its MSE gap is
0.001 or 10.0. The distribution of informative signal across pairs is completely lost.

### IPO (Identity Preference Optimisation, Azar et al. 2023)
Replace the log-sigmoid objective with a squared deviation from a fixed target margin:

```
L_IPO = (margin − 1/(2β))²
∂L/∂margin = 2·(margin − 1/(2β))
```

At margin=0: gradient = 2·(0 − 1/2β) = −1/β — **constant and non-zero**, independent
of the current margin value. IPO always provides a learning signal, even when the
policy has not yet moved away from the reference.

The fixed point of IPO is `margin = 1/(2β)`, meaning the model is trained to maintain
a preference margin of exactly `1/(2β)` on average. For task 4-3 with β=50 this
target is 0.01; for task 4-5 with β=5 it is 0.1.

The same logic applies to all tasks with small β×p50 products (tasks 1-1 through 2-3):
IPO is the default for all v8 tasks except task_4-4.

### Implementation
`ContinuousGPOLoss.__init__` and `compute` were extended with a `loss_type` parameter:

```python
_VALID_LOSS_TYPES = frozenset({"dpo", "ipo", "slic", "hinge"})

if self._loss_type == "ipo":
    ipo_target = 0.5 / self._beta                           # 1/(2β)
    gpo_loss = (margin - ipo_target).square().mean()
elif self._loss_type == "slic":
    slic_target = 0.5 / self._beta
    gpo_loss = F.relu(slic_target - margin).mean()
elif self._loss_type == "hinge":
    gpo_loss = F.relu(1.0 - margin).mean()
else:  # "dpo"
    gpo_loss = -F.logsigmoid(self._beta * margin).mean()
```

---

## 7. SLiC and Hinge Loss Variants (`22444b8`)

Added two additional loss types to `ContinuousGPOLoss` for future experimentation:

### SLiC (Zhao et al., 2023)
```
L_SLiC = max(0, δ − margin)    where δ = 1/(2β)
∂L/∂margin = −1 when margin < δ,  0 when margin ≥ δ
```
A max-margin hinge objective. Like IPO, the target threshold is `1/(2β)`. Unlike
IPO, the gradient is zero once the constraint is satisfied — SLiC is more conservative
and does not penalise already-separated pairs. Suitable when a fraction of pairs
already have large margins and should not be disturbed.

### Hinge
```
L_hinge = max(0, 1 − margin)
∂L/∂margin = −1 when margin < 1,  0 when margin ≥ 1
```
Fixed absolute margin target of 1, independent of β. Useful when β is used only for
diagnostic panels (β-effectiveness score, Fisher-weighted effective N) but you want
the training objective decoupled from the β scale.

Both variants use the same reference-anchored margin as DPO and IPO — `loss_type`
only selects the objective applied to that margin.

---

## 8. Frequency-Weighted Embeddings — Attempted and Reverted

### Attempt (`42e1506`)
Added harmonic rank weighting `w_d = 1/(rank_d + 1)` to the MSE distance in
`ContinuousGPOLoss`, hypothesising that low-frequency DCT coefficients carry more
NRMSE-relevant signal and should receive higher weight.

### Revert (`b9f4982`)
**Reason:** Mathematically incorrect.

By Parseval's theorem with `norm='ortho'`, `‖x_pred − x_true‖²_native =
‖z_pred − z_true‖²_DCT` exactly. NRMSE in native space = equal-weight MSE over DCT
coefficients. Any non-uniform weighting changes which minimum the loss finds — it
makes the model optimise a *different* objective from NRMSE, not a better-aligned one.

The DC coefficient has large *magnitude* but this does not mean its *prediction error*
is systematically larger. Weighting by magnitude confuses magnitude with error.

---

## 9. Schema Evolution

| Version | Pair format | Native arrays | Filtering support |
|---------|------------|---------------|-------------------|
| v1 | Native space (y_w, y_l raw arrays) | Stored in full | Blacklist only |
| v2 | Embedding space (y_w_emb, y_l_emb) | Not stored | Blacklist, min_margin_mse |
| v3 (current) | Embedding space | Compact scalars: native_nrmse, native_nmae, native_mse, embedding_gap | All filters inc. native_nrmse_percentiles, max_pairs_per_shot_percentile |

Schema v3 shards are 10–100× smaller than v1 because the full native arrays are not
persisted. The `GpoPairDataset` supports schema v2 and v3; v1 is read-only (stats
visualiser only, not usable for training).

---

## 10. Visualisation Script (`visualize_gpo_stats.py`)

Built incrementally to extract all analytically useful information from the raw `.npz`
pair shards without requiring model re-runs.

### Figure 1 — Standard overview (12 panels)
Panels 1–12: window counts (train/val split), embedding dimension consistency, MSE gap
boxplot per signal, preference margin distribution, cosine similarity distribution,
normalised margin distribution, per-coefficient absolute error profile (with top-5
annotated), window index temporal coverage, windows-per-shot histogram, shard size
distribution, zero-margin fraction bar chart, collection metadata summary table.

### Figure 2 — Extended diagnostics (8 panels, schema v2/v3)
Panels 13–20: preference-strength CDF (p50/p90/p99 marked), pair hardness tiers ×
temporal coverage (stacked histogram), embedding energy bias hexbin (‖y_w‖ vs ‖y_l‖),
direction vs magnitude error decomposition per signal, per-shot mean MSE gap histogram
(top-10 annotated), coefficient error × time heatmap (first 64 DCT coefficients × 20
temporal bins), ground-truth norm drift over time with model norm overlay, 2D PCA
diversity of y_w embeddings coloured by window_index.

### Figure 3 — Actionability diagnostics (3 panels)
Panels 21–23: β-effectiveness score σ(β×MSE_gap) histogram per signal (ideal:
centred near 0.73), pair quality 2D scatter (normalised margin × cosine similarity,
coloured by log MSE gap, quadrant-labelled), **shot-level outlier table** (top-15 worst
shots with n_windows, mean/p50/p95 MSE, mean normalised margin, mean cosine).

**This figure directly drove the shot blacklists in `gpo_tasks.yaml`:**
- Tasks 4-1/4-2: top-3 shots had mean MSE 45–54 (up to 188× p95)
- Task 4-4: all top-15 had n_windows=1 (single-window disruption captures)
- Task 4-5: top-3 had mean MSE 117–284
- Tasks 1-1, 2-2, 2-3: extreme fat tails confirmed; blacklists set accordingly

### Figure 4 — Training-signal quality (6 panels, schema v2/v3)
Panels 24–29:
- **Panel 24 — Lorenz curve / Gini coefficient:** Cumulative MSE gap concentration.
  Gini > 0.9 means a tiny fraction of hard pairs drives almost all gradient signal.
  (task_2-3: Gini=0.901, eff_N=28.9% → extreme concentration).
- **Panel 25 — Signed coefficient bias:** `E[y_l[d] − y_w[d]]` per DCT coefficient —
  reveals direction of systematic error per frequency mode.
- **Panel 26 — Fisher-weighted effective N:** `Σ 4σ(β·g)(1−σ(β·g))`. Pairs with gate
  ≈ 0.5 contribute weight 1; saturated or dead pairs contribute ≈ 0.
- **Panel 27 — Between/within-shot ANOVA:** One-way ANOVA decomposition of MSE
  variance. High between-shot fraction → blacklisting reduces variance meaningfully.
- **Panel 28 — Filter survival curve:** Fraction of pairs surviving each
  `min_margin_mse` threshold (log-spaced sweep). Used historically for task 4-3.
- **Panel 29 — β optimisation table:** Per-signal: current β, p50 MSE, β×p50,
  σ(β×p50), β_opt = 1/p50_MSE. Colour-coded green/amber/red by calibration quality.
  **Primary driver of v8 β values.**

---

## 11. Per-Task Dataset Filters

Determined from Figure 3 (shot outlier table) and Figure 4 (filter survival curve).
All v8 tasks use `native_nrmse_percentiles: [25.0, 95.0]` and
`max_pairs_per_shot_percentile: 90.0` in addition to any task-specific filters below.

### task_4-1 and task_4-2
`shot_blacklist: [18675, 14013, 14342]`
Three shots with mean MSE 45–54 and p95 up to 188×. They contain 1–3 windows each
and represent unusual/disrupted discharges, not typical plasma behaviour.

### task_4-3
Native NRMSE percentile filter [25, 95] replaces the historical `min_margin_mse: 0.01`.
The earlier filter cut 96% of pairs below p85 of the embedding gap distribution.
The native filter is more physically grounded: it removes pairs where the model
was trivially correct (bottom 25% NRMSE) or catastrophically wrong (top 5% NRMSE).

### task_4-4
`shot_blacklist: [18492, 14357, 17044, 12667, 15359, 16180, 15767, 15524, 14433,
                   12898, 20575, 20839, 14383, 17048, 17412]`
All 15 top-MSE shots had exactly `n_windows=1` — single-window captures of
disruptions. These contribute unreliable preference pairs and dominate the gradient.

### task_4-5
Four filters combined:
1. `shot_blacklist: [30339, 28311, 28268]` — mean MSE 117–284, p95 up to 1013
2. `min_window_index: 50` — early ramp-up phase where signal amplitude spikes 10×
   and model norm ≈ 0
3. `mse_gap_clip: 50.0` — hard-clip at ≈p90 to prevent gradient explosion from
   the extreme fat tail
4. `log_mse_gap: true` — `log(1 + MSE)` compression for remaining tail

### task_1-1
`shot_blacklist: [24463, 25327, 21296]`
Mean MSE 41/36/10 — 367/324/86× dataset mean (0.1125). Catastrophic fat tail.

### task_1-2
`shot_blacklist: [26921, 24828, 21359]`
Top-3 from outlier table.

### task_1-3
`shot_blacklist: [16914, 11822, 21004, 26846, 13202]`
Five shots excluded; all have mean MSE > 100× dataset mean (background p50=0.26).

### task_2-1
`shot_blacklist: [24463, 22046, 23931]`
Top-3; geometric-mean p50 MSE≈0.018 across signals.

### task_2-2
`shot_blacklist: [24828, 24463, 26921]`
Mean MSE 6.1/2.6/2.3 — 70/30/27× dataset mean. Between-shot variance is only 18–20%
of total, so blacklisting beyond 3 has diminishing returns.

### task_2-3
`shot_blacklist: [11822, 26259, 28240, 21004, 25094]`
Mean MSE 1050/680/557/544/452 — 292/189/155/151/126× dataset mean (3.595).
Gini=0.901: extreme concentration requiring 5-shot cut. Between-shot variance is only
13.2% — the majority of concentration is within-shot (temporal).

---

## 12. Key Decisions and Their Rationale

| Decision | Rationale |
|----------|-----------|
| Embedding space for GPO | Avoids decode at every training step; Parseval guarantees NRMSE alignment |
| Schema v3: store coefficients + scalar diagnostics | 10–100× smaller shards vs v1 native arrays; native NRMSE filter still enabled via scalar |
| Reference model (DPO-style KL) | Without it (v2), policy drifts freely and all tasks degrade |
| `sft_weight = 0.0` in v4 | SFT was the only gradient at t=0, biasing adapters toward hardest-pair subset |
| IPO for most tasks | Near-zero starting margin makes DPO sigmoid uniformly flat at 0.5 — no pair discrimination |
| IPO for reward-hacking tasks (v7) | DPO sigmoid saturates as margin grows → free drift away from y_w; IPO gradient is restorative |
| `embed_mse` anchor weight=0.2 | 50–85% of batches have no pairs; zero gradient → model drift confirmed as root cause of v4/v5 degradation |
| Freeze backbone, train only output adapters + modality heads | Reduces catastrophic forgetting; output adapters are the primary prediction layer |
| No frequency weighting | Parseval proves equal-weight MSE = NRMSE; weighting distorts the loss minimum |
| Native NRMSE percentile filter [25, 95] | Physically grounded; removes trivially-solved pairs and extreme outliers in native space |
| Per-shot cap at p90 | Prevents over-represented shots (common in disruption-heavy datasets) from dominating |
| Pair-only val-loss averaging | Diluted average was 17–37% lower than true GPO loss, causing premature early stopping |
| Revert frequency weighting | Parseval proves equal-weight MSE over DCT = NRMSE; weighting distorts minimum |

---

## 13. Current Open Work

1. **Launch v8 GPO training and evaluate all GPO-configured tasks** on CCC test split.
   Pair collection is confirmed for tasks 1-2, 1-3, 2-1 (artefacts in `gpo_v8/`).
   Collections for tasks 1-1, 2-2, 2-3 not yet confirmed in repository artefacts.
   Group 3 tasks (task_3-1 – task_3-3) are included in the base fine-tune pipeline;
   GPO configs will be added after their base runs complete and pairs are calibrated.

2. **Additional PairSource providers** — the `PairSource` protocol in `collect.py` is
   designed and fully documented, but no concrete implementation exists yet.
   It supports alternative dispreferred-response generators independent of model error.

3. **Iterative re-collection** — after a successful v8 GPO round, re-collect pairs
   from the GPO model (`y_l = ŷ_v8`). The stale collection-time anchor is the main
   structural limitation of single-round GPO: the model improves over the original
   `y_l` quickly, after which the dispreferred anchor no longer provides gradient.
   To execute: set `BASE_TAG` to the v8 GPO run-ID tag (the launcher derives the
   source model from `ft-<task>-scratch-mmt-<BASE_TAG>`) and use a new `GPO_TAG`.

4. **Backbone unfreezing** — if output-adapter-only GPO cannot improve NRMSE on a
   task, unfreeze the last 2 backbone layers at LR ~1e-5 (task_4-4 suggested as first
   test case given its clean 1-D signal and stable training dynamics).

5. **SLiC ablation** — once v8 results are known, test SLiC on tasks where IPO
   already pushes some pairs above the target margin (well-separated pairs receive
   no gradient from SLiC, preserving them while still pushing laggards).

---

## 14. File Map

### Python entrypoints

| File | Role |
|------|------|
| `scripts_mast/run_finetune.py` | Base supervised fine-tuning entrypoint |
| `scripts_mast/run_collect_gpo_pairs.py` | Pair collection entrypoint (9 CLI flags) |
| `scripts_mast/run_gpo_finetune.py` | GPO training entrypoint; `_GpoBatchInjector`; reads `GPO_RESUME` + `GPO_TASKS_YAML` from env |
| `scripts_mast/run_eval.py` | Evaluation entrypoint |
| `scripts_mast/calibrate_gpo_task.py` | Validate + calibrate β + patch YAML + plots + Markdown reports |
| `scripts_mast/compare_gpo_eval.py` | GPO vs base finetune comparison table + Markdown report |
| `scripts_mast/visualize_gpo_stats.py` | 4-figure / 29-panel dataset diagnostics |
| `scripts_mast/print_gpo_shot_outliers.py` | Shot outlier table (no-GPU, login-node safe) |
| `scripts_mast/validate_gpo_pairs.py` | Schema-v3 pre-flight validation (10 checks; standalone) |

### Configuration

| File | Role |
|------|------|
| `scripts_mast/configs/mmt/phases/gpo.yaml` | Phase-level GPO config (LR, schedule, freeze) |
| `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` | Per-task β, filters, loss_type overrides (v8) |
| `scripts_mast/configs/mmt/tasks/gpo_tasks_embed_mse_only.yaml` | Ablation: embed_mse anchor only |

### Cluster scripts

| File | Role |
|------|------|
| `cluster/ccc_finetune_task.sh` | Single-task base fine-tuning worker |
| `cluster/ccc_finetune_all.sh` | Batch base fine-tuning launcher (all 14 tasks; `SUBMIT_EVAL=0` default) |
| `cluster/ccc_collect_gpo_pairs.sh` | Single-task pair collection worker |
| `cluster/ccc_collect_gpo_all.sh` | Batch pair collection launcher |
| `cluster/ccc_gpo_finetune.sh` | Single-task GPO fine-tuning worker |
| `cluster/ccc_gpo_all.sh` | Batch GPO training + eval launcher (`SUBMIT_EVAL=1` default) |
| `cluster/ccc_eval_task.sh` | Single-task evaluation worker |
| `cluster/gpo_compare_all.sh` | Login-node: calibrate + reports + (optionally) compare |
| `cluster/gpo_pipeline_all.sh` | End-to-end pipeline orchestrator |

### Training stack

| File | Role |
|------|------|
| `src/mmt/train/losses/continuous_gpo.py` | `ContinuousGPOLoss` (dpo/ipo/slic/hinge) |
| `src/mmt/train/losses/aggregator.py` | Threads GPO kwargs; logs preference accuracy |
| `src/mmt/train/loop_utils.py` | `run_one_epoch` with pair-only val-loss averaging |
| `src/mmt/train/loop.py` | `train_finetune` entry point |
| `src/mmt/data/embeddings/dct3d.py` | DCT3D codec (ortho norm; Parseval holds) |
| `scripts_mast/mast_utils/gpo/collect.py` | Core pair-collection forward loop |
| `scripts_mast/mast_utils/gpo/writer.py` | `GpoPairWriter` (atomic shard writing) |
| `scripts_mast/mast_utils/gpo/dataset.py` | `GpoPairDataset` (5-filter, train/val split) |
| `scripts_mast/mast_utils/gpo/reconstruct.py` | Deterministic dataloader reconstruction |
