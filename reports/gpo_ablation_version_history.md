# GPO Ablation Version History
## What changed in each version, what failed, and why

> **Source:** `docs/gpo_implementation_log.md §3` lines 61–227;
> `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` header comments (lines 38–61)

---

## Summary Table

| Version | Key change | Outcome | Root cause identified |
|---------|------------|---------|----------------------|
| v1 | Basic DPO, β=0.1, no reference | No effect | β too small; gradient ≈ 0.05 everywhere |
| v2 | Per-task β calibration (β=5–50) | All tasks degraded +58–136% | No reference model; policy drifts freely away from y_w |
| v3 | DPO-style reference model; shot blacklists; sft_weight=0.3 | Still degraded +40–73% (4-1 to 4-4) | sft_weight=0.3 was the only gradient at t=0; trained adapters on hardest-pair subset → biased |
| v4 | sft_weight=0.0; 4 training infra bugs fixed | Val-loss dilution discovered | 46–85% pair-absent batches → artificial 17–37% val loss deflation; premature early stopping |
| v5 | Val-loss dilution fix; IPO for tasks 4-3, 4-5 | Partial improvement | Pair-absent batches still receive zero gradient → adapter drift |
| v6 | embed_mse anchor (weight=0.1) on all batches | Reward hacking identified in 4-1, 4-2 | DPO sigmoid saturates once margin grows; d_w rising while pref_acc rising |
| v7 | IPO for 4-1, 4-2; LR 5× reduction; embed_mse weight 0.1→0.2 | Stable training; pending final eval | — |
| v8 | Groups 1-x and 2-x onboarded; β re-calibration per signal; improved pair filtering | Training launched | — |

---

## v1 — First implementation (β = 0.1)

**Commit:** initial GPO implementation

**Change:** Basic `ContinuousGPOLoss` with unanchored margin, β=0.1 across all tasks.

**Outcome:** No measurable effect on any task.

**Root cause:**
```
β = 0.1  →  DPO gradient = β · σ(−β·margin) ≈ β · 0.5 = 0.05
```
The preference signal was present mathematically but orders of magnitude too small to
move any parameter.  GPO was effectively disabled.

*Source: `gpo_implementation_log.md` lines 63–73*

---

## v2 — Per-task β calibration

**Change:** Measured median MSE gap per task from collected pair shards.  Set β so that
`β × p50_MSE ≈ 0.7–0.9` (ideal DPO operating point).  No reference model.

**β values chosen at v2:**

| Task | Median MSE | β chosen | β×median |
|------|-----------|---------|---------|
| 4-1 | 0.0789 | 10.0 | 0.79 |
| 4-2 | 0.0786 | 10.0 | 0.79 |
| 4-3 | 0.0028 | 50.0 | 0.14 |
| 4-4 | 0.1095 | 8.0 | 0.88 |
| 4-5 | 0.0223 | 5.0 | 0.11 |

**Outcome:** All 5 tasks degraded — NRMSE increased by **+58–136%**.

**Root cause:** No reference model means the policy drifts freely.  GPO pushes `ŷ` away
from `y_l` but without a KL anchor it also moves away from `y_w` for hard windows,
particularly in later epochs where gradient comes only from the hardest, most unusual shots.

*Source: `gpo_implementation_log.md` lines 75–96*

---

## v3 — DPO-style reference model (KL constraint)

**Commits:** `94ad59c` (GPU bug), `7365f4d` (batch device bug)

**Change:** Added a frozen copy of the base finetune checkpoint as a reference model.
`_GpoBatchInjector` runs a `no_grad` forward pass on every batch, injecting
`batch["ref_preds"]`.  Reference-anchored margin:

```
margin = [d(ŷ, y_l) − d(ŷ_ref, y_l)] − [d(ŷ, y_w) − d(ŷ_ref, y_w)]
```

Also added per-task dataset filters (shot blacklists, `min_margin_mse`).
Kept `sft_weight=0.3`.

**Outcome:** Still degraded — **+40–73%** for tasks 4-1 to 4-4.  Task 4-5 near-neutral (+0.1%).

**Root cause:** At t=0 with a reference model, `margin = 0` for every pair because
`ŷ ≡ ŷ_ref` at initialisation.  The DPO gradient at `margin=0` is `β × 0.5` — non-zero
but *identical for all pairs regardless of their MSE gap*.  The only pair-discriminative
gradient was `sft_weight=0.3 × MSE(ŷ, y_w)`, which trained output adapters on the
hardest-window subset (disruptions, unusual discharges) — biased adaptation.

*Source: `gpo_implementation_log.md` lines 99–127*

---

## v4 — Pure GPO ablation (sft_weight = 0.0)

**Commit:** `c439db9` (sft_weight=0), `0bb0c4a` (infrastructure bugs)

**Change:** Set `sft_weight: 0.0` on all five tasks.

**Four training infrastructure bugs fixed:**

| Bug | Description | Fix |
|-----|-------------|-----|
| **A** | `y_w_emb`/`y_l_emb` on CPU, `output_mask` on CUDA — device error | Extended `move_batch_to_device` to move GPO dict tensors |
| **B** | GPO run re-tuned output embeddings → different `encoded_dim` → `load_best_weights` shape mismatch | Added `gpo_inherit_all_embeddings=True` flag |
| **C** | `build_window_data` called twice (train + val both pointing to same MAST dataset) | Pass only `{"train": mast_context}` |
| **D** | `deep_merge` appended GPO stage to finetune stages `["ft_heads", "ft_full"]` → 20 extra epochs | Force-replace `train.stages` after merging `gpo.yaml` |

*Source: `gpo_implementation_log.md` §§4, lines 134–149; `gpo_tasks.yaml` lines 38–41*

**Val-loss dilution discovered:** 17–37% artificial deflation (see gpo_infrastructure_bugs.md §Bug E).

---

## v5 — Val-loss dilution fix + IPO for weak-margin tasks

**Commits:** `3230453` (val-loss fix), `1c2d00b` (IPO for 4-3, 4-5)

**Change 1:** Val-loss now averaged over pair-containing batches only (see
`src/mmt/train/loop_utils.py`).

**Change 2:** IPO loss type for tasks 4-3 and 4-5 where `β×p50 ≪ 0.5`.

**Root cause of remaining degradation identified:** 46–85% of batches contained no GPO
pairs.  With `sft_weight=0.0` these batches produced zero gradient.  Output adapters
drifted on the non-pair majority, undoing the preference signal from the paired minority.

*Source: `gpo_implementation_log.md` lines 153–164; `gpo_tasks.yaml` lines 43–44*

---

## v6 — embed_mse reconstruction anchor on all batches

**Change:** Added `embed_mse` (weight=0.1) as a second loss term.  Unlike
`ContinuousGPOLoss` (zero on pair-absent batches), `embed_mse` fires on every batch —
providing a global reconstruction anchor that prevents adapter drift.

Also added SLiC and Hinge loss variants for future experimentation.

**Outcome:** Stable training on most tasks.  **Reward hacking identified** in tasks 4-1
and 4-2:
- `d_w` (distance from prediction to ground truth) rising monotonically
- `pref_acc` rising simultaneously
- Policy is moving `ŷ` away from `y_l` rather than toward `y_w`

**Also identified:** LR of 5e-4 on `modality_heads` and `output_adapters` is too high
for tasks 4-1, 4-2, 4-4 — overshooting visible in training dynamics.

*Source: `gpo_implementation_log.md` lines 166–181; `gpo_tasks.yaml` lines 45–62*

---

## v7 — IPO for reward-hacking tasks, LR reduction, embed_mse weight increase

**Changes:**
1. Tasks 4-1 and 4-2: `loss_type: dpo` → `loss_type: ipo`.
   IPO gradient `2·(margin − 1/(2β))` is restorative — penalises overshooting the
   target margin, pulling policy back toward `y_w` if it drifts.
2. LR reduced 5× on `modality_heads` and `output_adapters` (5e-4 → 1e-4) for all tasks.
3. `embed_mse` weight: 0.1 → 0.2.

*Source: `gpo_implementation_log.md` lines 184–193*

---

## v8 — Groups 1-x and 2-x onboarded (current)

**Changes:**
- Groups 1 and 2 onboarded alongside the existing group 4 configs — tasks 1-1, 1-2,
  1-3, 2-1, 2-2, 2-3 added to `gpo_tasks.yaml` with per-signal β calibration.
  Group 3 base fine-tuning is included in the pipeline; GPO configs will be added
  in a subsequent version once base runs and pair collections complete.
- β set per-signal using panel 29 of `visualize_gpo_stats.py` (`β_opt = 1/p50_MSE`).
- IPO for all new tasks.
- Pair filtering upgraded from `min_margin_mse` (legacy) to:
  - `native_nrmse_percentiles: [25.0, 95.0]` — physically grounded percentile filter
  - `max_pairs_per_shot_percentile: 90.0` — cap on over-represented shots

*Source: `gpo_implementation_log.md` lines 196–226; `gpo_tasks.yaml` lines 349–691*
