# GPO Key Design Decisions
## Algorithmic choices and their rationale

> **Source:** `docs/gpo_implementation_log.md §12` lines 556–573

---

## Decision Table

| Decision | Rationale |
|----------|-----------|
| **Embedding space for GPO** | Avoids decode at every training step (expensive for 2-D VAE outputs); Parseval's theorem guarantees NRMSE alignment with equal-weight DCT MSE |
| **Schema v3: store embedding coefficients + scalar diagnostics** | 10–100× smaller shards vs v1 (decoded native arrays); native NRMSE filter still enabled via pre-computed scalar diagnostic |
| **Reference model (DPO-style KL constraint)** | Without it (v2), policy drifts freely and all tasks degraded by +58–136% NRMSE |
| **`sft_weight = 0.0` (v4+)** | SFT was the *only* active gradient at t=0 (DPO margin = 0 at initialisation), biasing output adapters toward the hardest-window subset of the data |
| **IPO for most tasks** | Near-zero starting margin makes the DPO sigmoid uniformly flat at 0.5 — no pair discrimination; IPO's linear gradient works at any scale |
| **IPO for reward-hacking tasks (v7: 4-1, 4-2)** | DPO sigmoid saturates as margin grows → free drift away from `y_w`; IPO gradient `2·(margin − 1/(2β))` is restorative, penalising overshoot |
| **`embed_mse` anchor (weight=0.2) on all batches** | 50–85% of batches have no GPO pairs → zero gradient on those batches → adapter drift confirmed as root cause of v4/v5 degradation |
| **Freeze backbone; train only output adapters + modality heads** | Reduces catastrophic forgetting; output adapters are the primary prediction layer; backbone encodes shared physics representation |
| **No frequency weighting** | Parseval's theorem proves equal-weight MSE over DCT = NRMSE; frequency weighting distorts the loss minimum |
| **Native NRMSE percentile filter [25, 95]** | Physically grounded; removes trivially-solved pairs (bottom 25% — model already accurate) and extreme outliers in native space (top 5% — catastrophic failures, unreliable gradient) |
| **Per-shot cap at p90 of window count distribution** | Prevents over-represented shots (common in disruption-heavy datasets) from dominating the gradient; Lorenz/Gini analysis showed extreme concentration |
| **Pair-only val-loss averaging** | Diluted average (non-pair batches contribute zero loss) was 17–37% artificially lower than the true GPO val loss, causing premature early stopping |
| **Revert frequency weighting** | Second confirmation: a frequency-weighting experiment (commit `42e1506`) was reverted (`b9f4982`); Parseval is the definitive proof |

*Source: `docs/gpo_implementation_log.md §12` lines 556–573*

---

## Expanded Notes on Selected Decisions

### Why embedding space?

At inference the model predicts in DCT3D or VAE embedding space.  The output adapters
map backbone features → embedding coefficients, which are decoded to native signal space
only at the final step.  Running GPO in embedding space means:
- The preference loss fires directly on the representation the model optimises
- No expensive decoding of `y_w` and `y_l` per batch
- Parseval's theorem (for DCT3D, `norm='ortho'`) gives: `‖x‖²_native = ‖z‖²_DCT`,
  so MSE in embedding space is exactly proportional to MSE in native space

*Source: `docs/gpo_implementation_log.md` lines 19–33*

### Why val-loss dilution was critical

Before the fix (commit `3230453`), `run_one_epoch` averaged loss over **all batches**
including pair-absent ones (which return loss=0 from `ContinuousGPOLoss`).  With
46–85% pair-absent batches, the apparent val loss was:

```
apparent_val = true_val × (pair_hit_rate)   ≈ 0.63–0.54 × true_val
```

Early stopping saw this artificially low plateau and stopped training in <5 epochs.

Fix: val loss is averaged only over batches that contain ≥1 GPO pair.

*Source: `docs/gpo_implementation_log.md §5` lines 284–328*

### Why IPO over DPO for most tasks

DPO: `loss = −log σ(β · margin)`,  gradient ∝ `β · σ(−β · margin)`

At initialisation with a reference model, `ŷ = ŷ_ref` exactly, so `margin = 0` and
`σ(0) = 0.5`.  The gradient is `β × 0.5` — equal for all pairs regardless of their MSE gap.
No pair discrimination is possible.

IPO: `loss = (margin − 1/(2β))²`,  gradient = `2(margin − 1/(2β))`

At initialisation, `margin = 0` and the gradient is `-1/β` — non-zero, proportional to
the target, and pair-discriminative from epoch 1.

*Source: `docs/gpo_implementation_log.md §6` lines 330–380*

### Reward hacking (tasks 4-1, 4-2 in v6)

In v6 logs, `d_w` (distance from prediction to ground truth) rose **monotonically** while
`pref_acc` (fraction of pairs where `d(ŷ, y_l) > d(ŷ, y_w)`) also rose.  This means the
model was improving preference accuracy by moving `ŷ` *away from y_l* rather than *toward y_w*.

DPO sigmoid saturates once `β · margin > ~2` (gradient < 0.01), so the loss provides no
pull back toward `y_w` after the margin grows.

IPO's quadratic target `(margin − 1/(2β))²` penalises both too-small and too-large margins,
acting as a spring toward the target margin.

*Source: `docs/gpo_implementation_log.md §3 v6, v7` lines 166–193*
