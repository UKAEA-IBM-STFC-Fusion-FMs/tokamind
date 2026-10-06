# GPO Open Research Directions
## What remains to be done after the v8 training run

> **Source:** `docs/gpo_implementation_log.md §13` lines 576–600

---

## 1. Launch v8 GPO training and evaluate all tasks

Run the full pipeline for all GPO-configured tasks (groups 1, 2, 4 — and group 3 once
its GPO configs are added).  Collection is confirmed for tasks 1-2, 1-3, and 2-1
(diagnostic artefacts in `gpo_v8/`).  All tasks — including group 3 — undergo base
fine-tuning via `ccc_finetune_all.sh`; GPO configs for group 3 will be created after
calibration completes.

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_pipeline_all.sh
```

Expected result: a Δ% comparison table (NRMSE, NMAE, RMSE, MAE, n_shots) for all 11
tasks showing GPO vs base fine-tune.

*Source: `gpo_implementation_log.md §13` item 1; `docs/gpo_runbook.md §Stage 3`*

---

## 2. Additional PairSource providers

**Status:** Designed and documented; no concrete implementation.

The `PairSource` protocol in `scripts_mast/mast_utils/gpo/collect.py` is designed for
extensibility, including alternative dispreferred-response generators independent of model error.

Additional generators could identify failure modes that the model-error source misses.
Their implementation and evaluation are future work, not current capabilities.

*Source: `gpo_implementation_log.md §13` item 2*

---

## 3. Iterative re-collection (multi-round GPO)

**Status:** Infrastructure ready (`gpo_pipeline_all.sh` supports iterative rounds via
`GPO_TAG=v9`, etc.); not yet run.

After a successful v8 round, re-collect pairs from the GPO model so that
`y_l = ŷ_v8` (harder dispreferred anchor).  The stale collection-time anchor is the
main structural limitation of single-round GPO: once the model improves past the
original `y_l`, the dispreferred anchor no longer provides gradient.

Iterative re-collection is the standard mitigation in the RLHF literature (online DPO /
iterative DPO).

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v9 bash cluster/gpo_pipeline_all.sh
```

*Source: `gpo_implementation_log.md §13` item 3*

---

## 4. Backbone unfreezing experiment

**Status:** Not yet run.

If output-adapter-only GPO cannot improve NRMSE on a task, unfreeze the last 2 backbone
layers at LR ≈ 1e-5.  task_4-4 is the suggested first test case: clean 1-D signal
(`summary-ip`), stable training dynamics, and cleanest NRMSE baseline.

The current freeze policy (`token_encoder: true`, `backbone: true`) is conservative
and motivated by catastrophic-forgetting concerns.  A partial unfreeze ablation would
determine whether the backbone representation is a bottleneck for GPO.

*Source: `gpo_implementation_log.md §13` item 4*

---

## 5. SLiC ablation

**Status:** Loss variant implemented (`loss_type: slic`); not yet tested on v8 results.

SLiC (Sequence Likelihood Calibration, Zhao et al. 2023):
```
loss = max(0, 1 − β · margin)
```

Unlike IPO (which penalises all margins including overshoots), SLiC produces zero
gradient for well-separated pairs (`β · margin ≥ 1`).  This preserves already-good
pairs while still pushing laggard pairs.

Test candidate: tasks where IPO is pushing some pairs above the target margin (well
separated) — SLiC would focus the gradient on the remaining poorly-separated pairs.

*Source: `gpo_implementation_log.md §§7, 13` items 5; `gpo_tasks.yaml §SLiC` lines 382–408*

---

## 6. Group 3 tasks (task_3-1, task_3-2, task_3-3)

**Status:** Base fine-tuning included in pipeline; GPO configs not yet written.

Group 3 tasks are submitted via `ccc_finetune_all.sh` but have no entries in
`gpo_tasks.yaml`.  Once base fine-tune checkpoints are available, the calibration
step (Stage 2) will produce β recommendations and shot blacklists, and task configs
can be added.

*Source: `docs/gpo_runbook.md §Stage 0`; absence from `gpo_tasks.yaml`*
