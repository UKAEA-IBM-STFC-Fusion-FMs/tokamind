# Base Fine-Tune Evaluation Metrics
## Schema, methodology, and how to extract results as runs complete

> **Source:** `scripts_mast/run_eval.py`; `tokamark.evaluator.compute_metrics`;
> `scripts_mast/mast_utils/benchmark_imports.py`;
> `scripts_mast/configs/mmt/phases/eval.yaml`

---

## Purpose of this file

This document describes the evaluation metric schema and how to collect results, rather
than hard-coding point-in-time numbers.  As each task's base fine-tune run completes on
CCC, its metrics can be read from the standard path and appended to the results tables
below (initially empty).

---

## CSV Schema

Each completed eval run writes:
```
runs/<run_id>/eval/metrics/<task>/task_metrics.csv
```

One row per output signal, plus a task-level aggregate row, with these columns:

```
feature_name, n_shots,
NRMSE_mean, NRMSE_std_pop,
NMAE_mean,  NMAE_std_pop,
RMSE_mean,  RMSE_std_pop,
MAE_mean,   MAE_std_pop,
nan_fraction_mean, nan_fraction_std_pop
```

**Key metrics for the paper:** `NRMSE_mean`, `NMAE_mean`, `RMSE_mean`, `MAE_mean`, `n_shots`.
`nan_fraction_mean` and `*_std_pop` columns are diagnostic only.

*Source: `tokamark.evaluator.compute_metrics` via `scripts_mast/mast_utils/benchmark_imports.py`*

---

## Additional per-shot and CRPS outputs

The eval runner also writes:
- `shots_metrics.csv` — per-shot breakdown of the same metrics
- `crps_task_metrics.csv` — CRPS score at task level (when enabled)
- `crps_shot_metrics.csv` — CRPS score per shot

*Source: `scripts_mast/mast_utils/eval/benchmark_eval.py`; `scripts_mast/configs/mmt/phases/eval.yaml`*

---

## Eval Configuration

Source: `scripts_mast/configs/mmt/phases/eval.yaml`

| Parameter | Value |
|-----------|-------|
| `per_task` | `true` — writes `task_metrics.csv` per task |
| `per_shot` | `true` — writes `shots_metrics.csv` per shot |
| Eval split | test split of MAST dataset |

---

## How to run evaluation for a single task

```bash
# Standalone eval for one task (after base fine-tune completes):
TASK=task_4-1 \
  MODEL_RUN=ft-task_4-1-scratch-mmt-dct3d-embed-mse \
  bash cluster/ccc_eval_task.sh

# Or chain it automatically at the end of GPO training:
SUBMIT_EVAL=1 bash cluster/ccc_gpo_all.sh
```

*Source: `docs/gpo_runbook.md §Stage 3`; `cluster/ccc_eval_task.sh`*

---

## How to read the comparison table (GPO vs base)

After both base and GPO evals are complete:

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh --compare
```

This calls `scripts_mast/compare_gpo_eval.py` and prints per-task Δ% for
`NRMSE_mean`, `NMAE_mean`, `RMSE_mean`, `MAE_mean`, `n_shots`.

*Source: `scripts_mast/compare_gpo_eval.py`; `cluster/gpo_compare_all.sh`*

---

## Results placeholder — Base fine-tune

Fill in from `runs/<run_id>/eval/metrics/<task>/task_metrics.csv` as runs complete.

| Task | Run ID | n_shots | NRMSE_mean | NMAE_mean | RMSE_mean | MAE_mean |
|------|--------|---------|------------|-----------|-----------|----------|
| task_1-1 | ft-task_1-1-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_1-2 | ft-task_1-2-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_1-3 | ft-task_1-3-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_2-1 | ft-task_2-1-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_2-2 | ft-task_2-2-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_2-3 | ft-task_2-3-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_3-1 | ft-task_3-1-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_3-2 | ft-task_3-2-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_3-3 | ft-task_3-3-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_4-1 | ft-task_4-1-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_4-2 | ft-task_4-2-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_4-3 | ft-task_4-3-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_4-4 | ft-task_4-4-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |
| task_4-5 | ft-task_4-5-scratch-mmt-\<BASE_TAG\> | — | — | — | — | — |

---

## Results placeholder — GPO Δ% vs base

Fill in from `compare_gpo_eval.py` output after Stage 4 completes.

| Task | NRMSE Δ% | NMAE Δ% | RMSE Δ% | MAE Δ% | n_shots |
|------|----------|---------|---------|--------|---------|
| task_1-1 | — | — | — | — | — |
| task_1-2 | — | — | — | — | — |
| task_1-3 | — | — | — | — | — |
| task_2-1 | — | — | — | — | — |
| task_2-2 | — | — | — | — | — |
| task_2-3 | — | — | — | — | — |
| task_3-1 | *(GPO pending)* | — | — | — | — |
| task_3-2 | *(GPO pending)* | — | — | — | — |
| task_3-3 | *(GPO pending)* | — | — | — | — |
| task_4-1 | — | — | — | — | — |
| task_4-2 | — | — | — | — | — |
| task_4-3 | — | — | — | — | — |
| task_4-4 | — | — | — | — | — |
| task_4-5 | — | — | — | — | — |

`✓` = improvement (lower is better), `✗` = degradation. See comparison script output for markers.
