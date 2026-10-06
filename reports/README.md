# reports/

Research-grade artefacts extracted from the TokaMind / MAST Continuous GPO pipeline.
Intended as raw material for the paper — all design decisions, configurations, and
methodology are traceable to their source files.

## Static reference documents (hand-curated)

| File | Description | Primary source |
|------|-------------|----------------|
| `gpo_experiment_design.md` | Full experiment design: task taxonomy, model arch, pipeline stages, config decisions | `docs/gpo_implementation_log.md`, `docs/gpo_runbook.md` |
| `gpo_beta_calibration_table.md` | Per-task β calibration methodology, current values, loss type rationale, shot blacklists | `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`, `gpo_implementation_log.md §3 v8` |
| `gpo_pair_dataset_statistics.md` | Pair dataset schema, collection methodology, filtering strategy, diagnostic figures | `scripts_mast/mast_utils/gpo/`, `gpo_tasks.yaml §11`, `gpo_implementation_log.md §§9–11` |
| `gpo_base_eval_metrics.md` | Evaluation metric schema and placeholder tables — fill in as runs complete | `scripts_mast/run_eval.py`, `tokamark.evaluator`, `runs/*/eval/metrics/` |
| `gpo_key_design_decisions.md` | Narrative of key algorithmic decisions and their rationale | `docs/gpo_implementation_log.md §12` |
| `gpo_ablation_version_history.md` | Version-by-version ablation history (v1–v8): what changed, what failed, root causes | `docs/gpo_implementation_log.md §3` |
| `gpo_infrastructure_bugs.md` | Training infrastructure bugs found and fixed during development | `docs/gpo_implementation_log.md §§4–5` |
| `gpo_open_work.md` | Open research directions and next steps | `docs/gpo_implementation_log.md §13` |

## Auto-generated artefacts

All of the following are written automatically when `cluster/gpo_compare_all.sh` runs.
They accumulate across pipeline rounds and provide a complete audit trail.

### `gpo_stats/<task>/calibration_<task>_<gpo_tag>_<timestamp>.md`

Written by `scripts_mast/calibrate_gpo_task.py --report_dir`.
One file per task per calibration run.  Contains:
- Calibrated β and shot blacklist
- Per-signal statistics table (n_windows, p50 MSE, β_opt, dataset mean MSE)
- Top-shot outlier table with blacklist markers
- List of diagnostic PNG figures saved alongside

### `gpo_stats/<task>/*.png`

The 4-figure diagnostic PNG set copied from `PLOTS_DIR` alongside each calibration report:
- `gpo_stats__*.png` — standard 12-panel pair-distribution overview
- `gpo_extended__*.png` — extended diagnostics (Lorenz curve, effective N, tail analysis)
- `gpo_actionability__*.png` — actionable-pair fraction, filter survival, shot outlier panel
- `gpo_quality__*.png` — Lorenz, signed bias, ANOVA variance, β table (panel 29)

Source: `scripts_mast/calibrate_gpo_task.py` → `_save_plots()` → copied by `_write_calibration_report()`

### `gpo_stats/<task>/shot_outliers_<gpo_tag>.md`

Written by `cluster/gpo_compare_all.sh` (Step 3), which calls
`scripts_mast/print_gpo_shot_outliers.py` and captures stdout.
Contains the top-20 shots ranked by mean MSE gap, with `← BLACKLIST CANDIDATE` markers.
Useful for manual review of blacklist decisions.

### `gpo_comparison_<BASE_TAG>_<GPO_VERSION>_<timestamp>.md`

Written by `scripts_mast/compare_gpo_eval.py --report_dir`.
One file per comparison run (after eval completes).  Contains:
- A provenance header (timestamp, run tags, metrics)
- A **summary table** — one row per task, Δ% for every metric, verdict
- A **per-task detail section** — absolute Base/GPO values, Δ, Δ%, improvement marker

Triggered via:
```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh --compare
```
Or manually:
```bash
python scripts_mast/compare_gpo_eval.py \
  --tasks task_4-1 task_4-2 ... \
  --base_ft_tag dct3d-embed-mse \
  --gpo_version v8 \
  --report_dir reports/
```

## Conventions

- Every static document cites its **source file and line/section**.
- Auto-generated reports cite their provenance in the header.
- Run IDs follow the pattern:
  - Base: `ft-<task>-scratch-mmt-<BASE_TAG>`
  - GPO:  `ft-<task>-ws-ft-<task>-scratch-mmt-<BASE_TAG>-mmt-<GPO_TAG>`
- "GPO-configured" means an entry exists in `gpo_tasks.yaml`.  Tasks without an entry
  inherit phase-level defaults only; calibration will add their entry as runs complete.
