> **Protocol update:** New training requires the verified MAST shot protocol.
> Follow [CGPO shot protocol](cgpo_shot_protocol.md) for current collection and
> training commands. The legacy fractional split and cluster commands below
> must not be used unchanged for new experiments.

# GPO Fine-Tuning Runbook

Step-by-step instructions for running the Continuous GPO pipeline on CCC.
All commands are run from the **repo root** on the CCC login node unless noted otherwise.

---

## Quick start — full pipeline in one command

If base fine-tune checkpoints already exist for all tasks:

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_pipeline_all.sh
```

To also run base fine-tuning from scratch (e.g. reproducing on a new dataset):

```bash
RUN_FINETUNE=1 BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_pipeline_all.sh
```

The orchestrator:
1. (Optional) Submits base fine-tune jobs and waits for all to finish.
2. Submits pair-collection jobs and waits for all to finish.
3. Runs β calibration inline (CPU, login node) — patches `gpo_tasks.yaml` and writes
   calibration reports + shot-outlier tables to `reports/gpo_stats/<task>/`.
   **Note:** plots are skipped in the orchestrator (`--no_plots`); run Stage 2 manually
   if you need the full 4-figure diagnostic set.
4. Submits GPO training + eval jobs and waits for all to finish.
5. Runs final comparison inline — prints the Δ% table and writes a Markdown report to
   `reports/gpo_comparison_<BASE_TAG>_<GPO_TAG>_<timestamp>.md`.

---

## Script reference

### Workers (one job = one task)

| Script | Purpose |
|--------|---------|
| `cluster/ccc_finetune_task.sh` | Base supervised fine-tuning for a single task |
| `cluster/ccc_collect_gpo_pairs.sh` | GPO pair collection for a single task |
| `cluster/ccc_gpo_finetune.sh` | GPO fine-tuning for a single task |
| `cluster/ccc_eval_task.sh` | Standalone evaluation for a single task |

### Batch launchers (submit all tasks at once)

| Script | Purpose |
|--------|---------|
| `cluster/ccc_finetune_all.sh` | Submit base fine-tuning for all tasks (`SUBMIT_EVAL=0` by default) |
| `cluster/ccc_collect_gpo_all.sh` | Submit pair collection for all tasks |
| `cluster/ccc_gpo_all.sh` | Submit GPO training + eval for all tasks (`SUBMIT_EVAL=1` by default) |

### CPU-only login-node scripts (no job submission)

| Script | Purpose |
|--------|---------|
| `cluster/gpo_compare_all.sh` | Calibrate β + write reports + (optionally) compare eval |
| `cluster/gpo_pipeline_all.sh` | Full end-to-end orchestrator — chains all of the above |

### Python utilities

| Script | Purpose |
|--------|---------|
| `scripts_mast/calibrate_gpo_task.py` | Validate schema + calibrate β + patch `gpo_tasks.yaml` + save plots + write reports |
| `scripts_mast/validate_gpo_pairs.py` | Schema-v3 validation only (10 checks) |
| `scripts_mast/visualize_gpo_stats.py` | 4-figure / 29-panel diagnostic plots |
| `scripts_mast/compare_gpo_eval.py` | GPO vs base NRMSE comparison table + Markdown report |
| `scripts_mast/print_gpo_shot_outliers.py` | Shot outlier table (no matplotlib; login-node safe) |

---

## Pipeline overview

```
[Source dataset (MAST)]
      │
      ▼  Stage 0 (optional): ccc_finetune_all.sh
         SUBMIT_EVAL=0 by default; add SUBMIT_EVAL=1 to chain eval jobs.
[Base fine-tune checkpoints]
   runs/ft-<task>-scratch-mmt-<BASE_TAG>/
      │
      ▼  Stage 1: ccc_collect_gpo_all.sh
         Skips tasks where gpo_pairs_<GPO_TAG>/ already exists.
         Use GPO_OVERWRITE=1 to force re-collection.
[GPO preference-pair datasets]
   runs/ft-.../gpo_pairs_<GPO_TAG>/  (schema v3 .npz shards)
      │
      ▼  Stage 2: gpo_compare_all.sh  (login node, CPU)
         calibrate_gpo_task.py:
           - validates schema v3 (structural check on each shard)
           - computes β_opt = 1/p50_MSE per signal; geometric mean across signals
           - computes shot blacklist (shots > 10× dataset mean MSE; top-5 cap)
           - patches gpo_tasks.yaml in-place
           - saves 4-figure diagnostic PNGs to gpo_plots/
           - writes calibration_summary.json in each pair dir
           - writes reports/gpo_stats/<task>/calibration_<task>_<gpo_tag>_<ts>.md
           - copies diagnostic PNGs to reports/gpo_stats/<task>/
         print_gpo_shot_outliers.py  (per task):
           - writes reports/gpo_stats/<task>/shot_outliers_<GPO_TAG>.md
[gpo_tasks.yaml updated; reports/gpo_stats/ populated]
      │
      ▼  Stage 3: ccc_gpo_all.sh  (SUBMIT_EVAL=1 default)
         GPO training + chained eval per task.
[GPO model checkpoints + eval metrics]
   runs/ft-<task>-ws-ft-<task>-scratch-mmt-<BASE_TAG>-mmt-<GPO_TAG>/
      │
      ▼  Stage 4: gpo_compare_all.sh --compare  (login node, CPU)
         compare_gpo_eval.py:
           - prints Δ% table (NRMSE, NMAE, RMSE, MAE, n_shots) per task
           - writes reports/gpo_comparison_<BASE_TAG>_<GPO_TAG>_<timestamp>.md
      │
      ▼  (Optional) Stage 5: iterative re-collection
         Re-run Stages 1–4 with a new GPO_TAG (e.g. v9) and BASE_TAG pointing
         at the v8 GPO run IDs so y_l = ŷ_v8 (harder dispreferred anchor).
         The batch launchers always derive the collection source from BASE_TAG,
         so update BASE_TAG to the previous GPO run-ID tag for iterative rounds.
```

---

## Step-by-step (manual)

### Prerequisites

- Python environment `tokamind-env` activated on CCC.
- Conda at `/u/skorupskyy/miniconda3` (override with `MINIFORGE_ROOT=...`).
- Repo cloned and on the `cgpo` development branch.

---

### Stage 0 — Base fine-tuning (skip if checkpoints already exist)

Submit all tasks across all groups (groups 1–4, 14 tasks total):

```bash
TASKS="task_1-1 task_1-2 task_1-3 task_2-1 task_2-2 task_2-3 \
       task_3-1 task_3-2 task_3-3 \
       task_4-1 task_4-2 task_4-3 task_4-4 task_4-5" \
FINETUNE_TAG=dct3d-embed-mse \
bash cluster/ccc_finetune_all.sh
```

Run IDs: `ft-<task>-scratch-mmt-dct3d-embed-mse`

Monitor: `bjobs -u $USER`

To also chain evaluation jobs at the end of each fine-tune:
```bash
SUBMIT_EVAL=1 FINETUNE_TAG=dct3d-embed-mse bash cluster/ccc_finetune_all.sh
```

---

### Stage 1 — Collect GPO preference pairs

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/ccc_collect_gpo_all.sh
```

The script skips tasks where `gpo_pairs_v8/` already exists.  Force re-collection:
```bash
GPO_OVERWRITE=1 BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/ccc_collect_gpo_all.sh
```

Output: `runs/ft-<task>-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8/`

Monitor: `bjobs -u $USER`

Check completion:
```bash
grep -l "GPO dataset written" logs/collect-gpo-v8-*.out
```

---

### Stage 2 — Validate + calibrate β (login node, no GPU)

Run **after all Stage 1 jobs finish**:

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh
```

This script runs four sub-steps:

1. Discovers all `gpo_pairs_v8/` directories and runs structural validation
   (schema version, required `.npz` keys, non-empty shards) via `calibrate_gpo_task.py`.
2. Calibrates `β_opt = 1/p50_MSE` per signal; geometric mean across signals.
3. Determines shot blacklist (shots exceeding 10× dataset mean MSE; top-5 cap).
4. Patches `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` with new β and blacklists.
5. Saves 4-figure diagnostic PNGs to `gpo_plots/`.
6. Writes `calibration_summary.json` inside each pair directory.
7. Writes `reports/gpo_stats/<task>/calibration_<task>_v8_<ts>.md` (per task).
8. Copies the diagnostic PNGs into `reports/gpo_stats/<task>/`.
9. Runs `print_gpo_shot_outliers.py` per task; saves
   `reports/gpo_stats/<task>/shot_outliers_v8.md`.

Dry-run (no YAML write; calibration reports and shot-outlier tables are still written):
```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh --dry_run
```

Skip plot generation (headless — also skips PNG copy to reports/):
```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh --no_plots
```

**Review the calibration output.**  Key things to check:

- `β_calibrated` — the value now in `gpo_tasks.yaml`.  For IPO tasks the target margin
  `1/(2β)` should be close to the measured `p50_MSE`.
- `shot_blacklist` — review in `reports/gpo_stats/<task>/calibration_*.md` or the JSON.
  Add/remove shots manually in `gpo_tasks.yaml` if needed.
- `calibration_summary.json` inside each pair dir contains the full per-signal stats.

---

### Stage 3 — GPO training + evaluation

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/ccc_gpo_all.sh
```

`SUBMIT_EVAL=1` is the default — evaluation is chained automatically after training.
Training takes 3–24 h per task (task-dependent).

Monitor training logs:
```bash
grep "Epoch\|pref_acc\|pair_hit_rate\|d_w" logs/gpo-v8-task_4-2-*.out | tail -60
```

**Healthy training signals per epoch:**
```
Stage gpo | Epoch 8/200 | train=0.138, val=0.131 | no_improve=0/10
  val_ContinuousGPOLoss_0/pref_acc → rising from ~0.5
  val_ContinuousGPOLoss_0/d_w     → decreasing
  val_EmbedMSELoss_1_nonpair      → flat (non-pair windows stable)
  val_pair_hit_rate               → ≥ 0.05
```

**Warning signs:**
- `pref_acc` stays at 0.5 → β mis-calibrated; check `calibration_summary.json` (panel 29 data).
- `d_w` rising while `pref_acc` rising → reward hacking; switch `loss_type: dpo` → `ipo` in `gpo_tasks.yaml`.
- `EmbedMSELoss_1_nonpair` rising sharply → anchor too weak; increase `embed_mse weight` from 0.2 to 0.3.
- `pair_hit_rate < 0.05` → pair dataset too sparse; widen `native_nrmse_percentiles` e.g. `[10.0, 99.0]`.

---

### Stage 4 — Compare GPO vs base finetune

After all eval jobs finish:

```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/gpo_compare_all.sh --compare
```

Prints a Δ% table to the terminal and writes a Markdown report to `reports/`:

```
task_4-2
  Metric        Base     GPO    Δ (GPO−Base)   Δ %
  NRMSE_mean   0.249   0.231      −0.018       −7.2% ✓
  NMAE_mean    0.131   0.121      −0.010       −7.6% ✓
  ...
Report    : reports/gpo_comparison_dct3d-embed-mse_v8_20250101_120000.md
```

`✓` = improvement (lower is better), `✗` = degradation.

The Markdown report contains a cross-task summary table and a per-task detail section.
Override the output directory with `REPORTS_DIR=/path/to/dir`.

The script skips tasks that do not have both base and GPO eval results yet and prints
which directories are missing.

---

### Stage 5 — Iterative re-collection (next round)

If Stage 4 shows NRMSE reduction, re-collect pairs from the GPO model so that
`y_l = ŷ_v8` (harder dispreferred anchor):

```bash
# The collection launchers use BASE_TAG to derive the source model run ID.
# Point BASE_TAG at the GPO run-ID tag to collect from the GPO model.
BASE_TAG=dct3d-embed-mse-mmt-v8 GPO_TAG=v9 bash cluster/ccc_collect_gpo_all.sh
# Then re-run stages 2–4 with GPO_TAG=v9.
```

> **Note:** The batch launchers derive collection source as
> `ft-<task>-scratch-mmt-<BASE_TAG>`.  For iterative rounds set `BASE_TAG` to the
> portion of the previous GPO run ID after `ft-<task>-scratch-mmt-` (i.e. the suffix
> that makes the path resolve correctly).

---

## Environment variables reference

| Variable | Default | Scope | Description |
|----------|---------|-------|-------------|
| `TASKS` | all 14 tasks | all scripts | Space-separated task list |
| `BASE_TAG` | `dct3d-embed-mse` | all | Fine-tune tag embedded in run IDs |
| `GPO_TAG` | `v8` | collection/GPO/compare | Pair directory tag and GPO run version tag |
| `FINETUNE_TAG` | `dct3d-embed-mse` | `ccc_finetune_all.sh` | Tag appended to base fine-tune run IDs |
| `FINETUNE_INIT` | `scratch` | `ccc_finetune_all.sh` | `scratch` or `warmstart` |
| `FINETUNE_MODEL` | _(none)_ | `ccc_finetune_all.sh` | Source run ID for warmstart |
| `MODEL_PROFILE` | `mmt` | finetune/GPO workers | Model architecture profile |
| `EMB_PROFILE` | `dct3d` | finetune/GPO workers | Embedding profile |
| `GPO_OVERWRITE` | `0` | `ccc_collect_gpo_all.sh` | `1` to re-collect even if pairs already exist |
| `GPO_VAL_FRACTION` | `0.1` | `ccc_collect_gpo_all.sh` | Fraction of pairs held out for GPO val split |
| `GPO_SHARD_SIZE` | `2048` | `ccc_collect_gpo_all.sh` | Windows per `.npz` shard file |
| `GPO_RESUME` | `0` | `ccc_gpo_finetune.sh` | `1` to resume an interrupted GPO training run (env-only, not a CLI flag) |
| `GPO_TASKS_YAML` | _(none)_ | `ccc_gpo_finetune.sh`, `ccc_gpo_all.sh` | Path to alternate `gpo_tasks.yaml` (ablations). Env-only; forwarded by `ccc_gpo_all.sh` to workers. |
| `SUBMIT_EVAL` | `0` (finetune) / `1` (GPO) | batch launchers | `1` to chain eval jobs after training |
| `RUN_FINETUNE` | `0` | `gpo_pipeline_all.sh` | `1` to run Stage 0 (base fine-tuning) |
| `RUN_COLLECT` | `1` | `gpo_pipeline_all.sh` | `1` to run Stage 1 (collection) |
| `RUN_GPO` | `1` | `gpo_pipeline_all.sh` | `1` to run Stages 2–4 (calibrate + train + compare) |
| `DRY_RUN` | `0` | `gpo_pipeline_all.sh` | `1` to print commands without submitting |
| `POLL_INTERVAL` | `120` | `gpo_pipeline_all.sh` | Seconds between LSF job-status polling loops |
| `PLOTS_DIR` | `gpo_plots/` | `gpo_compare_all.sh` | Root directory for diagnostic PNGs |
| `REPORTS_DIR` | `reports/` | `gpo_compare_all.sh` | Root directory for all Markdown reports and copied PNGs |
| `RUNS_DIR` | `runs/` | all scripts | Path to the runs directory |
| `CONDA_ENV` | `tokamind-env` | all scripts | Conda environment name |
| `MINIFORGE_ROOT` | `/u/skorupskyy/miniconda3` | all scripts | Path to Miniforge/Miniconda installation |
| `LSF_QUEUE` | `normal` | cluster scripts | LSF queue name |
| `LSF_GPU` | `num=1:mode=exclusive_process:gmem=80G` | cluster scripts | GPU resource string |
| `LSF_NCPUS` | `32` | cluster scripts | CPU cores per job |

> **`GPO_RESUME` and `GPO_TASKS_YAML`** are environment-only overrides — they are not
> `argparse` flags.  `run_gpo_finetune.py` reads them via `os.environ.get(...)`.

---

## Manual interventions and experiments

All experiment-level overrides are applied via environment variables — no YAML editing required.

### Run a single task (not the full set)
```bash
TASKS="task_4-2" GPO_TAG=v8 BASE_TAG=dct3d-embed-mse \
  bash cluster/ccc_gpo_all.sh
```

### Fewer epochs (quick test)
Use a task-override file and `GPO_TASKS_YAML`:
```yaml
# gpo_tasks_quick.yaml — minimal epochs for a smoke test
tasks:
  task_4-2:
    train:
      stages:
        - name: gpo
          epochs: 10
```
```bash
GPO_TASKS_YAML=/path/to/gpo_tasks_quick.yaml \
  TASKS="task_4-2" GPO_TAG=v8-quick bash cluster/ccc_gpo_all.sh
```

### Ablation: embed_mse-only (no GPO term)
```bash
GPO_TASKS_YAML=scripts_mast/configs/mmt/tasks/gpo_tasks_embed_mse_only.yaml \
  TASKS="task_4-2" GPO_TAG=v8-ablation bash cluster/ccc_gpo_all.sh
```

### Override β for a single task without editing YAML
Write a minimal override file:
```yaml
# gpo_tasks_beta_override.yaml
tasks:
  task_4-4:
    train:
      loss:
        terms:
          - type: continuous_gpo
            beta: 5.0
            loss_type: ipo
            sft_weight: 0.0
            weight: 1.0
          - type: embed_mse
            weight: 0.2
```
```bash
GPO_TASKS_YAML=/path/to/gpo_tasks_beta_override.yaml \
  TASKS="task_4-4" GPO_TAG=v8-beta5 bash cluster/ccc_gpo_all.sh
```

### Resume an interrupted GPO run
```bash
GPO_TASK=task_4-2 \
  GPO_MODEL=ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  GPO_TAG=v8 \
  GPO_RESUME=1 \
  bash cluster/ccc_gpo_finetune.sh
```

### Re-calibrate β after re-collection (manual, targeted)
```bash
python scripts_mast/calibrate_gpo_task.py \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --blacklist_multiplier 15 \
  --blacklist_top_n 3 \
  --report_dir reports/ \
  --dry_run          # inspect first; remove --dry_run to write gpo_tasks.yaml
```

### Skip calibration and go straight to training
Manually verify `gpo_tasks.yaml` looks correct, then:
```bash
BASE_TAG=dct3d-embed-mse GPO_TAG=v8 bash cluster/ccc_gpo_all.sh
```

### Generate full diagnostic plots for a specific task
The orchestrator (`gpo_pipeline_all.sh`) skips plots with `--no_plots`.
To generate the full 4-figure set manually:
```bash
python scripts_mast/visualize_gpo_stats.py \
  --gpo_dir runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --save_dir gpo_plots/ \
  --no_show
```

---

## Troubleshooting

**Node failure during cache materialisation (UNKWN status)**
Pure hardware failure — resubmit the same command.
Check `bjobs -l <JOB_ID>` for the failure reason.

**NRMSE catastrophically worse (+thousands of %)**
The model collapsed on non-pair windows. Confirm `embed_mse` anchor is active
(`weight: 0.2` in `gpo_tasks.yaml`).  If it recurs, increase weight to 0.3.

**`val loss flat from epoch 1`**
Check `pair_hit_rate` in the logs. If < 0.05, widen `native_nrmse_percentiles`
(e.g. `[10.0, 99.0]`) or re-collect with `GPO_OVERWRITE=1`.

**`pref_acc` plateaus at 0.5 (IPO tasks)**
Confirm `loss_type: ipo` in `gpo_tasks.yaml`. Also check `calibration_summary.json`
— if `beta_calibrated` looks wrong, re-run Stage 2 with `--dry_run` to inspect.

**Reward hacking (`d_w` rising while `pref_acc` rising)**
Switch `loss_type: dpo` → `loss_type: ipo` for the affected task in `gpo_tasks.yaml`
and resubmit with `GPO_RESUME=1`.

**Early stopping fires at epoch 1**
Confirm `early_stop.patience: 10` in `gpo.yaml`.  Check that `val_fraction: 0.1`
is present in the collection config (set via `GPO_VAL_FRACTION` at collection time).

**`SameFileError` at embedding resolution**
`GPO_DIR` must not be inside the same run as `--model_source`.  Always supply a
tagged pair directory: `GPO_TAG=v8 bash cluster/ccc_gpo_finetune.sh`.

**Calibration writes wrong β for a task**
1. Inspect `reports/gpo_stats/<task>/calibration_*.md` or `calibration_summary.json`.
2. Edit `beta` manually in `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`.
3. Re-run Stage 3 — no need to re-collect.

**Task has no entry in `gpo_tasks.yaml` (new task group)**
Run Stage 2 — `calibrate_gpo_task.py` will add the task entry automatically.
If the task key cannot be inferred from the path, check that the pair directory
follows the convention `runs/ft-<task>-*/gpo_pairs*/`.

---

## File map

| File | Role |
|------|------|
| `scripts_mast/run_finetune.py` | Base fine-tuning entrypoint |
| `scripts_mast/run_collect_gpo_pairs.py` | Pair collection entrypoint (9 CLI flags) |
| `scripts_mast/run_gpo_finetune.py` | GPO training entrypoint; `_GpoBatchInjector`; reads `GPO_RESUME` and `GPO_TASKS_YAML` from env |
| `scripts_mast/run_eval.py` | Evaluation entrypoint |
| `scripts_mast/calibrate_gpo_task.py` | Validate + calibrate β + patch YAML + plots + per-task reports |
| `scripts_mast/compare_gpo_eval.py` | GPO vs base NRMSE comparison; `--report_dir` writes Markdown |
| `scripts_mast/visualize_gpo_stats.py` | 4-figure / 29-panel dataset diagnostics |
| `scripts_mast/print_gpo_shot_outliers.py` | Shot outlier table (no-GPU, login-node safe) |
| `scripts_mast/validate_gpo_pairs.py` | Schema-v3 pre-flight validation (10 checks; standalone) |
| `scripts_mast/configs/mmt/phases/gpo.yaml` | Phase-level GPO config (LR, schedule, freeze) |
| `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` | Per-task β, filters, loss_type overrides |
| `scripts_mast/configs/mmt/tasks/gpo_tasks_embed_mse_only.yaml` | Ablation: embed_mse anchor only |
| `cluster/ccc_finetune_task.sh` | Single-task base fine-tuning worker |
| `cluster/ccc_finetune_all.sh` | Batch base fine-tuning launcher (all groups; `SUBMIT_EVAL=0` default) |
| `cluster/ccc_collect_gpo_pairs.sh` | Single-task pair collection worker |
| `cluster/ccc_collect_gpo_all.sh` | Batch pair collection launcher (all groups) |
| `cluster/ccc_gpo_finetune.sh` | Single-task GPO fine-tuning worker |
| `cluster/ccc_gpo_all.sh` | Batch GPO training + eval launcher (`SUBMIT_EVAL=1` default) |
| `cluster/ccc_eval_task.sh` | Single-task evaluation worker |
| `cluster/gpo_compare_all.sh` | Login-node: calibrate β + write reports + (optionally) compare |
| `cluster/gpo_pipeline_all.sh` | Full end-to-end pipeline orchestrator |
| `src/mmt/train/losses/continuous_gpo.py` | `ContinuousGPOLoss` (dpo/ipo/slic/hinge) |
| `src/mmt/train/losses/aggregator.py` | Threads GPO kwargs; logs preference accuracy |
| `src/mmt/train/loop_utils.py` | `run_one_epoch` with pair-only val-loss averaging |
| `scripts_mast/mast_utils/gpo/collect.py` | Core pair-collection forward loop |
| `scripts_mast/mast_utils/gpo/writer.py` | `GpoPairWriter` (atomic shard writing) |
| `scripts_mast/mast_utils/gpo/dataset.py` | `GpoPairDataset` (5-filter, train/val split) |
