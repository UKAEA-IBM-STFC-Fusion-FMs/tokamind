> **Protocol update:** New training requires the verified MAST shot protocol.
> Follow [CGPO shot protocol](cgpo_shot_protocol.md) for current collection and
> training commands. The legacy fractional split and cluster commands below
> must not be used unchanged for new experiments.

# GPO Local Testing Runbook

Step-by-step instructions for running the full Continuous GPO pipeline **without
IBM CCC or any cluster scheduler**.  Every command calls the Python scripts
directly.  No `bsub`, no `bjobs`, no LSF.

All commands are run from the **repo root** with the `tokamind-env` conda
environment activated.

---

## Prerequisites

```bash
# Activate the environment
conda activate tokamind-env

# Confirm GPU / CPU availability
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"

# Confirm all entry-point scripts are importable
python -c "from mast_utils import load_experiment_config; print('OK')"
```

Thread-cap exports (prevent DCT3D / BLAS thrashing):

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

> **Tip:** Add the four exports to your shell's `~/.bashrc` / `~/.zshrc` or
> prefix every `python` invocation below with them when testing locally.

---

## Tag conventions used throughout this runbook

| Symbol | Meaning | Default used below |
|--------|---------|--------------------|
| `<TASK>` | task identifier | `task_4-2` |
| `<BASE_TAG>` | fine-tune run tag | `dct3d-embed-mse` |
| `<GPO_TAG>` | pair-dir tag and GPO run tag | `v8` |
| `<BASE_RUN>` | base fine-tune run ID | `ft-<TASK>-scratch-mmt-<BASE_TAG>` |
| `<GPO_RUN>` | GPO run ID | `ft-<TASK>-ws-<BASE_RUN>-mmt-<GPO_TAG>` |

All `runs/` output lives under the repo root by default.

---

## Stage 0 — Base fine-tuning

Fine-tunes the model from scratch for a single task.

```bash
python scripts_mast/run_finetune.py \
  --task         task_4-2 \
  --init         scratch \
  --model_profile mmt \
  --emb_profile  dct3d \
  --tag          dct3d-embed-mse
```

Output: `runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/`

Repeat for every task you need. To run all 14 tasks sequentially in one shell
loop (CPU/GPU depending on your machine):

```bash
for TASK in task_1-1 task_1-2 task_1-3 \
            task_2-1 task_2-2 task_2-3 \
            task_3-1 task_3-2 task_3-3 \
            task_4-1 task_4-2 task_4-3 task_4-4 task_4-5; do
  echo "=== Finetuning $TASK ==="
  python scripts_mast/run_finetune.py \
    --task "$TASK" \
    --init scratch \
    --model_profile mmt \
    --emb_profile dct3d \
    --tag dct3d-embed-mse
done
```

> **Note:** Full training typically requires a GPU with ≥ 16 GB VRAM per task.
> If you only want to verify the pipeline on one task, `task_1-2` is the
> lightest (16 GB RAM, 8 h walltime on a single A100 on CCC).

---

## Stage 1 — Collect GPO preference pairs

Run after the base fine-tune checkpoint is ready.

```bash
python scripts_mast/run_collect_gpo_pairs.py \
  --task           task_4-2 \
  --model_source   ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  --split          train \
  --val_fraction   0.1 \
  --train_fraction 1.0 \
  --shard_size     2048 \
  --tag            v8
```

Output: `runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8/`

Key flags:

| Flag | Purpose | Typical override |
|------|---------|-----------------|
| `--val_fraction` | Fraction of pairs held out for GPO validation | `0.1` |
| `--train_fraction` | Random subset of windows to collect from (useful for quick tests) | `0.1` for smoke test |
| `--shard_size` | Windows per `.npz` shard | `256` for quick test |
| `--tag` | Names the output sub-folder `gpo_pairs_<tag>/` | `v8` |
| `--overwrite` | Replace existing pair dir | add flag to re-collect |

**Quick smoke-test collection** (collects only 10 % of windows, small shards):

```bash
python scripts_mast/run_collect_gpo_pairs.py \
  --task           task_4-2 \
  --model_source   ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  --split          train \
  --val_fraction   0.1 \
  --train_fraction 0.1 \
  --shard_size     256 \
  --tag            v8-smoke \
  --overwrite
```

Loop for all tasks:

```bash
for TASK in task_1-1 task_1-2 task_1-3 \
            task_2-1 task_2-2 task_2-3 \
            task_4-1 task_4-2 task_4-3 task_4-4 task_4-5; do
  echo "=== Collecting pairs for $TASK ==="
  python scripts_mast/run_collect_gpo_pairs.py \
    --task           "$TASK" \
    --model_source   "ft-${TASK}-scratch-mmt-dct3d-embed-mse" \
    --split          train \
    --val_fraction   0.1 \
    --train_fraction 1.0 \
    --shard_size     2048 \
    --tag            v8
done
```

> Group 3 tasks (`task_3-1`, `task_3-2`, `task_3-3`) are excluded above because
> they have no GPO config in `gpo_tasks.yaml` yet.  Include them once their base
> runs complete and calibration adds their entries.

---

## Stage 2 — Validate pair datasets (optional but recommended)

Runs the full 10-check schema-v3 validation suite on every pair directory.
CPU-only, no GPU needed, completes in seconds.

```bash
python scripts_mast/validate_gpo_pairs.py \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8
```

Multiple directories at once:

```bash
python scripts_mast/validate_gpo_pairs.py \
  runs/ft-task_1-1-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_1-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8
```

Exit code `0` = all checks passed.  Any non-zero exit means a shard has a
genuine problem (NaN, shape mismatch, zero-gap pair, wrong schema version) that
would silently corrupt GPO training.

---

## Stage 3 — β calibration

Reads the pair shards, computes per-signal β, updates `gpo_tasks.yaml`, and
writes per-task reports.  CPU-only.

```bash
python scripts_mast/calibrate_gpo_task.py \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --plots_dir gpo_plots/ \
  --report_dir reports/
```

All 11 GPO-configured tasks at once:

```bash
python scripts_mast/calibrate_gpo_task.py \
  runs/ft-task_1-1-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_1-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_1-3-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_2-1-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_2-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_2-3-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-1-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-3-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-4-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  runs/ft-task_4-5-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --plots_dir gpo_plots/ \
  --report_dir reports/
```

What this writes:

| Output | Path |
|--------|------|
| Calibrated β + shot blacklist | `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` (patched in-place) |
| 4-figure diagnostic PNGs | `gpo_plots/<task>/` |
| JSON stats summary | `runs/.../gpo_pairs_v8/calibration_summary.json` |
| Markdown calibration report | `reports/gpo_stats/<task>/calibration_<task>_v8_<ts>.md` |
| PNGs copied for paper | `reports/gpo_stats/<task>/` |

Useful flags:

| Flag | Effect |
|------|--------|
| `--dry_run` | Print computed β values without writing `gpo_tasks.yaml` — safe for inspection |
| `--no_plots` | Skip matplotlib (no display / headless) |
| `--blacklist_multiplier 15` | Raise outlier threshold from 10× to 15× |
| `--blacklist_top_n 3` | Cap blacklist at 3 shots instead of 5 |

**Review step before proceeding.**  Open each
`reports/gpo_stats/<task>/calibration_*.md` and confirm:

- `β_calibrated` is in the expected range for the task (see table in
  `reports/gpo_beta_calibration_table.md`).
- `shot_blacklist` is not removing more than ~20 % of all shots.
- `p50_MSE` is non-zero and not unusually large.

---

## Stage 3b — Shot outlier inspection

Print the top-20 outlier shots per task (no matplotlib, login-node safe):

```bash
python scripts_mast/print_gpo_shot_outliers.py \
  runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --top 20
```

Save to a Markdown file for paper use:

```bash
{
  echo "# Shot Outlier Table — task_4-2 (v8)"
  echo ""
  python scripts_mast/print_gpo_shot_outliers.py \
    runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
    --top 20
} > reports/gpo_stats/task_4-2/shot_outliers_v8.md
```

---

## Stage 3c — Full diagnostic plots

Generate the 4-figure / 29-panel dataset diagnostic set for a task:

```bash
python scripts_mast/visualize_gpo_stats.py \
  --gpo_dir runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --save_dir gpo_plots/ \
  --no_show
```

Remove `--no_show` if you have a display and want the interactive matplotlib
window to open.

---

## Stage 4 — GPO fine-tuning

Runs the GPO training loop.  Requires a GPU.

```bash
python scripts_mast/run_gpo_finetune.py \
  --task         task_4-2 \
  --model_source ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  --gpo_dir      runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --model_profile mmt \
  --emb_profile  dct3d \
  --tag          v8
```

Output run ID: `ft-task_4-2-ws-ft-task_4-2-scratch-mmt-dct3d-embed-mse-mmt-v8`

Output path: `runs/ft-task_4-2-ws-ft-task_4-2-scratch-mmt-dct3d-embed-mse-mmt-v8/`

Environment-only overrides (not CLI flags):

```bash
# Resume an interrupted run
GPO_RESUME=1 python scripts_mast/run_gpo_finetune.py \
  --task         task_4-2 \
  --model_source ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  --gpo_dir      runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --tag          v8

# Use an alternate gpo_tasks.yaml (e.g. ablation)
GPO_TASKS_YAML=scripts_mast/configs/mmt/tasks/gpo_tasks_embed_mse_only.yaml \
  python scripts_mast/run_gpo_finetune.py \
  --task         task_4-2 \
  --model_source ft-task_4-2-scratch-mmt-dct3d-embed-mse \
  --gpo_dir      runs/ft-task_4-2-scratch-mmt-dct3d-embed-mse/gpo_pairs_v8 \
  --tag          v8-ablation
```

**Healthy training signals** — watch these in the terminal output per epoch:

```
pref_acc       → rising from ~0.5 toward 0.7+   (preference discrimination improving)
d_w            → decreasing                       (margin between preferred/dispreferred widening)
pair_hit_rate  → ≥ 0.05                           (enough paired windows per batch)
EmbedMSELoss_1_nonpair → flat                    (anchor not degrading)
```

**Warning signs and fixes:**

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `pref_acc` stuck at 0.5 | β mis-calibrated | Inspect `calibration_summary.json`; re-run Stage 3 with `--dry_run` |
| `d_w` rising while `pref_acc` rising | Reward hacking | Edit `loss_type: ipo` in `gpo_tasks.yaml` and restart |
| `EmbedMSELoss_1_nonpair` rising sharply | Anchor too weak | Increase `embed_mse weight` from `0.2` to `0.3` in `gpo_tasks.yaml` |
| `pair_hit_rate < 0.05` | Too few pairs reaching training batches | Widen `native_nrmse_percentiles` to `[10.0, 99.0]`; re-collect |
| Early stopping at epoch 1 | `val_fraction` missing or zero | Ensure `GPO_VAL_FRACTION=0.1` was set during collection |

---

## Stage 5 — Evaluation

Evaluates the GPO model against the test split and writes benchmark metrics.
Requires a GPU.

**GPO model:**

```bash
python scripts_mast/run_eval.py \
  --task         task_4-2 \
  --model_source ft-task_4-2-ws-ft-task_4-2-scratch-mmt-dct3d-embed-mse-mmt-v8
```

**Base fine-tune model** (needed for comparison):

```bash
python scripts_mast/run_eval.py \
  --task         task_4-2 \
  --model_source ft-task_4-2-scratch-mmt-dct3d-embed-mse
```

Output metrics CSV:
`runs/<run_id>/eval/metrics/task_4-2/task_metrics.csv`

Columns: `NRMSE_mean`, `NRMSE_std_pop`, `NMAE_mean`, `NMAE_std_pop`,
`RMSE_mean`, `RMSE_std_pop`, `MAE_mean`, `MAE_std_pop`,
`nan_fraction_mean`, `nan_fraction_std_pop`, `n_shots`

---

## Stage 6 — GPO vs base comparison

Reads both `task_metrics.csv` files, prints a Δ% table, and writes a Markdown
report.  CPU-only, no GPU needed.

```bash
python scripts_mast/compare_gpo_eval.py \
  --tasks    task_4-2 \
  --base_ft_tag dct3d-embed-mse \
  --gpo_version v8 \
  --report_dir  reports/
```

Multi-task (all 11 GPO-configured tasks):

```bash
python scripts_mast/compare_gpo_eval.py \
  --tasks task_1-1 task_1-2 task_1-3 \
          task_2-1 task_2-2 task_2-3 \
          task_4-1 task_4-2 task_4-3 task_4-4 task_4-5 \
  --base_ft_tag dct3d-embed-mse \
  --gpo_version v8 \
  --report_dir  reports/
```

Output: `reports/gpo_comparison_dct3d-embed-mse_v8_<timestamp>.md`

Terminal output looks like:

```
task_4-2
  Metric        Base     GPO    Δ (GPO−Base)   Δ %
  NRMSE_mean   0.249   0.231      −0.018       −7.2% ✓
  NMAE_mean    0.131   0.121      −0.010       −7.6% ✓
  ...
```

`✓` = improvement (lower is better), `✗` = degradation.

---

## Quick smoke-test: end-to-end in < 30 minutes

Tests the full pipeline logic with minimal data.  Substitute `task_1-2`
(smallest memory footprint):

```bash
TASK=task_1-2
BASE_TAG=dct3d-embed-mse
GPO_TAG=v8-smoke

# 1. Base fine-tune (few epochs for speed — use a minimal override config)
python scripts_mast/run_finetune.py \
  --task "$TASK" --init scratch \
  --model_profile mmt --emb_profile dct3d --tag "$BASE_TAG"

# 2. Pair collection (10 % of windows)
python scripts_mast/run_collect_gpo_pairs.py \
  --task "$TASK" \
  --model_source "ft-${TASK}-scratch-mmt-${BASE_TAG}" \
  --val_fraction 0.1 --train_fraction 0.1 \
  --shard_size 256 --tag "$GPO_TAG" --overwrite

# 3. Validate
python scripts_mast/validate_gpo_pairs.py \
  "runs/ft-${TASK}-scratch-mmt-${BASE_TAG}/gpo_pairs_${GPO_TAG}"

# 4. Calibrate (dry-run first)
python scripts_mast/calibrate_gpo_task.py \
  "runs/ft-${TASK}-scratch-mmt-${BASE_TAG}/gpo_pairs_${GPO_TAG}" \
  --no_plots --dry_run --report_dir reports/

# 4b. Write gpo_tasks.yaml for real
python scripts_mast/calibrate_gpo_task.py \
  "runs/ft-${TASK}-scratch-mmt-${BASE_TAG}/gpo_pairs_${GPO_TAG}" \
  --no_plots --report_dir reports/

# 5. GPO fine-tuning
python scripts_mast/run_gpo_finetune.py \
  --task "$TASK" \
  --model_source "ft-${TASK}-scratch-mmt-${BASE_TAG}" \
  --gpo_dir "runs/ft-${TASK}-scratch-mmt-${BASE_TAG}/gpo_pairs_${GPO_TAG}" \
  --tag "$GPO_TAG"

# 6. Evaluate both models
python scripts_mast/run_eval.py \
  --task "$TASK" --model_source "ft-${TASK}-scratch-mmt-${BASE_TAG}"

python scripts_mast/run_eval.py \
  --task "$TASK" \
  --model_source "ft-${TASK}-ws-ft-${TASK}-scratch-mmt-${BASE_TAG}-mmt-${GPO_TAG}"

# 7. Compare
python scripts_mast/compare_gpo_eval.py \
  --tasks "$TASK" \
  --base_ft_tag "$BASE_TAG" \
  --gpo_version "$GPO_TAG" \
  --report_dir reports/
```

---

## Environment variable reference

All variables below are optional — defaults are shown.

| Variable | Default | Used by | Description |
|----------|---------|---------|-------------|
| `GPO_RESUME` | `0` | `run_gpo_finetune.py` | `1` to resume interrupted GPO training (env-only, not a CLI flag) |
| `GPO_TASKS_YAML` | _(repo default)_ | `run_gpo_finetune.py` | Path to alternate `gpo_tasks.yaml` for ablations (env-only) |
| `OMP_NUM_THREADS` | _(system)_ | all | Cap to `1` to prevent BLAS thrashing |
| `MKL_NUM_THREADS` | _(system)_ | all | Cap to `1` |
| `OPENBLAS_NUM_THREADS` | _(system)_ | all | Cap to `1` |
| `NUMEXPR_NUM_THREADS` | _(system)_ | all | Cap to `1` |

---

## File map — scripts called in this runbook

| Script | Stage | Description |
|--------|-------|-------------|
| `scripts_mast/run_finetune.py` | 0 | Base supervised fine-tuning |
| `scripts_mast/run_collect_gpo_pairs.py` | 1 | Preference-pair collection |
| `scripts_mast/validate_gpo_pairs.py` | 2 | Schema-v3 validation (10 checks) |
| `scripts_mast/calibrate_gpo_task.py` | 3 | β calibration + YAML patch + reports |
| `scripts_mast/print_gpo_shot_outliers.py` | 3b | Shot outlier table (no GPU) |
| `scripts_mast/visualize_gpo_stats.py` | 3c | 4-figure / 29-panel diagnostics |
| `scripts_mast/run_gpo_finetune.py` | 4 | GPO fine-tuning |
| `scripts_mast/run_eval.py` | 5 | Benchmark evaluation |
| `scripts_mast/compare_gpo_eval.py` | 6 | GPO vs base Δ% table + Markdown report |

---

## Troubleshooting

**`RuntimeError: GPO pair index is empty`**
The `--task` and `--model_source` you passed to `run_gpo_finetune.py` do not
match the task used for collection.  Confirm the pair directory contains shards
for the correct signals, then re-run with the matching `--task` value.

**`calibrate_gpo_task.py` writes `β = inf` or `β = 0`**
The collected pair shards are empty or all pairs have zero MSE gap.  Run
`validate_gpo_pairs.py` to find the root cause.  Re-collect with
`--train_fraction 1.0` (do not subsample) and `--overwrite`.

**`compare_gpo_eval.py` reports "no tasks have eval results"`**
Both `runs/<base_run>/eval/metrics/<task>/task_metrics.csv` and
`runs/<gpo_run>/eval/metrics/<task>/task_metrics.csv` must exist.  Run Stage 5
for both models first.

**`pair_hit_rate = 0` during GPO training**
The `(shot_id, window_index)` keys in the pair dataset do not match what the
dataloader yields.  This usually means the pair directory was collected with a
different `--model_source` or a different data split.  Re-collect with the
exact `--model_source` and `--split train` flags.

**`pref_acc` never moves from 0.5 (even after 20 epochs)**
Check `calibration_summary.json` in the pair directory.  If `p50_MSE` is very
small (< 0.001), `β` will be very large and the IPO gradient saturates.  Add
`--dry_run` to the calibrate step and adjust `--blacklist_multiplier` or set β
manually in `gpo_tasks.yaml`.

**Out-of-memory during GPO training (local GPU)**
Reduce `batch_size` in `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml` for the
affected task, or reduce `shard_size` during collection so fewer pairs are
loaded into the index.
