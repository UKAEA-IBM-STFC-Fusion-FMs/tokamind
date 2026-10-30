# CGPO: Local Machine Smoke Test Runbook

This runbook guides researchers, reviewers, and collaborators who do **not** have access to the IBM CCC cluster or large-scale compute resources to run and verify the complete **Continuous Generalized Preference Optimization (CGPO)** pipeline locally on a personal machine (such as an Apple Silicon Mac or Linux workstation).

---

## 1. Prerequisites and Setup

### Environment
Ensure your Python environment has PyTorch, `zarr`, and `tokamark` installed:

```bash
# Verify environment
python -c "import torch, zarr, tokamark, mmt; print('All core libraries imported successfully!')"
```

> [!NOTE]
> On Apple Silicon Macs, PyTorch uses the `mps` device automatically when available, falling back cleanly to CPU where appropriate.

### Data Configuration: Remote S3 Vault vs. Local Directory

You do not need the full 9,000-shot MAST dataset. Setting `data.subset_size: 1` selects 1 shot per split:
- Train: Shot `#21719` (~72 windows)
- Val: Shot `#16431` (~63 windows)
- Test: Shot `#20509` (~70 windows)

Create or update `scripts_mast/configs/local_overrides.yaml` (which is gitignored and takes highest precedence).

#### Option A: Stream from the Open S3 Cloud Vault (Recommended for Reviewers)
Reviewers without local data can stream and cache shots directly from the public UKAEA/STFC S3 endpoint (`https://s3.echo.stfc.ac.uk/mast/tokamark/v1`):

```yaml
data:
  local: false                               # stream directly from public S3 vault
  local_path: null
  subset_size: 1                             # smoke-test on 1 shot per split
  cache:
    enable: true                             # materialize windows into in-memory cache
    num_workers: 0
    max_windows: null                        # CGPO protocol requires null (complete windows)
loader:
  batch_size: 16
  num_workers: 0
train:
  amp:
    enable: false
```

#### Option B: Load from Pre-Downloaded Local Zarr Directory
Keep the common settings above, replacing only the `data.local` and `data.local_path` values
if you already downloaded the required shots (e.g. via `download_tokamark_sample_data.ipynb`):
```yaml
data:
  local: true
  local_path: /path/to/your/local/mast/shots  # directory containing <shot_id>.zarr
  subset_size: 1
```

Local files must include **all shots selected from the official split**. With `subset_size: 1`,
this means the first selected train, validation and test shot, not any arbitrary three shots.
Missing local files are errors; they never silently redefine the benchmark split.
Shots that genuinely yield no valid test windows are rechecked individually before exclusion.

#### One-epoch schedules (both data options)

Do **not** put `train.stages` in the shared `local_overrides.yaml`: named stages are merged
and unknown names appended, so listing both `ft_scratch` and `gpo` runs both stages in each phase.

For this smoke test, edit the existing `task_1-1` entry in
`scripts_mast/configs/mmt/tasks/finetune_tasks.yaml`, retaining its other settings:

```yaml
tasks:
  task_1-1:
    train:
      stages:
        - name: ft_scratch
          epochs: 1
```

Separately, edit the existing `task_1-1` entry in
`scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`, again retaining its other settings:

```yaml
tasks:
  task_1-1:
    train:
      stages:
        - name: gpo
          epochs: 1
```

These are partial edits to the corresponding task, **not replacements for the full files**.
Use the chosen task ID instead of `task_1-1` when testing another task. Each phase now has
one one-epoch stage, with its own existing learning rates and freezing policy. Make these
edits **before calibration**, which freezes the effective CGPO recipe, and restore the
production epoch budgets before launching full experiments.

> [!IMPORTANT]
> The configuration hash is strictly tracked. Never edit `local_overrides.yaml` between Stage 2 (calibration) and Stage 3 (GPO training), as CGPO enforces configuration immutability.

---

## 2. Automated One-Command Execution

You can run the entire 5-stage pipeline with a single command via [`run_local_smoke_test.py`](../scripts_mast/run_local_smoke_test.py):

```bash
PYTHONPATH=src:scripts_mast python scripts_mast/run_local_smoke_test.py \
  --task task_1-1 \
  --tag local_smoke
```

This will run all stages sequentially:
1. **Stage 0**: Scratch fine-tune base model
2. **Stage 1**: Collect GPO pair trajectories
3. **Stage 2**: Calibrate beta heuristic and signal blacklist
4. **Stage 3**: GPO fine-tune warmstarted from base model
5. **Stage 4**: Verified test evaluation of both models
6. **Stage 5**: Produce side-by-side comparison report

---

## 3. Manual Step-by-Step Stage Execution

If you prefer to inspect intermediate artifacts between stages, execute each step individually:

### Stage 0: Base Model Fine-Tuning
Train the initial base model from scratch:
```bash
PYTHONPATH=src:scripts_mast python scripts_mast/run_finetune.py \
  --task task_1-1 \
  --init scratch \
  --emb_profile dct3d \
  --tag local_smoke
```
*Output:* Checkpoints and embedding indices under `runs/ft-task_1-1-scratch-mmt-local_smoke/`.

### Stage 1: Collect GPO Pairs
Collect counterfactual prediction windows from the base model:
```bash
PYTHONPATH=src:scripts_mast python scripts_mast/run_collect_gpo_pairs.py \
  --task task_1-1 \
  --model_source ft-task_1-1-scratch-mmt-local_smoke \
  --split both \
  --val_fraction 0 \
  --tag local_smoke \
  --overwrite
```
*Output:* Pair shards and `collection_config.json` under `runs/ft-task_1-1-scratch-mmt-local_smoke/gpo_pairs_local_smoke/`.

### Stage 2: CGPO Task Calibration
Compute the $\beta$ scale heuristic and generate immutable task configuration:
```bash
PYTHONPATH=src:scripts_mast python scripts_mast/calibrate_gpo_task.py \
  runs/ft-task_1-1-scratch-mmt-local_smoke/gpo_pairs_local_smoke \
  --output_dir cgpo_artifacts/local_smoke \
  --run_tag local_smoke \
  --no_plots
```
*Output:* `cgpo_artifacts/local_smoke/task_1-1/gpo_tasks.yaml` and `calibration_summary.json`.

### Stage 3: CGPO Fine-Tuning
Fine-tune the model with the Continuous GPO loss using the calibrated recipe:
```bash
GPO_TASKS_YAML=cgpo_artifacts/local_smoke/task_1-1/gpo_tasks.yaml \
PYTHONPATH=src:scripts_mast python scripts_mast/run_gpo_finetune.py \
  --task task_1-1 \
  --model_source ft-task_1-1-scratch-mmt-local_smoke \
  --gpo_dir runs/ft-task_1-1-scratch-mmt-local_smoke/gpo_pairs_local_smoke \
  --model_profile mmt \
  --emb_profile dct3d \
  --tag local_smoke
```
*Output:* Checkpoint under `runs/ft-task_1-1-ws-ft-task_1-1-scratch-mmt-local_smoke-mmt-local_smoke/`.

### Stage 4: Verified Test Evaluation
Evaluate both models against the official test protocol:

```bash
# Remove unverified preliminary log from Stage 0 if present
rm -rf runs/ft-task_1-1-scratch-mmt-local_smoke/eval_local_smoke

# 4a: Evaluate base model
PYTHONPATH=src:scripts_mast python scripts_mast/run_eval.py \
  --task task_1-1 \
  --model_source ft-task_1-1-scratch-mmt-local_smoke \
  --tag local_smoke \
  --cgpo_protocol runs/ft-task_1-1-scratch-mmt-local_smoke/gpo_pairs_local_smoke

# 4b: Evaluate CGPO model
PYTHONPATH=src:scripts_mast python scripts_mast/run_eval.py \
  --task task_1-1 \
  --model_source ft-task_1-1-ws-ft-task_1-1-scratch-mmt-local_smoke-mmt-local_smoke \
  --tag local_smoke \
  --cgpo_protocol runs/ft-task_1-1-scratch-mmt-local_smoke/gpo_pairs_local_smoke
```

### Stage 5: Comparison and Validation
Generate the side-by-side delta table and verify protocol conformance:

```bash
PYTHONPATH=src:scripts_mast python scripts_mast/compare_gpo_eval.py \
  --tasks task_1-1 \
  --base_ft_tag local_smoke \
  --gpo_version local_smoke \
  --eval_tag local_smoke \
  --require_verified_test
```

---

## 4. Expected Results Example

A typical local smoke test run on 1 shot will output:

```text
GPO vs Base Finetune Evaluation Comparison
Runs root : runs
Metrics   : ['NRMSE_mean', 'NMAE_mean', 'RMSE_mean', 'MAE_mean', 'n_shots']
Eval tags : base=local_smoke, gpo=local_smoke

──────────────────────────────────────────────────────────────────────────
  Task: task_1-1   base: ft-task_1-1-scratch-mmt-local_smoke
                  gpo:  ft-task_1-1-ws-ft-task_1-1-scratch-mmt-local_smoke-mmt-local_smoke
──────────────────────────────────────────────────────────────────────────
  Metric                 Base            GPO   Δ (GPO−Base)            Δ %
  ────────────────────────────────────────────────────────────────────────
  NRMSE_mean         0.558703       0.549832    -0.00887078       -1.59% ✓
  NMAE_mean          0.436367       0.429395    -0.00697141       -1.60% ✓
  RMSE_mean          0.466657        0.45824    -0.00841732       -1.80% ✓
  MAE_mean           0.372953       0.365784    -0.00716974       -1.92% ✓
  n_shots                   1              1             +0       +0.00%  
  ────────────────────────────────────────────────────────────────────────

  Verdict: ✓ GPO improved all metrics

Compared 1/1 tasks successfully; incomplete=[].
```

This confirms end-to-end functionality, embedding preservation, pair alignment, calibration, and optimization on standard hardware.
