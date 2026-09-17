# CGPO: verified cluster workflow

This runbook describes the corrected `cgpo` branch. Historical recipes and reports are not evidence
that older runs followed this protocol. See [the shot/resume protocol](cgpo_shot_protocol.md) for
configuration precedence, independent validation and checkpoint semantics.

## Cluster setup

Keep the Bash dispatchers in the external cluster folder, outside the checkout. They call `scripts_mast/gpo_pipeline.py`; paths, Python and
LSF resources belong in an external environment file. Copy `cgpo.env.example` from that external folder to a private environment file and replace its placeholders. Use an installed project environment on a filesystem shared
by login and compute nodes. Set compatible `scripts_mast/configs/local_overrides.yaml` **before**
collection/calibration. Paths may differ from the source machine; split, subset and representation
must agree with the source run. Do not commit machine overrides.

```bash
export CGPO_CLUSTER_ENV=/absolute/path/to/cgpo-cluster.env
export CGPO_LAUNCHER_DIR=/absolute/path/to/external/cluster
bash "$CGPO_LAUNCHER_DIR/gpo_pipeline_all.sh" --dry_run
```

Dry-run prints all five stages and exits successfully without running Python entrypoints, calling
LSF, creating directories or writing calibration/metrics files. It cannot prove that remote datasets,
packages, resources or credentials are available. The environment file must contain configuration
exports only, since Bash sources it even during dry-run.

The CCC checkout must contain the updated Python code, including `scripts_mast/gpo_pipeline.py`, and the external
launcher directory must be copied to CCC separately.

For a future controlled run, set `TASKS='task_1-1'`, a new `GPO_TAG`, and a matching artifact root in
that external file. `BASE_RUN` may select an absolute existing base run (one task only); otherwise
base runs are derived from `BASE_TAG`. With `RUN_FINETUNE=1`, the pipeline creates scratch base runs.
Remove `--dry_run` only when actually ready to submit. No real cluster experiment was launched as
part of the local implementation checks.

## Complete stage sequence

| Stage | Operation | Required result |
|---|---|---|
| 0 | Base training, or explicit reuse (`RUN_FINETUNE=0`) | Source YAML and checkpoint blocks |
| 1 | Collection, or explicit reuse (`RUN_COLLECT=0`) | Official train/validation pairs, disjoint test list and verified shard hashes |
| 2 | Calibration | Immutable task recipe and matching summary under the artifact root |
| 3 | Base evaluation, CGPO training, CGPO evaluation | Successful jobs and complete test manifests |
| 4 | Comparison only | Every requested task has finite metrics and matching verified test coverage |

Base evaluation is always included in the complete path. `RUN_GPO=0` explicitly skips training and
CGPO evaluation; comparison still requires existing valid outputs. Skipped stages, reused artifacts,
missing tasks and incomplete comparisons are printed explicitly; incomplete work exits nonzero.

Each submission must return exactly one positive job ID. The coordinator waits after each stage.
`DONE` is success, `EXIT` is failure; active jobs are polled. An absent job is checked against `bhist`;
without an explicit matching terminal record it is **unknown**, never successful. Failure, ambiguity
or polling timeout prevents downstream submission. Jobs already submitted in the same stage are
not automatically cancelled. Keep the login-side coordinator alive (for example in a managed shell
session); it submits workers directly rather than submitting the Bash dispatchers recursively.

All resource values are external: `LSF_QUEUE`, `LSF_NCPUS`, `LSF_MEM_GB`, `LSF_WALLTIME`, optional
`LSF_GPU`. Override individual stages with `FINETUNE_`, `COLLECT_`, `GPO_`, `BASE_EVAL_` or
`GPO_EVAL_` prefixes, such as `COLLECT_LSF_MEM_GB`. `RUNS_DIR` becomes `MMT_RUNS_DIR` for Python;
`LOG_DIR` and `ARTIFACT_DIR` choose other outputs. Paths and worker arguments are shell-quoted.

## Calibration and frozen configuration

`GPO_TASKS_YAML` chooses the **read-only input recipe**, including an ablation recipe. Calibration
merges the CGPO phase, task and local settings, computes statistics on **training shots only**, and
writes:

```text
<ARTIFACT_DIR>/<task>/gpo_tasks.yaml
<ARTIFACT_DIR>/<task>/calibration_summary.json
```

Optional `--report` and plots stay inside this task artifact. Missing task overrides use phase
defaults, explicitly logged and materialized in the artifact; the shared YAML is never patched.
Existing artifacts are immutable. The coordinator may reuse an artifact only after checking the
recipe, task, run tag, current config hashes, collection identity and matching summary.

```bash
python scripts_mast/calibrate_gpo_task.py /absolute/base/gpo_pairs_new-tag \
  --gpo_tasks_yaml scripts_mast/configs/mmt/tasks/gpo_tasks.yaml \
  --output_dir /absolute/calibration/new-tag --run_tag new-tag --no_plots
```

Training receives the generated task file as `GPO_TASKS_YAML`. Its preflight verifies both the config
inputs and effective frozen training settings, collection identity and target tag. Local overrides
that would undo calibrated beta/blacklist are rejected. Set local overrides first; changing configs
after calibration requires a new artifact and training tag. Keep artifacts outside `configs/`.

The beta rule `1 / median(collection embedding MSE gap)`, geometrically aggregated across signals,
is a **scale heuristic**, not an estimate of reference-relative margins or a demonstrated optimum.
At initialization policy and reference coincide, so reference-relative margins are zero. Calibration
preserves MSE-only recipes; it does not insert a preference term. Blacklist candidates are based only
on training-pair statistics. Review the summary before a costly run; a dry calibration uses the same
calculation without writing files. Automatic calibration requires untransformed gaps.

## Individual entrypoints and comparison

| Bash dispatcher | Operation |
|---|---|
| `ccc_finetune_all.sh`, `ccc_finetune_task.sh` | Base scratch training, wait and check source artifacts |
| `ccc_collect_gpo_all.sh`, `ccc_collect_gpo_pairs.sh` | Collect official train/validation pairs and validate |
| `gpo_compare_all.sh` | Calibrate selected tasks; `--compare` additionally compares existing evaluations |
| `ccc_gpo_all.sh`, `ccc_gpo_finetune.sh` | Train with existing frozen artifacts, then evaluate (`SUBMIT_EVAL=0` skips this evaluation explicitly) |
| `ccc_eval_task.sh` | Verified evaluation; `EVAL_KIND=base` or `gpo` |
| `gpo_pipeline_all.sh` | Complete stage sequence |

```bash
bash "$CGPO_LAUNCHER_DIR/gpo_compare_all.sh" --compare_only
```

`--compare_only` reads existing evaluation outputs and prints comparisons. It performs no calibration,
submission, plotting or report writes. Missing or stale outputs yield a nonzero exit status.

The strict path runs `run_eval.py --cgpo_protocol <pairs>` on both models. It checks collection
identity, task definition and official test assets, rejects non-test batches, requires all declared
test shots to produce windows and writes `cgpo_evaluation.json` only after metrics are produced.
The proof records observed window counts, resolved embedding settings, source/config snapshot hashes,
checkpoint hashes and metric-file hash. The full effective evaluation config is saved alongside it. `compare_gpo_eval.py --require_verified_test` rejects unequal settings, missing shots, stale
weights/CSVs and unequal or non-finite metric counts. Missing shots must be investigated; they are
not silently removed from the test protocol.

Use fresh evaluation tags for incomplete/stale evaluations. The coordinator can reuse a complete,
verified evaluation for the same collection and checkpoint. Untagged historical CSVs remain readable
through the standalone comparison command without strict mode, but do not satisfy this pipeline.

## Loss interpretation

Let `d(a,b)` be per-window embedding MSE, `p` the policy prediction, `r` the frozen source prediction,
`w` the observed target and `l` the collected source prediction. With a reference:

```text
m = [d(p,l) - d(p,w)] - [d(r,l) - d(r,w)]
DPO = -log sigmoid(beta*m)        dL/dm = -beta*sigmoid(-beta*m)
IPO = (m - 1/(2*beta))²         dL/dm = 2*(m - 1/(2*beta))
```

DPO has gradient `-beta/2` at zero margin and approaches zero at large positive margins. IPO's
margin derivative is zero at its target and penalizes overshoot; it does not have a universally
nonzero gradient. The frozen reference supplies an offset; this objective contains no explicit KL
penalty. With squared distances the relative margin is linear in `p`, leaving directions orthogonal
to `w-l` unconstrained. Neither DPO nor IPO guarantees absolute closeness to the target.

`sft_weight` adds a soft MSE penalty on paired windows. A separate `embed_mse` term supervises all
valid windows, including those without pairs. Their weights must be assessed on independent
validation; `train.checkpoint_metric: mse` selects checkpoints using that validation by default.

Without a reference, the implemented surrogate is `transform(d(w,l)) - d(p,w)`. `log_mse_gap` applies
`log1p` and `mse_gap_clip` clips that constant gap. They are incompatible with reference predictions
and now raise during configuration validation and at loss computation. Remove these options from
reference recipes; do not reinterpret them as effective gradient clipping.

## Resume and migration

1. Preserve historical runs and treat their reported metrics as unverified under this protocol.
2. Set source-compatible local overrides; remove clip/log from reference recipes.
3. Recollect with `--split both --val_fraction 0` and a new pair tag; no automatic conversion of old row splits.
4. Calibrate into a fresh external artifact root. Freeze objective, filters, selection metric and configs.
5. Use a new training tag and verified evaluations for both base and CGPO.
6. Resume only the identical interrupted training with `GPO_RESUME=1` and the same artifact/tag.

Strict resume validates before dataset preparation. It restores policy, optimizer, scheduler, scaler,
RNG/loader state, counters, early-stop state and history; the original reference must still match its
saved hashes. Missing/corrupt/legacy checkpoints fail. Exact continuation is supported at completed
epoch boundaries for cached map-style loaders without persistent workers. Objective/protocol/config
changes require a new run. See [protocol details](cgpo_shot_protocol.md) for limitations.

Legacy per-task Bash variables that selected a model/run (`GPO_MODEL`, `FINETUNE_MODEL`, etc.) are
rejected; use `BASE_RUN`, `MODEL_PROFILE`, `BASE_TAG` and the documented tags. Launch these dispatchers
with `bash`, not `bsub < script`. Unrelated pretraining/statistics helpers are outside this workflow.

## Local verification

```bash
PYTHONPATH=src:scripts_mast python -m pytest -q tests
```

Export `CGPO_LAUNCHER_DIR` to include the external Bash checks; otherwise those checks are explicitly skipped.

The suite covers scheduler success/failure/unknown states, ID validation, full no-write dry-run,
comparison-only behavior, frozen calibration, collection→training→evaluation→comparison wiring with
synthetic loaders, configuration rejection, metric invariance and deterministic CPU resume. This
establishes local orchestration and protocol behavior, not availability of CCC resources or empirical
model quality. No real experiment is necessary for these checks.
