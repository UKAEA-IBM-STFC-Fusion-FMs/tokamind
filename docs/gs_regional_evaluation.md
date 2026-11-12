# Limiter MSE and spatial error maps (evaluation only)

These diagnostics reuse existing checkpoints and decoded predictions. They do
not change training, losses, embeddings, benchmark scores or GS residuals.

## Configuration and execution

The GS entries in `scripts_mast/configs/mmt/tasks/eval_tasks.yaml` enable:

```yaml
eval:
  compute_metrics:
    psi_error_map:
      enable: true
    jtor_error_map:
      enable: true
    regional_mse:
      enable: true
```

Each option is independently disableable; omitted options default to disabled.
Tasks predicting only psi do not produce invented J_tor scores or maps.
Regional MSE scores only the fixed limiter region. It requires neither J_tor as
an auxiliary reference nor LCFS contours, output timestamps or `TopologyProvider`.
It therefore works for psi-only Tasks 1-3/2-3 without additional source reads.
The independent `gs_topology` option still fetches EFIT geometry if enabled;
disable it for a campaign that needs no topology diagnostics.

For the GS rebuttal campaign, keep the original **v1 Zarr** data, not the newer
NetCDF data or another dataset revision. On CCC the intended override is:

```yaml
data:
  local: true
  local_path: /proj/plasma-fusion/data/fairmast/tokamark/v1
```

Use the original splits, all intended test windows, preprocessing, precision and
checkpoint-selection policy. Check local overrides for accidental shot limits.
Use a GS-compatible TokaMark version: the newer native-adapter APIs are not
interchangeable with this branch. Record both code commits and the resolved
config. Tests also run against TokaMark `main` at
`146b0a0f87380bb16817f2e8758e84ce79ed45d0`; confirm checkpoint/data parity with
the original evaluations before launching.

```bash
python scripts_mast/run_eval.py \
  --task task_1-4-strong \
  --model_source /path/to/existing/training_run \
  --tag gs_limiter_v1
```

The run must include its original best checkpoint, config and required codec/
normalization artifacts. Do not retune them. Use a fresh tag (or `--tag-date`)
to avoid replacing any existing evaluation. Results go under the evaluation's
`metrics/<task>/` directory. No training command is needed.

## Maps: `psi_error_map.npz`, `jtor_error_map.npz`

Both use the same streaming implementation and cover the **full grid** in
`(R, Z)` array order, matching their stored `r` and `z` coordinate matrices.
They store `mse`, `rmse`, `mae`, signed `bias` (prediction minus reference),
per-cell `count`, `task_name`, `signal`, `native_units`, `full_grid`,
`n_fields_selected` and `n_fields_contributed`. All previous psi keys, filename,
summary key and public `PsiErrorMapMetric` call signature remain supported.

Only finite prediction/reference pairs contribute to a map. Zero-count pixels
are NaN. This preserves the existing psi-map convention. Regional MSE has the
stricter prediction-validity rule below. Neither output stores individual grids.

The map aggregates every selected window/time occurrence, including repeated
physical timestamps if windows overlap. Each pixel's MSE equals its accumulated
squared error divided by its count. For a static mask, a pooled regional MSE can
be recovered by summing `mse * count` over positive-count masked cells and
dividing by their total count. That is **not generally the task MSE below**, as
the map does not preserve per-shot/window means. A time-varying LCFS cannot be
applied retrospectively to the aggregate map.

## Regional tables and protocol

- `regional_mse_shots.csv`: one row per selected shot, signal and region.
- `regional_mse_task.csv`: one row per signal and region; includes population
  standard deviation across scored shot MSEs (`mse_std_pop`, not a standard error).
- `regional_mse_protocol.json`: aggregation, geometry, missing-data policy and units conventions.

The only CSV region is `limiter`: original `MAST_lim_mask_rz`, **without erosion**.
No full-grid or LCFS MSE rows are produced. Protocol schema version 2 records this
change and removes the former LCFS coverage counters. Existing `limiter` rows
retain their aggregation and coverage-hash conventions for the same finite data.

The existing eroded `GSEvalGrid.region` is unchanged for GS and smoothness
diagnostics. Their unsuffixed CSV columns use that eroded limiter. Their optional
`plasma_mask` columns are separate current-threshold diagnostics, not this new
MSE region. Pointwise MSE needs no derivative-stencil erosion. The limiter is not
the plasma boundary: it may include cells outside the LCFS.

For each predicted signal:

1. Compute a window MSE by pooling valid squared errors across limiter cells
   and that signal's target times. Each signal uses its own window mask.
2. Average scored window MSEs equally within each shot.
3. Average scored shot MSEs equally for the task.

These MSEs differ from the benchmark's full-grid errors by region as well as
aggregation. Even for the same region, mean shot MSE is not generally the square
of mean shot RMSE: squaring and averaging do not commute. Computation uses
float64 arithmetic; compare with tolerance rather than bitwise equality to a
float32 benchmark.
Raw MSE is in `(Wb/rad)^2` for psi and `(A/m^2)^2` for J_tor; do not combine
signals with different units or label these values NRMSE.

CSV coverage includes selected/scored windows and fields, valid cells,
non-finite reference cells inside the limiter and empty-field counts (no valid
limiter cells). A shot/signal with no valid scored limiter cells reports NaN,
never zero.
Finite selected targets inside the limiter with non-finite predictions (or
squared-error overflow) raise an error rather than improve the score through
exclusion. Cells outside the limiter, including non-finite predictions there,
do not affect this diagnostic. Unexpected regional computation errors fail the
evaluation; missing reference cells are reported through coverage.

`squared_error_sum / n_cells` is the alternative pooled MSE, not the `mse` column.
Per-shot `coverage_sha256` hashes the ordered scored window/slice indices and
finite region-cell masks. Compare these hashes and counts across data/strong/
weak with identical ordering and dataset/config provenance before interpreting
score differences. Hashes do not establish dataset identity on their own.

Inspect logs for incomplete coverage. If scored subsets differ, align the
comparison or rerun on a common subset; do not silently compare different samples.

## Validation

With a compatible TokaMark installed or prepended to `PYTHONPATH`:

```bash
PYTHONPATH=src:scripts_mast:/path/to/compatible-tokamark/src \
  python -m pytest tests/test_regional_evaluation.py -q
```

Tests cover orientation, un-eroded limiter boundaries, exterior-cell exclusion,
finite-value accounting, per-shot weighting, batch invariance, maps, config
validation and psi-only/signal-specific evaluation without auxiliary reads. They
do not replace a checkpoint/data smoke evaluation before a CCC campaign.
