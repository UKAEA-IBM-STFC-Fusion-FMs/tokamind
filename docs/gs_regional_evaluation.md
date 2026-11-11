# Regional MSE and spatial error maps (evaluation only)

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
Regional MSE requires the source equilibrium geometry for the LCFS component.
It reads the same source stores as the dataset, locally or remotely, using the
existing `TopologyProvider`, even if `gs_topology` is disabled.

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
config. Local tests of this change used TokaMark
`1a200f4588addaad2ae3d53d438d9e1946c09294`; confirm the campaign version against
the original evaluations before launching.

```bash
python scripts_mast/run_eval.py \
  --task task_1-4-strong \
  --model_source /path/to/existing/training_run \
  --tag gs_regions_v1
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

Regions are overlapping, not disjoint partitions:

| Region | Definition |
| --- | --- |
| `full_grid` | All grid centres with finite reference values |
| `limiter` | Original `MAST_lim_mask_rz`, **without erosion** |
| `lcfs` | Inside the reference EFIT LCFS polygon for the signal's own target time |

The existing eroded `GSEvalGrid.region` is unchanged for derivative diagnostics.
LCFS masks never use model predictions or the J_tor threshold mask. The provider
uses the nearest reference time within 1.5 times the median reference cadence;
an unavailable alignment is explicitly counted. No output-time fallback from
psi to J_tor is used. The reference contour is closed and grid centres on its
edges are included (1e-10 m tolerance). Non-finite, degenerate, self-intersecting
or out-of-grid polygons are invalid; LCFS is **not** silently clipped to limiter.
The shared provider removes invalid/zero-padded coordinate pairs before scoring,
as it already does for topology metrics.

For each signal and region:

1. Compute a window MSE by pooling valid squared errors across its grid cells
   and target times. Different LCFS sizes therefore weight times by valid cells
   within that window.
2. Average scored window MSEs equally within each shot.
3. Average scored shot MSEs equally for the task.

For complete finite data, the full-grid shot MSE equals the square of the
benchmark shot RMSE, up to numerical precision. The task MSE averages those
shot MSEs; it is **not** the square of the benchmark's mean shot RMSE. Both use
equal shot weights, but squaring and averaging do not commute. Computation uses
float64 arithmetic; compare with tolerance rather than bitwise equality to a
float32 benchmark.
Raw MSE is in `(Wb/rad)^2` for psi and `(A/m^2)^2` for J_tor; do not combine
signals with different units or label these values NRMSE.

CSV coverage includes selected/scored windows and fields, valid cells,
non-finite reference cells, missing/invalid LCFS and empty-region counts.
Missing/invalid LCFS excludes only that field from LCFS scoring, not full-grid
or limiter scoring. An entirely unscored region reports NaN, never zero.
Finite selected targets with non-finite predictions (or squared-error overflow)
raise an error, rather than yield an artificially improved score. Unexpected
regional computation/provider initialization errors also fail the evaluation;
normal missing reference data is reported through coverage.

`squared_error_sum / n_cells` is the alternative pooled MSE, not the `mse` column.
Per-shot `coverage_sha256` hashes the ordered scored window/slice indices and
finite region-cell masks. Compare these hashes and counts across data/strong/
weak with identical ordering and dataset/config provenance before interpreting
score differences. Hashes do not establish dataset identity on their own.

Inspect logs for incomplete coverage: a successful process alone does not mean
all LCFS references were available. Missing source stores produce missing LCFS
coverage, not a substitute plasma mask. If scored subsets differ, align the
comparison or rerun on a common subset; do not silently compare different samples.

## Validation

With a compatible TokaMark installed or prepended to `PYTHONPATH`:

```bash
PYTHONPATH=src:scripts_mast:/path/to/compatible-tokamark/src \
  python -m pytest tests/test_regional_evaluation.py -q
```

Tests cover orientation, boundary inclusion, invalid polygons, dynamic masks,
finite-value accounting, per-shot weighting, batch invariance, maps, config
validation and single-pass evaluator wiring with synthetic predictions. They
do not replace a checkpoint/data smoke evaluation before a CCC campaign.
