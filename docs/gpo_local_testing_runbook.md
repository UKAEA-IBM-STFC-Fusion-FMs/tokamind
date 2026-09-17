# Local CGPO verification without cluster jobs

Use the installed project Python environment from the repository root:

```bash
export CGPO_LAUNCHER_DIR=/absolute/path/to/external/cluster
PYTHONPATH=src:scripts_mast python -m pytest -q tests
PYTHON_BIN=/absolute/path/to/python TASKS='task_1-1' RUN_FINETUNE=1 \
  bash "$CGPO_LAUNCHER_DIR/gpo_pipeline_all.sh" --dry_run
PYTHON_BIN=/absolute/path/to/python TASKS='task_1-1' \
  bash "$CGPO_LAUNCHER_DIR/gpo_compare_all.sh" --compare_only --dry_run
```

`test_cgpo_pipeline.py` simulates LSF responses, validates submission IDs and failure propagation,
checks all Bash entrypoints, asserts that a full dry-run writes nothing and verifies test manifests.
`test_cgpo_protocol.py` exercises all 14 task configurations and a synthetic entrypoint smoke test
through collection, real calibration, CGPO configuration/loaders, both evaluations and strict
comparison. Heavy MAST/model operations are stubbed in that smoke test; no real experiment runs.
`test_cgpo_metrics_resume.py` uses the production epoch/training loop with small CPU models to test
masked/sparse metrics, batch partitioning and deterministic interrupted/resumed training.

For real data commands, immutable calibration, verified evaluation, cluster resources and migration,
use [the current runbook](gpo_runbook.md). Old procedures that patch the shared YAML, create reports
in dry-run or treat IPO as having an always-nonzero gradient are superseded.

A dry-run cannot validate remote data access or installed compute-node dependencies. Its purpose is
to check the complete command graph without submissions or writes. Resource and local-data settings
must be prepared on CCC before a later real experiment.

Without `CGPO_LAUNCHER_DIR`, only the external Bash checks are explicitly skipped. Launchers remain outside the repository.
