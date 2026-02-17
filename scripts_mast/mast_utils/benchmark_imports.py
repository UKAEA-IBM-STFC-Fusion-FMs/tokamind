"""
Centralized imports from the benchmark package (MAST_benchmark).

Purpose
-------
This module is intentionally strict: it FAILS FAST at import time if:
- the benchmark package is not installed / importable, OR
- the benchmark refactored and moved/renamed required symbols.

This makes benchmark refactors a "single-file fix":
update ONLY this module to match the new benchmark API.

Recommended usage
-----------------
Instead of importing benchmark symbols directly in run_*.py, do:

    from scripts_mast.mast_utils.benchmark_imports import (
        initialize_MAST_dataset,
        initialize_model_dataset_iterable,
        TaskModelTransformWrapperIterable,
        get_train_test_val_shots,
        get_task_metadata,
        benchmark_get_task_config,
        WindowMetricsWriter,
        compute_task_metrics,
    )

Notes
-----
- As of the 'wrapper_to_iterable' branch, the benchmark now provides window-level
  iterables via TaskModelTransformWrapperIterable (IterableDataset).
- The old shot-level wrapper (initialize_model_dataset) is deprecated.
- All MMT code now uses initialize_model_dataset_iterable exclusively.
"""

from __future__ import annotations


# 1) Make the "package not importable" error explicit and readable.
try:
    import MAST_benchmark  # noqa: F401
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "benchmark package 'MAST_benchmark' is not importable. "
        "Install the benchmark package (or ensure it's on PYTHONPATH)."
    ) from e


# 2) Import the exact symbols we rely on. If benchmark refactors, this block breaks
#    and the error points here (single-file fix).
try:
    from MAST_benchmark.data import initialize_MAST_dataset, initialize_model_dataset_iterable
    from MAST_benchmark.tools.Task_Model_Wrapper_Iterable import TaskModelTransformWrapperIterable
    from MAST_benchmark.data_split import get_train_test_val_shots
    from MAST_benchmark.tasks import get_task_metadata
    from MAST_benchmark.tasks import get_task_config as benchmark_get_task_config
    from MAST_benchmark.evaluator import WindowMetricsWriter, compute_task_metrics
except Exception as e:
    raise ImportError(
        "Failed to import required symbols from benchmark package 'MAST_benchmark'.\n"
        "This likely means the benchmark repo refactored its Python API.\n"
        "Update scripts_mast/mast_utils/benchmark_imports.py to match the new locations/names.\n"
        f"Original error: {type(e).__name__}: {e}"
    ) from e


__all__ = [
    "initialize_MAST_dataset",
    "initialize_model_dataset_iterable",
    "TaskModelTransformWrapperIterable",
    "get_train_test_val_shots",
    "get_task_metadata",
    "benchmark_get_task_config",
    "WindowMetricsWriter",
    "compute_task_metrics",
]
