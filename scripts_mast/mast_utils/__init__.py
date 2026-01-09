"""
MAST integration utilities (scripts_mast).

This package contains the project-specific glue code that connects the
dataset-agnostic `mmt/` core library to the MAST benchmark stack.

Key modules
-----------
- benchmark_imports.py : centralized baseline (MAST_benchmark) imports
- config_loader.py     : convention-based YAML merge into ExperimentConfig
- task_definition.py   : resolve benchmark/local task definition by task name
- task_signals.py      : convert task definition -> signals_by_role for MMT core
- pipeline_helpers.py  : shared helpers for run_*.py entrypoints
"""

from .config_loader import load_experiment_config
from .task_definition import load_task_definition
from .task_signals import build_signals_by_role_from_task_definition

from .pipeline_helpers import (
    setup_device_and_mp,
    extract_signal_stats,
    build_default_transform,
    build_window_datasets,
    make_collate_fn,
)

__all__ = [
    # Config
    "load_experiment_config",
    # Task definition + signals
    "load_task_definition",
    "build_signals_by_role_from_task_definition",
    # Pipeline helpers
    "setup_device_and_mp",
    "extract_signal_stats",
    "build_default_transform",
    "build_window_datasets",
    "make_collate_fn",
]
