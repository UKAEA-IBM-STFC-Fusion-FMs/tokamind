"""
MAST integration utilities (scripts_mast).

This package contains the project-specific glue code that connects the dataset-agnostic `mmt/` core library to the MAST
benchmark stack.

Key modules
-----------
- benchmark_imports.py : centralized baseline (MAST_benchmark) imports
- config/              : convention-based YAML merge into ExperimentConfig
- task_definition.py   : resolve benchmark/local task definition by task name
- task_signals.py      : convert task definition -> signals_by_role for MMT core
- pipeline_ops.py      : low-level transforms/collate helpers for run_*.py
- entry_helpers.py     : high-level run_*.py orchestration helpers
- embedding_resolution.py : pretrain/finetune/eval embedding resolution
- benchmark_eval.py       : benchmark-aligned evaluation runner
- tune_dct3d.py        : DCT3D tuning step (run + load overrides)
"""

from .config import load_experiment_config
from .task_definition import load_task_definition
from .task_signals import build_signals_by_role_from_task_definition

from .pipeline_ops import (
    setup_device_and_mp,
    extract_signal_stats,
    build_default_transform,
    make_collate_fn,
)
from .entry_helpers import (
    init_run_context,
    build_mast_datasets,
    build_window_data,
    build_model_and_optional_warmstart,
)
from .embedding_resolution import (
    resolve_pretrain_embeddings,
    resolve_finetune_embeddings,
    resolve_eval_embeddings,
)
from .benchmark_eval import evaluate_benchmark_and_diagnostics

from .tune_dct3d import run_dct3d_tuning, load_embeddings_overrides


# ----------------------------------------------------------------------------------------------------------------------

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
    "make_collate_fn",
    # Entry script helpers
    "init_run_context",
    "build_mast_datasets",
    "build_window_data",
    "build_model_and_optional_warmstart",
    # Embedding resolution
    "resolve_pretrain_embeddings",
    "resolve_finetune_embeddings",
    "resolve_eval_embeddings",
    # Evaluation
    "evaluate_benchmark_and_diagnostics",
    # Embedding tuning
    "run_dct3d_tuning",
    "load_embeddings_overrides",
]
