"""
Configuration loading and merging for scripts_mast experiments.

This package provides convention-based YAML configuration assembly for pretrain/finetune/eval phases, with support for
task-specific overrides, CLI parameter injection, and warm-start inheritance.

Key modules
-----------
- loader.py        : top-level experiment config loading orchestration
- merge.py         : YAML loading and deep-merge utilities
- inheritance.py   : source model config inheritance for warmstart/eval
- cli_overrides.py : CLI parameter injection (--model, --init, --tag, etc.)
- finalize.py      : path computation and config snapshot persistence
- ids.py           : run-id and model-id naming conventions

Configuration hierarchy
-----------------------
Configs are assembled in this order (later overrides earlier):
1. scripts_mast/configs/common/embeddings.yaml
2. scripts_mast/configs/common/{phase}.yaml
3. scripts_mast/configs/tasks_overrides/{task}/{phase}_overrides.yaml
4. scripts_mast/configs/tasks_overrides/{task}/embeddings_overrides/{profile}.yaml
5. CLI overrides (--model, --init, --tag, --run_id)
6. Source model inheritance (warmstart/eval only)

"""

from .loader import load_experiment_config


# ----------------------------------------------------------------------------------------------------------------------

__all__ = ["load_experiment_config"]
