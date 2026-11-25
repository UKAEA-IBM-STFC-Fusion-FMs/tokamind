"""
Convention-based experiment config assembly (dataset/benchmark agnostic).

This package provides convention-based YAML configuration assembly for
pretrain/finetune/eval phases, with support for task-specific overrides, CLI
parameter injection, and warm-start inheritance. It is shared by integration
layers (e.g. ``scripts_mast``), which supply their own
``configs_root`` and a thin wrapper around :func:`load_experiment_config`.

Key modules
-----------
- loader.py        : top-level experiment config loading orchestration
- merge.py         : YAML loading and deep-merge utilities
- inheritance.py   : source model config inheritance for warmstart/eval
- cli_overrides.py : CLI parameter injection (--model_source, --init, --tag, etc.)
- finalize.py      : path computation and config snapshot persistence
- ids.py           : run-id and model-id naming conventions

Configuration hierarchy
-----------------------
Configs are assembled in this order (later overrides earlier), all relative to
the caller-supplied ``configs_root``:
1. {model_profile}/embeddings/{profile}/_default.yaml, or {model_profile}/embeddings/dct3d/_default.yaml for task-only profiles
2. {model_profile}/phases/{phase}.yaml
3. {model_profile}/tasks/{phase}_tasks.yaml["tasks"][{task}]
4. {model_profile}/embeddings/{profile}/{task}.yaml
5. CLI overrides (--model_profile, --model_source, --init, --tag, --run_id)
6. Source model inheritance (warmstart/eval only)
"""

from .loader import load_experiment_config


# ----------------------------------------------------------------------------------------------------------------------

__all__ = ["load_experiment_config"]
