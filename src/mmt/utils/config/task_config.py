"""
Task-config helpers.

This module loads a task configuration YAML (either a FAIRMAST `config_task_*.yaml`
or an in-repo `pretrain_task_*.yaml`) and applies a small set of overrides coming
from our `ExperimentConfig`.

Overrides supported (when provided in `cfg.data`):
  - subset_of_shots
  - local
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.config.loader import ExperimentConfig


def build_task_config(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Load the task config (either FAIRMAST config_task_*.yaml or an in-repo pretrain_task_*.yaml)
    and apply simple overrides from our MMT ExperimentConfig.

    Currently we override (when provided in cfg.data):
      - subset_of_shots
      - local

    Returns
    -------
    config_task : dict
        Task configuration dict ready to be passed to
        initialize_datasets_and_metadata_for_task().
    """
    task_cfg_path = Path(cfg.task_config_path)  # <- rename field in loader

    if not task_cfg_path.exists():
        raise FileNotFoundError(
            f"Task config not found: {task_cfg_path}. "
            "Check experiment_base.yaml: task_config: <path>."
        )

    with task_cfg_path.open("r") as f:
        config_task = yaml.safe_load(f) or {}

    data_cfg = cfg.data or {}

    subset = data_cfg.get("subset_of_shots")
    if subset is not None:
        config_task["subset_of_shots"] = subset

    local_flag = data_cfg.get("local")
    if local_flag is not None:
        config_task["local"] = local_flag

    return config_task
