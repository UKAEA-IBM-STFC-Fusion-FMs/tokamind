"""
Bridge helpers between MMT and the FAIRMAST baseline repository.

This module only provides a small helper to override a few
fields in the FAIRMAST baseline task config (config_task_*.yaml) using
values from our ExperimentConfig:

  - subset_of_shots
  - local

"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

from mmt.utils.config_loader import ExperimentConfig


def build_baseline_task_config(cfg: ExperimentConfig) -> Dict[str, Any]:
    """
    Load the FAIRMAST baseline task config (config_task_*.yaml) and
    apply simple overrides from our MMT config.

    Currently we override:
      - subset_of_shots
      - local

    If these keys are not present in cfg.data, the original values
    from the baseline config are kept.

    Parameters
    ----------
    cfg :
        Unified ExperimentConfig produced by load_experiment_config().

    Returns
    -------
    config_task :
        Dictionary representing the baseline task configuration, ready
        to be passed to initialize_datasets_and_metadata_for_task().
    """
    baseline_cfg_path: Path = cfg.baseline_config_path

    with baseline_cfg_path.open("r") as f:
        config_task = yaml.safe_load(f) or {}

    data_cfg = cfg.data or {}

    subset = data_cfg.get("subset_of_shots", None)
    if subset is not None:
        config_task["subset_of_shots"] = subset

    local_flag = data_cfg.get("local", None)
    if local_flag is not None:
        config_task["local"] = local_flag

    return config_task
