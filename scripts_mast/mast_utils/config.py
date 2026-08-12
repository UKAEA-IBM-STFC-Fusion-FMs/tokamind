"""
MAST config loading: a thin wrapper over the shared mmt experiment-config loader.

The convention-based assembly logic lives in ``mmt.utils.config.experiment`` so
it can be shared across integration layers. This module binds it to the MAST
config tree and owns MAST-specific validation and TokaMark split inheritance.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

from mmt.utils.config.experiment import load_experiment_config as _load_experiment_config
from mmt.utils.config.experiment.inheritance import load_source_run_config_yaml
from mmt.utils.config.schema import ExperimentConfig


MAST_CONFIGS_ROOT = "scripts_mast/configs"
MAST_DATA_SPLITS = {"random", "temporal"}
logger = logging.getLogger("mmt.ConfigLoader")


def _inherit_mast_data_split(merged: MutableMapping[str, Any], phase: str) -> None:
    """
    Inherit the source run's TokaMark split for MAST warm-start and evaluation.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged experiment configuration, modified in place.
    phase : str
        Experiment phase.

    Returns
    -------
    None
    """

    if phase not in {"finetune", "eval"}:
        return

    model_source = merged.get("model_source") or {}
    source_run_dir = model_source.get("run_dir")
    if source_run_dir is None:
        return

    source_cfg = load_source_run_config_yaml(model_run_dir=Path(str(source_run_dir)))
    source_data = source_cfg.get("data") or {}
    if "split" not in source_data:
        return

    source_split = source_data["split"]
    if phase == "finetune":
        requested_split = (merged.get("data") or {}).get("split")
        if requested_split is not None and requested_split != source_split:
            logger.warning(
                "Finetune warmstart requested data.split=%r, but source run uses data.split=%r. "
                "Using source split for consistency.",
                requested_split,
                source_split,
            )
        merged.setdefault("data", {})["split"] = source_split

    model_source["data_split"] = source_split


def _validate_smoothness_eval_config(raw_cfg: MutableMapping[str, Any]) -> None:
    """Validate MAST-specific smoothness diagnostic settings when they are configured for evaluation."""

    eval_cfg = raw_cfg.get("eval") or {}
    if not isinstance(eval_cfg, Mapping):
        raise TypeError("MAST eval configuration must be a mapping.")
    compute_metrics = eval_cfg.get("compute_metrics") or {}
    if not isinstance(compute_metrics, Mapping):
        raise TypeError("MAST eval.compute_metrics must be a mapping.")
    smoothness_cfg = compute_metrics.get("smoothness")
    if smoothness_cfg is None:
        return
    if not isinstance(smoothness_cfg, Mapping):
        raise TypeError("MAST eval.compute_metrics.smoothness must be a mapping.")

    enable = smoothness_cfg.get("enable", False)
    if not isinstance(enable, bool):
        raise TypeError("MAST eval.compute_metrics.smoothness.enable must be bool.")
    if "sigma" not in smoothness_cfg:
        return
    sigma = smoothness_cfg["sigma"]
    if isinstance(sigma, bool) or not isinstance(sigma, (int, float)):
        raise TypeError("MAST eval.compute_metrics.smoothness.sigma must be a finite positive number.")
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("MAST eval.compute_metrics.smoothness.sigma must be finite and positive.")


def _validate_psi_error_map_eval_config(raw_cfg: MutableMapping[str, Any]) -> None:
    """Validate the optional full-grid psi error-map diagnostic configuration."""

    compute_metrics = (raw_cfg.get("eval") or {}).get("compute_metrics") or {}
    psi_error_map_cfg = compute_metrics.get("psi_error_map")
    if psi_error_map_cfg is None:
        return
    if not isinstance(psi_error_map_cfg, Mapping):
        raise TypeError("MAST eval.compute_metrics.psi_error_map must be a mapping.")
    enable = psi_error_map_cfg.get("enable", False)
    if not isinstance(enable, bool):
        raise TypeError("MAST eval.compute_metrics.psi_error_map.enable must be bool.")


def _validate_gs_metrics_eval_config(raw_cfg: MutableMapping[str, Any]) -> None:
    """Validate the optional Grad--Shafranov equation/operator diagnostic configuration."""

    compute_metrics = (raw_cfg.get("eval") or {}).get("compute_metrics") or {}
    gs_metrics_cfg = compute_metrics.get("gs_metrics")
    if gs_metrics_cfg is None:
        return
    if not isinstance(gs_metrics_cfg, Mapping):
        raise TypeError("MAST eval.compute_metrics.gs_metrics must be a mapping.")
    enable = gs_metrics_cfg.get("enable", False)
    if not isinstance(enable, bool):
        raise TypeError("MAST eval.compute_metrics.gs_metrics.enable must be bool.")


def load_experiment_config(*args: Any, **kwargs: Any) -> ExperimentConfig:
    """
    Load a MAST experiment config (defaults ``configs_root`` to the MAST tree).

    Parameters
    ----------
    *args : Any
        Positional arguments forwarded to ``mmt.utils.config.experiment.load_experiment_config``.
    **kwargs : Any
        Keyword arguments forwarded to the shared loader. ``configs_root`` defaults
        to ``scripts_mast/configs`` when not supplied.

    Returns
    -------
    ExperimentConfig
        Resulting experiment configuration object.
    """

    kwargs.setdefault("configs_root", MAST_CONFIGS_ROOT)
    kwargs.setdefault("integration_hook", _inherit_mast_data_split)
    return _load_experiment_config(*args, **kwargs)


def validate_mast_config(cfg: Any) -> None:
    """
    Validate configuration fields owned by the MAST integration.

    Parameters
    ----------
    cfg : Any
        Fully merged MAST experiment configuration.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If a required MAST data-source field is missing.
    TypeError
        If a MAST-specific field has an unsupported type.
    ValueError
        If a shot subset is not positive or a TokaMark split is unsupported.
    """

    raw_cfg = getattr(cfg, "raw", cfg)
    if not isinstance(raw_cfg, MutableMapping):
        raise TypeError("MAST configuration must be a mutable mapping or expose a mutable `.raw` mapping.")

    data_cfg = raw_cfg.get("data")
    if not isinstance(data_cfg, MutableMapping):
        raise TypeError("MAST config must define data as a mapping.")

    if "local" not in data_cfg:
        raise KeyError("MAST config is missing required field data.local.")
    if not isinstance(data_cfg["local"], bool):
        raise TypeError(f"MAST data.local must be bool, got {type(data_cfg['local']).__name__}.")

    if "subset_size" not in data_cfg:
        raise KeyError("MAST config is missing required field data.subset_size.")
    subset_size = data_cfg["subset_size"]
    if subset_size is not None and (
        not isinstance(subset_size, int) or isinstance(subset_size, bool) or subset_size <= 0
    ):
        raise ValueError(f"MAST data.subset_size must be a positive integer or null, got {subset_size!r}.")

    phase = str(raw_cfg.get("phase"))
    if phase in {"pretrain", "finetune"}:
        split = data_cfg.get("split", "random")
        if split is None or (isinstance(split, str) and not split.strip()):
            split = "random"
        if not isinstance(split, str):
            raise TypeError(f"MAST data.split must be a string, got {type(split).__name__}.")
        split = split.strip().lower()
        if split not in MAST_DATA_SPLITS:
            raise ValueError(f"Unsupported MAST data.split={split!r}. Allowed values are: {sorted(MAST_DATA_SPLITS)}.")
        data_cfg["split"] = split
    elif phase == "eval":
        if "split" in data_cfg:
            raise ValueError(
                "For MAST phase='eval', data.split must not be set; evaluation uses model_source.data_split "
                "inherited from the source run."
            )
        model_source = raw_cfg.get("model_source") or {}
        source_split = model_source.get("data_split")
        if not isinstance(source_split, str):
            raise TypeError("MAST evaluation requires model_source.data_split inherited from the source run.")
        source_split = source_split.strip().lower()
        if source_split not in MAST_DATA_SPLITS:
            raise ValueError(
                f"Unsupported MAST model_source.data_split={source_split!r}. "
                f"Allowed values are: {sorted(MAST_DATA_SPLITS)}."
            )
        model_source["data_split"] = source_split
        _validate_smoothness_eval_config(raw_cfg=raw_cfg)
        _validate_psi_error_map_eval_config(raw_cfg=raw_cfg)
        _validate_gs_metrics_eval_config(raw_cfg=raw_cfg)


__all__ = ["load_experiment_config", "validate_mast_config", "MAST_CONFIGS_ROOT"]
