"""
Tokamark split-aware asset resolution and metadata adaptation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import yaml

from .benchmark_imports import (
    RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
    TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
    RANDOM_SPLIT_SIGNALS_STATS_FILE,
    TEMPORAL_SPLIT_SIGNALS_STATS_FILE,
    RANDOM_SPLIT_OUTLIER_METADATA_FILE,
    TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
)


# ----------------------------------------------------------------------------------------------------------------------
def resolve_split_assets(split: str) -> dict[str, str]:
    """
    Resolve Tokamark asset paths for a given split.

    Parameters
    ----------
    split : str
        Split name ("random" or "temporal").

    Returns
    -------
    dict[str, str]
        Mapping with keys:
        - data_splits_file_path
        - stats_metadata_file_path
        - outlier_metadata_file

    Raises
    ------
    ValueError
        If split is not supported.

    """

    s = str(split).strip().lower()
    if s == "random":
        return {
            "data_splits_file_path": RANDOM_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
            "stats_metadata_file_path": RANDOM_SPLIT_SIGNALS_STATS_FILE,
            "outlier_metadata_file": RANDOM_SPLIT_OUTLIER_METADATA_FILE,
        }
    if s == "temporal":
        return {
            "data_splits_file_path": TEMPORAL_SPLIT_TOKAMARK_DATA_SPLITS_FILE,
            "stats_metadata_file_path": TEMPORAL_SPLIT_SIGNALS_STATS_FILE,
            "outlier_metadata_file": TEMPORAL_SPLIT_OUTLIER_METADATA_FILE,
        }

    raise ValueError(f"Unsupported data.split={split!r}. Allowed values: ['random', 'temporal'].")


# ----------------------------------------------------------------------------------------------------------------------
def apply_signal_stats_override(
    dict_task_metadata: dict[str, Any],
    *,
    stats_metadata_file_path: str,
) -> None:
    """
    Overwrite mean/std in role-scoped task metadata using split-specific stats YAML.

    Parameters
    ----------
    dict_task_metadata : dict[str, Any]
        Task metadata returned by tokamark.tasks.get_task_metadata().
    stats_metadata_file_path : str
        Split-specific signal stats YAML path.

    Returns
    -------
    None

    """

    with open(stats_metadata_file_path, "r", encoding="utf-8") as f:
        stats = yaml.safe_load(f) or {}
    if not isinstance(stats, Mapping):
        raise TypeError(
            f"Signals stats metadata must be a mapping (dict), got {type(stats).__name__} from "
            f"{stats_metadata_file_path!r}."
        )

    for role in ("input", "actuator", "output"):
        role_meta = dict_task_metadata.get(role, {})
        if not isinstance(role_meta, dict):
            continue
        for signal_name, signal_meta in role_meta.items():
            if not isinstance(signal_meta, dict):
                continue
            split_meta = stats.get(signal_name)
            if not isinstance(split_meta, Mapping):
                continue
            if "mean" in split_meta:
                signal_meta["mean"] = split_meta["mean"]
            if "std" in split_meta:
                signal_meta["std"] = split_meta["std"]
