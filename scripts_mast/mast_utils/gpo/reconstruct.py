"""
scripts_mast.mast_utils.gpo.reconstruct

Deterministic reconstruction of the dataloader used during GPO pair collection.

Design
------
Each GPO pair dataset directory contains a ``collection_config.json`` file
written by :class:`~scripts_mast.mast_utils.gpo.writer.GpoPairWriter`.  The
file stores the minimal recipe needed to rebuild the exact window-level
dataloader that was used when the pairs were collected:

    {
        "schema_version": 3,
        "run_dir":     "/abs/path/to/runs/ft-task_4-1-scratch-mmt",
        "run_id":      "ft-task_4-1-scratch-mmt",
        "task":        "task_4-1",
        "split":       "train",
        "val_fraction": 0.1,
        "pair_sources": ["model_error"],
        "multi_signal": "joint"
    }

Schema v1 collections used ``"split": "test"`` or ``"val"``. Schema versions
1 through 3 are accepted; v1 datasets are usable for stats analysis but not for
GPO training (native-space shards, no embeddings).

Given this file the reconstruction helper:

1. Loads ``collection_config.json`` from the GPO pairs directory.
2. Calls ``load_experiment_config(phase="eval", model_source=run_id)`` to
   get the full merged config (preprocess, embeddings, loader, …) exactly
   as it was during collection — the eval phase re-reads the saved run YAML
   which is the source of truth for every representation-defining setting.
3. Initialises the MAST dataset for the recorded split.
4. Builds the transform pipeline, optional cache, collate function, and
   DataLoader — reproducing the same window stream.

The returned dataloader produces windows that, when indexed by the
``(shot_id, window_index)`` stored in the .npz shards, recover the exact
input context ``x`` for each GPO pair.

Public API
----------
load_collection_config(gpo_dir)
    Parse and validate ``collection_config.json``.

reconstruct_dataloader(gpo_dir, **loader_overrides)
    Full reconstruction returning a ready-to-use DataLoader.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

logger = logging.getLogger("mmt.GPO")

_SUPPORTED_SCHEMA_VERSIONS = {1, 2, 3}
_REQUIRED_KEYS = {"run_dir", "run_id", "task", "split"}


# ======================================================================================================================
# Config loading
# ======================================================================================================================


def load_collection_config(gpo_dir: str | Path) -> dict[str, Any]:
    """
    Load and validate ``collection_config.json`` from a GPO pairs directory.

    Parameters
    ----------
    gpo_dir : str | Path
        Path to the GPO pairs directory (the folder containing the .npz shards
        and the two JSON files).

    Returns
    -------
    dict[str, Any]
        Parsed collection config.

    Raises
    ------
    FileNotFoundError
        If ``collection_config.json`` is not present in ``gpo_dir``.
    ValueError
        If the file is missing required keys or has an unsupported schema
        version.
    """
    gpo_dir = Path(gpo_dir)
    cc_path = gpo_dir / "collection_config.json"

    if not cc_path.is_file():
        raise FileNotFoundError(
            f"collection_config.json not found in GPO pairs directory.\n"
            f"  expected: {cc_path}\n"
            "  This file is written by run_collect_gpo_pairs.py (schema_version >= 1).\n"
            "  Datasets collected with older code do not have it; re-run collection."
        )

    with cc_path.open(encoding="utf-8") as f:
        cfg = json.load(f)

    version = cfg.get("schema_version")
    if version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"collection_config.json has unsupported schema_version={version!r}. "
            f"Supported: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}."
        )

    missing = _REQUIRED_KEYS - set(cfg)
    if missing:
        raise ValueError(
            f"collection_config.json is missing required keys: {sorted(missing)}. "
            f"Re-run run_collect_gpo_pairs.py to regenerate the dataset."
        )

    split = cfg["split"]
    if split not in ("train", "val", "test", "both"):
        raise ValueError(f"collection_config.json has unexpected split={split!r}. Expected train, val, test, or both.")

    return cfg


# ======================================================================================================================
# Dataloader reconstruction
# ======================================================================================================================


def reconstruct_dataloader(
    gpo_dir: str | Path,
    *,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> DataLoader:
    """
    Reconstruct the DataLoader used during GPO pair collection.

    Reads ``collection_config.json``, replays ``load_experiment_config`` for
    the eval phase with the recorded ``run_id``, and rebuilds the MAST dataset
    + window pipeline + DataLoader for the recorded split.

    The resulting DataLoader is deterministic: iterating it in order and looking
    up windows by ``(shot_id, window_index)`` recovers the exact input context
    ``x`` for every pair in the GPO dataset.

    Parameters
    ----------
    gpo_dir : str | Path
        GPO pairs directory (contains ``collection_config.json``).
    batch_size : int | None
        Override the batch size from the saved config.  ``None`` keeps the
        saved value.
    num_workers : int | None
        Override ``loader.num_workers``.  ``None`` keeps the saved value.

    Returns
    -------
    DataLoader
        Ready-to-use DataLoader that reproduces the collection window stream.
    """
    # Import here to avoid circular imports at module level; these are
    # integration-layer imports that depend on tokamark being installed.
    from mast_utils import (
        load_experiment_config,
        validate_mast_config,
        load_task_definition,
        build_signals_by_role_from_task_definition,
        build_mast_datasets,
        build_window_data,
        resolve_eval_embeddings,
    )
    from mmt.utils import validate_config

    cc = load_collection_config(gpo_dir=gpo_dir)

    run_id: str = cc["run_id"]
    task: str = cc["task"]
    split: str = cc["split"]

    logger.info(
        "Reconstructing GPO dataloader: run_id=%s task=%s split=%s",
        run_id,
        task,
        split,
    )

    # ------------------------------------------------------------------
    # 1. Rebuild the merged config (eval phase re-reads the saved run YAML
    #    and inherits every representation-defining setting from it).
    # ------------------------------------------------------------------
    from .protocol import PROTOCOL, apply_source_contract, collection_contract, validate_collection

    new_protocol = cc.get("protocol") == PROTOCOL
    kwargs = {}
    if new_protocol:
        kwargs["integration_hook"] = lambda merged, phase: apply_source_contract(
            merged, "scripts_mast/configs", collection=True
        )
    cfg_mmt = load_experiment_config(task=task, phase="eval", model_source=cc["run_dir"], save_config=False, **kwargs)
    validate_config(cfg=cfg_mmt)
    if new_protocol:
        cfg_mmt.data.pop("split")
    validate_mast_config(cfg=cfg_mmt)

    # Apply caller overrides to the loader config.
    if batch_size is not None:
        cfg_mmt.raw["loader"]["batch_size"] = int(batch_size)
    if num_workers is not None:
        cfg_mmt.raw["loader"]["num_workers"] = int(num_workers)

    cfg_data = cfg_mmt.data
    cfg_task = load_task_definition(task_key=task)

    # ------------------------------------------------------------------
    # 2. Build MAST dataset for the recorded split.
    # _inherit_mast_data_split writes the source run's split into
    # cfg_mmt.raw["model_source"]["data_split"] during the eval hook.
    # For train/val branches we inject it into cfg_data so
    # build_mast_datasets receives the correct split identifier.
    # ------------------------------------------------------------------
    cfg_model_source = cfg_mmt.raw.get("model_source") or {}
    data_split = cfg_model_source.get("data_split")

    if split == "both":
        if not new_protocol:
            raise ValueError("split=both requires a verified CGPO shot protocol.")
        cfg_data["split"] = data_split
        validate_collection(cc, collection_contract(cfg_mmt.raw, cfg_task), gpo_dir)
        dict_task_metadata, train, val, _test = build_mast_datasets(
            cfg_task=cfg_task,
            cfg_data=cfg_data,
            phase="finetune",
            cfg_model_source=cfg_model_source,
        )
        mast_datasets = {"train": train, "val": val}
        loader_split = "train"
    elif split == "test":
        dict_task_metadata, _train, _val, mast_split = build_mast_datasets(
            cfg_task=cfg_task,
            cfg_data=cfg_data,
            phase="eval",
            cfg_model_source=cfg_model_source,
        )
        mast_datasets = {"test": mast_split}
        loader_split = "test"
    elif split == "train":
        cfg_data_train = {**cfg_data, "split": data_split}
        dict_task_metadata, mast_split, _val, _test = build_mast_datasets(
            cfg_task=cfg_task,
            cfg_data=cfg_data_train,
            phase="finetune",
            cfg_model_source=cfg_model_source,
        )
        mast_datasets = {"train": mast_split}
        loader_split = "train"
    else:  # val
        cfg_data_val = {**cfg_data, "split": data_split}
        dict_task_metadata, _train, mast_split, _test = build_mast_datasets(
            cfg_task=cfg_task,
            cfg_data=cfg_data_val,
            phase="finetune",
            cfg_model_source=cfg_model_source,
        )
        mast_datasets = {"val": mast_split}
        loader_split = "val"

    # ------------------------------------------------------------------
    # 3. Signal specs + embeddings.
    # ------------------------------------------------------------------
    from pathlib import Path as _Path

    train_run_dir = _Path(cfg_mmt.model_source["run_dir"])
    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task=cfg_task,
        dict_metadata=dict_task_metadata,
    )
    signal_specs, codecs = resolve_eval_embeddings(
        cfg_mmt=cfg_mmt,
        signals_by_role=signals_by_role,
        dict_task_metadata=dict_task_metadata,
        train_run_dir=train_run_dir,
    )

    # ------------------------------------------------------------------
    # 4. Window data + DataLoader.
    # ------------------------------------------------------------------
    window_data = build_window_data(
        cfg_mmt=cfg_mmt,
        mast_datasets=mast_datasets,
        dict_task_metadata=dict_task_metadata,
        cfg_task=cfg_task,
        signal_specs=signal_specs,
        codecs=codecs,
        phase="eval",
        window_test_mode=False if new_protocol else None,
    )

    if split == "both":
        return _CombinedLoaders(window_data["train"]["loader"], window_data["val"]["loader"])
    return window_data[loader_split]["loader"]


class _CombinedLoaders:
    """Re-iterable view of official train and validation context loaders."""

    def __init__(self, train, val):
        self.loaders = (train, val)

    def __iter__(self):
        for loader in self.loaders:
            yield from loader

    def __len__(self):
        return sum(len(loader) for loader in self.loaders)
