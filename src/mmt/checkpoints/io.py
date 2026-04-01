"""
Low-level checkpoint I/O utilities.

This module provides atomic save/load helpers used by the checkpointing system to safely write and read model state,
metadata, and auxiliary files. It contains no model-specific logic and is shared across checkpoint workflows (save,
resume, warm-start).
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Callable
from typing import Any, Optional, IO

import torch


# ======================================================================================================================
# Low-level atomic save/load
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def atomic_save(obj: Mapping[str, Any], path: str) -> None:
    """
    Save input state object (mapping/dict) at target path.

    Parameters
    ----------
    obj : Any
        Input state object (mapping/dict) to be saved ar target path `path`.
    path : str
        Target path to save input object `obj`.

    Returns
    -------
    None

    """

    os.makedirs(name=os.path.dirname(path), exist_ok=True)
    final_ext = os.path.splitext(path)[1] or ".pt"

    with tempfile.NamedTemporaryFile(
        suffix=final_ext + ".tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp = f.name
        torch.save(obj, tmp)

    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------------
def atomic_json_save(obj: Mapping[str, Any], path: str) -> None:
    """
    Save input object (mapping/dict) at target path in ".json.tmp" format.

    Parameters
    ----------
    obj : Mapping[str, Any]
        Input object (mapping/dict) to be saved ar target path `path`.
    path : str
        Target path to save input object `obj`.

    Returns
    -------
    None

    """

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json.tmp", dir=os.path.dirname(path), delete=False) as f:
        tmp = f.name
        json.dump(obj, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------------------------------------------
def torch_load(
    path: str | os.PathLike[str] | IO[bytes],
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
) -> Any:
    """
    Load an object (saved with "torch.save") from target path. Proxy for "torch.load(...)".

    Parameters
    ----------
    path : str | os.PathLike[str] | IO[bytes]
        Target path of the torch object to be loaded.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as `map_location` parameter of `torch.load()`.
        Optional. Default: "cpu".

    Returns
    -------
    Any
        Loaded object.

    """

    return torch.load(f=path, map_location=map_location)


# ----------------------------------------------------------------------------------------------------------------------
def torch_load_full(
    path: str | os.PathLike[str] | IO[bytes],
    map_location: Callable | torch.device | str | dict[str, str] | None = "cpu",
) -> Any:
    """
    Load an object (saved with "torch.save") from target path. Proxy for "torch.load(...)" with `weights_only=False`.
    If

    Parameters
    ----------
    path : str | os.PathLike[str] | IO[bytes]
        Target path of the torch object to be loaded.
    map_location : Callable | torch.device | str | dict[str, str] | None
        Same as `map_location` parameter of `torch.load()`.
        Optional. Default: "cpu".

    Returns
    -------
    Any
        Loaded object.

    """

    return torch.load(f=path, map_location=map_location, weights_only=False)


# ----------------------------------------------------------------------------------------------------------------------
def best_or_latest_dir(run_dir: str) -> Optional[str]:
    """
    Return best or latest subdirectory within target run directory.

    Parameters
    ----------
    run_dir : str
        Target run directory.

    Returns
    -------
    Optional[str]
        Either best or latest subdirectory within target run directory, if available, otherwise None.

    """

    best = os.path.join(run_dir, "checkpoints", "best")
    if os.path.isdir(best):
        return best

    latest = os.path.join(run_dir, "checkpoints", "latest")
    if os.path.isdir(latest):
        return latest

    return None
