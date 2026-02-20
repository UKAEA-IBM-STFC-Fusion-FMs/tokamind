"""
Low-level checkpoint I/O utilities.

This module provides atomic save/load helpers used by the checkpointing
system to safely write and read model state, metadata, and auxiliary
files. It contains no model-specific logic and is shared across
checkpoint workflows (save, resume, warm-start).
"""

from __future__ import annotations
import json
import os
import tempfile
from typing import Any, Dict, Optional

import torch


# ------------------------------------------------------------------
# Low-level atomic save/load
# ------------------------------------------------------------------


def atomic_save(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    final_ext = os.path.splitext(path)[1] or ".pt"

    with tempfile.NamedTemporaryFile(
        suffix=final_ext + ".tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp = f.name
        torch.save(obj, tmp)

    os.replace(tmp, path)


def atomic_json_save(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json.tmp",
        dir=os.path.dirname(path),
        delete=False,
    ) as f:
        tmp = f.name
        json.dump(obj, f)
    os.replace(tmp, path)


def tload(path: str, map_location="cpu") -> Any:
    return torch.load(path, map_location=map_location)


def torch_load_full(path: str, map_location="cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def best_or_latest_dir(run_dir: str) -> Optional[str]:
    best = os.path.join(run_dir, "checkpoints", "best")
    latest = os.path.join(run_dir, "checkpoints", "latest")
    if os.path.isdir(best):
        return best
    if os.path.isdir(latest):
        return latest
    return None
