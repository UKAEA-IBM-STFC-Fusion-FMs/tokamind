"""
strict.py — Checkpoint integrity and configuration checks before training resume.

This module defines the shared checkpoint verification protocol used by the MMT training loop. It provides:

    • A training/configuration signature that excludes only the resume flag.
    • A versioned SHA-256 manifest covering the latest state and the best checkpoint.
    • Read-only verification of required files, metadata and serialized training state.

CGPO calls inspect_resume() before constructing MAST datasets or the policy model. The checkpoint API also calls it
before restoring weights and optimizer state. CGPO-specific pair and source/reference identity is supplied through
the training configuration; this module does not load MAST data or implement a separate CGPO checkpoint format.

Responsibilities and limits
---------------------------
api.py writes/restores checkpoints; rng.py handles global and DataLoader random state. This module verifies that a
resume point is complete and compatible. The manifest is written last, so partial checkpoint overwrites are rejected;
it does not provide rollback or repair. Only completed-epoch checkpoints using the supported manifest version can
be resumed. Changes to the objective or protocol require a new run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import atomic_json_save, torch_load_full
from .rng import validate_rng_state

VERSION = 1


# ----------------------------------------------------------------------------------------------------------------------
def training_signature(train_cfg: Mapping[str, Any], loader_cfg: Mapping[str, Any]) -> str:
    """
    Compute a stable SHA-256 signature for the effective training and loader configuration.

    Parameters
    ----------
    train_cfg : Mapping[str, Any]
        Effective training configuration, including stages, losses and any caller-provided provenance identity.
        Only the ``resume`` flag is excluded from comparison.
    loader_cfg : Mapping[str, Any]
        Effective DataLoader configuration, including batch and sampling settings.

    Returns
    -------
    str
        Hexadecimal configuration digest. Neither input mapping is modified.
    """

    config = copy.deepcopy(dict(train_cfg))
    config.pop("resume", None)
    payload = {"train": config, "loader": dict(loader_cfg)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
def _hash(path: Path) -> str:
    """
    Compute the SHA-256 digest of a checkpoint file without loading it into memory.

    Parameters
    ----------
    path : pathlib.Path
        Existing file to read.

    Returns
    -------
    str
        Hexadecimal content digest.

    Raises
    ------
    OSError
        If the file cannot be opened or read.
    """

    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
def write_manifest(directory: str | Path, files: Iterable[str], best_files: Iterable[str]) -> None:
    """
    Publish the integrity manifest after all checkpoint files have been written.

    Parameters
    ----------
    directory : str | pathlib.Path
        The ``checkpoints/latest`` directory.
    files : Iterable[str]
        Basenames of the latest model blocks, metadata and training-state files.
    best_files : Iterable[str]
        Basenames of the model blocks and metadata in the sibling ``best`` directory.

    Returns
    -------
    None

    Raises
    ------
    OSError
        If a required checkpoint file cannot be read or the manifest cannot be written.

    Notes
    -----
    The manifest itself is replaced atomically. A failure while writing earlier checkpoint files can still leave
    a mixed checkpoint; the next verification rejects it instead of resuming from inconsistent state.
    """

    root = Path(directory)
    best = root.parent / "best"
    atomic_json_save(
        {
            "version": VERSION,
            "files": {name: _hash(root / name) for name in files},
            "best_files": {name: _hash(best / name) for name in best_files},
        },
        str(root / "resume_manifest.json"),
    )


# ----------------------------------------------------------------------------------------------------------------------
def inspect_resume(
    run_dir: str | Path,
    *,
    signature: str | None = None,
    block_names: Iterable[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Verify a complete resume point without constructing datasets or changing model and RNG state.

    Parameters
    ----------
    run_dir : str | pathlib.Path
        Existing run directory containing ``checkpoints/latest`` and ``checkpoints/best``.
    signature : str | None
        Expected training/configuration digest. When provided, it must match the saved signature.
        Optional. Default: None.
    block_names : Iterable[str] | None
        Model block names whose checkpoint files must be present in both latest and best manifests.
        Optional. Default: None.

    Returns
    -------
    tuple[dict[str, Any], dict[str, Any]]
        Verified latest metadata and training state containing the signature, history and loader RNG states.

    Raises
    ------
    FileNotFoundError
        If the manifest or a required checkpoint file is missing.
    ValueError
        If the manifest version, hashes, metadata, configuration signature or state structure are incompatible.
    RuntimeError
        If a serialized Torch state cannot be decoded or an RNG state cannot be validated.

    Notes
    -----
    Serialized files are read on CPU. Model tensor compatibility and optimizer loading are checked subsequently
    by the checkpoint API. Streaming datasets and persistent-worker RNG state cannot be resumed exactly.
    """
    block_names = tuple(block_names) if block_names is not None else None
    root = Path(run_dir) / "checkpoints" / "latest"
    manifest_path = root / "resume_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing verified resume checkpoint: {manifest_path}. Use a new run tag.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != VERSION or not isinstance(manifest.get("files"), dict):
        raise ValueError("Unsupported resume checkpoint protocol; use a new run tag.")
    files = manifest["files"]
    required = {"meta.json", "rng.pt", "training_state.pt", "optimizer.pt", "scheduler.pt", "scaler.pt"}
    if block_names is not None:
        required.update(f"{name}.pt" for name in block_names)
    if not required <= files.keys():
        raise ValueError(f"Incomplete resume manifest: missing {sorted(required - files.keys())}")
    best_files = manifest.get("best_files", {})
    expected_best = {"meta.json"} | {f"{name}.pt" for name in (block_names or [])}
    if not isinstance(best_files, dict) or not expected_best <= best_files.keys():
        raise ValueError("Incomplete best checkpoint manifest.")
    for name, expected_hash in best_files.items():
        path = root.parent / "best" / name
        if Path(name).name != name or not path.is_file():
            raise FileNotFoundError(f"Missing or invalid best checkpoint file: {name}")
        if _hash(path) != expected_hash:
            raise ValueError(f"Best checkpoint is corrupt or inconsistent with latest: {name}.")
    loaded = {}
    for name, expected_hash in files.items():
        if Path(name).name != name or not (root / name).is_file():
            raise FileNotFoundError(f"Missing or invalid resume file: {name}")
        if _hash(root / name) != expected_hash:
            raise ValueError(f"Corrupt/incomplete resume checkpoint: {name} checksum mismatch.")
        if name.endswith(".pt"):
            value = torch_load_full(root / name, map_location="cpu")
            if not isinstance(value, dict):
                raise ValueError(f"Invalid resume state: {name}")
            if name in {"rng.pt", "training_state.pt", "optimizer.pt", "scheduler.pt", "scaler.pt"}:
                loaded[name] = value
            del value
    meta = json.loads((root / "meta.json").read_text())
    for key in ("epoch", "global_step", "bad_epochs", "stage_index", "epoch_in_stage"):
        if not isinstance(meta.get(key), int) or meta[key] < 0:
            raise ValueError(f"Invalid resume metadata: {key}")
    if not isinstance(meta.get("training_complete"), bool):
        raise ValueError("Resume checkpoint must declare whether training is complete.")
    if meta.get("checkpoint_metric") not in {"mse", "objective"}:
        raise ValueError("Resume checkpoint is missing its selection metric.")
    if meta["epoch"] < 1 or meta["epoch_in_stage"] < 1:
        raise ValueError("Resume checkpoint must follow a completed epoch.")
    if not isinstance(meta.get("best_val_so_far"), (int, float)) or not np.isfinite(meta["best_val_so_far"]):
        raise ValueError("Invalid best validation metric in resume checkpoint.")
    state = loaded["training_state.pt"]
    if signature is not None and state.get("signature") != signature:
        raise ValueError("Resume training configuration/objective/protocol changed; use a new run tag.")
    if not isinstance(state.get("history", {}).get("stages"), dict):
        raise ValueError("Resume checkpoint is missing training history.")
    for name in ("train", "val"):
        loader_state = state.get("loaders", {}).get(name)
        if not isinstance(loader_state, dict):
            raise ValueError(f"Resume checkpoint is missing {name} loader state.")
        if "unsupported" in loader_state:
            raise ValueError(f"Cannot resume {name} loader: {loader_state['unsupported']}")
        for value in loader_state.values():
            torch.Generator().set_state(value.cpu())
    validate_rng_state(loaded["rng.pt"])
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt"):
        if not isinstance(loaded[name], dict):
            raise ValueError(f"Invalid resume state: {name}")
    if not {"state", "param_groups"} <= loaded["optimizer.pt"].keys():
        raise ValueError("Invalid optimizer resume state.")
    if "last_epoch" not in loaded["scheduler.pt"]:
        raise ValueError("Invalid scheduler resume state.")
    return meta, state
