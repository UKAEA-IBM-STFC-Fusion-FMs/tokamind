"""
protocol.py — Configuration, provenance and shot-split contract for the MAST CGPO pipeline.

This module defines the CGPO-specific contract shared by pair collection, pair validation, calibration,
reconstruction and fine-tuning. It is not part of the ordinary MAST training path. Its responsibilities are:

    • Inheriting model, embeddings, preprocessing and benchmark split from the source run.
    • Reapplying local overrides while rejecting changes to source assumptions or run identity.
    • Recording disjoint official train/validation/test shot lists and benchmark asset hashes.
    • Fingerprinting source checkpoint, embedding and collected pair files.
    • Creating or validating the immutable pre-resolution CGPO request snapshot.

Configuration precedence
------------------------
The source run owns the representation and shot split. Local overrides are applied last, but may only change
settings compatible with that contract. Collection and training require deterministic collation and complete
windows; validation pairs never define the training split or its filtering thresholds.

Scope and limits
----------------
PROTOCOL identifies the collection format and split contract. Legacy or changed collections are rejected rather
than migrated. These helpers verify configuration and artifact identity; they do not load model weights, restore
optimizer/RNG state or validate the training checkpoint manifest. Those operations belong to mmt.checkpoints.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from mmt.utils.config.experiment.inheritance import load_source_run_config_yaml
from mmt.utils.config.experiment.loader import _normalize_preprocess_chunks
from mmt.utils.config.experiment.merge import deep_merge, load_yaml

# Collection contract version; independent of the training checkpoint manifest version.
PROTOCOL = "mast-shot-split-v1"


# ----------------------------------------------------------------------------------------------------------------------
def digest(value):
    """
    Compute a stable SHA-256 digest of a JSON-serializable configuration value.

    Parameters
    ----------
    value : Any
        Configuration or manifest value. Mapping keys are sorted during serialization; objects unsupported by JSON
        are converted to strings.

    Returns
    -------
    str
        Hexadecimal digest of the serialized value.
    """

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
def file_digest(path):
    """
    Compute the SHA-256 digest of an artifact without loading the whole file into memory.

    Parameters
    ----------
    path : str | pathlib.Path
        Existing file whose contents are fingerprinted.

    Returns
    -------
    str
        Hexadecimal content digest.

    Raises
    ------
    OSError
        If the artifact cannot be opened or read.
    """

    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


# ----------------------------------------------------------------------------------------------------------------------
def source_contract(merged):
    """
    Read the source run's model, representation, preprocessing and benchmark split.

    Parameters
    ----------
    merged : Mapping[str, Any]
        Effective CGPO configuration containing ``task`` and ``model_source.run_dir``.

    Returns
    -------
    dict[str, Any]
        Independent copies of the source ``model``, ``embeddings`` and normalized ``preprocess`` sections, plus
        ``data.split`` and ``data.subset_size``. Machine-local paths are excluded from this contract.

    Raises
    ------
    ValueError
        If the source task differs from the requested CGPO task.
    OSError
        If the source configuration cannot be read.
    KeyError
        If a required source or request field is missing.

    Notes
    -----
    The supplied configuration is not modified. Preprocessing normalization applies to the loaded source copy.
    """
    source = load_source_run_config_yaml(model_run_dir=Path(merged["model_source"]["run_dir"]))
    _normalize_preprocess_chunks(source)
    if source.get("task") != merged["task"]:
        raise ValueError("CGPO requires the same task as its source checkpoint.")
    return {
        "model": copy.deepcopy(source["model"]),
        "embeddings": copy.deepcopy(source["embeddings"]),
        "preprocess": copy.deepcopy(source["preprocess"]),
        "data": {"split": source["data"]["split"], "subset_size": source["data"].get("subset_size")},
    }


# ----------------------------------------------------------------------------------------------------------------------
def _contract_conflicts(expected, actual, prefix=""):
    """
    Describe mismatched configuration leaves using dotted paths and explicit absent-value markers.

    Parameters
    ----------
    expected : dict
        Source contract or identity fields to preserve.
    actual : dict
        Effective fields after applying local overrides.
    prefix : str
        Parent path used when descending into nested mappings.
        Optional. Default: an empty string.

    Returns
    -------
    list[str]
        Conflict descriptions containing each field path, source value and override value. A missing field is
        distinguished from an explicit None value; matching mappings produce an empty list.
    """
    missing = object()
    conflicts = []
    for key in sorted(expected.keys() | actual.keys()):
        path = f"{prefix}.{key}" if prefix else str(key)
        before, after = expected.get(key, missing), actual.get(key, missing)
        if isinstance(before, dict) and isinstance(after, dict):
            conflicts.extend(_contract_conflicts(before, after, path))
        elif before != after:
            before_text = "<absent>" if before is missing else repr(before)
            after_text = "<absent>" if after is missing else repr(after)
            conflicts.append(f"{path}: source={before_text}, override={after_text}")
    return conflicts


# ----------------------------------------------------------------------------------------------------------------------
def apply_source_contract(merged, configs_root, *, collection=False):
    """
    Apply source-owned settings and compatible local overrides to a CGPO configuration in place.

    Parameters
    ----------
    merged : MutableMapping[str, Any]
        Merged phase/task configuration. Model, embeddings, preprocessing and split settings are replaced by the
        source contract before local overrides are reapplied.
    configs_root : str | pathlib.Path
        Directory containing the optional ``local_overrides.yaml`` file.
    collection : bool
        Whether the configuration is for pair collection. Collection enables caching unless locally overridden
        and rejects evaluation signal dropping. Training marks all embedding roles for source-codec reuse.
        Optional. Default: False.

    Returns
    -------
    dict[str, Any]
        Source contract used to constrain the effective configuration.

    Raises
    ------
    ValueError
        If local overrides change run identity or source-owned fields, or if collection/training uses signal
        dropping, stochastic collation, window truncation or ``loader.drop_last=True`` contrary to the protocol.
    OSError
        If source or local configuration files cannot be read.
    KeyError
        If a required configuration field is missing.

    Notes
    -----
    This function mutates ``merged`` as it proceeds. Callers must discard that configuration if validation raises;
    there is no rollback of partially applied overrides.
    """

    contract = source_contract(merged)
    identity = {k: copy.deepcopy(merged.get(k)) for k in ("task", "model_source", "run_id", "phase")}
    merged.update({k: copy.deepcopy(contract[k]) for k in ("model", "embeddings", "preprocess")})
    merged.setdefault("data", {}).update(contract["data"])
    merged["model_source"]["data_split"] = contract["data"]["split"]
    identity["model_source"] = copy.deepcopy(merged["model_source"])
    local = Path(configs_root) / "local_overrides.yaml"
    if local.is_file():
        merged.update(deep_merge(base=dict(merged), override=load_yaml(local)))
    identity_conflicts = _contract_conflicts(identity, {k: merged.get(k) for k in identity})
    if identity_conflicts:
        raise ValueError("Local overrides conflict with CGPO run identity: " + "; ".join(identity_conflicts))
    actual = {k: merged[k] for k in ("model", "embeddings", "preprocess")}
    actual["data"] = {k: merged["data"].get(k) for k in contract["data"]}
    conflicts = _contract_conflicts(contract, actual)
    if conflicts:
        raise ValueError("Local overrides conflict with the CGPO source contract: " + "; ".join(conflicts))
    if not collection:
        # All roles, including outputs, reuse source codecs; never retune CGPO embeddings.
        merged["gpo_inherit_all_embeddings"] = True
    if collection:
        # Match the default CGPO cache precision; local overrides remain authoritative.
        local_cfg = load_yaml(local) if local.is_file() else {}
        if "enable" not in local_cfg.get("data", {}).get("cache", {}):
            merged["data"]["cache"]["enable"] = True
        if any((merged.get("eval", {}).get("drop") or {}).values()):
            raise ValueError("CGPO pair collection cannot drop model signals.")
    if any((merged["data"]["cache"].get("max_windows") or {}).values()):
        raise ValueError("CGPO shot protocol requires complete windows; cache.max_windows must be null.")
    # Offline pairs must see the same unaugmented context during both passes.
    for key, value in merged.get("collate", {}).items():
        if key.startswith("p_drop") and value:
            raise ValueError(f"CGPO requires deterministic collation: {key} must be zero/empty.")
    if merged.get("loader", {}).get("drop_last"):
        raise ValueError("CGPO requires loader.drop_last=false.")
    return contract


# ----------------------------------------------------------------------------------------------------------------------
def benchmark_manifest(data):
    """
    Resolve the official MAST shot lists and fingerprint the assets defining the benchmark split.

    Parameters
    ----------
    data : Mapping[str, Any]
        Data configuration containing ``split`` and optional ``subset_size``.

    Returns
    -------
    dict[str, Any]
        Protocol identifier, sorted integer shot lists for train/validation/test and hashes of the resolved assets.

    Raises
    ------
    ValueError
        If the resolved shot lists are empty, duplicated or overlap across splits.
    OSError
        If a required benchmark asset cannot be read.

    Notes
    -----
    Split membership comes from the official benchmark assets, not a random partition of collected pair rows.
    """

    from mast_utils.benchmark_imports import get_train_test_val_shots
    from mast_utils.tokamark_split import resolve_split_assets

    assets = resolve_split_assets(data["split"])
    train, test, val = get_train_test_val_shots(
        max_index=data.get("subset_size"), data_splits_file_path=assets["data_splits_file_path"]
    )
    shots = {k: sorted(int(s) for s in v) for k, v in (("train", train), ("val", val), ("test", test))}
    validate_shots(shots)
    return {"protocol": PROTOCOL, "shots": shots, "assets": {k: file_digest(v) for k, v in assets.items()}}


# ----------------------------------------------------------------------------------------------------------------------
def validate_shots(shots):
    """
    Require nonempty, duplicate-free and mutually disjoint train, validation and test shot lists.

    Parameters
    ----------
    shots : Mapping[str, Sequence[int]]
        Shot identifiers under the required ``train``, ``val`` and ``test`` keys.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If a split is absent or empty, contains duplicate identifiers, or shares a shot with another split.
    """

    for name in ("train", "val", "test"):
        if not shots.get(name) or len(shots[name]) != len(set(shots[name])):
            raise ValueError(f"Missing, empty or duplicate {name} shots in CGPO manifest.")
    tr, va, te = (set(shots[k]) for k in ("train", "val", "test"))
    if tr & va or tr & te or va & te:
        raise ValueError("CGPO train, validation and test shots must be disjoint.")


# ----------------------------------------------------------------------------------------------------------------------
def source_fingerprint(run_dir):
    """
    Fingerprint the selected source checkpoint files and all available embedding artifacts.

    Parameters
    ----------
    run_dir : str | pathlib.Path
        Source run directory containing checkpoint and embedding subdirectories.

    Returns
    -------
    dict[str, str]
        Artifact paths relative to the source run, mapped to SHA-256 content digests.

    Raises
    ------
    ValueError
        If the selected checkpoint directory contains no ``.pt`` weight files.
    OSError
        If a discovered artifact cannot be read.

    Notes
    -----
    The ``checkpoints/best`` directory is selected when present; otherwise ``checkpoints/latest`` is used. An existing
    but empty best directory is rejected. This function checks artifact identity, not model tensor compatibility.
    """

    root = Path(run_dir)
    checkpoint = root / "checkpoints" / "best"
    if not checkpoint.is_dir():
        checkpoint = root / "checkpoints" / "latest"
    files = sorted(p for folder in (checkpoint, root / "embeddings") for p in folder.rglob("*") if p.is_file())
    if not any(p.suffix == ".pt" for p in files if checkpoint in p.parents):
        raise ValueError(f"No source checkpoint weights in {checkpoint}")
    return {str(p.relative_to(root)): file_digest(p) for p in files}


# ----------------------------------------------------------------------------------------------------------------------
def collection_contract(merged, task_definition):
    """
    Build the provenance contract that collection and subsequent CGPO consumers must share.

    Parameters
    ----------
    merged : Mapping[str, Any]
        Effective configuration with task, model source, data/cache settings and benchmark split.
    task_definition : Mapping[str, Any]
        Resolved task definition to fingerprint alongside the source artifacts.

    Returns
    -------
    dict[str, Any]
        Task identity and definition digest, source artifact hashes, source representation contract, effective cache
        dtype and official shot-split manifest. Cache dtype is None when caching is disabled.

    Raises
    ------
    ValueError
        If source-task identity, checkpoint availability or shot-split validation fails.
    OSError
        If a required configuration or artifact cannot be read.
    """

    return {
        "task": merged["task"],
        "task_definition": digest(task_definition),
        "source": source_fingerprint(merged["model_source"]["run_dir"]),
        "representation": source_contract(merged),
        "cache_dtype": merged["data"]["cache"].get("dtype") if merged["data"]["cache"]["enable"] else None,
        "split_manifest": benchmark_manifest(merged["data"]),
    }


# ----------------------------------------------------------------------------------------------------------------------
def validate_collection(cc, expected, gpo_dir=None):
    """
    Verify a collection's protocol and provenance, optionally checking its pair-shard contents.

    Parameters
    ----------
    cc : Mapping[str, Any]
        Saved collection configuration with protocol, contract and shard hashes.
    expected : Mapping[str, Any]
        Expected contract, normally produced from the effective configuration by ``collection_contract()``.
    gpo_dir : str | pathlib.Path | None
        Pair directory to inspect. When provided, hashes for every top-level ``*.npz`` shard must exactly match the
        saved mapping, and at least one shard must exist.
        Optional. Default: None, which skips shard-file verification.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the protocol is unsupported, provenance differs, shot lists are invalid, or pair shards are absent,
        added, removed or modified relative to the saved manifest.
    OSError
        If a shard cannot be read.
    """

    if cc.get("protocol") != PROTOCOL:
        raise ValueError(
            "Legacy CGPO collection has no verified shot split. Recollect with --split both and a new tag."
        )
    if cc.get("contract") != expected:
        raise ValueError(
            "CGPO collection contract differs from task/source/embeddings/windows/split/cache dtype. Recollect."
        )
    validate_shots(cc["contract"]["split_manifest"]["shots"])
    if gpo_dir is not None:
        actual = {p.name: file_digest(p) for p in sorted(Path(gpo_dir).glob("*.npz"))}
        if not actual or cc.get("shards") != actual:
            raise ValueError("CGPO shard contents differ from their collection manifest; recollect.")


# ----------------------------------------------------------------------------------------------------------------------
def save_run_snapshot(cfg, *, resume=False):
    """
    Create the immutable CGPO request snapshot, or verify it when resuming an existing run.

    Parameters
    ----------
    cfg : ExperimentConfig
        Effective pre-resolution configuration exposing ``paths["run_dir"]`` and ``raw``. The run directory must
        already exist; new runs require it to be empty.
    resume : bool
        Verify the existing snapshot against ``cfg.raw`` instead of writing a new file.
        Optional. Default: False.

    Returns
    -------
    None

    Raises
    ------
    FileExistsError
        If a new run would reuse a nonempty directory or overwrite an existing request snapshot.
    ValueError
        If resume is requested without ``gpo_request.yaml`` or with a different saved configuration.
    OSError
        If the run directory or snapshot cannot be accessed.

    Notes
    -----
    The file is created exclusively and never overwritten. It records the request before embedding resolution;
    the later resolved configuration snapshot is managed by the embedding-resolution code. Snapshot agreement
    alone does not establish that a valid training checkpoint is available for resume.
    """
    root = Path(cfg.paths["run_dir"])
    path = root / "gpo_request.yaml"
    if resume:
        if not path.is_file():
            raise ValueError(
                f"Cannot resume CGPO run {root}: gpo_request.yaml is missing. "
                "Use a new tag to start a fresh run; the existing directory is preserved."
            )
        if yaml.safe_load(path.read_text()) != cfg.raw:
            raise ValueError("CGPO resume requires an identical saved configuration; use a new tag for changes.")
        return
    if any(root.iterdir()):
        if not path.is_file():
            raise FileExistsError(
                f"CGPO run directory already exists without gpo_request.yaml: {root}. "
                "Use a new tag to start a fresh run; these files cannot be resumed and are preserved."
            )
        raise FileExistsError(f"CGPO run already exists: {root}. Use a new tag, or GPO_RESUME=1 for the same run.")
    with path.open("x") as stream:
        yaml.safe_dump(cfg.raw, stream, sort_keys=False)


# ----------------------------------------------------------------------------------------------------------------------
def config_fingerprint(configs_root):
    """
    Fingerprint the configuration inputs used by calibration and CGPO training.

    Parameters
    ----------
    configs_root : str | pathlib.Path
        Configuration tree, including any machine-local overrides.

    Returns
    -------
    dict[str, str]
        Relative YAML paths and SHA-256 digests. Calibration artifacts must live outside this tree.
    """
    root = Path(configs_root)
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob("*.yaml")) if p.is_file()}


# ----------------------------------------------------------------------------------------------------------------------
def require_config_subset(expected, actual, path="configuration"):
    """
    Reject overrides that change fields frozen in a calibration recipe.

    Parameters
    ----------
    expected, actual : Any
        Frozen fields and effective values. Extra mapping keys in actual are allowed for inherited defaults.
    path : str
        Dotted path used in errors.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If an expected field differs or is missing.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Frozen calibration field missing: {path}.{key}; create a new calibration/run tag.")
            require_config_subset(value, actual[key], f"{path}.{key}")
    elif expected != actual:
        raise ValueError(f"Frozen calibration field changed: {path}; create a new calibration/run tag.")


# ----------------------------------------------------------------------------------------------------------------------
def verified_test_batches(loader, test_shots, counts):
    """
    Record actual evaluation windows and reject any shot outside the frozen test split.

    Parameters
    ----------
    loader : Iterable[dict]
        Evaluation batches containing shot_id metadata.
    test_shots : Sequence[int]
        Official test membership from the collection manifest.
    counts : MutableMapping[str, int]
        Per-shot window counts, updated in place while batches are consumed.

    Yields
    ------
    dict
        Original unmodified batch.

    Raises
    ------
    ValueError
        If a batch includes a training, validation or unknown shot.
    """
    allowed = set(test_shots)
    for batch in loader:
        for shot in batch["shot_id"]:
            shot = int(shot)
            if shot not in allowed:
                raise ValueError(f"Evaluation contains non-test shot {shot}.")
            key = str(shot)
            counts[key] = counts.get(key, 0) + 1
        yield batch


# ----------------------------------------------------------------------------------------------------------------------
def validate_evaluation_manifest(run_dir, eval_dir, task):
    """
    Verify complete test coverage and the identity of evaluated weights and saved metrics.

    Parameters
    ----------
    run_dir, eval_dir : str | pathlib.Path
        Model run and its tagged evaluation directory.
    task : str
        Requested task identifier.

    Returns
    -------
    dict
        Verified evaluation manifest for comparison with another run.

    Raises
    ------
    ValueError
        If identity, test coverage or file contents differ from the saved proof.
    OSError
        If the manifest, checkpoint or metric files are missing.
    """
    root = Path(eval_dir)
    manifest = json.loads((root / "cgpo_evaluation.json").read_text())
    validate_shots(manifest["split_manifest"]["shots"])
    counts = manifest["test_window_counts"]
    if (
        manifest.get("format") != "cgpo-evaluation-v1"
        or manifest.get("task") != task
        or not manifest.get("complete")
        or set(map(int, counts)) != set(manifest["split_manifest"]["shots"]["test"])
        or any(not isinstance(n, int) or n <= 0 for n in counts.values())
        or manifest["source_config_digest"] != file_digest(Path(run_dir) / f"{Path(run_dir).name}.yaml")
        or manifest["config_digest"] != file_digest(root / "evaluation_config.yaml")
        or manifest["checkpoint"] != source_fingerprint(run_dir)
        or manifest["metrics_digest"] != file_digest(root / "metrics" / task / "task_metrics.csv")
    ):
        raise ValueError(f"Incomplete or stale CGPO evaluation: {root}")
    return manifest
