"""Reproducible CGPO collection contract and official MAST shot splits."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from mmt.utils.config.experiment.inheritance import load_source_run_config_yaml
from mmt.utils.config.experiment.loader import _normalize_preprocess_chunks
from mmt.utils.config.experiment.merge import deep_merge, load_yaml

PROTOCOL = "mast-shot-split-v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_contract(merged):
    """Source owns the model, representation, windows and benchmark split."""
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


def _contract_conflicts(expected, actual, prefix=""):
    """Describe conflicting leaves without conflating omission with an override."""
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


def apply_source_contract(merged, configs_root, *, collection=False):
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


def benchmark_manifest(data):
    from mast_utils.benchmark_imports import get_train_test_val_shots
    from mast_utils.tokamark_split import resolve_split_assets

    assets = resolve_split_assets(data["split"])
    train, test, val = get_train_test_val_shots(
        max_index=data.get("subset_size"), data_splits_file_path=assets["data_splits_file_path"]
    )
    shots = {k: sorted(int(s) for s in v) for k, v in (("train", train), ("val", val), ("test", test))}
    validate_shots(shots)
    return {"protocol": PROTOCOL, "shots": shots, "assets": {k: file_digest(v) for k, v in assets.items()}}


def validate_shots(shots):
    for name in ("train", "val", "test"):
        if not shots.get(name) or len(shots[name]) != len(set(shots[name])):
            raise ValueError(f"Missing, empty or duplicate {name} shots in CGPO manifest.")
    tr, va, te = (set(shots[k]) for k in ("train", "val", "test"))
    if tr & va or tr & te or va & te:
        raise ValueError("CGPO train, validation and test shots must be disjoint.")


def source_fingerprint(run_dir):
    root = Path(run_dir)
    checkpoint = root / "checkpoints" / "best"
    if not checkpoint.is_dir():
        checkpoint = root / "checkpoints" / "latest"
    files = sorted(p for folder in (checkpoint, root / "embeddings") for p in folder.rglob("*") if p.is_file())
    if not any(p.suffix == ".pt" for p in files if checkpoint in p.parents):
        raise ValueError(f"No source checkpoint weights in {checkpoint}")
    return {str(p.relative_to(root)): file_digest(p) for p in files}


def collection_contract(merged, task_definition):
    return {
        "task": merged["task"],
        "task_definition": digest(task_definition),
        "source": source_fingerprint(merged["model_source"]["run_dir"]),
        "representation": source_contract(merged),
        "cache_dtype": merged["data"]["cache"].get("dtype") if merged["data"]["cache"]["enable"] else None,
        "split_manifest": benchmark_manifest(merged["data"]),
    }


def validate_collection(cc, expected, gpo_dir=None):
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


def save_run_snapshot(cfg, *, resume=False):
    """Never overwrite a previous run's resolved configuration."""
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
