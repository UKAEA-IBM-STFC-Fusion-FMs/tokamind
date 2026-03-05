"""
Source-run inheritance and finetune model semantics.

This module handles config inheritance from source models for warmstart/eval:
- Resolves source model directories (run_id or path)
- Loads source run config snapshots
- Optionally inherits preprocessing settings (chunk, trim_chunks)
- Applies finetune model semantics (scratch vs warmstart)
- Merges source model config with current overrides

Key concepts:
- Warmstart: load source model weights/config, keep finetune preprocess from
  current task config
- Scratch: use model_scratch architecture, no source inheritance
- Eval: always inherits from source model (weights + config + embeddings),
  including preprocess chunk/trim settings
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

from mmt.utils.paths import get_repo_root

from .merge import deep_merge, load_yaml, resolve_from_repo_root

logger = logging.getLogger("mmt.ConfigLoader")


def resolve_run_id_to_run_dir(run_id: str) -> Path:
    """Resolve a training run id to <repo_root>/runs/<run_id>."""
    s = str(run_id).strip()
    if not s:
        raise ValueError(
            "model_source.run_id must be a non-empty run id (folder name under <repo_root>/runs/)."
        )

    p = Path(s)
    if p.is_absolute() or len(p.parts) != 1:
        raise ValueError(
            "model_source.run_id must be a run id (folder name under <repo_root>/runs/), "
            "e.g. 'pretrain_base'. Do not include 'runs/' or any path separators."
        )

    return (get_repo_root() / "runs" / p.parts[0]).resolve()


def resolve_model_source_dir(
    model_source: Dict[str, Any],
    *,
    phase: str,
) -> tuple[Path, str | None]:
    """Resolve model_source to absolute source run directory."""
    if not isinstance(model_source, dict):
        raise TypeError("Config key 'model_source' must be a mapping (dict).")

    model_path = model_source.get("model_path", None)
    run_id = model_source.get("run_id", None)

    if model_path is not None:
        mp = str(model_path).strip()
        if not mp:
            raise ValueError(
                "model_source.model_path, if provided, must be a non-empty path string."
            )
        path = resolve_from_repo_root(mp)
        if not path.is_dir():
            raise FileNotFoundError(
                f"{phase} phase requires model_source.model_path to point to an existing directory.\n"
                f"Got: {path}"
            )
        return path, None

    if run_id is None:
        raise ValueError(
            f"{phase} phase requires model_source.run_id (a training run id under <repo_root>/runs/) "
            "or model_source.model_path (external run directory)."
        )

    return resolve_run_id_to_run_dir(str(run_id)), str(run_id).strip()


def load_source_run_config_yaml(model_run_dir: Path) -> Dict[str, Any]:
    """Load saved merged source run config from <run_dir>/<run_id>.yaml."""
    src_cfg_path = model_run_dir / f"{model_run_dir.name}.yaml"
    if not src_cfg_path.is_file():
        raise FileNotFoundError(
            "Required source run config YAML not found.\n"
            "Warm-start and evaluation require the saved merged config at:\n"
            f"  {src_cfg_path}\n"
        )
    return load_yaml(src_cfg_path)


def inherit_preprocess_chunk_trim(
    merged: Dict[str, Any],
    src_cfg: Dict[str, Any],
    *,
    allow_override: bool,
) -> None:
    """Inherit preprocess.chunk and preprocess.trim_chunks from source config."""
    src_pre = src_cfg.get("preprocess")
    if not isinstance(src_pre, dict):
        raise KeyError(
            "Source run config is missing required mapping: 'preprocess'.\n"
            "Expected 'preprocess.chunk' and 'preprocess.trim_chunks' to be present."
        )
    if "chunk" not in src_pre or "trim_chunks" not in src_pre:
        raise KeyError(
            "Source run config is missing required keys under 'preprocess'.\n"
            "Expected: preprocess.chunk and preprocess.trim_chunks"
        )

    merged_pre = merged.get("preprocess")
    if merged_pre is None:
        merged_pre = {}
        merged["preprocess"] = merged_pre
    if not isinstance(merged_pre, dict):
        raise TypeError("Config key 'preprocess' must be a mapping (dict).")

    override_chunk = None
    override_trim = None
    if allow_override:
        override_chunk = copy.deepcopy(merged_pre.get("chunk"))
        override_trim = copy.deepcopy(merged_pre.get("trim_chunks"))

    merged_pre["chunk"] = copy.deepcopy(src_pre["chunk"])
    merged_pre["trim_chunks"] = copy.deepcopy(src_pre["trim_chunks"])

    if allow_override:
        if override_chunk is not None:
            if isinstance(override_chunk, dict) and isinstance(
                merged_pre["chunk"], dict
            ):
                merged_pre["chunk"] = deep_merge(merged_pre["chunk"], override_chunk)
            else:
                merged_pre["chunk"] = override_chunk

        if override_trim is not None:
            if isinstance(override_trim, dict) and isinstance(
                merged_pre["trim_chunks"], dict
            ):
                merged_pre["trim_chunks"] = deep_merge(
                    merged_pre["trim_chunks"], override_trim
                )
            else:
                merged_pre["trim_chunks"] = override_trim


def apply_finetune_model_semantics(merged: Dict[str, Any], *, init_mode: str) -> None:
    """Materialize canonical `model` for scratch and validate warmstart blocks.

    New finetune semantics:
      - `model_scratch`: full architecture base used for scratch init
      - `finetune_model_overrides`: overrides applied in both scratch and warmstart
      - `warmstart.model_overrides`: additional overrides applied only in warmstart
    """
    if "model" in merged:
        raise ValueError(
            "Finetune config now uses explicit keys: "
            "'model_scratch', 'finetune_model_overrides', and 'warmstart.model_overrides'. "
            "Remove top-level 'model' from finetune configs."
        )

    common_overrides = merged.get("finetune_model_overrides")
    if common_overrides is None:
        common_overrides = {}
        merged["finetune_model_overrides"] = common_overrides
    if not isinstance(common_overrides, dict):
        raise TypeError("finetune_model_overrides must be a mapping (dict).")

    if init_mode == "scratch":
        model_scratch = merged.get("model_scratch")
        if not isinstance(model_scratch, dict):
            raise TypeError(
                "Finetune init=scratch requires 'model_scratch' to be defined as a mapping (dict)."
            )
        merged["model"] = deep_merge(copy.deepcopy(model_scratch), common_overrides)
        return

    if init_mode == "warmstart":
        warm_cfg = merged.get("warmstart")
        if warm_cfg is None:
            warm_cfg = {}
            merged["warmstart"] = warm_cfg
        if not isinstance(warm_cfg, dict):
            raise TypeError("warmstart must be a mapping (dict).")

        ws_model_overrides = warm_cfg.get("model_overrides")
        if ws_model_overrides is None:
            warm_cfg["model_overrides"] = {}
        elif not isinstance(ws_model_overrides, dict):
            raise TypeError("warmstart.model_overrides must be a mapping (dict).")
        return

    raise ValueError(f"Unsupported finetune init mode: {init_mode!r}")


def inherit_from_source_model(merged: Dict[str, Any], *, phase: str) -> None:
    """Load source run config/checkpoints and inherit config for warmstart/eval."""
    model_source = merged.get("model_source", {})
    if not isinstance(model_source, dict):
        raise TypeError("model_source must be set for finetune/eval phases")

    src_run_dir, src_run_id_for_yaml = resolve_model_source_dir(
        model_source, phase=phase
    )
    src_cfg = load_source_run_config_yaml(src_run_dir)

    ckpt_root = src_run_dir / "checkpoints"
    best_dir = ckpt_root / "best"
    latest_dir = ckpt_root / "latest"
    if not best_dir.is_dir() and not latest_dir.is_dir():
        raise FileNotFoundError(
            f"No checkpoints found in {src_run_dir}/checkpoints/\n"
            f"Expected: {best_dir} or {latest_dir}"
        )

    if "model" not in src_cfg:
        raise KeyError(f"Source config missing 'model' key: {src_run_dir}")

    if phase == "finetune":
        common_overrides = merged.get("finetune_model_overrides", {})
        if not isinstance(common_overrides, dict):
            raise TypeError("finetune_model_overrides must be a mapping (dict).")

        warm_cfg = merged.get("warmstart", {})
        if not isinstance(warm_cfg, dict):
            raise TypeError("warmstart must be a mapping (dict).")
        ws_model_overrides = warm_cfg.get("model_overrides", {})
        if ws_model_overrides is None:
            ws_model_overrides = {}
        if not isinstance(ws_model_overrides, dict):
            raise TypeError("warmstart.model_overrides must be a mapping (dict).")

        merged["model"] = copy.deepcopy(src_cfg["model"])
        if common_overrides:
            merged["model"] = deep_merge(merged["model"], common_overrides)
        if ws_model_overrides:
            merged["model"] = deep_merge(merged["model"], ws_model_overrides)
        # Finetune warmstart: keep preprocess settings from current merged config
        # (common + task overrides), do not force source chunk/trim inheritance.
    else:
        if "embeddings" not in src_cfg:
            raise KeyError(f"Source config missing 'embeddings' key: {src_run_dir}")
        merged["model"] = copy.deepcopy(src_cfg["model"])
        merged["embeddings"] = copy.deepcopy(src_cfg["embeddings"])
        merged["embeddings_profile"] = src_cfg.get(
            "embeddings_profile",
            merged.get("embeddings_profile"),
        )
        inherit_preprocess_chunk_trim(merged, src_cfg, allow_override=False)

    merged["model_source"]["run_dir"] = str(src_run_dir)
    if src_run_id_for_yaml:
        merged["model_source"]["run_id"] = src_run_id_for_yaml
