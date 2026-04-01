"""
Embedding resolution orchestration for pretrain, finetune, and eval phases.

This module owns the phase-level embedding decisions:
- pretrain: optionally tune and snapshot DCT3D overrides
- finetune: choose source/config mode, validate inherited roles, trigger retunes
- eval: load the training run's resolved embedding artifacts

It converts the merged experiment config plus task signal definitions into the final embedding artifacts and
codec-ready signal specs used by each run phase.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import yaml

from mmt.data import build_signal_specs, build_codecs
from mmt.data.signal_spec import SignalSpecRegistry
from mmt.utils.config.schema import ExperimentConfig

from .tune_dct3d import (
    run_dct3d_tuning,
    load_embeddings_overrides,
)


# ----------------------------------------------------------------------------------------------------------------------

logger = logging.getLogger("mmt.EmbeddingResolution")


# ======================================================================================================================
# Config Snapshot Helper
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def save_config_snapshot(
    cfg_mmt: ExperimentConfig,
    run_dir: Path,
    logger_inst: logging.Logger | None = None,
) -> Path:
    """
    Save config snapshot to run_dir/{run_id}.yaml.

    This helper unifies the config snapshot saving pattern used in both  pretrain and finetune phases after embedding
    resolution.

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config (ExperimentConfig object).
    run_dir : Path
        Run directory where config snapshot will be saved.
    logger_inst : logging.Logger
        Logger for logging the save location.
        Optional. Default: None.

    Returns
    -------
    Path
        Path to the saved config file.

    """

    config_snapshot_path = run_dir / f"{cfg_mmt.run_id}.yaml"
    with config_snapshot_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_mmt.raw, f, sort_keys=False)

    if logger_inst is not None:
        logger_inst.info("Saved config snapshot → %s", config_snapshot_path)

    return config_snapshot_path


# ======================================================================================================================
# Other supporting methods
# ======================================================================================================================


# ----------------------------------------------------------------------------------------------------------------------
def stage_task_used_dct3d_artifacts_from_source(  # NOSONAR - Ignore cognitive complexity
    source_run_dir: Path, run_dir: Path, signals_by_role: Mapping[str, Any]
) -> bool:
    """
    Stage only task-used DCT3D artifacts from a source run.

    The destination finetune run receives only the source artifacts required for the current task. This includes
    task-used signals that may later be re-tuned, since their files and YAML entries are overwritten in-place by
    the tuning step.

    Parameters
    ----------
    source_run_dir : Path
        Source training run directory.
    run_dir : Path
        Destination finetune run directory.
    signals_by_role : Mapping[str, Any]
        Task-used signals keyed by role. Each role value may be either a mapping of signal name -> modality or an
        iterable of signal names.

    Returns
    -------
    bool
        True if the source `embeddings/` folder exists, else False.

    """

    src_emb = source_run_dir / "embeddings"
    if not src_emb.exists():
        return False

    src_overrides = load_embeddings_overrides(run_dir=source_run_dir)
    filtered_overrides: dict = {}
    dst_emb = run_dir / "embeddings"

    for role, role_signals in signals_by_role.items():
        role_overrides = src_overrides.get(role)
        if not isinstance(role_overrides, dict):
            continue

        if isinstance(role_signals, dict):
            signal_names = role_signals.keys()
        else:
            signal_names = role_signals

        for sig_name in signal_names:
            if sig_name not in role_overrides:
                continue

            sig_override = role_overrides[sig_name]
            filtered_overrides.setdefault(role, {})[sig_name] = sig_override

            if not isinstance(sig_override, dict):
                continue

            encoder_kwargs = sig_override.get("encoder_kwargs")
            if not isinstance(encoder_kwargs, dict):
                continue

            coeff_indices_path = encoder_kwargs.get("coeff_indices_path")
            if not isinstance(coeff_indices_path, str):
                continue

            src_indices = src_emb / coeff_indices_path
            if not src_indices.exists():
                continue

            dst_indices = dst_emb / coeff_indices_path
            dst_indices.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src=src_indices, dst=dst_indices)

    dst_emb.mkdir(parents=True, exist_ok=True)
    dct3d_yaml_path = dst_emb / "dct3d.yaml"
    with dct3d_yaml_path.open(mode="w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"embeddings": {"per_signal_overrides": filtered_overrides}},
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info(
        "Staged task-used DCT3D artifacts from %s -> %s",
        src_emb,
        dst_emb,
    )

    return True


# ----------------------------------------------------------------------------------------------------------------------
def resolve_pretrain_embeddings(
    cfg_mmt: ExperimentConfig,
    signals_by_role: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    run_dir: Path,
    cfg_task: Mapping[str, Any],
) -> tuple[SignalSpecRegistry, dict]:
    """
    Resolve embeddings for pretrain phase with optional tuning.

    This function handles the pretrain embedding workflow:
    1. Check if any roles need tuning
    2. If yes: build initial signal_specs, run tuning, merge overrides, save config
    3. Build final signal_specs with tuned config
    4. Build codecs from embeddings_dir
    5. Return (signal_specs, codecs)

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    signals_by_role : Mapping[str, Any]
        Dict mapping role -> list of signal names.
    dict_task_metadata : Mapping[str, Any]
        Task metadata from get_task_metadata().
    run_dir : Path
        Training run directory.
    cfg_task : Mapping[str, Any]
        Benchmark task definition (dictionary from load_task_definition()).

    Returns
    -------
    tuple[SignalSpecRegistry, dict]
        (signal_specs, codecs) ready for model construction.

    """

    cfg_tune_emb = cfg_mmt.embeddings.get("tune_embeddings", {})
    roles_cfg = cfg_tune_emb.get("roles", {})
    roles_to_tune = [r for r in ("input", "actuator", "output") if roles_cfg.get(r, True)]

    if roles_to_tune:
        # Step 1: build initial signal_specs with default (spatial) config for tuning
        signal_specs = build_signal_specs(
            embeddings_cfg=cfg_mmt.embeddings,
            signals_by_role=signals_by_role,
            dict_metadata=dict_task_metadata,
            chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
            log_summary=False,
        )

        # Step 2: tune DCT3D coefficients and save indices to run_dir/embeddings/
        logger.info("")
        per_signal_overrides = run_dct3d_tuning(
            cfg_mmt=cfg_mmt,
            signal_specs=signal_specs,
            cfg_task=cfg_task,
            dict_task_metadata=dict_task_metadata,
            run_dir=run_dir,
            roles=roles_to_tune,
        )

        # Step 3: update in-memory config with rank-mode overrides
        cfg_mmt.raw["embeddings"].setdefault("per_signal_overrides", {})
        for role, sigs in per_signal_overrides.items():
            cfg_mmt.raw["embeddings"]["per_signal_overrides"].setdefault(role, {})
            cfg_mmt.raw["embeddings"]["per_signal_overrides"][role].update(sigs)

        # Step 4: save config snapshot to capture tuned per_signal_overrides
        save_config_snapshot(cfg_mmt=cfg_mmt, run_dir=run_dir, logger_inst=logger)

    # Step 5: (re)build signal_specs with tuned config (rank-mode dims)
    logger.info("")
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
    )

    # Step 6: build codecs — indices live in run_dir/embeddings/
    embeddings_dir = run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


# ----------------------------------------------------------------------------------------------------------------------
def _validate_inherited_embeddings_strict(  # NOSONAR - Ignore cognitive complexity
    per_signal_overrides: Mapping[str, Any],
    signals_by_role: Mapping[str, Any],
    roles_to_inherit: list[str],
    signal_specs: SignalSpecRegistry,
) -> None:
    """
    Strict validation: check signal-level DCT3D parameters for inherited roles.

    For each inherited role:
    1. Role must exist in overrides
    2. For each signal in that role with encoder_name='dct3d':
       - Signal must exist in overrides[role]
       - Must have 'encoder_name': 'dct3d'
       - Must have 'encoder_kwargs' with required keys:
         * selection_mode='rank'
         * coeff_indices_path
         * coeff_shape
         * num_coeffs

    Parameters
    ----------
    per_signal_overrides : Mapping[str, Any]
        Loaded overrides from source run's dct3d.yaml.
    signals_by_role : Mapping[str, Any]
        Dict mapping role -> list of signal names for current task.
    roles_to_inherit : list[str]
        List of roles that should be inherited (not re-tuned).
    signal_specs : SignalSpecRegistry
        Signal spec registry (built with default config before inheritance).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If validation fails with detailed message about missing/invalid entries.
    TypeError
        If validation fails with detailed message about invalid types.

    """

    for role in roles_to_inherit:
        role_signals = signals_by_role.get(role, [])
        if not role_signals:
            # Nothing to inherit/validate for this role in the current task.
            continue

        # Check role exists
        if role not in per_signal_overrides:
            raise ValueError(
                f"embeddings.mode=source: inherited role '{role}' has no entries in source embeddings/dct3d.yaml. "
                f"Source model may not have used DCT3D rank-mode tuning for this role. "
                f"Switch to embeddings.mode=config to use current config defaults instead, or set "
                f"tune_embeddings.roles.{role}=true to re-tune from scratch."
            )

        role_overrides = per_signal_overrides[role]

        # Check each DCT3D signal in this role
        for sig_name in role_signals:
            spec = signal_specs.get(role, sig_name)
            if (spec is None) or (spec.encoder_name != "dct3d"):
                continue  # Skip non-DCT3D signals

            # Signal must exist in overrides
            if sig_name not in role_overrides:
                raise ValueError(
                    f"embeddings.mode=source: inherited role '{role}' is missing signal "
                    f"'{sig_name}' in source embeddings/dct3d.yaml. Expected all DCT3D signals "
                    f"for inherited roles to have rank-mode overrides. Available signals in "
                    f"source: {list(role_overrides.keys())}."
                )

            sig_override = role_overrides[sig_name]

            # Validate structure
            if not isinstance(sig_override, dict):
                raise TypeError(
                    f"embeddings.mode=source: invalid override for {role}:{sig_name}. "
                    f"Expected dict, got {type(sig_override).__name__}"
                )

            # Check encoder_name
            if sig_override.get("encoder_name") != "dct3d":
                raise TypeError(
                    f"embeddings.mode=source: {role}:{sig_name} has encoder_name='{sig_override.get('encoder_name')}', "
                    f"expected 'dct3d'."
                )

            # Check encoder_kwargs
            kwargs = sig_override.get("encoder_kwargs")
            if not isinstance(kwargs, dict):
                raise TypeError(
                    f"embeddings.mode=source: {role}:{sig_name} missing or invalid 'encoder_kwargs' (expected dict)."
                )

            # Check required kwargs fields
            required_fields = [
                "selection_mode",
                "coeff_indices_path",
                "coeff_shape",
                "num_coeffs",
            ]
            missing = [f for f in required_fields if f not in kwargs]
            if missing:
                raise ValueError(
                    f"embeddings.mode=source: {role}:{sig_name} encoder_kwargs missing required fields: {missing}."
                )

            # Check selection_mode is 'rank'
            if kwargs["selection_mode"] != "rank":
                raise ValueError(
                    f"embeddings.mode=source: {role}:{sig_name} has selection_mode='{kwargs['selection_mode']}', "
                    f"expected 'rank' for inherited embeddings."
                )

    logger.info(
        "Strict validation passed: all DCT3D signals for inherited roles %s have valid rank-mode overrides",
        roles_to_inherit,
    )


# ----------------------------------------------------------------------------------------------------------------------
def resolve_finetune_embeddings(  # NOSONAR - Ignore cognitive complexity
    cfg_mmt: ExperimentConfig,
    signals_by_role: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    run_dir: Path,
    cfg_task: Mapping[str, Any],
) -> tuple[SignalSpecRegistry, dict]:
    """
    Resolve embeddings for finetune phase with mode/tune/inherit logic.

    This function handles the complex finetune embedding workflow:
    - mode=source: stage task-used source DCT3D artifacts, inherit/retune per
      role with strict validation
    - mode=config: use config defaults directly (no source artifacts)

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    signals_by_role : Mapping[str, Any]
        Dict mapping role -> list of signal names.
    dict_task_metadata : Mapping[str, Any]
        Task metadata from get_task_metadata().
    run_dir : Path
        Finetune run directory.
    cfg_task : Mapping[str, Any]
        Benchmark task definition (dictionary from load_task_definition()).

    Returns
    -------
    tuple[SignalSpecRegistry, dict]
        (signal_specs, codecs) ready for model construction.

    Raises
    ------
    ValueError
        If mode is invalid, or if mode=config with roles_to_tune, or if  strict validation fails for inherited
        embeddings.
    FileNotFoundError
        If mode=source, a source run is configured, source embeddings do not exist, and roles need inheriting.

    """

    cfg_emb = cfg_mmt.embeddings
    cfg_tune_emb = cfg_emb.get("tune_embeddings", {})
    roles_cfg = cfg_tune_emb.get("roles", {})
    emb_mode = cfg_emb.get("mode", "source")

    # Validate mode
    if emb_mode not in ["source", "config"]:
        raise ValueError(f"embeddings.mode must be 'source' or 'config', got '{emb_mode}'")

    roles_to_tune = [r for r in ["input", "actuator", "output"] if roles_cfg.get(r, False)]

    # Validate mode=config does not have roles_to_tune
    if (emb_mode == "config") and roles_to_tune:
        raise ValueError(
            f"embeddings.mode=config does not support re-tuning roles (got tune_embeddings.roles={roles_to_tune}). "
            "Set all roles to false or switch to mode=source."
        )

    if emb_mode == "source":
        roles_to_inherit = [r for r in ["input", "actuator", "output"] if r not in roles_to_tune]
        logger.info("")
        logger.info(
            "Embeddings mode=source | retune=%s | inherit=%s",
            roles_to_tune or "none",
            roles_to_inherit or "none",
        )

        # Step 1: Optional source inheritance path
        model_source_cfg = cfg_mmt.raw.get("model_source")
        source_run_dir = None
        if isinstance(model_source_cfg, dict):
            run_dir_src = model_source_cfg.get("run_dir")
            if run_dir_src:
                source_run_dir = Path(str(run_dir_src))

        if source_run_dir is not None:
            src_emb = source_run_dir / "embeddings"
            source_embeddings_available = stage_task_used_dct3d_artifacts_from_source(
                source_run_dir=source_run_dir,
                run_dir=run_dir,
                signals_by_role=signals_by_role,
            )

            if (not source_embeddings_available) and roles_to_inherit:
                raise FileNotFoundError(
                    f"embeddings.mode=source requires source embeddings at {src_emb} for inherited roles "
                    f"{roles_to_inherit}. Ensure the source model was trained with DCT3D rank-mode tuning, or set all "
                    f"tune_embeddings.roles to true to re-tune from scratch, or switch to embeddings.mode=config."
                )

            # Step 2: Load inherited overrides and perform strict validation
            per_signal_overrides = load_embeddings_overrides(run_dir=run_dir)

            if roles_to_inherit:
                # Build initial signal_specs for validation (with default config)
                signal_specs_for_validation = build_signal_specs(
                    embeddings_cfg=cfg_mmt.embeddings,
                    signals_by_role=signals_by_role,
                    dict_metadata=dict_task_metadata,
                    chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
                    log_summary=False,
                )

                # Strict validation: check signal-level parameters
                _validate_inherited_embeddings_strict(
                    per_signal_overrides=per_signal_overrides,
                    signals_by_role=signals_by_role,
                    roles_to_inherit=roles_to_inherit,
                    signal_specs=signal_specs_for_validation,
                )

            # Step 3: Merge inherited overrides into config
            if per_signal_overrides:
                cfg_mmt.raw["embeddings"].setdefault("per_signal_overrides", {})
                for role, sigs in per_signal_overrides.items():
                    cfg_mmt.raw["embeddings"]["per_signal_overrides"].setdefault(role, {})
                    cfg_mmt.raw["embeddings"]["per_signal_overrides"][role].update(sigs)
        elif roles_to_inherit:
            logger.info(
                "Embeddings mode=source without source model: inherited roles %s will use current config defaults.",
                roles_to_inherit,
            )

        # Step 4: Re-tune selected roles (overwrites their files in run_dir/embeddings/)
        if roles_to_tune:
            signal_specs = build_signal_specs(
                embeddings_cfg=cfg_mmt.embeddings,
                signals_by_role=signals_by_role,
                dict_metadata=dict_task_metadata,
                chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
                log_summary=False,
            )
            logger.info("Re-tuning DCT3D embeddings for roles: %s", roles_to_tune)
            new_overrides = run_dct3d_tuning(
                cfg_mmt=cfg_mmt,
                signal_specs=signal_specs,
                cfg_task=cfg_task,
                dict_task_metadata=dict_task_metadata,
                run_dir=run_dir,
                roles=roles_to_tune,
            )
            cfg_mmt.raw["embeddings"].setdefault("per_signal_overrides", {})
            for role, sigs in new_overrides.items():
                cfg_mmt.raw["embeddings"]["per_signal_overrides"].setdefault(role, {})
                cfg_mmt.raw["embeddings"]["per_signal_overrides"][role].update(sigs)

        # Step 5: Save config snapshot to capture final per_signal_overrides
        save_config_snapshot(cfg_mmt=cfg_mmt, run_dir=run_dir, logger_inst=logger)

    else:  # -> I.e., emb_mode is "config"
        logger.info("")
        logger.info("Embeddings mode=config: using emb_profile config directly (no source artifacts)")

    # Step 6: (Re)build signal_specs with final config
    logger.info("")
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
    )

    # Step 7: Build codecs — indices live in run_dir/embeddings/
    embeddings_dir = run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


# ----------------------------------------------------------------------------------------------------------------------
def resolve_eval_embeddings(
    cfg_mmt: ExperimentConfig,
    signals_by_role: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    train_run_dir: Path,
) -> tuple:
    """Resolve embeddings for eval phase from training run.

    This function loads the embeddings configuration from the training run
    and builds signal_specs + codecs for evaluation.

    Parameters
    ----------
    cfg_mmt : ExperimentConfig
        Merged experiment config.
    signals_by_role : Mapping[str, Any]
        Dict mapping role -> list of signal names.
    dict_task_metadata : Mapping[str, Any]
        Task metadata from get_task_metadata().
    train_run_dir : Path
        Training run directory to load embeddings from.

    Returns
    -------
    tuple
        (signal_specs, codecs) ready for model construction.

    """

    # Load per-signal rank-mode overrides from training run
    per_signal_overrides = load_embeddings_overrides(train_run_dir)

    if not per_signal_overrides:
        logger.warning(
            "No rank-mode embeddings found in %s. "
            "Signal specs will use config defaults — verify this matches training.",
            train_run_dir / "embeddings",
        )

    # Merge overrides into config
    if per_signal_overrides:
        cfg_mmt.raw["embeddings"].setdefault("per_signal_overrides", {})
        for role, sigs in per_signal_overrides.items():
            cfg_mmt.raw["embeddings"]["per_signal_overrides"].setdefault(role, {})
            cfg_mmt.raw["embeddings"]["per_signal_overrides"][role].update(sigs)

    # Build signal_specs with loaded config
    logger.info("")
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=cfg_mmt.preprocess["chunk"]["chunk_length"],
    )

    # Build codecs — indices live in training run's embeddings folder
    embeddings_dir = train_run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs
