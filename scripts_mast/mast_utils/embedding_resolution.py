"""
Embedding resolution orchestration for pretrain, finetune, and eval phases.

This module handles the high-level embedding workflow logic:
- Pretrain: optional tuning + config snapshot
- Finetune: mode/tune/inherit logic with strict validation
- Eval: loading embeddings from training run

Core functions:
---------------
save_config_snapshot()              — Unified config snapshot saving
resolve_pretrain_embeddings()       — Pretrain embedding orchestration
resolve_finetune_embeddings()       — Finetune mode/tune/inherit with validation
resolve_eval_embeddings()           — Eval embedding loading from training run
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import yaml

from mast_utils.tune_dct3d import run_dct3d_tuning, load_embeddings_overrides

from mmt.data import build_signal_specs, build_codecs

logger = logging.getLogger("mmt.EmbeddingResolution")


# ------------------------------------------------------------------
# Config Snapshot Helper
# ------------------------------------------------------------------


def save_config_snapshot(
    cfg_mmt,
    run_dir: Path,
    logger_inst: logging.Logger | None = None,
) -> Path:
    """Save config snapshot to run_dir/{run_id}.yaml.

    This helper unifies the config snapshot saving pattern used in both
    pretrain and finetune phases after embedding resolution.

    Parameters
    ----------
    cfg_mmt:
        Merged experiment config (ExperimentConfig object).
    run_dir:
        Run directory where config snapshot will be saved.
    logger_inst:
        Optional logger for logging the save location.

    Returns
    -------
    Path
        Path to the saved config file.
    """
    config_snapshot_path = run_dir / f"{cfg_mmt.run_id}.yaml"
    with config_snapshot_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_mmt.raw, f, sort_keys=False)

    if logger_inst is not None:
        logger_inst.info("Saved config snapshot → %s", config_snapshot_path)

    return config_snapshot_path


# ------------------------------------------------------------------
# Embedding Resolution Helpers
# ------------------------------------------------------------------


def resolve_pretrain_embeddings(
    cfg_mmt,
    signals_by_role: dict,
    dict_task_metadata: dict,
    run_dir: Path,
    cfg_task,
) -> tuple:
    """Resolve embeddings for pretrain phase with optional tuning.

    This function handles the pretrain embedding workflow:
    1. Check if any roles need tuning
    2. If yes: build initial signal_specs, run tuning, merge overrides, save config
    3. Build final signal_specs with tuned config
    4. Build codecs from embeddings_dir
    5. Return (signal_specs, codecs)

    Parameters
    ----------
    cfg_mmt:
        Merged experiment config.
    signals_by_role:
        Dict mapping role -> list of signal names.
    dict_task_metadata:
        Task metadata from get_task_metadata().
    run_dir:
        Training run directory.
    cfg_task:
        Benchmark task definition (from load_task_definition()).

    Returns
    -------
    tuple
        (signal_specs, codecs) ready for model construction.
    """
    cfg_tune_emb = cfg_mmt.embeddings.get("tune_embeddings", {})
    roles_cfg = cfg_tune_emb.get("roles", {})
    roles_to_tune = [
        r for r in ("input", "actuator", "output") if roles_cfg.get(r, True)
    ]

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
        save_config_snapshot(cfg_mmt, run_dir, logger)

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
    codecs = build_codecs(signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


def _validate_inherited_embeddings_strict(
    per_signal_overrides: dict,
    signals_by_role: dict,
    roles_to_inherit: list[str],
    signal_specs,
) -> None:
    """Strict validation: check signal-level DCT3D parameters for inherited roles.

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
    per_signal_overrides:
        Loaded overrides from source run's dct3d.yaml.
    signals_by_role:
        Dict mapping role -> list of signal names for current task.
    roles_to_inherit:
        List of roles that should be inherited (not re-tuned).
    signal_specs:
        Signal spec registry (built with default config before inheritance).

    Raises
    ------
    ValueError
        If validation fails with detailed message about missing/invalid entries.
    """
    for role in roles_to_inherit:
        # Check role exists
        if role not in per_signal_overrides:
            raise ValueError(
                f"embeddings.mode=source: inherited role '{role}' has no entries in "
                f"source embeddings/dct3d.yaml. Source model may not have used DCT3D "
                f"rank-mode tuning for this role. Switch to embeddings.mode=config to "
                f"use current config defaults instead, or set tune_embeddings.roles.{role}=true "
                f"to re-tune from scratch."
            )

        role_overrides = per_signal_overrides[role]
        role_signals = signals_by_role.get(role, [])

        # Check each DCT3D signal in this role
        for sig_name in role_signals:
            spec = signal_specs.get(role, sig_name)
            if spec is None or spec.encoder_name != "dct3d":
                continue  # Skip non-DCT3D signals

            # Signal must exist in overrides
            if sig_name not in role_overrides:
                raise ValueError(
                    f"embeddings.mode=source: inherited role '{role}' is missing signal "
                    f"'{sig_name}' in source embeddings/dct3d.yaml. Expected all DCT3D signals "
                    f"for inherited roles to have rank-mode overrides. Available signals in "
                    f"source: {list(role_overrides.keys())}"
                )

            sig_override = role_overrides[sig_name]

            # Validate structure
            if not isinstance(sig_override, dict):
                raise ValueError(
                    f"embeddings.mode=source: invalid override for {role}:{sig_name}. "
                    f"Expected dict, got {type(sig_override).__name__}"
                )

            # Check encoder_name
            if sig_override.get("encoder_name") != "dct3d":
                raise ValueError(
                    f"embeddings.mode=source: {role}:{sig_name} has encoder_name="
                    f"'{sig_override.get('encoder_name')}', expected 'dct3d'"
                )

            # Check encoder_kwargs
            kwargs = sig_override.get("encoder_kwargs")
            if not isinstance(kwargs, dict):
                raise ValueError(
                    f"embeddings.mode=source: {role}:{sig_name} missing or invalid "
                    f"'encoder_kwargs' (expected dict)"
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
                    f"embeddings.mode=source: {role}:{sig_name} encoder_kwargs missing "
                    f"required fields: {missing}"
                )

            # Check selection_mode is 'rank'
            if kwargs["selection_mode"] != "rank":
                raise ValueError(
                    f"embeddings.mode=source: {role}:{sig_name} has selection_mode="
                    f"'{kwargs['selection_mode']}', expected 'rank' for inherited embeddings"
                )

    logger.info(
        "Strict validation passed: all DCT3D signals for inherited roles %s have valid "
        "rank-mode overrides",
        roles_to_inherit,
    )


def resolve_finetune_embeddings(
    cfg_mmt,
    signals_by_role: dict,
    dict_task_metadata: dict,
    run_dir: Path,
    cfg_task,
) -> tuple:
    """Resolve embeddings for finetune phase with mode/tune/inherit logic.

    This function handles the complex finetune embedding workflow:
    - mode=source: copy source embeddings, inherit/retune per role with strict validation
    - mode=config: use config defaults directly (no source artifacts)

    Parameters
    ----------
    cfg_mmt:
        Merged experiment config.
    signals_by_role:
        Dict mapping role -> list of signal names.
    dict_task_metadata:
        Task metadata from get_task_metadata().
    run_dir:
        Finetune run directory.
    cfg_task:
        Benchmark task definition (from load_task_definition()).

    Returns
    -------
    tuple
        (signal_specs, codecs) ready for model construction.

    Raises
    ------
    ValueError
        If mode is invalid, or if mode=config with roles_to_tune, or if
        strict validation fails for inherited embeddings.
    FileNotFoundError
        If mode=source, a source run is configured, source embeddings don't
        exist, and roles need inheriting.
    """
    cfg_emb = cfg_mmt.embeddings
    cfg_tune_emb = cfg_emb.get("tune_embeddings", {})
    roles_cfg = cfg_tune_emb.get("roles", {})
    emb_mode = cfg_emb.get("mode", "source")

    # Validate mode
    if emb_mode not in ("source", "config"):
        raise ValueError(
            f"embeddings.mode must be 'source' or 'config', got '{emb_mode}'"
        )

    roles_to_tune = [
        r for r in ("input", "actuator", "output") if roles_cfg.get(r, False)
    ]

    # Validate mode=config doesn't have roles_to_tune
    if emb_mode == "config" and roles_to_tune:
        raise ValueError(
            f"embeddings.mode=config does not support re-tuning roles "
            f"(got tune_embeddings.roles={roles_to_tune}). "
            "Set all roles to false or switch to mode=source."
        )

    if emb_mode == "source":
        roles_to_inherit = [
            r for r in ("input", "actuator", "output") if r not in roles_to_tune
        ]
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
                source_run_dir = Path(run_dir_src)

        if source_run_dir is not None:
            src_emb = source_run_dir / "embeddings"
            dst_emb = run_dir / "embeddings"

            if src_emb.exists():
                shutil.copytree(src_emb, dst_emb, dirs_exist_ok=True)
                logger.info("Copied embeddings from %s → %s", src_emb, dst_emb)
            elif roles_to_inherit:
                raise FileNotFoundError(
                    f"embeddings.mode=source requires source embeddings at {src_emb} "
                    f"for inherited roles {roles_to_inherit}. "
                    "Ensure the source model was trained with DCT3D rank-mode tuning, "
                    "or set all tune_embeddings.roles to true to re-tune from scratch, "
                    "or switch to embeddings.mode=config."
                )

            # Step 2: Load inherited overrides and perform strict validation
            per_signal_overrides = load_embeddings_overrides(run_dir)

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
                    per_signal_overrides,
                    signals_by_role,
                    roles_to_inherit,
                    signal_specs_for_validation,
                )

            # Step 3: Merge inherited overrides into config
            if per_signal_overrides:
                cfg_mmt.raw["embeddings"].setdefault("per_signal_overrides", {})
                for role, sigs in per_signal_overrides.items():
                    cfg_mmt.raw["embeddings"]["per_signal_overrides"].setdefault(role, {})
                    cfg_mmt.raw["embeddings"]["per_signal_overrides"][role].update(sigs)
        elif roles_to_inherit:
            logger.info(
                "Embeddings mode=source without source model: inherited roles %s "
                "will use current config defaults.",
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
        save_config_snapshot(cfg_mmt, run_dir, logger)

    else:  # emb_mode == "config"
        logger.info("")
        logger.info(
            "Embeddings mode=config: using emb_profile config directly (no source artifacts)"
        )

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
    codecs = build_codecs(signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


def resolve_eval_embeddings(
    cfg_mmt,
    signals_by_role: dict,
    dict_task_metadata: dict,
    train_run_dir: Path,
) -> tuple:
    """Resolve embeddings for eval phase from training run.

    This function loads the embeddings configuration from the training run
    and builds signal_specs + codecs for evaluation.

    Parameters
    ----------
    cfg_mmt:
        Merged experiment config.
    signals_by_role:
        Dict mapping role -> list of signal names.
    dict_task_metadata:
        Task metadata from get_task_metadata().
    train_run_dir:
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
    codecs = build_codecs(signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs
