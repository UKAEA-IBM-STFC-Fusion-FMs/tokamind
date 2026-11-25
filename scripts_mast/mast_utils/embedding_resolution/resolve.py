"""
Pretrain / finetune / eval embedding-resolution orchestration.

The phase-level entrypoints that convert a merged experiment config plus task signal definitions into final
embedding artifacts and codec-ready signal specs. Each phase composes the lower layers — select signals, plan the
tune/source/config policy, stage and validate inherited artifacts, tune the remainder, then build specs and codecs:
- pretrain: tune (and snapshot) rank artifacts for every non-config signal
- finetune: apply the source/tune policy, validate inherited signals, retune the rest
- eval: reuse the training run's resolved artifacts
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mmt.data import build_signal_specs, build_codecs
from mmt.data.signal_spec import SignalSpecRegistry
from mmt.utils.config.experiment.inheritance import load_source_run_config_yaml
from mmt.utils.config.schema import ExperimentConfig

from ..tune_dct3d import run_dct3d_tuning, load_embeddings_overrides

from .signals import (
    _dct3d_signal_names_by_role,
    _filter_embedding_overrides_by_signal_names,
    _signal_names_from_overrides,
    _subtract_signal_names_by_role,
    _signal_policy_roles,
)
from .policy import (
    _uses_dct3d_tuning_policy,
    _resolve_dct3d_signal_policy,
)
from .artifacts import (
    _merge_embedding_overrides,
    _profile_embedding_overrides,
    stage_task_used_dct3d_artifacts_from_source,
    _validate_inherited_embeddings_strict,
)


logger = logging.getLogger("mmt.EmbeddingResolution")


def _chunk_lengths(cfg_mmt: ExperimentConfig) -> tuple[float, dict[str, float]]:
    """Return canonical input and actuator chunk durations from ``preprocess.chunks``."""

    chunks = cfg_mmt.preprocess["chunks"]
    input_length = float(chunks["input"]["chunk_length"])
    return input_length, {"input": input_length, "actuator": float(chunks["actuator"]["chunk_length"])}


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
    logger_inst : logging.Logger | None
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
        logger_inst.info("Saved config snapshot -> %s", config_snapshot_path)

    return config_snapshot_path


def resolve_pretrain_embeddings(
    cfg_mmt: ExperimentConfig,
    signals_by_role: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    run_dir: Path,
    cfg_task: Mapping[str, Any],
) -> tuple[SignalSpecRegistry, dict]:
    """
    Resolve embeddings for pretrain phase with optional DCT3D tuning.

    This function handles the pretrain embedding workflow:
    1. Check if any DCT3D signals need tuning
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

    signals_to_tune: dict[str, set[str]] = {}
    roles_to_tune: list[str] = []

    if _uses_dct3d_tuning_policy(cfg_mmt=cfg_mmt):
        # Step 1: build initial signal_specs with default (spatial) config for tuning
        input_length, chunk_lengths = _chunk_lengths(cfg_mmt)
        signal_specs_for_policy = build_signal_specs(
            embeddings_cfg=cfg_mmt.embeddings,
            signals_by_role=signals_by_role,
            dict_metadata=dict_task_metadata,
            chunk_length_sec=input_length,
            chunk_length_sec_by_role=chunk_lengths,
            log_summary=False,
        )

        dct3d_signals = _dct3d_signal_names_by_role(
            signal_specs=signal_specs_for_policy,
            signals_by_role=signals_by_role,
        )
        manual_config_signals = _signal_names_from_overrides(
            overrides=_profile_embedding_overrides(cfg_mmt.raw["embeddings"]),
            signal_names_by_role=dct3d_signals,
        )
        signals_to_tune = _subtract_signal_names_by_role(
            signal_names_by_role=dct3d_signals,
            names_to_remove=manual_config_signals,
        )
        roles_to_tune = _signal_policy_roles(signals_to_tune)
        roles_from_config = _signal_policy_roles(manual_config_signals)

        logger.info("")
        logger.info(
            "Embeddings policy | tune=%s | config=%s",
            {role: sorted(signals_to_tune[role]) for role in roles_to_tune} or "none",
            {role: sorted(manual_config_signals[role]) for role in roles_from_config} or "none",
        )

        if roles_to_tune:
            # Step 2: tune DCT3D coefficients and save indices to run_dir/embeddings/
            logger.info("")
            per_signal_overrides = run_dct3d_tuning(
                cfg_mmt=cfg_mmt,
                signal_specs=signal_specs_for_policy,
                cfg_task=cfg_task,
                dict_task_metadata=dict_task_metadata,
                run_dir=run_dir,
                roles=roles_to_tune,
                signal_names_by_role=signals_to_tune,
            )

            # Step 3: update in-memory config with rank-mode overrides
            _merge_embedding_overrides(cfg_mmt.raw["embeddings"], per_signal_overrides)

            # Step 4: save config snapshot to capture tuned per_signal_overrides
            save_config_snapshot(cfg_mmt=cfg_mmt, run_dir=run_dir, logger_inst=logger)

    # Step 5: (re)build signal_specs with tuned config (rank-mode dims)
    logger.info("")
    input_length, chunk_lengths = _chunk_lengths(cfg_mmt)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=input_length,
        chunk_length_sec_by_role=chunk_lengths,
    )

    # Step 6: build codecs — indices live in run_dir/embeddings/
    embeddings_dir = run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


def resolve_finetune_embeddings(  # NOSONAR - Ignore cognitive complexity
    cfg_mmt: ExperimentConfig,
    signals_by_role: Mapping[str, Any],
    dict_task_metadata: Mapping[str, Any],
    run_dir: Path,
    cfg_task: Mapping[str, Any],
) -> tuple[SignalSpecRegistry, dict]:
    """
    Resolve embeddings for finetune phase with fixed DCT3D tune/source policy.

    This function handles the complex finetune embedding workflow:
    - warmstart DCT3D: source existing input/actuator signals and tune missing input/actuator plus outputs
    - scratch DCT3D: tune all DCT3D signals
    - explicit per-signal profile overrides: use config values without DCT3D tuning/source inheritance
    - non-DCT3D profiles: use config/profile values without DCT3D tuning

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
        If DCT3D policy is invalid, or if strict validation fails for inherited embeddings.
    FileNotFoundError
        If source mode is selected and source embeddings are unavailable.

    """

    original_profile_overrides = _profile_embedding_overrides(cfg_mmt.raw["embeddings"])

    # Build initial signal specs from profile/default config. These specs decide which task signals are DCT3D and are
    # also used for source validation before any inherited/tuned rank-mode overrides are merged.
    input_length, chunk_lengths = _chunk_lengths(cfg_mmt)
    signal_specs_for_policy = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=input_length,
        chunk_length_sec_by_role=chunk_lengths,
        log_summary=False,
    )

    model_source_cfg = cfg_mmt.raw.get("model_source")
    source_run_dir = None
    if isinstance(model_source_cfg, dict):
        run_dir_src = model_source_cfg.get("run_dir")
        if run_dir_src:
            source_run_dir = Path(str(run_dir_src))

    dct3d_signals = _dct3d_signal_names_by_role(
        signal_specs=signal_specs_for_policy,
        signals_by_role=signals_by_role,
    )
    manual_config_signals = _signal_names_from_overrides(
        overrides=original_profile_overrides,
        signal_names_by_role=dct3d_signals,
    )
    dct3d_policy_signals = _subtract_signal_names_by_role(
        signal_names_by_role=dct3d_signals,
        names_to_remove=manual_config_signals,
    )

    source_overrides = {}
    if _uses_dct3d_tuning_policy(cfg_mmt=cfg_mmt):
        init_mode = str((cfg_mmt.raw.get("cli") or {}).get("init", "")).lower()
        if (init_mode == "warmstart") and (source_run_dir is not None):
            source_overrides = load_embeddings_overrides(run_dir=source_run_dir)

        signal_policy = _resolve_dct3d_signal_policy(
            cfg_mmt=cfg_mmt,
            dct3d_signals=dct3d_policy_signals,
            source_overrides=source_overrides,
        )
    else:
        signal_policy = {"config": dct3d_policy_signals}

    signals_to_tune = signal_policy.get("tune", {})
    signals_to_inherit = signal_policy.get("source", {})
    signals_from_config = {role: set(names) for role, names in signal_policy.get("config", {}).items()}
    for role, signal_names in manual_config_signals.items():
        signals_from_config.setdefault(role, set()).update(signal_names)

    roles_to_tune = _signal_policy_roles(signals_to_tune)
    roles_to_inherit = _signal_policy_roles(signals_to_inherit)
    roles_from_config = _signal_policy_roles(signals_from_config)

    logger.info("")
    logger.info(
        "Embeddings policy | tune=%s | source=%s | config=%s",
        {role: sorted(signals_to_tune[role]) for role in roles_to_tune} or "none",
        {role: sorted(signals_to_inherit[role]) for role in roles_to_inherit} or "none",
        {role: sorted(signals_from_config[role]) for role in roles_from_config} or "none",
    )

    # Step 1: Optional source inheritance path for signals explicitly marked as source.
    if roles_to_inherit:
        if source_run_dir is None:
            raise FileNotFoundError(
                f"DCT3D warmstart source policy selected source signals {signals_to_inherit}, "
                "but no model_source.run_dir is set."
            )

        src_emb = source_run_dir / "embeddings"
        source_embeddings_available = stage_task_used_dct3d_artifacts_from_source(
            source_run_dir=source_run_dir,
            run_dir=run_dir,
            signals_by_role=signals_to_inherit,
        )

        if not source_embeddings_available:
            raise FileNotFoundError(
                f"DCT3D warmstart source policy requires source embeddings at {src_emb} for source signals "
                f"{signals_to_inherit}. Ensure the source model was trained with DCT3D rank-mode tuning, "
                "or use manual per-signal overrides for these signals."
            )

        # Step 2: Load inherited overrides and perform strict validation.
        per_signal_overrides = load_embeddings_overrides(run_dir=run_dir)

        _validate_inherited_embeddings_strict(
            per_signal_overrides=per_signal_overrides,
            signal_names_by_role=signals_to_inherit,
            signal_specs=signal_specs_for_policy,
        )

        inherited_overrides = _filter_embedding_overrides_by_signal_names(per_signal_overrides, signals_to_inherit)
        if inherited_overrides:
            _merge_embedding_overrides(cfg_mmt.raw["embeddings"], inherited_overrides)

    # Step 3: Re-tune selected signals (overwrites their files in run_dir/embeddings/).
    if roles_to_tune:
        logger.info(
            "Tuning DCT3D embeddings for signals: %s",
            {role: sorted(signals_to_tune[role]) for role in roles_to_tune},
        )
        new_overrides = run_dct3d_tuning(
            cfg_mmt=cfg_mmt,
            signal_specs=signal_specs_for_policy,
            cfg_task=cfg_task,
            dict_task_metadata=dict_task_metadata,
            run_dir=run_dir,
            roles=roles_to_tune,
            signal_names_by_role=signals_to_tune,
        )
        _merge_embedding_overrides(cfg_mmt.raw["embeddings"], new_overrides)

    # Step 4: Re-apply explicit profile overrides last so user/config choices win over computed artifacts.
    _merge_embedding_overrides(cfg_mmt.raw["embeddings"], original_profile_overrides)

    # Step 5: Save config snapshot to capture final per_signal_overrides.
    save_config_snapshot(cfg_mmt=cfg_mmt, run_dir=run_dir, logger_inst=logger)

    # Step 6: (Re)build signal_specs with final config
    logger.info("")
    input_length, chunk_lengths = _chunk_lengths(cfg_mmt)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=input_length,
        chunk_length_sec_by_role=chunk_lengths,
    )

    # Step 7: Build codecs — indices live in run_dir/embeddings/
    embeddings_dir = run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs


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

    original_profile_overrides = _profile_embedding_overrides(cfg_mmt.raw["embeddings"])

    # Evaluation does not merge a task-specific embedding profile. Reuse the
    # source snapshot first so VAE overrides remain aligned with the trained
    # model, then let saved DCT rank artifacts refine that mapping.
    source_cfg = load_source_run_config_yaml(model_run_dir=train_run_dir)
    source_embeddings = source_cfg.get("embeddings") or {}
    if not isinstance(source_embeddings, Mapping):
        raise TypeError(f"Source run {train_run_dir} has a non-mapping embeddings block.")
    source_overrides = source_embeddings.get("per_signal_overrides") or {}
    if not isinstance(source_overrides, Mapping):
        raise TypeError(f"Source run {train_run_dir} has non-mapping embeddings.per_signal_overrides.")
    _merge_embedding_overrides(cfg_mmt.raw["embeddings"], source_overrides)

    # Load per-signal rank-mode overrides from the training run.
    per_signal_overrides = load_embeddings_overrides(train_run_dir)

    if not per_signal_overrides:
        logger.warning(
            "No rank-mode embeddings found in %s. "
            "Signal specs will use config defaults — verify this matches training.",
            train_run_dir / "embeddings",
        )

    # Merge computed artifacts first, then re-apply explicit profile overrides so config choices win.
    if per_signal_overrides:
        _merge_embedding_overrides(cfg_mmt.raw["embeddings"], per_signal_overrides)
    _merge_embedding_overrides(cfg_mmt.raw["embeddings"], original_profile_overrides)

    # Build signal_specs with loaded config
    logger.info("")
    input_length, chunk_lengths = _chunk_lengths(cfg_mmt)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_task_metadata,
        chunk_length_sec=input_length,
        chunk_length_sec_by_role=chunk_lengths,
    )

    # Build codecs — indices live in training run's embeddings folder
    embeddings_dir = train_run_dir / "embeddings"
    codecs = build_codecs(signal_specs=signal_specs, config_dir=embeddings_dir)

    return signal_specs, codecs
