"""
DCT3D embedding tuning utility for MAST.

run_dct3d_tuning()         — build a MAST dataset subsample, stream windows through
                             TuneRankedDCT3DTransform, and save results to
                             runs/<run_id>/embeddings/.

load_embeddings_overrides() — read per-signal rank-mode overrides from a run folder
                              (used by finetune and eval).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import yaml

from mast_utils.benchmark_imports import (
    initialize_MAST_dataset,
    initialize_model_dataset_iterable,
    get_train_test_val_shots,
)

from mmt.data import (
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    TuneRankedDCT3DTransform,
    ComposeTransforms,
)

logger = logging.getLogger("mmt.TuneRankedDCT3D")


def run_dct3d_tuning(
    cfg_mmt,
    signal_specs,
    cfg_task,
    dict_task_metadata: Mapping,
    run_dir: Path,
    roles: list[str] | None = None,
) -> dict:
    """Tune DCT3D rank-mode embeddings and save results to the run folder.

    Builds a small MAST dataset subsample, streams windows through
    TuneRankedDCT3DTransform to accumulate per-coefficient energies E[c_i²],
    then selects top-K coefficients meeting the explained-variance threshold
    for each signal.

    Outputs written to ``run_dir/embeddings/``:
      - ``dct3d_indices/<role>_<signal_name>.npy``  — selected coefficient indices
      - ``dct3d.yaml``                               — per-signal rank-mode overrides

    Parameters
    ----------
    cfg_mmt:
        Merged experiment config. Reads ``embeddings.tune_embeddings`` for
        tuning params and ``preprocess`` for chunk/window settings.
    signal_specs:
        Signal spec registry (built from default spatial embeddings config).
    cfg_task:
        Benchmark task definition (from load_task_definition()).
    dict_task_metadata:
        Task metadata dict (from get_task_metadata()).
    run_dir:
        Training run directory. Results saved to ``run_dir/embeddings/``.
    roles:
        Roles to tune. Defaults to ``["input", "actuator", "output"]``.

    Returns
    -------
    dict
        Per-signal overrides to merge into
        ``cfg_mmt.raw["embeddings"]["per_signal_overrides"]``.
        Structure: ``{role: {signal_name: {encoder_name, encoder_kwargs}}}``.
    """
    if roles is None:
        roles = ["input", "actuator", "output"]

    cfg_data = cfg_mmt.data
    cfg_prep = cfg_mmt.preprocess
    cfg_tune = cfg_mmt.embeddings.get("tune_embeddings", {})
    cfg_objective = cfg_tune.get("objective", {})

    n_shots = cfg_tune.get("n_shots", 100)
    max_windows = cfg_tune.get("max_windows", 15000)
    local_flag = cfg_data.get("local", True)
    max_budget_cfg = cfg_objective.get("max_budget", {})
    budget_summary = (
        {r: max_budget_cfg.get(r) for r in roles}
        if isinstance(max_budget_cfg, Mapping)
        else max_budget_cfg
    )
    guardrails_cfg = cfg_tune.get("guardrails") or {}
    guardrails_state = "enabled" if guardrails_cfg.get("enable") else "disabled"

    logger.info(
        "n_shots=%d | max_windows=%d | roles=%s | budgets=%s | guardrails=%s",
        n_shots,
        max_windows,
        ",".join(roles),
        budget_summary,
        guardrails_state,
    )

    # ------------------------------------------------------------------
    # Dataset: subsample training shots for tuning
    # ------------------------------------------------------------------
    train_shots, _, _ = get_train_test_val_shots(
        max_index=n_shots,
        shuffle=True,
        seed=cfg_mmt.seed,
    )

    mast_dataset = initialize_MAST_dataset(
        cfg_task,
        train_shots,
        local_flag=local_flag,
        use_std_scaling=True,
        return_incomplete_shots=True,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Transform pipeline
    # ------------------------------------------------------------------
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    tune_transform = TuneRankedDCT3DTransform(
        signal_specs=signal_specs,
        thresholds=cfg_objective.get("thresholds", {}),
        max_budget=max_budget_cfg,
        roles=roles,
        guardrails=guardrails_cfg,
    )

    transform_pipeline = ComposeTransforms(
        [
            ChunkWindowsTransform(
                dict_metadata=dict_task_metadata,
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks.get("stride"),
            ),
            SelectValidWindowsTransform(
                min_valid_inputs_actuators=cfg_valid_win["min_valid_inputs_actuators"],
                min_valid_chunks=cfg_valid_win["min_valid_chunks"],
                min_valid_outputs=cfg_valid_win["min_valid_outputs"],
                window_stride_sec=cfg_valid_win["window_stride_sec"],
            ),
            TrimChunksTransform(max_chunks=cfg_trim["max_chunks"]),
            tune_transform,
        ]
    )

    ds_windows = initialize_model_dataset_iterable(
        mast_dataset,
        dict_task_metadata,
        cfg_task,
        model_specific_transform=transform_pipeline,
        test_mode=False,
        shuffle_windows=False,
        shuffle_buffer_size=512,
        verbose=False,
    )

    assert ds_windows is not None, "Window iterable should not be None"

    # ------------------------------------------------------------------
    # Stream windows to accumulate energies
    # ------------------------------------------------------------------
    n_windows_total = 0
    for _ in ds_windows:
        n_windows_total += 1
        if max_windows is not None and n_windows_total >= max_windows:
            break

    logger.info("Tuning: processed %d windows", n_windows_total)

    # ------------------------------------------------------------------
    # Select best and log
    # ------------------------------------------------------------------
    best = tune_transform.select_best()

    n_signals = 0
    n_guardrail_up = 0
    n_budget_capped = 0
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            n_signals += 1
            if bool(info.get("guardrail_increased_k", False)):
                n_guardrail_up += 1
            if bool(info.get("budget_capped", False)):
                n_budget_capped += 1

            logger.debug(
                "[%s:%s] shape=%s K_target=%s K_final=%d expl_energy=%.4f flags={guardrail_up:%s,budget_cap:%s}",
                role,
                name,
                info["coeff_shape"],
                info.get("k_target", "-"),
                info["num_coeffs"],
                info["explained_energy"],
                bool(info.get("guardrail_increased_k", False)),
                bool(info.get("budget_capped", False)),
            )
    logger.info(
        "DCT3D tuning summary: signals=%d guardrail_up=%d budget_capped=%d",
        n_signals,
        n_guardrail_up,
        n_budget_capped,
    )

    # ------------------------------------------------------------------
    # Save indices and config
    # ------------------------------------------------------------------
    emb_dir = run_dir / "embeddings"
    indices_dir = emb_dir / "dct3d_indices"
    indices_dir.mkdir(parents=True, exist_ok=True)

    per_signal_overrides: dict = {}

    for role, by_sig in best.items():
        for name, info in by_sig.items():
            spec = signal_specs.get(role, name)
            if spec is None or spec.encoder_name != "dct3d":
                continue

            coeff_indices = info["coeff_indices"]
            filename = f"{role}_{name}.npy"
            np.save(indices_dir / filename, coeff_indices)

            h, w, t = info["coeff_shape"]
            indices_3d = np.unravel_index(coeff_indices, (h, w, t))
            flags: list[str] = []
            if bool(info.get("guardrail_increased_k", False)):
                flags.append("guardrail_up")
            if bool(info.get("budget_capped", False)):
                if bool(info.get("budget_violated_for_guardrails", False)):
                    flags.append("budget_cap_guardrail")
                else:
                    flags.append("budget_cap")

            per_signal_overrides.setdefault(role, {})[name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": {
                    "selection_mode": "rank",
                    "coeff_indices_path": f"dct3d_indices/{filename}",
                    "coeff_shape": list(info["coeff_shape"]),
                    "num_coeffs": int(info["num_coeffs"]),
                    "explained_energy": float(info["explained_energy"]),
                    "dim_distribution": {
                        "unique_h": int(len(np.unique(indices_3d[0]))),
                        "unique_w": int(len(np.unique(indices_3d[1]))),
                        "unique_t": int(len(np.unique(indices_3d[2]))),
                    },
                    "tuning_info": {
                        "target": float(info["target"]),
                        "k_target": int(info.get("k_target", info["num_coeffs"])),
                        "guardrail_min_k": int(info.get("guardrail_min_k", 0)),
                        "k_after_guardrails": int(
                            info.get("k_after_guardrails", info["num_coeffs"])
                        ),
                        "k_final": int(info["num_coeffs"]),
                        "n_windows": int(info["n_windows"]),
                        "max_budget": None
                        if info.get("max_budget") is None
                        else int(info["max_budget"]),
                        "flags": flags,
                        "tuned_in_run_id": str(run_dir.name),
                    },
                },
            }

    # Merge with any existing overrides (e.g. inherited roles from source) so that
    # a partial retune does not drop roles that were not re-tuned this run.
    dct3d_yaml_path = emb_dir / "dct3d.yaml"
    if dct3d_yaml_path.exists():
        with dct3d_yaml_path.open(encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing_overrides = existing.get("embeddings", {}).get(
            "per_signal_overrides", {}
        )
        for role, sigs in per_signal_overrides.items():
            existing_overrides.setdefault(role, {}).update(sigs)
        merged_overrides = existing_overrides
    else:
        merged_overrides = per_signal_overrides

    with dct3d_yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"embeddings": {"per_signal_overrides": merged_overrides}},
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    logger.info("Saved tuned overrides → %s", dct3d_yaml_path)
    logger.info("Saved coefficient indices → %s", indices_dir)

    return per_signal_overrides


def load_embeddings_overrides(run_dir: Path) -> dict:
    """Load per-signal rank-mode overrides from a run's embeddings folder.

    Parameters
    ----------
    run_dir:
        Training run directory. Reads ``run_dir/embeddings/dct3d.yaml``.

    Returns
    -------
    dict
        ``{role: {signal_name: {encoder_name, encoder_kwargs}}}``.
        Returns ``{}`` if the file does not exist (e.g. run used spatial
        encoding and no tuning was performed).
    """
    dct3d_yaml = run_dir / "embeddings" / "dct3d.yaml"
    if not dct3d_yaml.exists():
        return {}

    with dct3d_yaml.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data.get("embeddings", {}).get("per_signal_overrides", {})
