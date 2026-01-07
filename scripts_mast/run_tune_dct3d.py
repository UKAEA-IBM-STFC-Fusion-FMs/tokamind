from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import yaml

from scripts.pipelines.utils.preprocessing_utils import (
    initialize_datasets_and_metadata_for_task,
)
from scripts.pipelines.utils.utils import initialize_model_dataset

from scripts_mast.mast_utils import (
    build_task_config,
    build_signals_by_role_from_task_config,
)

from mmt.utils.config import load_experiment_config
from mmt.utils import set_seed, setup_logging
from mmt.data import (
    build_signal_specs,
    ChunkWindowsTransform,
    SelectValidWindowsTransform,
    TrimChunksTransform,
    TuneDCT3DTransform,
    ComposeTransforms,
    WindowStreamedDataset,
)

DEBUG_MODE = False
CONFIGS_ROOT = "scripts_mast/configs"


def parse_args_tune_dct3d() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune DCT3D embedding parameters for a given task."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="task_2-1",
        help="Task folder name under scripts_mast/configs/tasks/<task>/",
    )
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    # ------------------------------------------------------------------
    # Device / multiprocessing
    # ------------------------------------------------------------------
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Needed for DataLoader(num_workers>0) on some platforms
    mp.set_start_method("spawn", force=True)

    # ------------------------------------------------------------------
    # Load merged config (NEW: convention-based)
    # ------------------------------------------------------------------
    args = parse_args_tune_dct3d()
    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="tune_dct3d",
        configs_root=CONFIGS_ROOT,
    )

    if "tune_dct3d" not in cfg_mmt.raw:
        raise KeyError("Missing 'tune_dct3d' section in config.")

    # Small sub-configs for readability
    cfg_prep = cfg_mmt.preprocess
    cfg_chunks = cfg_prep["chunk"]
    cfg_trim = cfg_prep["trim_chunks"]
    cfg_valid_win = cfg_prep["valid_windows"]

    cfg_tune = cfg_mmt.raw["tune_dct3d"]

    # Baseline task config (with overrides such as subset_of_shots/local)
    cfg_task = build_task_config(cfg_mmt)

    # ------------------------------------------------------------------
    # Seed + logging
    # ------------------------------------------------------------------
    set_seed(cfg_mmt.seed, deterministic=True, warn_only=True)

    # For tune_dct3d, loader sets run_dir to: scripts_mast/configs/tasks/<task>/
    logger = setup_logging(
        cfg_mmt.paths["run_dir"],
        logger_name="mmt",
        filename=None,
        console=True,
    )
    logger.setLevel("DEBUG" if DEBUG_MODE else "INFO")

    logger = logging.getLogger("mmt.TuneDCT3D")
    logger.info(f"task = {cfg_mmt.task} | phase = {cfg_mmt.phase} | device = {device}")

    # ------------------------------------------------------------------
    # Baseline datasets + metadata
    # ------------------------------------------------------------------
    datasets_shots_raw, dict_metadata = initialize_datasets_and_metadata_for_task(
        cfg_task
    )

    # Build signals_by_role from baseline config + metadata and signal specs
    signals_by_role = build_signals_by_role_from_task_config(cfg_task, dict_metadata)
    signal_specs = build_signal_specs(
        embeddings_cfg=cfg_mmt.embeddings,
        signals_by_role=signals_by_role,
        dict_metadata=dict_metadata,
        chunk_length_sec=cfg_chunks["chunk_length"],
    )

    # ------------------------------------------------------------------
    # Model-specific transform chain (shot -> windows)
    # ------------------------------------------------------------------
    tune_transform = TuneDCT3DTransform(
        signal_specs=signal_specs,
        keep_h=cfg_tune["search_space"]["keep_h"],
        keep_w=cfg_tune["search_space"]["keep_w"],
        keep_t=cfg_tune["search_space"]["keep_t"],
        thresholds=cfg_tune["objective"]["thresholds"],
    )

    mmt_transform_map = ComposeTransforms(
        [
            ChunkWindowsTransform(
                dict_metadata=dict_metadata,
                chunk_length_sec=cfg_chunks["chunk_length"],
                stride_sec=cfg_chunks["stride"],
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

    # ------------------------------------------------------------------
    # Shot-level dataset (wrapped) for tuning
    # ------------------------------------------------------------------
    ds_shots_full = initialize_model_dataset(
        datasets_shots_raw.get("train"),
        dict_metadata,
        cfg_task,
        model_specific_transform=mmt_transform_map,
        verbose=False,
    )

    # ------------------------------------------------------------------
    # Random shot selection (optional)
    # ------------------------------------------------------------------
    cfg_sampling = cfg_tune["sampling"]
    n_shots = cfg_sampling.get("n_shots")

    all_indices = np.arange(len(ds_shots_full))
    if n_shots is not None:
        rng = np.random.default_rng(cfg_mmt.seed)
        sel_indices = rng.choice(
            all_indices, size=min(n_shots, len(all_indices)), replace=False
        )
        ds_shots_selected = [ds_shots_full[i] for i in sel_indices]
    else:
        sel_indices = all_indices
        ds_shots_selected = ds_shots_full

    logger.info(
        "ds_shots_full=%d, requested n_shots=%s, selected=%d, sel_indices=%s",
        len(ds_shots_full),
        str(n_shots),
        len(ds_shots_selected),
        np.array(sel_indices).tolist() if n_shots is not None else None,
    )

    # ------------------------------------------------------------------
    # Window-level dataset (streaming)
    # ------------------------------------------------------------------
    ds_windows = WindowStreamedDataset(
        ds_shots_selected,
        shuffle_shots=False,
        seed=cfg_mmt.seed,
    )

    # ------------------------------------------------------------------
    # Shot exploration
    # ------------------------------------------------------------------
    max_windows = cfg_sampling.get("max_windows", None)

    n_windows_total = 0
    for _window in ds_windows:
        n_windows_total += 1
        if max_windows is not None and n_windows_total >= max_windows:
            break

    logger.info("Processed windows: total=%d", n_windows_total)

    # ------------------------------------------------------------------
    # Select and print best configs
    # ------------------------------------------------------------------
    best = tune_transform.select_best()

    logger.info("Selected configurations:")
    for role, by_sig in best.items():
        for name, info in by_sig.items():
            logger.info(
                f"[{role}:{name}] "
                f"shape={info['rep_shape']} "
                f"keep=({info['eff_keep_h']},{info['eff_keep_w']},{info['eff_keep_t']}) "
                f"rmse={info['rmse_mean_windows']:.4e} "
                f"eff_dim={info['effective_dim']}"
            )

    # ------------------------------------------------------------------
    # Write tuned overrides to: tasks/<task>/embeddings_overrides.yaml
    # (ONLY per-signal overrides; no defaults)
    # ------------------------------------------------------------------
    overrides_out = {"embeddings": {"per_signal_overrides": {}}}

    for role, by_sig in best.items():
        for name, info in by_sig.items():
            spec = signal_specs.get(role, name)
            if spec is None:
                continue
            if spec.encoder_name != "dct3d":
                continue

            tuned_keep_h = int(info["eff_keep_h"])
            tuned_keep_w = int(info["eff_keep_w"])
            tuned_keep_t = int(info["eff_keep_t"])

            # Compare against current baseline kwargs used for this run
            base_kwargs = dict(spec.encoder_kwargs or {})
            base_keep_h = int(base_kwargs.get("keep_h", tuned_keep_h))
            base_keep_w = int(base_kwargs.get("keep_w", tuned_keep_w))
            base_keep_t = int(base_kwargs.get("keep_t", tuned_keep_t))

            # Only write an override if something actually changes
            if (tuned_keep_h, tuned_keep_w, tuned_keep_t) == (
                base_keep_h,
                base_keep_w,
                base_keep_t,
            ):
                continue

            new_kwargs = copy.deepcopy(base_kwargs)
            new_kwargs.update(
                {"keep_h": tuned_keep_h, "keep_w": tuned_keep_w, "keep_t": tuned_keep_t}
            )

            overrides_out["embeddings"]["per_signal_overrides"].setdefault(role, {})
            overrides_out["embeddings"]["per_signal_overrides"][role][name] = {
                "encoder_name": "dct3d",
                "encoder_kwargs": new_kwargs,
            }

    out_path = Path(cfg_mmt.paths["run_dir"]) / "embeddings_overrides.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "# Auto-generated by scripts_mast/run_tune_dct3d.py\n"
        f"# task: {cfg_mmt.task}\n"
        "# Merged on top of config/common/embeddings.yaml\n"
        "# Contains ONLY per-signal overrides.\n\n"
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(overrides_out, f, sort_keys=False, default_flow_style=False)

    logger.info("Wrote tuned embedding overrides to %s", out_path)


if __name__ == "__main__":
    main()
