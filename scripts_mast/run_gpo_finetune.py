"""
GPO fine-tuning entrypoint for TokaMind.

This script:
  1. Loads the pre-trained checkpoint (finetune warmstart) as model_source.
  2. Merges the GPO phase config (phases/gpo.yaml) and per-task overrides
     (tasks/gpo_tasks.yaml) on top of the standard finetune-warmstart base via
     the integration_hook parameter of load_experiment_config.
  3. Rebuilds the MAST window dataloader (same split used during collection).
  4. Wraps it with GPO pair injection so each batch contains y_w_emb / y_l_emb.
  5. Runs the standard train_finetune() loop with the continuous_gpo loss term.

GPO dataset layout expected at:
    runs/<model_source>/gpo_pairs/
        collection_config.json
        <signal>__shard_000000.npz  ...

Usage
-----
    python scripts_mast/run_gpo_finetune.py \\
        --task task_4-4 \\
        --model_source ft-task_4-4-scratch-mmt-embed_mse

Optional flags
--------------
    --gpo_dir          path to gpo_pairs/ directory  (default: <model_source>/gpo_pairs)
    --model_profile    mmt        (default: mmt)
    --tag              experiment tag appended to the run ID
    --emb_profile      embedding profile               (default: dct3d)
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import torch
from mast_utils import (
    build_mast_datasets,
    build_model_and_optional_warmstart,
    build_signals_by_role_from_task_definition,
    build_window_data,
    extract_signal_stats,
    init_run_context,
    load_experiment_config,
    load_task_definition,
    resolve_finetune_embeddings,
    validate_mast_config,
)
from mast_utils.gpo import load_collection_config
from mast_utils.gpo.dataset import GpoPairDataset
from mast_utils.gpo.protocol import (
    apply_source_contract,
    collection_contract,
    digest,
    save_run_snapshot,
    validate_collection,
)

from mmt.checkpoints import load_best_weights
from mmt.data import build_decoders
from mmt.train import train_finetune
from mmt.train.loop_utils import move_batch_to_device
from mmt.utils import sdpa_math_only_ctx, validate_config

log = logging.getLogger("mmt.GPO")


# ======================================================================================================================
# GPO DataLoader wrapper
# ======================================================================================================================


class _GpoBatchInjector:
    """
    Wraps a MAST DataLoader and injects ``y_w_emb`` / ``y_l_emb`` (and
    optionally ``ref_preds``) tensors (keyed by signal_id) into each batch dict.

    Exposes ``__iter__`` and ``__len__`` so it can be passed directly to
    ``train_finetune()`` in place of a DataLoader.  ``__len__`` delegates to
    the underlying MAST loader so the training loop can compute epoch lengths.

    The injection is based on matching ``(shot_id, window_index)`` — windows
    without a matching GPO pair are still passed through unchanged; the
    ``ContinuousGPOLoss`` skips them gracefully when ``y_l_emb`` is absent.

    Parameters
    ----------
    mast_loader : torch.utils.data.DataLoader
        The reconstructed MAST DataLoader (yields standard batch dicts).
    gpo_ds : GpoPairDataset
        GPO pair dataset for the corresponding split (train or val).
    signal_id_map : dict[str, int]
        Signal name → signal_id mapping for routing y_w_emb / y_l_emb into the
        correct keyed dict expected by the LossAggregator.
    ref_model : torch.nn.Module | None
        Frozen reference model.  When not None, a ``no_grad`` forward pass is
        run on each batch and the predictions are injected as ``batch["ref_preds"]``
        (dict[signal_id → Tensor(B, D)]).  Default: None (no reference model).
    device : torch.device | None
        Target device.  Required when ``ref_model`` is not None so the batch
        can be moved to the model's device before the reference forward pass.
        When ``ref_model`` is None this argument is ignored.
    """

    def __init__(
        self,
        mast_loader: torch.utils.data.DataLoader,
        gpo_ds: GpoPairDataset,
        signal_id_map: dict[str, int],
        ref_model: torch.nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._mast_loader = mast_loader
        self._signal_id_map = signal_id_map
        self._ref_model = ref_model
        self._device = device

        # Build a fast lookup: (shot_id, window_index) → {signal_id: (y_w, y_l)}
        log.info("Building GPO pair index (%d windows)…", len(gpo_ds))
        self._pair_index: dict[tuple[int, int], dict[int, tuple[torch.Tensor, torch.Tensor]]] = {}
        n_unmatched = 0
        for item in gpo_ds:
            key = (int(item["shot_id"]), int(item["window_index"]))
            sig_id = signal_id_map.get(item["signal_name"])
            if sig_id is None:
                n_unmatched += 1
                continue
            self._pair_index.setdefault(key, {})[sig_id] = (
                item["y_w_emb"],  # (D,)
                item["y_l_emb"],  # (D,)
            )
        log.info("GPO pair index built: %d unique (shot, window) keys", len(self._pair_index))
        if n_unmatched > 0:
            log.warning(
                "GPO pair index: %d/%d windows had no matching signal_id "
                "(signal names in GPO dataset did not match output_signal_id_map keys). "
                "Available map keys: %s",
                n_unmatched,
                len(gpo_ds),
                sorted(signal_id_map.keys()),
            )
        if not self._pair_index:
            raise RuntimeError(
                "GPO pair index is empty — no GPO pairs could be matched to output signals. "
                f"GPO dataset signal names must match output signal names. "
                f"output_signal_id_map keys: {sorted(signal_id_map.keys())}. "
                "Check that --task and --model_source match the dataset used for collection."
            )

    def __len__(self) -> int:
        return len(self._mast_loader)

    def __iter__(self):
        for batch in self._mast_loader:
            # ------------------------------------------------------------------
            # Capture (shot_id, window_index) from the CPU batch BEFORE any
            # device transfer.  move_batch_to_device does not move Python lists,
            # but reading them here makes the intent explicit and future-proofs
            # against any upstream change that might convert them to tensors.
            # int() normalises numpy scalars, Python ints, and numeric strings.
            # ------------------------------------------------------------------
            shot_ids = batch["shot_id"]  # list[int | np.int64 | str] (B,)
            window_idxs = batch["window_index"]  # list[int | np.int64 | str] (B,)
            B = len(shot_ids)

            # ------------------------------------------------------------------
            # Optional reference model forward pass.
            # The MAST dataloader yields CPU tensors; move them to the target
            # device before calling the ref model (which lives on GPU).
            # ref_preds are therefore already on device when yielded.
            # ------------------------------------------------------------------
            if self._ref_model is not None and self._device is not None:
                batch = move_batch_to_device(batch=batch, device=self._device)
                with torch.no_grad():
                    ref_out = self._ref_model(batch)
                # ref_out["pred"]: dict[int, Tensor(B, D)]
                batch["ref_preds"] = ref_out.get("pred", {})

            # Collect y_w_emb / y_l_emb and a pair mask per signal across batch rows.
            y_w_by_sig: dict[int, list[tuple[int, torch.Tensor]]] = {}
            y_l_by_sig: dict[int, list[tuple[int, torch.Tensor]]] = {}

            for b in range(B):
                key = (int(shot_ids[b]), int(window_idxs[b]))
                pairs = self._pair_index.get(key)
                if pairs is None:
                    continue
                for sig_id, (y_w, y_l) in pairs.items():
                    y_w_by_sig.setdefault(sig_id, []).append((b, y_w))
                    y_l_by_sig.setdefault(sig_id, []).append((b, y_l))

            if y_w_by_sig:
                # Stack into (B, D) tensors and provide an explicit pair mask.
                # Unfilled rows retain zero placeholders, but the pair mask keeps
                # them out of ContinuousGPOLoss for mixed paired/unpaired batches.
                # NOTE: each signal may have a different embedding dim (D), so
                # allocate the buffer per-signal from the first entry of that
                # signal rather than reusing a single global D_ref.
                batch_y_w: dict[int, torch.Tensor] = {}
                batch_y_l: dict[int, torch.Tensor] = {}
                batch_pair_mask: dict[int, torch.Tensor] = {}

                for sig_id, entries_w in y_w_by_sig.items():
                    entries_l = y_l_by_sig.get(sig_id, [])
                    D_sig = entries_w[0][1].shape[0]
                    buf_w = torch.zeros(B, D_sig, dtype=torch.float32)
                    buf_l = torch.zeros(B, D_sig, dtype=torch.float32)
                    pair_mask = torch.zeros(B, dtype=torch.bool)
                    for b_idx, y_w in entries_w:
                        buf_w[b_idx] = y_w
                    for b_idx, y_l in entries_l:
                        buf_l[b_idx] = y_l
                        pair_mask[b_idx] = True
                    batch_y_w[sig_id] = buf_w
                    batch_y_l[sig_id] = buf_l
                    batch_pair_mask[sig_id] = pair_mask

                batch["y_w_emb"] = batch_y_w
                batch["y_l_emb"] = batch_y_l
                batch["gpo_pair_mask"] = batch_pair_mask

            yield batch


# ======================================================================================================================
# CLI
# ======================================================================================================================


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPO fine-tuning for a trained TokaMind model.",
        allow_abbrev=False,
    )
    parser.add_argument("--task", type=str, default="_test", help="Task identifier.")
    parser.add_argument(
        "--model_source", type=str, default="_test", help="Run ID of the trained model to GPO-finetune."
    )
    parser.add_argument(
        "--gpo_dir", type=str, default=None, help="Path to gpo_pairs/ directory. Default: <model_source>/gpo_pairs."
    )
    parser.add_argument("--model_profile", type=str, default="mmt", help="Model profile (mmt).")
    parser.add_argument("--tag", type=str, default=None, help="Optional experiment tag for the output run ID.")
    parser.add_argument("--emb_profile", type=str, default="dct3d", help="Embedding profile.")
    return parser.parse_args()


# ======================================================================================================================
# Main
# ======================================================================================================================


def apply_gpo_config(
    merged: MutableMapping[str, Any], phase: str, *, configs_root: str, task: str, model_profile: str
) -> None:
    """Merge gpo.yaml and gpo_tasks.yaml on top of the base finetune-warmstart config."""
    from pathlib import Path as _Path

    from mmt.utils.config.experiment.merge import _load_task_block, deep_merge, load_yaml

    cr = _Path(configs_root)
    model_profile_name = str(merged.get("model_profile", model_profile))
    phases_dir = cr / model_profile_name / "phases"
    tasks_dir = cr / model_profile_name / "tasks"

    # Preserve keys that were already resolved by inject_cli_overrides_finetune
    # (e.g. run_id) before merging gpo.yaml, which carries null placeholders.
    _preserve = {k: merged[k] for k in ("run_id",) if k in merged and merged[k] is not None}

    # Merge GPO phase config (overrides train schedule, loss, freeze).
    # NOTE: deep_merge merges train.stages lists by stage *name* — it appends any
    # new stage name rather than replacing the whole list.  gpo.yaml has a single
    # "gpo" stage; finetune_warmstart.yaml has "ft_heads" + "ft_full".  Without
    # explicit replacement the merged config would run all three stages (20 extra
    # epochs before GPO even starts).  We force-replace the stages list after
    # merging so only the GPO stage runs.
    gpo_phase_path = phases_dir / "gpo.yaml"
    if gpo_phase_path.is_file():
        gpo_phase = load_yaml(gpo_phase_path)
        merged.update(deep_merge(base=dict(merged), override=gpo_phase))
        # Force the stages list to only what gpo.yaml specifies.
        gpo_stages = (gpo_phase.get("train") or {}).get("stages")
        if gpo_stages is not None:
            merged.setdefault("train", {})["stages"] = gpo_stages

    # Merge per-task GPO overrides.
    # GPO_TASKS_YAML env var allows ablations to swap in an alternate tasks file
    # without modifying code (e.g. GPO_TASKS_YAML=.../gpo_tasks_embed_mse_only.yaml).
    import os as _os

    _tasks_yaml_override = _os.environ.get("GPO_TASKS_YAML")
    gpo_tasks_path = _Path(_tasks_yaml_override) if _tasks_yaml_override else tasks_dir / "gpo_tasks.yaml"
    task_overrides = _load_task_block(path=gpo_tasks_path, task=task, phase="gpo")
    if task_overrides:
        merged.update(deep_merge(base=dict(merged), override=task_overrides))

    # Restore any preserved keys that were nulled out by the phase/task yamls.
    merged.update(_preserve)
    apply_source_contract(merged, configs_root)


def main() -> None:
    args = _parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Config.
    # GPO uses phase="finetune" + finetune_init="warmstart" so the standard
    # config loader applies source-run inheritance and run-ID generation correctly.
    # The integration_hook merges gpo.yaml on top (overriding train schedule,
    # loss terms, freeze policy) and applies per-task overrides from gpo_tasks.yaml.
    # ------------------------------------------------------------------------------------------------------------------
    configs_root = "scripts_mast/configs"

    cfg_mmt = load_experiment_config(
        task=args.task,
        phase="finetune",
        model_profile=args.model_profile,
        embeddings_profile=args.emb_profile,
        model_source=args.model_source,
        tag=args.tag,
        finetune_init="warmstart",
        configs_root=configs_root,
        integration_hook=lambda merged, phase: apply_gpo_config(
            merged, phase, configs_root=configs_root, task=args.task, model_profile=args.model_profile
        ),
        save_config=False,
    )
    validate_config(cfg=cfg_mmt)
    validate_mast_config(cfg=cfg_mmt)

    # GPO needs model_source to construct the policy/reference architecture.
    # Set resume only after validation, which correctly rejects a declarative
    # combination of train.resume=true and model_source.
    import os

    resume_requested = os.environ.get("GPO_RESUME") == "1"

    cfg_data = cfg_mmt.data
    cfg_loader = cfg_mmt.loader
    cfg_train = cfg_mmt.train

    cfg_task = load_task_definition(task_key=args.task)

    # ------------------------------------------------------------------------------------------------------------------
    # GPO pairs directory
    # ------------------------------------------------------------------------------------------------------------------
    train_run_dir = Path(str(cfg_mmt.model_source["run_dir"]))
    gpo_dir = Path(args.gpo_dir) if args.gpo_dir else train_run_dir / "gpo_pairs"

    if not gpo_dir.is_dir():
        raise SystemExit(
            f"GPO pairs directory not found: {gpo_dir}\nRun run_collect_gpo_pairs.py first to generate the dataset."
        )

    cc = load_collection_config(gpo_dir=gpo_dir)
    expected_contract = collection_contract(cfg_mmt.raw, cfg_task)
    validate_collection(cc, expected_contract, gpo_dir)
    cfg_mmt.raw["gpo_provenance"] = {
        "protocol": cc["protocol"],
        "collection": cc,
        "collection_digest": digest(cc),
        "pairs_dir": str(gpo_dir.resolve()),
    }
    save_run_snapshot(cfg_mmt, resume=resume_requested)
    if resume_requested:
        cfg_mmt.train["resume"] = True
    device, _ = init_run_context(cfg_mmt=cfg_mmt, phase="finetune")

    cfg_model_source = cfg_mmt.raw.get("model_source")
    dict_task_metadata, _mast_train, _mast_val, _mast_test = build_mast_datasets(
        cfg_task=cfg_task,
        cfg_data=cfg_data,
        phase="finetune",
        cfg_model_source=cfg_model_source,
    )

    # ------------------------------------------------------------------------------------------------------------------
    # Signal specs + embeddings
    # run_dir must be the GPO run directory (where embeddings are staged/written),
    # not train_run_dir (the source model).  resolve_finetune_embeddings derives
    # source_run_dir from cfg_mmt.raw["model_source"]["run_dir"] internally, so
    # passing train_run_dir as run_dir makes src == dst and shutil.copy2 crashes
    # with SameFileError.  Same pattern as run_finetune.py:160.
    # ------------------------------------------------------------------------------------------------------------------
    gpo_run_dir = Path(str(cfg_mmt.paths["run_dir"]))
    signals_by_role = build_signals_by_role_from_task_definition(
        cfg_task=cfg_task,
        dict_metadata=dict_task_metadata,
    )
    signal_specs, codecs = resolve_finetune_embeddings(
        cfg_mmt=cfg_mmt,
        signals_by_role=signals_by_role,
        dict_task_metadata=dict_task_metadata,
        run_dir=gpo_run_dir,
        cfg_task=cfg_task,
    )
    output_decoders = build_decoders(registry=signal_specs, codecs=codecs, role="output")

    # signal_name (safe form, slashes → hyphens) → signal_id mapping for GPO pair injection.
    # GpoPairDataset stores the safe filename stem as signal_name, so the map must use the
    # same safe form to guarantee the lookup in _GpoBatchInjector always succeeds.
    output_signal_id_map: dict[str, int] = {
        spec.name.replace("/", "-").replace("\\", "-"): int(spec.signal_id)
        for spec in signal_specs.specs_for_role("output")
    }

    # ------------------------------------------------------------------------------------------------------------------
    # Separate official MAST shot sets for every loss, including the MSE anchor.
    # ------------------------------------------------------------------------------------------------------------------
    window_data = build_window_data(
        cfg_mmt=cfg_mmt,
        mast_datasets={"train": _mast_train, "val": _mast_val},
        dict_task_metadata=dict_task_metadata,
        cfg_task=cfg_task,
        signal_specs=signal_specs,
        codecs=codecs,
        phase="finetune",
        output_decoders=output_decoders,
    )
    mast_train_loader = window_data["train"]["loader"]

    mast_val_loader = window_data["val"]["loader"]

    # ------------------------------------------------------------------------------------------------------------------
    # GPO pair datasets — optional per-task dataset filters from config
    # ------------------------------------------------------------------------------------------------------------------
    # Filters are read from train.gpo_dataset in the task YAML:
    #   train:
    #     gpo_dataset:
    #       shot_blacklist: [18675, 14013, 14342]   # task_4-1/4-2
    #       native_nrmse_percentiles: [25.0, 95.0]  # schema-v3 V8 selection band
    #       max_pairs_per_shot_percentile: 90.0     # schema-v3 V8 coverage cap
    _gpo_ds_cfg: dict = cfg_train.get("gpo_dataset") or {}
    _shot_blacklist: set[int] | None = (
        {int(s) for s in _gpo_ds_cfg["shot_blacklist"]} if _gpo_ds_cfg.get("shot_blacklist") else None
    )
    _min_margin_mse: float | None = (
        float(_gpo_ds_cfg["min_margin_mse"]) if _gpo_ds_cfg.get("min_margin_mse") is not None else None
    )
    _min_window_index: int | None = (
        int(_gpo_ds_cfg["min_window_index"]) if _gpo_ds_cfg.get("min_window_index") is not None else None
    )
    _native_nrmse_percentiles: tuple[float, float] | None = (
        (float(_gpo_ds_cfg["native_nrmse_percentiles"][0]), float(_gpo_ds_cfg["native_nrmse_percentiles"][1]))
        if _gpo_ds_cfg.get("native_nrmse_percentiles") is not None
        else None
    )
    _max_pairs_per_shot_percentile: float | None = (
        float(_gpo_ds_cfg["max_pairs_per_shot_percentile"])
        if _gpo_ds_cfg.get("max_pairs_per_shot_percentile") is not None
        else None
    )

    gpo_train_ds = GpoPairDataset(
        gpo_dir=gpo_dir,
        split="train",
        shot_blacklist=_shot_blacklist,
        min_margin_mse=_min_margin_mse,
        min_window_index=_min_window_index,
        native_nrmse_percentiles=_native_nrmse_percentiles,
        max_pairs_per_shot_percentile=_max_pairs_per_shot_percentile,
    )
    gpo_val_ds = GpoPairDataset(
        gpo_dir=gpo_dir,
        split="val",
        shot_blacklist=_shot_blacklist,
        min_margin_mse=_min_margin_mse,
        min_window_index=_min_window_index,
        native_nrmse_percentiles=_native_nrmse_percentiles,
        max_pairs_per_shot_percentile=_max_pairs_per_shot_percentile,
        filter_state=gpo_train_ds.filter_state,
    )

    import json

    filter_path = gpo_run_dir / "gpo_filters.json"
    filter_json = json.dumps(gpo_train_ds.filter_state, sort_keys=True, indent=2)
    if resume_requested:
        if not filter_path.is_file() or filter_path.read_text() != filter_json:
            raise ValueError("Training-fitted CGPO filters differ from the saved run.")
    else:
        with filter_path.open("x") as stream:
            stream.write(filter_json)

    # ------------------------------------------------------------------------------------------------------------------
    # Model — load best checkpoint from the source finetune run
    # ------------------------------------------------------------------------------------------------------------------
    model = build_model_and_optional_warmstart(
        cfg_mmt=cfg_mmt, signal_specs=signal_specs, device=device, skip_warmstart=True
    )
    if resume_requested:
        log.info(
            "GPO resume requested; policy weights will be restored from %s/checkpoints/latest.",
            gpo_run_dir,
        )
    else:
        epoch_best, best_val_src, _ = load_best_weights(
            run_dir=str(train_run_dir), model=model, map_location=str(device)
        )
        log.info(
            "Loaded source checkpoint: run=%s epoch=%s best_val=%s",
            train_run_dir.name,
            epoch_best,
            best_val_src,
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Reference model (optional) — frozen copy of the base checkpoint.
    # Enabled by train.use_reference_model: true in gpo.yaml (default: false for
    # backwards compatibility).  The reference model shares the same architecture
    # and weights as the policy model at t=0; it is kept frozen throughout training
    # to provide the DPO-style KL constraint:
    #   margin = [d(ŷ,y_l) − d(ŷ_ref,y_l)] − [d(ŷ,y_w) − d(ŷ_ref,y_w)]
    # ------------------------------------------------------------------------------------------------------------------
    use_ref_model: bool = bool(cfg_train.get("use_reference_model", False))
    ref_model: torch.nn.Module | None = None

    if use_ref_model:
        ref_model = build_model_and_optional_warmstart(
            cfg_mmt=cfg_mmt, signal_specs=signal_specs, device=device, skip_warmstart=True
        )
        load_best_weights(run_dir=str(train_run_dir), model=ref_model, map_location=str(device))
        # Freeze all parameters — reference model never accumulates gradients.
        for param in ref_model.parameters():
            param.requires_grad_(False)
        ref_model.eval()
        ref_model.to(device)
        log.info("Reference model loaded and frozen on %s (use_reference_model=true).", device)

    # Re-build injectors now that the optional ref_model is known.
    gpo_train_loader = _GpoBatchInjector(
        mast_train_loader, gpo_train_ds, output_signal_id_map, ref_model=ref_model, device=device
    )
    gpo_val_loader = _GpoBatchInjector(
        mast_val_loader, gpo_val_ds, output_signal_id_map, ref_model=ref_model, device=device
    )

    # ------------------------------------------------------------------------------------------------------------------
    # GPO fine-tune
    # ------------------------------------------------------------------------------------------------------------------
    log.info("GPO run dir: %s", gpo_run_dir)

    with sdpa_math_only_ctx():
        result = train_finetune(
            model=model,
            train_loader=gpo_train_loader,
            val_loader=gpo_val_loader,
            run_dir=str(gpo_run_dir),
            train_cfg=cfg_train,
            loader_cfg=cfg_loader,
            output_decoders=output_decoders,
            signal_stats=extract_signal_stats(dict_metadata=dict_task_metadata),
        )

    log.info("GPO fine-tuning complete: %s", result)


# ======================================================================================================================
if __name__ == "__main__":
    main()
