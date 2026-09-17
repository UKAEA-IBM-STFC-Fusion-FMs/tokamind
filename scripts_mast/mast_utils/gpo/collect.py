"""
scripts_mast.mast_utils.gpo.collect

Core forward loop for GPO preference-pair collection.

Drives a single inference pass over a dataloader, calling
:class:`GpoPairWriter` for each batch.  The loop mirrors the structure of
:func:`~scripts_mast.mast_utils.eval.benchmark_eval.evaluate_benchmark_and_diagnostics`
but writes pairs instead of computing metrics.

Schema v3: pairs are stored in **embedding (coefficient) space**.
The ground-truth embedding ``y_w_emb`` comes from ``batch["output_emb"]``
(the codec-encoded target already present in the collated batch), and the
model-prediction embedding ``y_l_emb`` comes from the model's ``out["pred"]``
dict. Per-window decoded native-space error diagnostics are also recorded for
model-error pairs; full native arrays are not stored.

Split / validation fraction
---------------------------
Collection runs over the ``train`` split by default. A ``val_fraction`` of
retained windows is reserved as GPO validation data by the downstream dataset
through a fixed-seed global shuffle. The fraction is stored in
``collection_config.json`` for the downstream GPO DataLoader.

The GPO source for dispreferred responses ``y_l_emb`` is always the *model's own
prediction in embedding space*.  Additional ``PairSource`` providers can be
registered for additional synthetic or domain-specific sources in
future iterations — see the ``extra_sources`` parameter.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch.utils.data import DataLoader

from mmt.data.standardization import destandardize_numpy
from mmt.train.loop_utils import move_batch_to_device
from mmt.utils.amp_utils import amp_ctx_for_model

from .writer import GpoPairWriter

logger = logging.getLogger("mmt.GPO")

_LOG_INTERVAL = 10_000


# ======================================================================================================================
# Extension point: physics-informed pair sources
# ======================================================================================================================


@runtime_checkable
class PairSource(Protocol):
    """
    Protocol for pluggable dispreferred-response generators.

    A ``PairSource`` receives the raw batch and the ground-truth embedding
    arrays for one step and may produce an alternative ``y_l_emb`` for any
    subset of output signals.  Return an empty dict to skip a signal.

    This is the extension point for physics-informed sources, e.g.:

    * Injecting synthetic noise calibrated to a known physical bound.

    The default collector uses only the model-error source
    (``y_l_emb = model_pred_emb``), which does not require any ``PairSource``.
    """

    def generate(
        self,
        batch: Mapping[str, Any],
        y_w_emb: Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Generate dispreferred embedding responses for a batch.

        Parameters
        ----------
        batch : Mapping[str, Any]
            The raw collated batch (on CPU, before device transfer).
        y_w_emb : Mapping[str, np.ndarray]
            Ground-truth output embeddings in coefficient space, shape ``(B, D)``,
            keyed by signal name.

        Returns
        -------
        dict[str, np.ndarray]
            Signal-name → dispreferred embedding array ``(B, D)``.
            Return an empty dict or omit a key to leave that signal uncovered.
        """
        ...


# ======================================================================================================================
# Main collection loop
# ======================================================================================================================


def collect_gpo_pairs(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id_to_name: Mapping[int, str],
    run_dir: Path,
    task_name: str,
    split: str,
    val_fraction: float = 0.0,
    train_fraction: float = 1.0,
    multi_signal: str = "joint",
    out_dir: Path | None = None,
    amp_enabled: bool = True,
    shard_size: int = 2048,
    overwrite: bool = False,
    extra_sources: list[PairSource] | None = None,
    native_decoders: Mapping[str, torch.nn.Module] | None = None,
    native_stats: Mapping[str, Mapping[str, float]] | None = None,
    protocol_contract: dict | None = None,
) -> dict[str, Any]:
    """
    Collect GPO preference pairs from a trained model and write them to disk.

    For each window in ``dataloader`` the function:

    1. Runs a forward pass to get model-prediction embeddings (``y_l_emb``).
    2. Reads ground-truth embeddings from ``batch["output_emb"]`` (``y_w_emb``).
    3. Writes ``(y_w_emb, y_l_emb)`` pairs via :class:`GpoPairWriter`.
    4. Optionally calls any ``extra_sources`` to produce additional ``y_l_emb``
       arrays from physics-informed perturbations of ``y_w_emb``.

    Both ``y_w_emb`` and ``y_l_emb`` are standardised coefficient vectors of
    shape ``(B, D)`` — the same space used by the training loss. When native
    decoders and statistics are supplied, predictions are additionally decoded
    transiently to record native-space diagnostics.

    Output layout under ``out_dir`` (default ``run_dir/gpo_pairs``)::

        metadata.json           — human-readable provenance summary
        collection_config.json  — machine-readable reconstruction recipe
        <signal_name>__shard_000000.npz
        ...

    Parameters
    ----------
    model : torch.nn.Module
        Trained model in eval mode.
    dataloader : DataLoader
        Window-level dataloader (``train`` split recommended).
    device : torch.device
    id_to_name : Mapping[int, str]
        Signal-id → signal-name mapping.
    run_dir : Path
        Training run directory (e.g. ``runs/ft-task_4-1-scratch-mmt``).
        Recorded in provenance; also the default parent for the output when
        ``out_dir`` is not given.
    task_name : str
        Task identifier (e.g. ``"task_4-1"``), stored in provenance and in
        ``collection_config.json`` for deterministic reconstruction.
    split : str
        Dataset split used to build ``dataloader`` (``"train"`` recommended;
        ``"val"`` is accepted for small-data experiments).
        Stored in ``collection_config.json``.
    val_fraction : float
        Fraction of collected windows to hold out as GPO validation data.
        Applied at read-time by :class:`~scripts_mast.mast_utils.gpo.dataset.GpoPairDataset`
        (random global shuffle then fractional cut).
        Must be in ``[0.0, 1.0)``.  Default: ``0.0`` (no validation split).
    train_fraction : float
        Fraction of the dataloader batches to process.  ``1.0`` (default) collects
        from every batch.  Values < 1.0 are applied by the caller before passing
        ``dataloader`` to this function; this parameter is recorded in provenance
        for reproducibility.  Must be in ``(0.0, 1.0]``.
    multi_signal : str
        How multi-output tasks are handled by the downstream GPO loss.
        ``"joint"`` (default) computes a single combined loss over all signals.
        ``"independent"`` trains a separate loss per signal.
    out_dir : Path | None
        Directory the dataset is written to.  Default: ``run_dir/gpo_pairs``.
        Pass an explicit path (e.g. a ``--tag``-suffixed dir) to override.
    amp_enabled : bool
        Whether to use AMP in the forward pass.  Default: True.
    shard_size : int
        Target windows per .npz shard; a shard may exceed it by up to one batch.
        Default: 2048.
    overwrite : bool
        Replace ``out_dir`` if it already exists instead of raising.  Default: False.
    extra_sources : list[PairSource] | None
        Optional list of physics-informed ``PairSource`` instances.  Default: None.
    native_decoders, native_stats
        Output decoders and benchmark signal statistics used to record decoded
        native-space error diagnostics for model-error pairs. Both must be
        supplied together, or neither may be supplied.

    Returns
    -------
    dict[str, Any]
        ``{"out_dir": str, "total_windows": {signal_name: int}}``.
    """
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction!r}.")
    if (native_decoders is None) != (native_stats is None):
        raise ValueError("native_decoders and native_stats must be supplied together.")

    run_dir = Path(run_dir)
    run_id = run_dir.name
    out_dir = Path(out_dir) if out_dir is not None else run_dir / "gpo_pairs"

    provenance: dict[str, Any] = {
        "task": task_name,
        "split": split,
        "val_fraction": float(val_fraction),
        "train_fraction": float(train_fraction),
        "multi_signal": multi_signal,
        "run_dir": str(run_dir),
        "run_id": run_id,
        "pair_sources": ["model_error"] + [type(s).__name__ for s in (extra_sources or [])],
    }

    if protocol_contract is not None:
        from .protocol import PROTOCOL

        provenance.update(protocol=PROTOCOL, contract=protocol_contract)

    writer = GpoPairWriter(out_dir=out_dir, provenance=provenance, shard_size=shard_size, overwrite=overwrite)

    model.eval()
    n_windows = 0
    next_log_at = _LOG_INTERVAL

    # Discard the staging dir if collection fails before finalize(), so a crashed
    # run does not leave an orphaned temp directory behind.
    try:
        with torch.no_grad():
            for batch in dataloader:
                # ----------------------------------------------------------------
                # 1. Move batch to device and run the model.
                # ----------------------------------------------------------------
                batch_dev = move_batch_to_device(batch=batch, device=device)

                with amp_ctx_for_model(model=model, enable=amp_enabled):
                    out = model(batch_dev)

                # out["pred"]: dict[int, Tensor(B, D)] — model predictions in
                # standardised coefficient space.
                y_pred_emb_id: dict[int, torch.Tensor] = out.get("pred", {})

                # batch["output_emb"]: dict[int, Tensor(B, D)] — ground-truth
                # embeddings already computed by the collate/transform pipeline.
                y_true_emb_id: dict[int, torch.Tensor] = batch_dev.get("output_emb", {})

                # batch["output_mask"]: dict[int, Tensor(B,)] — validity mask.
                y_mask_id: dict[int, torch.Tensor] = batch_dev.get("output_mask", {})

                # Per-window metadata (CPU numpy).
                shot_ids = np.asarray(batch["shot_id"])
                window_indices = np.asarray(batch["window_index"])

                B = len(shot_ids)
                n_windows += B

                if n_windows >= next_log_at:
                    logger.info("Collected %d windows so far", next_log_at)
                    next_log_at += _LOG_INTERVAL

                # ----------------------------------------------------------------
                # Source 1 (always): model-error pairs in embedding space.
                #   y_w_emb = ground-truth coefficients (from batch)
                #   y_l_emb = model-prediction coefficients (from model output)
                # ----------------------------------------------------------------
                # Build name-keyed numpy views for the sources below.
                y_w_emb_named: dict[str, np.ndarray] = {}

                for sig_id, y_true_emb_t in y_true_emb_id.items():
                    if sig_id not in id_to_name:
                        continue
                    if sig_id not in y_pred_emb_id:
                        continue
                    if sig_id not in y_mask_id:
                        continue

                    name = id_to_name[sig_id]

                    # Move to CPU float32 numpy.
                    y_w_emb = y_true_emb_t.detach().cpu().float().numpy()  # (B, D)
                    y_l_emb = y_pred_emb_id[sig_id].detach().cpu().float().numpy()  # (B, D)
                    mask = y_mask_id[sig_id].detach().cpu().numpy()

                    native_nrmse = native_nmae = native_mse = None
                    if native_decoders is not None and native_stats is not None:
                        if name not in native_decoders or name not in native_stats:
                            raise ValueError(f"Missing native decoder or statistics for output signal {name!r}.")
                        y_true_native_t = batch_dev["output_native"].get(sig_id)
                        if y_true_native_t is None:
                            raise ValueError(f"Missing native target for output signal {name!r} (id={sig_id}).")
                        y_true_native = destandardize_numpy(
                            arr=y_true_native_t.detach().cpu().float().numpy(),
                            mean=native_stats[name]["mean"],
                            std=native_stats[name]["std"],
                        )
                        with torch.no_grad():
                            y_pred_native_std = native_decoders[name](torch.from_numpy(y_l_emb))
                        y_pred_native = destandardize_numpy(
                            arr=y_pred_native_std.detach().cpu().float().numpy(),
                            mean=native_stats[name]["mean"],
                            std=native_stats[name]["std"],
                        )

                        # Native targets use NaN padding for missing measurements. Mirror
                        # NativeSparseMSELoss by averaging only observed positions, and do
                        # not persist pairs whose prediction is non-finite at an observed
                        # position or whose target has no observed positions.
                        valid = np.isfinite(y_true_native).reshape(B, -1)
                        y_true_flat = np.where(valid, y_true_native.reshape(B, -1), 0.0)
                        y_pred_flat = np.where(valid, y_pred_native.reshape(B, -1), 0.0)
                        eligible = valid.any(axis=1) & np.isfinite(y_pred_flat).all(axis=1)
                        n_valid = valid.sum(axis=1, dtype=np.float64)
                        diff = y_pred_flat - y_true_flat
                        safe_n_valid = np.where(eligible, n_valid, 1.0)
                        native_mse = (np.sum(diff**2, axis=1, dtype=np.float64) / safe_n_valid).astype(np.float32)
                        native_nmae = (
                            np.sum(np.abs(diff), axis=1, dtype=np.float64) / safe_n_valid / native_stats[name]["std"]
                        ).astype(np.float32)
                        native_nrmse = (np.sqrt(native_mse) / native_stats[name]["std"]).astype(np.float32)

                        mask_1d = np.asarray(mask, dtype=bool)
                        if mask_1d.ndim > 1:
                            mask_1d = mask_1d.reshape(mask_1d.shape[0], -1).any(axis=1)
                        mask = mask_1d & eligible

                    y_w_emb_named[name] = y_w_emb  # keep for extra_sources

                    writer.add_batch(
                        output_name=name,
                        y_w_emb=y_w_emb,
                        y_l_emb=y_l_emb,
                        mask=mask,
                        shot_ids=shot_ids,
                        window_indices=window_indices,
                        native_nrmse=native_nrmse,
                        native_nmae=native_nmae,
                        native_mse=native_mse,
                    )

                # ----------------------------------------------------------------
                # Source 2+: physics-informed extra sources (extensible).
                # Each PairSource returns {signal_name: y_l_emb_array}.
                # y_w_emb is always the ground-truth embedding from the batch.
                # ----------------------------------------------------------------
                if extra_sources:
                    for source in extra_sources:
                        source_pairs = source.generate(
                            batch=batch,
                            y_w_emb=y_w_emb_named,
                        )
                        for out_name, y_l_extra_emb in source_pairs.items():
                            # Look up the signal_id for the mask.
                            sig_id_for_name = next(
                                (sid for sid, nm in id_to_name.items() if nm == out_name),
                                None,
                            )
                            if sig_id_for_name is None or sig_id_for_name not in y_mask_id:
                                continue
                            if out_name not in y_w_emb_named:
                                continue
                            mask_extra = y_mask_id[sig_id_for_name].detach().cpu().numpy()
                            writer.add_batch(
                                output_name=f"{out_name}__src_{type(source).__name__}",
                                y_w_emb=y_w_emb_named[out_name],
                                y_l_emb=y_l_extra_emb,
                                mask=mask_extra,
                                shot_ids=shot_ids,
                                window_indices=window_indices,
                            )

    except BaseException:
        writer.abort()
        raise

    logger.info("GPO collection complete: %d total windows processed", n_windows)
    return writer.finalize()
