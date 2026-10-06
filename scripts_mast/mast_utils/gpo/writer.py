"""
scripts_mast.mast_utils.gpo.writer

Writes GPO preference pairs to sharded NumPy .npz files.

Schema version 3 (current)
---------------------------
Each shard stores pairs in **embedding (coefficient) space** so that the GPO
training loop can work directly in the same space as the model's training loss,
without requiring a decode step at every training iteration:

    shard_NNNNNN.npz
        shot_id          : int64   (B,)
        window_index     : int64   (B,)
        y_w_emb          : float32 (B, D)  — ground-truth embedding (preferred)
        y_l_emb          : float32 (B, D)  — model-prediction embedding (dispreferred)
        native_nrmse     : float32 (B,)    — decoded native-space RMSE / benchmark signal std
        native_nmae      : float32 (B,)    — decoded native-space MAE / benchmark signal std
        native_mse       : float32 (B,)    — decoded native-space MSE
        embedding_gap    : float32 (B,)    — mean squared winner/loser embedding distance
        emb_dim          : int     — scalar, embedding dimension D (convenience check)

Native-space arrays (y_w, y_l) are **not** stored in v3 shards. They can be
recovered on demand via the reconstruction dataloader + codec when needed for
additional diagnostics or physics-aware loss terms.

Schema version 1 (legacy)
--------------------------
Earlier shards store pairs in native physical units:

    shard_NNNNNN.npz
        shot_id          : int64   (B,)
        window_index     : int64   (B,)
        y_w              : float32 (B, *native_shape)
        y_l              : float32 (B, *native_shape)

Version 1 shards are readable by the stats visualizer for backwards
compatibility but cannot be used for GPO training without re-collection.

Two JSON files are written once at finalization:

metadata.json
    Provenance record: task, split, val_fraction, model source, schema version,
    total windows per signal.  Human-readable; not used for reconstruction.

collection_config.json
    Minimal machine-readable reconstruction recipe:

        schema_version : int    — 3 for new collections
        run_dir        : str    — absolute path to the source training run
        run_id         : str    — run ID (basename of run_dir)
        task           : str    — task identifier (e.g. "task_4-1")
        split          : str    — dataset split used ("train")
        val_fraction   : float  — fraction of windows held out for GPO validation
        pair_sources   : list   — ["model_error"] + any extra physics sources
        multi_signal   : str    — "joint" (combined loss) or "independent"
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("mmt.GPO")

# Bump this when the on-disk layout changes in a backwards-incompatible way.
_SCHEMA_VERSION = 3

_COLLECTION_CONFIG_KEYS = (
    "run_dir", "run_id", "task", "split", "val_fraction", "pair_sources", "multi_signal",
)


# ======================================================================================================================
class GpoPairWriter:
    """
    Streams GPO preference pairs (y_w_emb, y_l_emb) per output signal to sharded .npz files.

    Schema v3: pairs are stored in **embedding space** (standardised coefficient
    vectors of shape ``(B, D)``), matching the space where the GPO loss is computed.

    One writer instance covers one collection run (one task + one model).
    Call :meth:`add_batch` for each dataloader batch, then :meth:`finalize` once
    at the end to flush and write metadata.json.

    Shards are staged in a temporary sibling directory during collection and the
    final ``out_dir`` is swapped in only on :meth:`finalize` (rename-aside the old
    dataset, rename staging into place, then drop the backup).  A crashed or
    interrupted run never leaves a half-written dataset in place.

    Parameters
    ----------
    out_dir : Path
        Directory the finished dataset is promoted to.  Staged atomically on
        :meth:`finalize`.
    provenance : dict[str, Any]
        Free-form provenance dict written verbatim into metadata.json (task name,
        model source, split, val_fraction, etc.).
    shard_size : int
        Target number of windows per shard file; a shard is flushed once the
        buffer reaches this size, so a single flush may exceed it by up to one
        batch.  Default: 2048.
    overwrite : bool
        If ``out_dir`` already exists: raise ``FileExistsError`` when False,
        replace it when True.  Default: False.
    """

    def __init__(
        self,
        out_dir: Path,
        provenance: dict[str, Any],
        shard_size: int = 2048,
        overwrite: bool = False,
    ) -> None:
        self._out_dir = Path(out_dir)
        self._overwrite = bool(overwrite)

        if self._out_dir.exists() and not self._overwrite:
            raise FileExistsError(
                f"GPO output directory already exists: {self._out_dir}. "
                "Pass overwrite=True (--overwrite) to replace it, or choose a different --tag."
            )

        self._provenance = provenance
        self._shard_size = int(shard_size)

        # Stage shards in a temp sibling dir on the same filesystem so the final
        # promotion in finalize() is an atomic os.replace (rename) rather than a
        # slow, non-atomic copy across filesystems.
        self._out_dir.parent.mkdir(parents=True, exist_ok=True)
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix=f".{self._out_dir.name}.", suffix=".tmp", dir=self._out_dir.parent)
        )

        # Per-signal accumulators: signal_name -> list of row dicts
        self._buffers: dict[str, list[dict[str, Any]]] = {}

        # Per-signal shard counter (for zero-padded filenames)
        self._shard_counters: dict[str, int] = {}

        # Total windows written per signal (for metadata)
        self._totals: dict[str, int] = {}

    # ------------------------------------------------------------------------------------------------------------------
    def add_batch(
        self,
        *,
        output_name: str,
        y_w_emb: np.ndarray,
        y_l_emb: np.ndarray,
        mask: np.ndarray,
        shot_ids: np.ndarray,
        window_indices: np.ndarray,
        native_nrmse: np.ndarray | None = None,
        native_nmae: np.ndarray | None = None,
        native_mse: np.ndarray | None = None,
    ) -> None:
        """
        Buffer one batch of (y_w_emb, y_l_emb) pairs for ``output_name``.

        Only windows where ``mask[b]`` is True are stored.

        Parameters
        ----------
        output_name : str
            Canonical signal name (e.g. ``"soft_x_rays-horizontal_cam_lower"``).
        y_w_emb : np.ndarray, shape (B, D)
            Ground-truth embedding coefficients — the *preferred* response.
        y_l_emb : np.ndarray, shape (B, D)
            Model-prediction embedding coefficients — the *dispreferred* response.
        mask : np.ndarray, shape (B,) or broadcastable
            Boolean validity mask.  Only True rows are stored.
        shot_ids : np.ndarray, shape (B,)
        window_indices : np.ndarray, shape (B,)
        native_nrmse, native_nmae, native_mse : np.ndarray | None
            Per-window decoded native-space diagnostics.  ``None`` is written as
            NaN and is reserved for future pair sources that cannot be decoded.
        """
        mask_1d = np.asarray(mask, dtype=bool)
        if mask_1d.ndim > 1:
            mask_1d = mask_1d.reshape(mask_1d.shape[0], -1).any(axis=1)

        valid_idx = np.nonzero(mask_1d)[0]
        if len(valid_idx) == 0:
            return

        def _diagnostic_or_nan(values: np.ndarray | None) -> np.ndarray:
            if values is None:
                return np.full(len(shot_ids), np.nan, dtype=np.float32)
            if len(values) != len(shot_ids):
                raise ValueError(f"Native diagnostic length {len(values)} != batch size {len(shot_ids)}.")
            return np.asarray(values, dtype=np.float32)

        buf = self._buffers.setdefault(output_name, [])
        buf.append(
            {
                "shot_id":      shot_ids[valid_idx].astype(np.int64),
                "window_index": window_indices[valid_idx].astype(np.int64),
                "y_w_emb":      y_w_emb[valid_idx].astype(np.float32),
                "y_l_emb":      y_l_emb[valid_idx].astype(np.float32),
                "native_nrmse": _diagnostic_or_nan(native_nrmse)[valid_idx],
                "native_nmae":  _diagnostic_or_nan(native_nmae)[valid_idx],
                "native_mse":   _diagnostic_or_nan(native_mse)[valid_idx],
                "embedding_gap": ((y_w_emb - y_l_emb) ** 2).mean(axis=1)[valid_idx].astype(np.float32),
            }
        )

        # Flush completed shards eagerly to bound RAM usage.
        self._maybe_flush(output_name)

    # ------------------------------------------------------------------------------------------------------------------
    def finalize(self) -> dict[str, Any]:
        """
        Flush remaining buffered rows and write metadata.json + collection_config.json.

        Returns
        -------
        dict[str, Any]
            Summary: ``{"out_dir": str, "total_windows": {signal_name: int}}``.
        """
        for output_name in list(self._buffers.keys()):
            self._flush(output_name, force=True)

        metadata: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "total_windows": self._totals,
            **self._provenance,
        }
        meta_path = self._staging_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Write the compact reconstruction recipe.  Only the fields needed by
        # reconstruct_dataloader are extracted; the rest lives in metadata.json.
        collection_config: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            **{k: self._provenance[k] for k in _COLLECTION_CONFIG_KEYS if k in self._provenance},
        }
        cc_path = self._staging_dir / "collection_config.json"
        with cc_path.open("w", encoding="utf-8") as f:
            json.dump(collection_config, f, indent=2)

        # Promote the fully-written staging dir to the final location without ever
        # leaving the previous dataset unrecoverable. POSIX cannot atomically
        # replace a non-empty directory in one call, so:
        #   1. rename any existing dataset aside to a unique backup sibling,
        #   2. rename staging into place,
        #   3. delete the backup only after (2) succeeds.
        # A crash between (1) and (2) leaves the old dataset recoverable in the
        # backup dir rather than destroyed; if (2) raises, we restore the backup.
        backup_dir: Path | None = None
        if self._out_dir.exists():
            # overwrite=True is guaranteed here; __init__ raises otherwise.
            backup_dir = self._out_dir.parent / f"{self._staging_dir.name}.backup"
            os.replace(self._out_dir, backup_dir)

        try:
            os.replace(self._staging_dir, self._out_dir)
        except BaseException:
            if backup_dir is not None and not self._out_dir.exists():
                os.replace(backup_dir, self._out_dir)  # restore the previous dataset
            raise

        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)

        logger.info("GPO dataset written to %s | windows per signal: %s", self._out_dir, self._totals)
        return {"out_dir": str(self._out_dir), "total_windows": dict(self._totals)}

    # ------------------------------------------------------------------------------------------------------------------
    def abort(self) -> None:
        """
        Discard the staging directory without promoting it.

        Call this if collection fails before :meth:`finalize` so the temp staging
        dir is not orphaned.  Safe to call more than once and after a successful
        :meth:`finalize` (staging has already been renamed away by then).
        """
        staging = getattr(self, "_staging_dir", None)
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)

    # ------------------------------------------------------------------------------------------------------------------
    def _maybe_flush(self, output_name: str) -> None:
        """Flush a shard for ``output_name`` if the buffer has reached ``shard_size``."""
        buf = self._buffers.get(output_name, [])
        total_rows = sum(len(b["shot_id"]) for b in buf)
        if total_rows >= self._shard_size:
            self._flush(output_name, force=False)

    # ------------------------------------------------------------------------------------------------------------------
    def _flush(self, output_name: str, *, force: bool) -> None:
        """Write one shard for ``output_name`` from the current buffer contents."""
        buf = self._buffers.get(output_name)
        if not buf:
            return

        total_rows = sum(len(b["shot_id"]) for b in buf)
        if (not force) and (total_rows < self._shard_size):
            return

        shot_id      = np.concatenate([b["shot_id"]      for b in buf], axis=0)
        window_index = np.concatenate([b["window_index"] for b in buf], axis=0)
        y_w_emb      = np.concatenate([b["y_w_emb"]      for b in buf], axis=0)
        y_l_emb      = np.concatenate([b["y_l_emb"]      for b in buf], axis=0)
        native_nrmse = np.concatenate([b["native_nrmse"] for b in buf], axis=0)
        native_nmae  = np.concatenate([b["native_nmae"]  for b in buf], axis=0)
        native_mse   = np.concatenate([b["native_mse"]   for b in buf], axis=0)
        embedding_gap = np.concatenate([b["embedding_gap"] for b in buf], axis=0)

        shard_idx = self._shard_counters.get(output_name, 0)
        # Replace slashes in signal names so they are safe as filename parts.
        safe_name = output_name.replace("/", "-").replace("\\", "-")
        shard_path = self._staging_dir / f"{safe_name}__shard_{shard_idx:06d}.npz"
        np.savez_compressed(
            shard_path,
            shot_id=shot_id,
            window_index=window_index,
            y_w_emb=y_w_emb,
            y_l_emb=y_l_emb,
            native_nrmse=native_nrmse,
            native_nmae=native_nmae,
            native_mse=native_mse,
            embedding_gap=embedding_gap,
            emb_dim=np.int64(y_w_emb.shape[1]),
        )

        n = len(shot_id)
        self._totals[output_name] = self._totals.get(output_name, 0) + n
        self._shard_counters[output_name] = shard_idx + 1
        self._buffers[output_name] = []

        logger.debug("Flushed shard %s (%d windows, emb_dim=%d)", shard_path.name, n, y_w_emb.shape[1])
