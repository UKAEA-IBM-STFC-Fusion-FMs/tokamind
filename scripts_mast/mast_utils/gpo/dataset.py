"""
scripts_mast.mast_utils.gpo.dataset
====================================

PyTorch Dataset that serves GPO preference pairs stored in schema-v2/v3 .npz
shards for the GPO fine-tuning loop.

Design
------
The dataset is *index-only*: it does not reconstruct the MAST window context
``x`` at load time. Instead it exposes the embedding-space pairs
``(y_w_emb, y_l_emb)`` directly from the shards, so the GPO training loop can
use the same ``embed_mse``-compatible batch dict structure — just with the two
additional keys ``y_w_emb`` and ``y_l_emb`` injected on top.

For the GPO loss the model is called on the *input context* to get its current
prediction ``ŷ_emb = model(x)``, and then the loss is computed against the
stored ``y_w_emb`` / ``y_l_emb``. The input context ``x`` is recovered by
pairing this dataset with a second DataLoader built from the reconstructed MAST
window stream (see :func:`~scripts_mast.mast_utils.gpo.reconstruct.reconstruct_dataloader`).

In *embedding-only mode* (default), only ``y_w_emb`` and ``y_l_emb`` are
loaded from the shards — no MAST context reconstruction is required.  This is
the mode used by the GPO training loop (the loop itself drives the forward
pass through the MAST DataLoader and injects the embedding pairs into the
batch at training time).

Train / val split
-----------------
New collections use the persisted official MAST shot split (mast-shot-split-v1).
Validation receives filter_state fitted on training pairs. No validation samples
contribute to percentile thresholds or pair caps.

For legacy analysis only: ``collection_config.json`` stores ``val_fraction`` (e.g. 0.1). After all
configured pair filters are applied, rows are globally shuffled with a fixed
seed and the final ``val_fraction`` becomes the validation set. The split is
therefore deterministic and is not biased toward particular shard or window
positions.

Usage
-----
::

    from mast_utils.gpo.dataset import GpoPairDataset

    ds_train = GpoPairDataset(gpo_dir="runs/.../gpo_pairs", split="train")
    ds_val   = GpoPairDataset(gpo_dir="runs/.../gpo_pairs", split="val", filter_state=ds_train.filter_state)

    loader_train = DataLoader(ds_train, batch_size=256, shuffle=True)

Each item returned is a dict::

    {
        "shot_id":      int64   — scalar
        "window_index": int64   — scalar
        "y_w_emb":      float32 (D,)  — preferred embedding
        "y_l_emb":      float32 (D,)  — dispreferred embedding
        "signal_name":  str     — safe signal name (slashes replaced with hyphens,
                                   matching the shard filename stem written by GpoPairWriter)
    }
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger("mmt.GPO")


# ======================================================================================================================
class GpoPairDataset(Dataset):
    """
    Map-style dataset over GPO preference-pair shards (schema v2/v3).

    Loads all ``.npz`` shards from ``gpo_dir`` for a given ``signal_name``
    (or for all signals when ``signal_name=None``), applies the train/val
    split, and exposes individual windows as dicts.

    Parameters
    ----------
    gpo_dir : str | Path
        Directory containing the .npz shards, ``metadata.json``, and
        ``collection_config.json``.
    split : str
        ``"train"`` or ``"val"``. The split is determined by the
        ``val_fraction`` stored in ``collection_config.json`` after a fixed-seed
        global shuffle of all retained rows.
    signal_name : str | None
        Name of the output signal to load (e.g. ``"summary-ip"``).
        When ``None``, all signals found in the directory are merged and
        returned as a flat list of windows.  Each item then carries a
        ``"signal_name"`` key so downstream code can identify the origin.
        Default: ``None`` (all signals).
    val_fraction : float | None
        Override the ``val_fraction`` from ``collection_config.json``.
        ``None`` (default) reads the stored value.  Useful for ablations.
    shot_blacklist : set[int] | None
        Shot IDs to exclude entirely.  Any row whose ``shot_id`` is in this
        set is dropped before the train/val split.  Default: ``None`` (no
        exclusions).  Use for pathological shots identified via the shot
        outlier table (panel 23 of ``visualize_gpo_stats.py``).
    min_margin_mse : float | None
        Minimum MSE preference gap ``‖y_w − y_l‖²/D`` a pair must have to
        be retained.  Rows below this threshold are treated as near-perfect
        predictions and dropped (they carry no learning signal).  Applied
        per-row at load time by peeking at the stored embeddings.
        Default: ``None`` (no threshold).  Recommended for task_4-3 where
        ~96% of pairs are near-zero.
    min_window_index : int | None
        Minimum ``window_index`` value (inclusive).  Rows with a temporal
        position below this threshold are dropped.  Useful for task_4-5
        to exclude early-ramp-up windows dominated by transient amplitude
        outliers.  Default: ``None`` (no filter).
    native_nrmse_percentiles : tuple[float, float] | None
        Per-signal lower and upper percentile bounds for schema-v3 decoded
        native-NRMSE diagnostics. Rows outside the inclusive interval are
        dropped. Default: ``None`` (no native-space filter).
    max_pairs_per_shot_percentile : float | None
        Per-signal percentile used to cap retained rows per shot after all
        other filters. Over-cap shots retain an evenly distributed deterministic
        subset ordered by ``window_index``. Default: ``None`` (no cap).
    """

    def __init__(
        self,
        gpo_dir: str | Path,
        split: str = "train",
        signal_name: str | None = None,
        val_fraction: float | None = None,
        shot_blacklist: set[int] | None = None,
        min_margin_mse: float | None = None,
        min_window_index: int | None = None,
        native_nrmse_percentiles: tuple[float, float] | None = None,
        max_pairs_per_shot_percentile: float | None = None,
        filter_state: dict | None = None,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}.")

        self._gpo_dir = Path(gpo_dir)
        self._split = split

        # ------------------------------------------------------------------
        # Load collection config to determine val_fraction and schema.
        # ------------------------------------------------------------------
        cc = self._load_collection_config()
        from .protocol import PROTOCOL, validate_shots

        protocol = cc.get("protocol")
        if protocol is not None and protocol != PROTOCOL:
            raise ValueError(f"Unsupported CGPO protocol: {protocol}")
        shot_splits = None
        if protocol == PROTOCOL:
            shot_splits = cc["contract"]["split_manifest"]["shots"]
            validate_shots(shot_splits)
            if val_fraction not in (None, 0.0):
                raise ValueError("A persisted MAST shot split cannot be overridden by val_fraction.")
            if split == "val" and filter_state is None:
                raise ValueError("Validation requires filter_state fitted on the training pairs.")
        schema_version = cc.get("schema_version", 1)

        if schema_version not in (1, 2, 3):
            raise ValueError(
                f"GpoPairDataset only supports schema versions 1 through 3, got {schema_version!r}. "
                "Re-collect the dataset with run_collect_gpo_pairs.py."
            )
        if schema_version == 1:
            raise ValueError(
                "GpoPairDataset requires embedding-space shards (y_w_emb / y_l_emb arrays). "
                "The dataset in this directory was collected with schema v1 (native-space y_w / y_l). "
                "Re-run run_collect_gpo_pairs.py to generate a v2 or v3 dataset."
            )

        effective_val_fraction = float(val_fraction if val_fraction is not None else cc.get("val_fraction", 0.0))
        if not (0.0 <= effective_val_fraction < 1.0):
            raise ValueError(f"val_fraction must be in [0.0, 1.0), got {effective_val_fraction!r}.")

        self._val_fraction = effective_val_fraction
        self._multi_signal: str = str(cc.get("multi_signal", "joint"))

        # Normalise optional filters.
        self._shot_blacklist: frozenset[int] = frozenset(shot_blacklist) if shot_blacklist else frozenset()
        self._min_margin_mse: float | None = float(min_margin_mse) if min_margin_mse is not None else None
        self._min_window_index: int | None = int(min_window_index) if min_window_index is not None else None
        self._native_nrmse_percentiles = native_nrmse_percentiles
        self._max_pairs_per_shot_percentile = max_pairs_per_shot_percentile
        if native_nrmse_percentiles is not None:
            if len(native_nrmse_percentiles) != 2:
                raise ValueError("native_nrmse_percentiles must contain exactly two values.")
            lo, hi = native_nrmse_percentiles
            if not (0.0 <= lo < hi <= 100.0):
                raise ValueError("native_nrmse_percentiles must satisfy 0 <= lower < upper <= 100.")
            if schema_version < 3:
                raise ValueError("native_nrmse_percentiles requires schema-v3 GPO pairs with native diagnostics.")
        if max_pairs_per_shot_percentile is not None and not (0.0 < max_pairs_per_shot_percentile <= 100.0):
            raise ValueError("max_pairs_per_shot_percentile must be in (0, 100].")

        # ------------------------------------------------------------------
        # Discover shard files for the requested signal(s).
        # ------------------------------------------------------------------
        shard_files = sorted(self._gpo_dir.glob("*.npz"))
        if not shard_files:
            raise FileNotFoundError(
                f"No .npz shards found in {self._gpo_dir}. Run run_collect_gpo_pairs.py to generate the dataset."
            )

        if signal_name is not None:
            safe = signal_name.replace("/", "-").replace("\\", "-")
            shard_files = [f for f in shard_files if f.stem.startswith(f"{safe}__shard_")]
            if not shard_files:
                raise FileNotFoundError(
                    f"No shards found for signal {signal_name!r} in {self._gpo_dir}. "
                    f"Available shards: {[f.name for f in sorted(self._gpo_dir.glob('*.npz'))]}"
                )

        # ------------------------------------------------------------------
        # Load all shards and build a flat row index, then apply the split.
        # ------------------------------------------------------------------
        # Candidate entries retain the metadata needed by every configured
        # filter: (shard_path, row_in_shard, safe_signal_name, shot_id,
        # window_index, native_nrmse). Keeping this in memory avoids reopening
        # shards per row for percentile filtering and per-shot capping.
        n_dropped_blacklist = 0
        n_dropped_margin = 0
        n_dropped_window = 0
        n_dropped_native = 0
        n_dropped_shot_cap = 0
        candidates: list[tuple[Path, int, str, int, int, float]] = []
        native_nrmse_by_signal: dict[str, list[float]] = {}

        allowed_shots = set(shot_splits[split]) if shot_splits else set()
        collection_shots = set(shot_splits["train"]) | set(shot_splits["val"]) if shot_splits else set()
        for shard_path in shard_files:
            # Extract the safe signal name from the filename:
            #   "<safe_signal>__shard_NNNNNN.npz"  →  sname_safe = "<safe_signal>"
            stem = shard_path.stem
            if "__shard_" in stem:
                sname_safe = stem[: stem.rfind("__shard_")]
            else:
                sname_safe = stem
            # When a specific signal_name was requested, override with its safe form.
            sname = signal_name.replace("/", "-").replace("\\", "-") if signal_name is not None else sname_safe

            # Load each shard once to apply per-row filters. ``shot_id`` and
            # ``window_index`` are retained for the subsequent per-shot cap.
            need_embeddings = self._min_margin_mse is not None
            need_native_nrmse = self._native_nrmse_percentiles is not None

            with np.load(shard_path, allow_pickle=False) as npz:
                shot_ids_arr = npz["shot_id"]  # (B,)
                window_idx_arr = npz["window_index"]  # (B,)
                n_rows = len(shot_ids_arr)

                if need_embeddings:
                    yw_arr = npz["y_w_emb"].astype(np.float32)  # (B, D)
                    yl_arr = npz["y_l_emb"].astype(np.float32)  # (B, D)
                    mse_gap_arr = ((yw_arr - yl_arr) ** 2).mean(axis=1)  # (B,)
                else:
                    mse_gap_arr = None
                if need_native_nrmse:
                    native_nrmse_arr = npz["native_nrmse"].astype(np.float32)
                else:
                    native_nrmse_arr = None

            for row_idx in range(n_rows):
                # Split BEFORE fitting any filters: all signals of a shot stay together.
                shot_id = int(shot_ids_arr[row_idx])
                if shot_splits is not None and shot_id not in allowed_shots:
                    if shot_id not in collection_shots:
                        raise ValueError(f"Pair shot {shot_id} is outside the train/val manifest.")
                    continue
                # Filter 1: shot blacklist
                window_index = int(window_idx_arr[row_idx])
                if self._shot_blacklist and shot_id in self._shot_blacklist:
                    n_dropped_blacklist += 1
                    continue
                # Filter 2: minimum window index (exclude early ramp-up)
                if self._min_window_index is not None and window_index < self._min_window_index:
                    n_dropped_window += 1
                    continue
                # Filter 3: minimum MSE preference gap (drop near-zero pairs)
                if need_embeddings and float(mse_gap_arr[row_idx]) < self._min_margin_mse:  # type: ignore[index]
                    n_dropped_margin += 1
                    continue
                native_nrmse = float(native_nrmse_arr[row_idx]) if native_nrmse_arr is not None else float("nan")
                candidates.append((shard_path, row_idx, sname, shot_id, window_index, native_nrmse))
                if np.isfinite(native_nrmse):
                    native_nrmse_by_signal.setdefault(sname, []).append(native_nrmse)

        native_bounds: dict[str, tuple[float, float]] = {}
        if filter_state is not None:
            native_bounds = filter_state["native_bounds"]
        elif self._native_nrmse_percentiles is not None:
            for sname, values in native_nrmse_by_signal.items():
                lo, hi = np.percentile(np.asarray(values), self._native_nrmse_percentiles)
                native_bounds[sname] = (float(lo), float(hi))

        retained_candidates: list[tuple[Path, int, str, int, int, float]] = []
        for candidate in candidates:
            shard_path, row_idx, sname, shot_id, window_index, native_nrmse = candidate
            if self._native_nrmse_percentiles is not None:
                lo, hi = native_bounds.get(sname, (float("nan"), float("nan")))
                if not (np.isfinite(native_nrmse) and lo <= native_nrmse <= hi):
                    n_dropped_native += 1
                    continue
            retained_candidates.append(candidate)

        caps: dict[str, int] = {}
        if self._max_pairs_per_shot_percentile is not None:
            rows_by_signal_shot: dict[tuple[str, int], list[tuple[Path, int, str, int, int, float]]] = {}
            for candidate in retained_candidates:
                rows_by_signal_shot.setdefault((candidate[2], candidate[3]), []).append(candidate)
            counts_by_signal: dict[str, list[int]] = {}
            for (sname, _shot_id), rows in rows_by_signal_shot.items():
                counts_by_signal.setdefault(sname, []).append(len(rows))
            caps = (
                filter_state["caps"]
                if filter_state is not None
                else {
                    sname: max(1, int(np.ceil(np.percentile(counts, self._max_pairs_per_shot_percentile))))
                    for sname, counts in counts_by_signal.items()
                }
            )
            retained_candidates = []
            for (sname, _shot_id), rows in rows_by_signal_shot.items():
                rows.sort(key=lambda candidate: candidate[4])
                if sname not in caps:
                    raise ValueError(f"No training-fitted pair cap for validation signal {sname}")
                cap = caps[sname]
                if len(rows) > cap:
                    indices = np.linspace(0, len(rows) - 1, cap, dtype=int)
                    selected = [rows[i] for i in indices]
                    n_dropped_shot_cap += len(rows) - len(selected)
                else:
                    selected = rows
                retained_candidates.extend(selected)

        all_rows = [
            (path, row_idx, sname)
            for path, row_idx, sname, _shot_id, _window_index, _native_nrmse in retained_candidates
        ]

        self.filter_state = {"native_bounds": native_bounds, "caps": caps}
        if shot_splits is not None:
            self._index = all_rows
        else:
            # Legacy readers retain their original semantics for historical analysis.
            # The training entrypoint rejects these collections explicitly.
            rng = random.Random(42)
            rng.shuffle(all_rows)
            val_n = round(len(all_rows) * effective_val_fraction)
            train_n = len(all_rows) - val_n
            self._index = all_rows[:train_n] if split == "train" else all_rows[train_n:]

        if n_dropped_blacklist or n_dropped_margin or n_dropped_window or n_dropped_native or n_dropped_shot_cap:
            logger.info(
                "GpoPairDataset | dropped %d blacklisted-shot + %d low-margin + %d early-window + %d native-NRMSE + %d shot-cap rows",
                n_dropped_blacklist,
                n_dropped_margin,
                n_dropped_window,
                n_dropped_native,
                n_dropped_shot_cap,
            )
        if native_bounds:
            logger.info("GpoPairDataset | native-NRMSE percentile bounds: %s", native_bounds)
        if caps:
            logger.info("GpoPairDataset | per-signal pairs-per-shot caps: %s", caps)
        logger.info(
            "GpoPairDataset | dir=%s split=%s signal=%s windows=%d val_fraction=%.3f",
            self._gpo_dir,
            split,
            signal_name or "all",
            len(self._index),
            effective_val_fraction,
        )

        # Cache loaded shards to avoid re-reading the same .npz file for each row.
        # The cache is keyed by shard_path (str) and holds the full npz arrays.
        self._shard_cache: dict[str, dict[str, np.ndarray]] = {}

    # ------------------------------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_path, row, sname = self._index[idx]
        arrays = self._load_shard(shard_path)

        return {
            "shot_id": int(arrays["shot_id"][row]),
            "window_index": int(arrays["window_index"][row]),
            "y_w_emb": torch.from_numpy(arrays["y_w_emb"][row].copy()),  # (D,)
            "y_l_emb": torch.from_numpy(arrays["y_l_emb"][row].copy()),  # (D,)
            "signal_name": sname,
        }

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def split(self) -> str:
        """The split this dataset was constructed for: ``"train"`` or ``"val"``."""
        return self._split

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def val_fraction(self) -> float:
        """Effective validation fraction (read from config or overridden at init)."""
        return self._val_fraction

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def multi_signal(self) -> str:
        """Multi-signal strategy from collection config (``"joint"`` or ``"independent"``)."""
        return self._multi_signal

    # ------------------------------------------------------------------------------------------------------------------
    def _load_collection_config(self) -> dict[str, Any]:
        """Load and return ``collection_config.json`` from the GPO directory."""
        cc_path = self._gpo_dir / "collection_config.json"
        if not cc_path.is_file():
            raise FileNotFoundError(
                f"collection_config.json not found in {self._gpo_dir}. "
                "This file is written by run_collect_gpo_pairs.py. "
                "Re-run collection or pass val_fraction explicitly."
            )
        with cc_path.open(encoding="utf-8") as f:
            return json.load(f)

    # ------------------------------------------------------------------------------------------------------------------
    def _load_shard(self, shard_path: Path) -> dict[str, np.ndarray]:
        """Return (possibly cached) array dict for a shard."""
        key = str(shard_path)
        if key not in self._shard_cache:
            npz = np.load(shard_path, allow_pickle=False)
            self._shard_cache[key] = {
                "shot_id": npz["shot_id"],
                "window_index": npz["window_index"],
                "y_w_emb": npz["y_w_emb"],
                "y_l_emb": npz["y_l_emb"],
            }
        return self._shard_cache[key]


# ======================================================================================================================
def build_gpo_dataloaders(
    gpo_dir: str | Path,
    *,
    signal_name: str | None = None,
    batch_size: int = 256,
    num_workers: int = 0,
    val_fraction: float | None = None,
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """
    Convenience factory: build train + val DataLoaders from a GPO pairs directory.

    Parameters
    ----------
    gpo_dir : str | Path
        GPO pairs directory (contains ``collection_config.json`` and ``.npz`` shards).
    signal_name : str | None
        If given, load only the named signal's shards.  Otherwise merges all signals.
    batch_size : int
        Batch size for both loaders.  Default: 256.
    num_workers : int
        Number of DataLoader worker processes.  Default: 0.
    val_fraction : float | None
        Override the stored ``val_fraction``.  ``None`` reads from config.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        ``(train_loader, val_loader)``
    """
    from torch.utils.data import DataLoader

    ds_train = GpoPairDataset(gpo_dir=gpo_dir, split="train", signal_name=signal_name, val_fraction=val_fraction)
    ds_val = GpoPairDataset(
        gpo_dir=gpo_dir,
        split="val",
        signal_name=signal_name,
        val_fraction=val_fraction,
        filter_state=ds_train.filter_state,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    return train_loader, val_loader
