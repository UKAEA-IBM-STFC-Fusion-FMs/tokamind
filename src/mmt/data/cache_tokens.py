# mmt/data/cache_tokens.py

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
from torch.utils.data import DataLoader, Dataset
import psutil
import os

import logging

logger = logging.getLogger("mmt.Cache")


# ----------------------------------------------------------------------------- #
# Dataset class to cache
# ----------------------------------------------------------------------------- #
class TokenizedWindowDataset(Dataset):
    """
    Simple in-RAM dataset of *tokenized windows*.

    Each item is a single window dict as produced by BuildTokensTransform, e.g.:

        {
            "shot_id": ...,
            "window_index": ...,
            "emb_chunks": [np.ndarray(D), ...],
            "pos": np.ndarray(L,),
            "id": np.ndarray(L,),
            "mod": np.ndarray(L,),
            "role": np.ndarray(L,),
            "signal_name": np.ndarray(L,),
            "outputs_emb": {signal_id: np.ndarray(D), ...},
            "outputs_shapes": {signal_id: shape, ...},
            "outputs_names": {signal_id: name, ...},
            # optionally (eval): "output_native" kept by EmbedChunks + collate
            ...
        }

    This dataset is designed to be fed directly to MMTCollate, which expects
    each batch element to be either:

      • a single window dict, or
      • an iterable of window dicts (for shot-level grouping).

    Here we choose the simpler option: *one window per item*.
    """

    def __init__(self, windows: Iterable[Dict[str, Any]]) -> None:
        self._windows: List[Dict[str, Any]] = list(windows)
        if len(self._windows) == 0:
            logger.warning(
                "[TokenizedWindowDataset] Created with 0 windows. "
                "Downstream collate may raise on empty batches."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._windows[idx]


# ----------------------------------------------------------------------------- #
# Helper to monitor RAM usage
# ----------------------------------------------------------------------------- #
def _get_ram_gb() -> float:
    """Return current RAM usage of this process in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


# ----------------------------------------------------------------------------- #
# Collate function used *only* for parallel caching
# Must be top-level so that multiprocessing can pickle it
# ----------------------------------------------------------------------------- #
def _cache_collate_identity(batch):
    """
    Collate function that simply unwraps the batch-of-one produced during
    parallel caching. Required to avoid PyTorch’s default batching behaviour.
    """
    return batch[0]


# ----------------------------------------------------------------------------- #
# Helper dataset wrapper for multiprocessing caching
# ----------------------------------------------------------------------------- #
class FlattenedStreamingDataset:
    """
    Wraps a TaskModelTransformWrapper so that __getitem__ returns a *list*
    of window dicts instead of a generator.

    - PyTorch DataLoader with num_workers > 0 **cannot pickle generators**.
    - TaskModelTransformWrapper normally returns a generator per shot.
    - This wrapper forces generator materialisation in the main process
      and returns a list, which *is* picklable.
    """

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        if item is None:
            return []

        # If streaming dataset returns a dict → wrap it
        if isinstance(item, dict):
            return [item]

        # Otherwise item is an iterable (generator/list) → fully materialize
        return [w for w in item if w is not None]


# ----------------------------------------------------------------------------- #
# Main caching helper
# ----------------------------------------------------------------------------- #
def materialize_tokenized_split_to_ram(
    streaming_dataset,
    max_windows: Optional[int] = None,
    num_workers_cache: int = 0,
):
    """
    Materialise a *streaming* MMT dataset (TaskModelTransformWrapper + transforms)
    into a RAM-backed TokenizedWindowDataset.

    This function executes the full model-specific transform chain once per window:

        ChunkWindowsTransform
        → SelectValidWindows
        → TrimChunksTransform
        → EmbedChunksTransform
        → BuildTokensTransform

    and stores the resulting tokenized window dicts in memory.

    Why materialisation?
    --------------------
    The streaming dataset applies all transforms at every __getitem__ call.
    For training/fine-tuning we want to pay preprocessing cost *once* and then
    reuse the tokenized windows efficiently during training epochs.

    Parallel caching
    ----------------
    If num_workers_cache > 0, a temporary DataLoader is used to process windows
    in parallel. Because multiprocessing cannot pickle generators, we wrap the
    dataset with FlattenedStreamingDataset so that each worker receives a list
    of window dicts instead of a generator.

    Parameters
    ----------
    streaming_dataset : Dataset
        A TaskModelTransformWrapper or similar streaming dataset whose
        __getitem__ returns either a dict (single window) or a generator/list
        of window dicts.

    max_windows : int or None
        Optional cap on collected windows. Useful for debugging smaller subsets.

    num_workers_cache : int
        >0  → enable parallel preprocessing using DataLoader workers
        0   → run caching in the main process (deterministic, slower)

    Returns
    -------
    TokenizedWindowDataset
        A dataset storing exactly *one window dict per item*, ready for fast
        training with MMTCollate.
    """
    # ------------------------------------------------------------------ #
    # Parallel path — DataLoader with workers (fastest)
    # ------------------------------------------------------------------ #
    if num_workers_cache > 0:
        flat_windows: List[Dict[str, Any]] = []

        # Wrap streaming dataset so workers receive lists, not generators
        ds_flat = FlattenedStreamingDataset(streaming_dataset)

        loader = DataLoader(
            ds_flat,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers_cache,
            collate_fn=_cache_collate_identity,
        )

        log_interval = 100  # log RAM every 100 windows

        for idx, item in enumerate(loader):
            if item is None:
                continue

            # item is now List[window_dict]
            for w in item:
                flat_windows.append(w)

                # Optional upper bound for debugging
                if max_windows is not None and len(flat_windows) >= max_windows:
                    break

            if max_windows is not None and len(flat_windows) >= max_windows:
                break

            # Periodic RAM logging
            if idx % log_interval == 0:
                logger.debug(
                    "[TokenCache] Cached %d windows so far (RAM: %.3f GB)",
                    len(flat_windows),
                    _get_ram_gb(),
                )

        logger.info(
            "[TokenCache] Finished caching %d tokenized windows using %d workers (Final RAM: %.3f GB)",
            len(flat_windows),
            num_workers_cache,
            _get_ram_gb(),
        )
        return TokenizedWindowDataset(flat_windows)

    # ------------------------------------------------------------------ #
    # Single-process path — fallback
    # ------------------------------------------------------------------ #
    flat_windows: List[Dict[str, Any]] = []
    n_items = len(streaming_dataset)
    log_interval = 100

    for idx in range(n_items):
        item = streaming_dataset[idx]
        if item is None:
            continue

        # Normalize → iterable of window dicts
        if isinstance(item, dict):
            item = [item]

        for w in item:
            if w is None:
                continue
            flat_windows.append(w)

            if max_windows is not None and len(flat_windows) >= max_windows:
                break

        if max_windows is not None and len(flat_windows) >= max_windows:
            break

        if idx % log_interval == 0:
            logger.debug(
                "[TokenCache] Single-process cached %d windows (RAM: %.3f GB)",
                len(flat_windows),
                _get_ram_gb(),
            )

    logger.info(
        "[TokenCache] Materialized %d tokenized windows from %d items (single process, Final RAM: %.3f GB)",
        len(flat_windows),
        n_items,
        _get_ram_gb(),
    )
    return TokenizedWindowDataset(flat_windows)
