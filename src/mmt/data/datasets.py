"""mmt.data.window_cached_dataset

RAM-backed dataset and helpers for cached *window-level* MMT data.

This module provides:

- WindowCachedDataset:
    A simple torch.utils.data.Dataset storing pre-tokenised **windows**
    in memory, ready to be fed directly to MMTCollate.

- materialize_tokenized_split_to_ram():
    A helper that runs the full model-specific transform chain once per
    window on top of a *shot-level* dataset adapter and returns a
    WindowCachedDataset.

Internally it also defines FlattenedStreamingDataset and a trivial collate
function used only during the parallel caching step.

The goal is to keep a clean separation between:

- shot-level datasets (shot -> iterable of windows), and
- cached, window-level datasets (index -> single window dict),

while keeping a symmetric, user-friendly API in the train scripts.

Notes on shuffling
------------------
- WindowCachedDataset is a map-style Dataset. When used with a standard
  DataLoader, shuffling is applied at the *window* level.
- During caching/materialisation itself, we can optionally shuffle *shots*
  (the order of indices into the shot-level dataset adapter) to avoid bias
  when using max_windows caps.

Dtype / memory
--------------
We accept only:
  - None
  - "float16"
  - "float32"

The dtype cast is applied only to cached *embedding* arrays (token embeddings
and output embeddings), never to native outputs.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import logging
import os
import random

import numpy as np
import psutil
from torch.utils.data import DataLoader, Dataset, IterableDataset

logger = logging.getLogger("mmt.Cache")

# We print the number of total preprocessed windows every _LOG_INTERVAL windows.
_LOG_INTERVAL = 50000

# -----------------------------------------------------------------------------
# Small RAM helper
# -----------------------------------------------------------------------------


def get_ram_gb() -> float:
    """Return the current RAM usage of this process in GiB."""
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


# -----------------------------------------------------------------------------
# Dtype helpers (cache only)
# -----------------------------------------------------------------------------


def _normalize_cache_dtype(dtype: Optional[str]) -> Optional[np.dtype]:
    """Normalize a user-provided dtype spec.

    Accepted values
    ---------------
    - None
    - "float16"
    - "float32"

    Returns
    -------
    np.dtype | None
        The normalized numpy dtype, or None meaning "do not cast".
    """

    if dtype is None:
        return None

    s = dtype.strip().lower()
    if s == "float16":
        return np.dtype(np.float16)
    if s == "float32":
        return np.dtype(np.float32)

    raise ValueError(
        f"Unsupported cache dtype={dtype!r}. Expected 'float16' or 'float32' (or null)."
    )


def _cast_window_embeddings_inplace(window: Dict[str, Any], *, dtype: np.dtype) -> None:
    """Cast cached *embedding* arrays in-place.

    This intentionally only touches *embeddings*:
    - token embeddings (window['emb_chunks'])
    - output embeddings (window['output_emb'])

    Native outputs (window['output'] values) are not modified.
    """

    # Token embeddings (ragged list)
    emb_chunks = window.get("emb_chunks")
    if isinstance(emb_chunks, list):
        for i, arr in enumerate(emb_chunks):
            if arr is None:
                continue
            emb_chunks[i] = np.asarray(arr, dtype=dtype)

    # Output embeddings (dict sid -> embedding)
    out_emb = window.get("output_emb")
    if isinstance(out_emb, dict):
        for sid, arr in list(out_emb.items()):
            if arr is None:
                continue
            out_emb[sid] = np.asarray(arr, dtype=dtype)

    # (Defensive) legacy fields if present
    emb_out = window.get("embedded_output")
    if isinstance(emb_out, dict):
        for sid, arr in list(emb_out.items()):
            if arr is None:
                continue
            emb_out[sid] = np.asarray(arr, dtype=dtype)


# -----------------------------------------------------------------------------
# In-RAM dataset of tokenised windows
# -----------------------------------------------------------------------------


class WindowCachedDataset(Dataset):
    """Simple in-RAM dataset of *tokenised windows*."""

    def __init__(self, windows: Iterable[Dict[str, Any]]) -> None:
        self._windows: List[Dict[str, Any]] = list(windows)
        if len(self._windows) == 0:
            logger.warning(
                "[WindowCachedDataset] Created with 0 windows. "
                "Downstream collate / loaders may raise on empty datasets."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._windows[idx]

    @classmethod
    def from_streaming(
        cls,
        streaming_dataset: Any,
        max_windows: int | None = None,
        num_workers_cache: int = 0,
        *,
        shuffle_shots: bool = False,
        seed: int = 0,
        dtype: Optional[str] = None,
    ) -> "WindowCachedDataset":
        """Create a WindowCachedDataset from an IterableDataset.

        Parameters
        ----------
        streaming_dataset:
            An IterableDataset that yields windows (e.g., TaskModelTransformWrapperIterable).
        max_windows:
            Optional cap on number of windows to cache.
        num_workers_cache:
            Number of DataLoader workers for parallel caching.
        shuffle_shots:
            If True, shuffle windows before caching.
        seed:
            Random seed for shuffling (only used if shuffle_shots=True).
        dtype:
            Optional dtype for cached embeddings ("float16" or "float32").

        Returns
        -------
        WindowCachedDataset
            Materialized in-RAM dataset.
        """
        return materialize_tokenized_split_to_ram(
            streaming_dataset=streaming_dataset,
            max_windows=max_windows,
            num_workers_cache=num_workers_cache,
            shuffle_shots=shuffle_shots,
            seed=seed,
            dtype=dtype,
        )


# -----------------------------------------------------------------------------
# Shot-batched wrapper for efficient IPC during parallel caching
# -----------------------------------------------------------------------------


class _ShotBatchedIterableDataset(IterableDataset):
    """Wrapper that batches windows by shot to reduce IPC overhead.

    Instead of yielding individual windows (causing many small pickle operations),
    this collects all windows from each shot and yields them as a list.
    This matches the old map-style dataset behavior where each worker returned
    List[window_dict] per shot, dramatically reducing serialization overhead.
    """

    def __init__(self, base_iterable: IterableDataset):
        super().__init__()
        self.base_iterable = base_iterable

    def __iter__(self):
        current_shot_id = None
        current_batch = []

        for window in self.base_iterable:
            shot_id = window.get("shot_id")

            # If shot changed and we have accumulated windows, yield the batch
            if shot_id != current_shot_id and current_batch:
                yield current_batch
                current_batch = []

            current_shot_id = shot_id
            current_batch.append(window)

        # Yield final batch if any
        if current_batch:
            yield current_batch


# -----------------------------------------------------------------------------
# Collate function used for parallel caching
# -----------------------------------------------------------------------------


def _cache_collate_identity(batch):
    """Identity collate for caching (batch_size=1, just unwrap the single item)."""
    return batch[0]


# -----------------------------------------------------------------------------
# Main caching helper
# -----------------------------------------------------------------------------


def materialize_tokenized_split_to_ram(
    *,
    streaming_dataset: Any,
    max_windows: Optional[int] = None,
    num_workers_cache: int = 0,
    shuffle_shots: bool = False,
    seed: int = 0,
    dtype: Optional[str] = None,
) -> WindowCachedDataset:
    """Materialise windows from an IterableDataset into a RAM-backed window dataset.

    This function uses DataLoader with multiple workers to parallelize window
    processing from an IterableDataset. Each worker processes a subset of shots,
    and windows are collected into RAM. Shuffling happens after caching.

    Parameters
    ----------
    streaming_dataset:
        An IterableDataset that yields windows directly (e.g., TaskModelTransformWrapperIterable).
        The IterableDataset's __iter__ method handles worker splitting internally.
    max_windows:
        Optional cap on number of windows to cache.
    num_workers_cache:
        Number of DataLoader workers for parallel processing. Each worker processes
        a subset of shots. Higher values improve throughput (e.g., 32 workers).
    shuffle_shots:
        If True, shuffle windows AFTER caching (uses seed for reproducibility).
        Note: IterableDataset doesn't support DataLoader shuffle, so shuffling
        happens post-caching on the materialized list.
    seed:
        Random seed for post-caching shuffle (only used if shuffle_shots=True).
    dtype:
        Optional dtype for cached embeddings ("float16" or "float32").

    Returns
    -------
    WindowCachedDataset
        In-RAM dataset of windows.

    Notes
    -----
    - Parallelization: Workers process different shots simultaneously via IterableDataset
    - Shuffling: Post-caching shuffle (fast, operates on already-processed windows)
    - Load balancing: IterableDataset.__iter__ handles worker splitting internally
    """

    dtype_np = _normalize_cache_dtype(dtype)

    # Note on shuffling: IterableDataset doesn't support DataLoader shuffling.
    # Shuffling must be configured in the IterableDataset itself (e.g., shuffle_windows=True).
    # The shuffle_shots parameter here will shuffle the final cached list instead.

    def _iter_windows() -> Iterable[Dict[str, Any]]:
        # Use DataLoader for parallel iteration if num_workers_cache > 0
        if num_workers_cache > 0:
            # Wrap IterableDataset to batch windows by shot for efficient IPC
            batched_dataset = _ShotBatchedIterableDataset(streaming_dataset)
            loader = DataLoader(
                batched_dataset,
                batch_size=1,
                num_workers=int(num_workers_cache),
                collate_fn=_cache_collate_identity,
                prefetch_factor=1,
                persistent_workers=False,
            )
            # Each batch is a list of windows from one shot
            for window_list in loader:
                for window in window_list:
                    yield window
        else:
            # Single-process: iterate directly
            yield from streaming_dataset

    flat_windows: List[Dict[str, Any]] = []

    for n, w in enumerate(_iter_windows(), start=1):
        if dtype_np is not None:
            _cast_window_embeddings_inplace(w, dtype=dtype_np)

        flat_windows.append(w)

        if max_windows is not None and n >= int(max_windows):
            break

        if n % _LOG_INTERVAL == 0:
            logger.info(
                "Cached %d windows so far (RAM: %.3f GB)",
                len(flat_windows),
                get_ram_gb(),
            )

    # Shuffle after caching if requested (since IterableDataset doesn't support DataLoader shuffle)
    if shuffle_shots and flat_windows:
        logger.info("Shuffling %d cached windows (seed=%d)...", len(flat_windows), seed)
        random.Random(int(seed)).shuffle(flat_windows)

    logger.info(
        "Materialised %d tokenised windows (num_workers_cache=%d, shuffle_shots=%s, dtype=%s, Final RAM: %.3f GB)",
        len(flat_windows),
        int(num_workers_cache),
        bool(shuffle_shots),
        str(dtype) if dtype is not None else "none",
        get_ram_gb(),
    )

    return WindowCachedDataset(flat_windows)
