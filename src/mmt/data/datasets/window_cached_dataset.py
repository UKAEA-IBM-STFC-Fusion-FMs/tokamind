"""
RAM-backed dataset and helpers for cached *window-level* MMT data.

This module provides:

- WindowCachedDataset:
    A simple torch.utils.data.Dataset storing pre-tokenised **windows**
    in memory, ready to be fed directly to MMTCollate.

- materialize_tokenized_split_to_ram():
    A helper that runs the full model-specific transform chain once per
    window on top of a *streaming* dataset (typically a
    TaskModelTransformWrapper) and returns a WindowCachedDataset.

Internally it also defines FlattenedStreamingDataset and a trivial collate
function used only during the parallel caching step.

The goal is to give a clean separation between:

- streaming, shot-based datasets (shot -> iterable of windows), and
- cached, window-based datasets (item -> single window dict),

while keeping a symmetric, user-friendly API in the train scripts.

Note on shuffling
-----------------
When used with a standard torch.utils.data.DataLoader, shuffling is applied
at the *window* level (since this is a map-style dataset).  A batch may
therefore contain windows coming from many different shots.

This differs from WindowStreamedDataset (IterableDataset), where the
DataLoader `shuffle=True` flag is ignored by PyTorch, and shot-level
shuffling must be enabled explicitly by passing `shuffle_shots=True` at
dataset construction time.

"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence
import logging
import os

import psutil
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("mmt.Cache")


# ----------------------------------------------------------------------------- #
# Small RAM helper
# ----------------------------------------------------------------------------- #
def get_ram_gb() -> float:
    """
    Return the current RAM usage of this process in GiB.

    Notes
    -----
    - This uses psutil.Process(os.getpid()).memory_info().rss, i.e. the
      resident set size (RSS) of the *current* Python process.
    - The result is expressed in GiB (1024**3 bytes).
    """
    return psutil.Process(os.getpid()).memory_info().rss / (1024**3)


# ----------------------------------------------------------------------------- #
# In-RAM dataset of tokenised windows
# ----------------------------------------------------------------------------- #
class WindowCachedDataset(Dataset):
    """
    Simple in-RAM dataset of *tokenised windows*.

    Parameters
    ----------
    windows:
        Any iterable of window dictionaries as produced by BuildTokensTransform.
        The iterable is fully materialised into a list at construction time.

    Each item must be a single window dict, with a structure similar to:

    .. code-block:: python

        {
            "shot_id": ...,
            "window_index": ...,
            "emb_chunks": [np.ndarray(D), ...],
            "pos": np.ndarray(L,),          # token positions
            "id": np.ndarray(L,),           # signal IDs
            "mod": np.ndarray(L,),          # modality IDs
            "role": np.ndarray(L,),         # role IDs
            "signal_name": np.ndarray(L,),  # optional string IDs
            "outputs_emb": {signal_id: np.ndarray(D), ...},
            "outputs_shapes": {signal_id: shape, ...},
            "outputs_names": {signal_id: name, ...},
            # optionally (e.g. eval): "output_native" etc.
            ...
        }

    Notes
    -----
    - This dataset is designed to be fed directly to MMTCollate, which in the
      new pipeline expects **one window dict per batch element**.
    - Construction will log a warning if created empty, as downstream code
      typically assumes a non-empty dataset.
    """

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
        streaming_dataset,
        max_windows: int | None = None,
        num_workers_cache: int = 0,
    ) -> "WindowCachedDataset":
        return materialize_tokenized_split_to_ram(
            streaming_dataset=streaming_dataset,
            max_windows=max_windows,
            num_workers_cache=num_workers_cache,
        )


# ----------------------------------------------------------------------------- #
# Collate function used *only* for parallel caching
# ----------------------------------------------------------------------------- #
def _cache_collate_identity(batch):
    """
    Collate function used during parallel caching only.

    The temporary DataLoader used in materialize_tokenized_split_to_ram()
    always has batch_size=1 and wraps the streaming dataset with
    FlattenedStreamingDataset, so that each worker receives *one* list of
    window dicts per batch.

    This function simply unwraps that batch-of-one and returns the list as-is.

    Parameters
    ----------
    batch:
        Sequence with a single element: a list of window dicts.

    Returns
    -------
    list of dict
        The underlying list of window dictionaries.
    """
    return batch[0]


# ----------------------------------------------------------------------------- #
# Wrapper for multiprocessing caching
# ----------------------------------------------------------------------------- #
class FlattenedStreamingDataset(Dataset):
    """
    Wrap a streaming, shot-based dataset so __getitem__ returns a *list* of
    window dicts instead of a generator.

    Motivation
    ----------
    - In the current pipeline, TaskModelTransformWrapper (and similar)
      return, for each shot index, either:
        * a generator / iterable of window dicts, or
        * a single window dict, or
        * None (if the shot yields no valid windows).
    - PyTorch DataLoader with num_workers > 0 **cannot pickle generators**.
    - To cache windows in parallel, workers must receive picklable objects.

    This wrapper normalises that behaviour:

    - If ds[idx] is None        -> return [].
    - If ds[idx] is a dict      -> return [dict].
    - If ds[idx] is iterable    -> fully materialise into a list of dicts.

    That list is picklable and can be safely passed to worker processes.
    """

    def __init__(self, ds: Sequence[Any]) -> None:
        self.ds = ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        item = self.ds[idx]
        if item is None:
            return []

        if isinstance(item, dict):
            return [item]

        return [w for w in item if w is not None]


# ----------------------------------------------------------------------------- #
# Main caching helper
# ----------------------------------------------------------------------------- #
def materialize_tokenized_split_to_ram(
    streaming_dataset: Sequence[Any],
    max_windows: Optional[int] = None,
    num_workers_cache: int = 0,
) -> WindowCachedDataset:
    """
    Materialise a *streaming* MMT dataset into a RAM-backed WindowCachedDataset.

    This function executes the full model-specific transform chain once per
    window:

        ChunkWindowsTransform
        → SelectValidWindows
        → TrimChunksTransform
        → EmbedChunksTransform
        → BuildTokensTransform

    and stores the resulting tokenised window dicts in memory.

    Parameters
    ----------
    streaming_dataset:
        Any dataset-like object that supports __len__ and __getitem__,
        typically a TaskModelTransformWrapper already wrapped with the
        model-specific transforms (chunking, embedding, token building).

        Each __getitem__(idx) is expected to return:
          - a window dict,
          - an iterable of window dicts, or
          - None (no valid windows for that shot).

    max_windows:
        Optional cap on the total number of windows to collect.  Useful for
        debugging or quick experiments on a subset of the data.

    num_workers_cache:
        - If > 0, use a temporary DataLoader with the given number of workers
          to parallelise preprocessing.  Generators are made picklable via
          FlattenedStreamingDataset.
        - If 0, run caching in the main process (deterministic, simpler).

    Returns
    -------
    WindowCachedDataset
        A dataset storing exactly *one window dict per item*, ready for fast
        train with MMTCollate (window-level batches).

    Notes
    -----
    - This is typically called once per split (train/val/test) at the
      beginning of a run when caching is enabled.
    - Logging includes periodic RAM usage (GiB) to help monitor footprint.
    """
    if num_workers_cache > 0:
        flat_windows: List[Dict[str, Any]] = []

        ds_flat = FlattenedStreamingDataset(streaming_dataset)

        loader = DataLoader(
            ds_flat,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers_cache,
            collate_fn=_cache_collate_identity,
        )

        log_interval = 100

        for idx, item in enumerate(loader):
            if item is None:
                continue

            for w in item:  # item is List[window_dict]
                flat_windows.append(w)
                if max_windows is not None and len(flat_windows) >= max_windows:
                    break

            if max_windows is not None and len(flat_windows) >= max_windows:
                break

            if idx % log_interval == 0:
                logger.debug(
                    "Cached %d windows so far (RAM: %.3f GB)",
                    len(flat_windows),
                    get_ram_gb(),
                )

        logger.info(
            "Finished caching %d tokenised windows using %d workers "
            "(Final RAM: %.3f GB)",
            len(flat_windows),
            num_workers_cache,
            get_ram_gb(),
        )
        return WindowCachedDataset(flat_windows)

    # single-process path
    flat_windows: List[Dict[str, Any]] = []
    n_items = len(streaming_dataset)
    log_interval = 100

    for idx in range(n_items):
        item = streaming_dataset[idx]
        if item is None:
            continue

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
                "Single-process cached %d windows (RAM: %.3f GB)",
                len(flat_windows),
                get_ram_gb(),
            )

    logger.info(
        "Materialised %d tokenised windows from %d items "
        "(single process, Final RAM: %.3f GB)",
        len(flat_windows),
        n_items,
        get_ram_gb(),
    )
    return WindowCachedDataset(flat_windows)
