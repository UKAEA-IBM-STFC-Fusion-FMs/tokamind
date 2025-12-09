"""
Streaming, window-level dataset for the MMT pipeline.

This module provides:

- WindowStreamedDataset:
    An `IterableDataset` that wraps a *shot-based* dataset (typically a
    `TaskModelTransformWrapper` + model-specific transforms) and exposes a
    flat stream of **window dicts**, one per iteration.

Motivation
----------
In the current pipeline, the baseline bridge + transforms operate at the
level of **shots**:

    shot_idx → iterable of window dicts (or a single window dict, or None)

For training, however, we want to think purely in terms of *windows*:

    - `batch_size` is "number of windows per batch"
    - collate operates on List[window_dict]
    - caching and non-caching paths share the same semantics

`WindowStreamedDataset` provides this bridge: it flattens a shot-based
dataset into an iterable of window-level samples, without materialising
all windows in memory.

Usage
-----
Typical usage in a training script:

.. code-block:: python

    from mmt.data.datasets.window_streamed_dataset import WindowStreamedDataset
    from mmt.data.datasets.window_cached_dataset import WindowCachedDataset

    if cfg.cache.enabled:
        dataset = WindowCachedDataset(...)
    else:
        dataset = WindowStreamedDataset(task_model_wrapper, shuffle_shots=True)

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,   # always: windows per batch
        collate_fn=MMTCollate(...),
        num_workers=cfg.train.num_workers,
    )

Notes
-----
- This dataset is *streaming*: it does not store windows in RAM, it only
  keeps a reference to the underlying shot dataset.
- Each yielded item is a single window dict as produced by
  `BuildTokensTransform`.
- `shot_id` and `window_index` fields are preserved and can be used for
  evaluation / grouping, but they are treated as metadata.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Sequence
import logging
import random

from torch.utils.data import IterableDataset, get_worker_info

logger = logging.getLogger("mmt.Data")


class WindowStreamedDataset(IterableDataset):
    """
    Flatten a shot-based dataset into a stream of window-level samples.

    Parameters
    ----------
    shot_dataset:
        A sequence-like dataset such that `shot_dataset[idx]` returns:
          - a single window dict, or
          - an iterable of window dicts (generator / list / tuple), or
          - None (if the shot yields no valid windows).

        In practice this is usually a `TaskModelTransformWrapper` wrapped
        with the model-specific transforms:

            ChunkWindowsTransform
            → SelectValidWindowsTransform
            → TrimChunksTransform
            → EmbedChunksTransform
            → BuildTokensTransform

    shuffle_shots:
        If True, the order of shot indices is shuffled *within each worker's
        slice* on every call to `__iter__`.  This gives a simple form of
        shot-level shuffling without materialising all windows.

        Note PyTorch ignores `shuffle=True` in DataLoader when used with IterableDataset.
        Shot-level shuffling is available only when `shuffle_shots=True` is passed
        at construction time.  Window-level shuffling is not performed in streaming
        mode.

    seed:
        Optional base seed used when `shuffle_shots=True` to initialise a
        per-worker Random instance.  If None, a deterministic default seed
        (0) is used.  You can change this between epochs by constructing a
        new dataset instance, or by passing epoch-dependent seeds.

    Yields
    ------
    dict
        A single window dictionary per iteration, with structure compatible
        with `MMTCollate` and the model.  All metadata fields such as
        "shot_id" and "window_index" are preserved untouched.

    Notes
    -----
    - This class is designed to be paired with `WindowCachedDataset` so that
      both cached and streaming paths expose the same semantics:
        * item = window dict
        * batch_size = number of windows per batch
    """

    def __init__(
        self,
        shot_dataset: Sequence[Any],
        shuffle_shots: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.shot_dataset = shot_dataset
        self.shuffle_shots = shuffle_shots
        self.seed = 0 if seed is None else int(seed)

    def _iter_shot_indices(self) -> Iterable[int]:
        """
        Yield the shot indices assigned to the current worker.

        Uses the standard PyTorch pattern to split the global index range
        across multiple workers when used with a DataLoader(num_workers>0).
        """
        n_shots = len(self.shot_dataset)
        if n_shots == 0:
            return []

        worker = get_worker_info()
        if worker is None:
            # Single-process data loading
            start, end = 0, n_shots
        else:
            # Multi-process: even split of the index range among workers
            num_workers = worker.num_workers
            worker_id = worker.id
            per_worker = (n_shots + num_workers - 1) // num_workers  # ceil div
            start = worker_id * per_worker
            end = min(start + per_worker, n_shots)

        indices = list(range(start, end))

        if self.shuffle_shots and indices:
            # Per-worker deterministic RNG
            # We offset the seed by worker ID if present, so different workers
            # still see different permutations of their local slice.
            worker = get_worker_info()
            worker_id = 0 if worker is None else worker.id
            rng = random.Random(self.seed + worker_id)
            rng.shuffle(indices)

        return indices

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over all windows in the underlying shot dataset.

        For each shot index assigned to the current worker:

        - If shot_dataset[idx] is None:
              skip.
        - If it returns a dict:
              yield that single window dict.
        - If it returns an iterable:
              iterate and yield each non-None window dict.

        This method never materialises all windows at once; it yields them
        lazily, one by one.
        """
        for idx_shot in self._iter_shot_indices():
            try:
                item = self.shot_dataset[idx_shot]
            except Exception:
                logger.exception(
                    "[WindowStreamedDataset] Error reading shot index %d", idx_shot
                )
                continue

            if item is None:
                continue

            # Single-window case: directly yield the dict
            if isinstance(item, dict):
                yield item
                continue

            # General case: iterable of windows
            try:
                for w in item:
                    if w is None:
                        continue
                    yield w
            except TypeError:
                # item is not iterable – this is likely a bug in the upstream
                # dataset; we log it to help debugging.
                logger.error(
                    "[WindowStreamedDataset] Expected dict or iterable of dicts, "
                    "got type %s for shot index %d",
                    type(item),
                    idx_shot,
                )
                continue

    def __len__(self) -> int:
        """
        Approximate epoch length for compatibility with code that calls len().

        We return the number of *shots* in the underlying dataset. This is not
        equal to the number of window-level batches, but is sufficient for:
          - logging,
          - approximate scheduling (warmup, total_steps),
          - code that expects a finite length for IterableDataset.

        For exact, window-level control over steps per epoch you should prefer
        the cached (map-style) path, where len(dataset) == num_windows.
        """
        return len(self.shot_dataset)
