"""
Streaming, window-level dataset for the MMT pipeline.

This module provides:

- WindowStreamedDataset:
    An IterableDataset that wraps a *shot-level* dataset adapter and exposes a
    flat stream of **window dicts**, one per iteration.

Motivation
----------
In the MMT pipeline, dataset adapters and transforms often operate at the
level of **shots**:

    shot_idx -> iterable of window dicts (or a single window dict, or None)

For training, however, we want to think purely in terms of *windows*:

    - batch_size is "number of windows per batch"
    - collate operates on List[window_dict]
    - caching and non-caching paths share the same semantics

WindowStreamedDataset provides this bridge: it flattens a shot-level dataset
into an iterable of window-level samples, without materialising all windows
in memory.

Shot-level adapter contract
---------------------------
The wrapped `shot_dataset` must support __len__ and __getitem__. Each
__getitem__(idx) must return one of:

  - a single window dict,
  - an iterable of window dicts (list/tuple/generator),
  - None (if the shot yields no valid windows).

Each yielded item is expected to be a window dict compatible with MMTCollate
and the model (typically produced by BuildTokensTransform).

Notes
-----
- This dataset is streaming: it does not store windows in RAM.
- DataLoader(shuffle=True) is ignored for IterableDataset by PyTorch.
  If you want shuffling in streaming mode, enable shot-level shuffling with
  shuffle_shots=True (window-level shuffling is not performed).
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Sequence
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

        In practice this is usually a dataset adapter already wrapped
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
        new dataset instance. When using persistent workers, this class also
        mixes an internal epoch counter into the shuffle seed so that shot
        order changes between epochs even if the same dataset instance is
        reused across epochs.

    Yields
    ------
    dict
        A single window dictionary per iteration, with structure compatible
        with `MMTCollate` and the model.  All metadata fields such as
        "shot_id" and "window_index" are preserved untouched.

    Notes
    -----
    - WindowStreamedDataset is an IterableDataset. __len__ returns the number of shots,
      NOT the number of windows. In streaming train mode, this value is ignored,
      and epoch length is controlled by loader.streaming.batches_per_epoch.
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

        # Epoch counter used to make streaming shot shuffling change between epochs.
        # This is important when DataLoader(persistent_workers=True) is used, because
        # otherwise each worker would see the same shot order every epoch.
        self._epoch: int = 0

    def _iter_shot_indices(self, *, epoch: int) -> Sequence[int]:
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
            # Per-worker deterministic RNG.
            #
            # NOTE: With persistent workers, each worker's dataset instance is reused
            # across epochs. If we seeded only with (self.seed + worker_id), each
            # worker would see the exact same shot order every epoch.
            #
            # We therefore mix in an epoch counter to force epoch-to-epoch reshuffling.
            worker_id = 0 if worker is None else worker.id
            # Large odd constant (prime) to reduce collisions for small epoch values.
            rng_seed = self.seed + worker_id + epoch * 1_000_003
            rng = random.Random(rng_seed)
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
        # WindowStreamedDataset.__iter__

        epoch = self._epoch
        self._epoch += 1

        for idx_shot in self._iter_shot_indices(epoch=epoch):
            try:
                item = self.shot_dataset[idx_shot]
            except Exception:
                logger.exception(
                    "[WindowStreamedDataset] Error reading shot index %d", idx_shot
                )
                continue

            if item is None:
                continue

            # Allow single-window dict for robustness (cheap, avoids iter(dict) bug)
            if isinstance(item, dict):
                yield item
                continue

            # Strict: must be iterable of window dicts
            try:
                it = iter(item)
            except TypeError as e:
                raise TypeError(
                    f"[WindowStreamedDataset] Expected iterable of window dicts (or a dict/None), "
                    f"got {type(item)} for shot index {idx_shot}"
                ) from e

            for w in it:
                if w is not None:
                    yield w

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
