"""Partitioned HDF5 handle pool for ``.mpco`` files.

A recorder output consists of N partition files (``results.part-0.mpco``,
``results.part-1.mpco``, …). The library opens and closes an
``h5py.File`` per query today, which is measurable overhead on datasets
with many partitions and many queries. ``Hdf5PartitionPool`` holds open
handles and hands them back on demand.

The Phase 1 landing introduces the class with ``pool_size=0`` as the
default, which preserves the current open-per-call semantics — this is a
*no-op* wrapper on the default path. The pool becomes load-bearing in
Phase 2 once the query engines consume it, at which point
``MPCODataSet`` will flip the default to ``min(16, n_partitions)``.

This class is **thread-unsafe by design**. h5py is not thread-safe
without SWMR; parallelism in the library is process-level via
``concurrent.futures.ProcessPoolExecutor`` (see ``AggregationEngine``).

Performance notes
-----------------
- ``pool_size=0``: the pool opens a fresh ``h5py.File`` per ``open()`` /
  ``with_partition()`` call and closes it when the caller closes it.
  Behaviorally identical to a plain ``h5py.File(path, "r")``.
- ``pool_size>0``: up to ``pool_size`` handles are kept open. The LRU
  policy evicts the least-recently-used handle when the cap is hit.
- ``close_all()`` releases every open handle and clears the LRU. Call on
  script teardown or ``MPCODataSet.__exit__``.
"""
from __future__ import annotations

import logging
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import h5py


logger = logging.getLogger(__name__)


class Hdf5PartitionPool:
    """Owns the open ``h5py.File`` handles for one recorder output.

    Parameters
    ----------
    partition_paths : dict[int, Path | str]
        Mapping from partition index to the absolute path of that partition's
        ``.mpco`` file. Typically comes from
        ``ModelInfo._get_file_list_for_results_name(extension='mpco')``.
    pool_size : int, optional
        Maximum number of handles kept open. ``0`` (the default) disables
        pooling entirely — every ``open()`` opens a fresh handle and the
        caller is responsible for closing it. Any positive value enables
        an LRU cache of up to ``pool_size`` handles.

    Thread-safety
    -------------
    Not thread-safe. h5py is not safe for concurrent access without SWMR
    mode; use process-level parallelism instead.
    """

    __slots__ = ("_paths", "_pool_size", "_lru")

    def __init__(self, partition_paths: dict[int, Path | str], pool_size: int = 0):
        if pool_size < 0:
            raise ValueError(f"pool_size must be >= 0, got {pool_size}")

        self._paths: dict[int, Path] = {
            int(idx): Path(p) for idx, p in partition_paths.items()
        }
        self._pool_size: int = int(pool_size)
        # OrderedDict so we can move items to the end on access (LRU).
        self._lru: "OrderedDict[int, h5py.File]" = OrderedDict()

    def __repr__(self) -> str:
        return (
            f"Hdf5PartitionPool(n_partitions={len(self._paths)}, "
            f"pool_size={self._pool_size}, n_open={len(self._lru)})"
        )

    def __len__(self) -> int:
        """Number of registered partitions (not the number of open handles)."""
        return len(self._paths)

    def __contains__(self, partition_idx: object) -> bool:
        """True iff ``partition_idx`` is a known partition index."""
        return partition_idx in self._paths

    @property
    def pool_size(self) -> int:
        """The maximum number of simultaneously-open handles."""
        return self._pool_size

    @property
    def n_open(self) -> int:
        """The number of handles currently open in the LRU."""
        return len(self._lru)

    @property
    def partition_indices(self) -> list[int]:
        """Sorted list of registered partition indices."""
        return sorted(self._paths)

    def path_for(self, partition_idx: int) -> Path:
        """Return the absolute path to partition ``partition_idx``.

        Raises
        ------
        KeyError
            If ``partition_idx`` is not a registered partition.
        """
        try:
            return self._paths[int(partition_idx)]
        except KeyError:
            raise KeyError(
                f"Unknown partition index {partition_idx!r}; "
                f"known indices: {self.partition_indices}"
            ) from None

    def open(self, partition_idx: int) -> h5py.File:
        """Return an open ``h5py.File`` for ``partition_idx``.

        With ``pool_size=0`` every call opens a fresh handle; the caller
        is responsible for closing it (or using the ``with_partition``
        context manager, which handles that automatically).

        With ``pool_size>0`` a cached handle is returned if available;
        otherwise a new one is opened and added to the LRU. The caller
        must **not** close a pooled handle manually — use ``close_all()``
        (or drop the pool) to release handles.

        Raises
        ------
        KeyError
            If ``partition_idx`` is not a registered partition.
        OSError
            If the underlying file cannot be opened.
        """
        path = self.path_for(partition_idx)

        if self._pool_size == 0:
            # Open-per-call — Phase-1 default preserves legacy behavior.
            return h5py.File(path, "r")

        # Pooled path.
        key = int(partition_idx)
        if key in self._lru:
            self._lru.move_to_end(key)  # mark as most recently used
            return self._lru[key]

        handle = h5py.File(path, "r")
        self._lru[key] = handle
        if len(self._lru) > self._pool_size:
            evicted_key, evicted_handle = self._lru.popitem(last=False)
            try:
                evicted_handle.close()
            except Exception:  # pragma: no cover - defensive close
                logger.debug("Failed to close evicted handle for partition %s", evicted_key)
        return handle

    @contextmanager
    def with_partition(self, partition_idx: int) -> Iterator[h5py.File]:
        """Context manager that yields an open ``h5py.File``.

        Under ``pool_size=0`` the handle is closed on block exit. Under
        ``pool_size>0`` the handle stays resident in the LRU and the
        block exit is a no-op (close happens on eviction or
        ``close_all``).

        Example
        -------
        >>> with pool.with_partition(0) as h5:
        ...     nodes = h5["/MODEL_STAGE[1]/MODEL/NODES"][...]
        """
        handle = self.open(partition_idx)
        try:
            yield handle
        finally:
            if self._pool_size == 0:
                handle.close()

    def close_all(self) -> None:
        """Close every handle currently held in the LRU and clear it.

        Safe to call multiple times. Has no effect on ``pool_size=0``
        pools (which hold no handles).
        """
        while self._lru:
            _, handle = self._lru.popitem(last=False)
            try:
                handle.close()
            except Exception:  # pragma: no cover - defensive close
                logger.debug("Failed to close pooled handle")
