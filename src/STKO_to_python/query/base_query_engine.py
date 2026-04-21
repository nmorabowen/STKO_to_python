"""BaseResultsQueryEngine — abstract template for result-set fetches.

Template-method base class (single inheritance, no mixins) that captures
concerns shared by :class:`NodalResultsQueryEngine` and
:class:`ElementResultsQueryEngine`:

* **MultiIndex reuse**: step axes (per stage) and ID axes (per selection)
  cached as ``pandas.Index`` to avoid O(n_steps × n_ids) reconstruction
  per fetch.
* **Chunk-sorted fancy indexing**: ``_chunk_sorted_take`` sorts requested
  IDs by their on-disk position before the fancy index so reads align
  with HDF5 chunking; restores original order on return. 2-5x on wide
  selections; zero-cost on single-ID fetches.
* **Result LRU cache**: optional, on by default (``cache_size=32``).
  Cache keys are ``(stage, result, component, ids_hash, step_slice)``.
  Set ``cache_size=0`` to disable.

Vectorization policy
--------------------
No per-row Python loops in any hot path. All result assembly uses numpy
fancy indexing + ``pd.MultiIndex.from_product``. This is enforced by a
unit test that patches ``pd.DataFrame.iterrows`` to raise during fetch.

Thread-safety
-------------
The engine is **not** thread-safe: the caches are plain dicts and h5py
itself is not safe across threads without SWMR. Parallelism belongs at
the process level (``ProcessPoolExecutor``) in aggregation code, not
here.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Hashable, Optional, Sequence

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..io.partition_pool import Hdf5PartitionPool
    from ..io.format_policy import MpcoFormatPolicy
    from ..selection.resolver import SelectionSetResolver

logger = logging.getLogger(__name__)


DEFAULT_CACHE_SIZE = 32


class BaseResultsQueryEngine(ABC):
    """Abstract template for result-set fetches.

    Subclasses implement :meth:`fetch` and reuse the helpers here.

    Parameters
    ----------
    dataset:
        Parent ``MPCODataSet`` instance. Held by reference; the engine
        does not own it. Used only to read already-built attributes
        (e.g. ``dataset.model_stages``) — no HDF5 access via dataset.
    pool:
        Shared :class:`Hdf5PartitionPool` for HDF5 handle reuse.
    policy:
        :class:`MpcoFormatPolicy` for path templates.
    resolver:
        :class:`SelectionSetResolver` for name/id → id-array resolution.
    cache_size:
        LRU capacity for fetch-result caching. ``0`` disables caching.
        Default is :data:`DEFAULT_CACHE_SIZE` (32).
    """

    __slots__ = (
        "_dataset",
        "_pool",
        "_policy",
        "_resolver",
        "_cache_size",
        "_result_cache",
        "_step_axis_cache",
        "_id_axis_cache",
        "__weakref__",
    )

    def __init__(
        self,
        *,
        dataset: "MPCODataSet",
        pool: "Hdf5PartitionPool",
        policy: "MpcoFormatPolicy",
        resolver: "SelectionSetResolver",
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        if cache_size < 0:
            raise ValueError(f"cache_size must be >= 0, got {cache_size!r}")
        self._dataset = dataset
        self._pool = pool
        self._policy = policy
        self._resolver = resolver
        self._cache_size = int(cache_size)
        self._result_cache: "OrderedDict[Hashable, pd.DataFrame]" = OrderedDict()
        self._step_axis_cache: dict[str, pd.Index] = {}
        self._id_axis_cache: dict[Hashable, pd.Index] = {}

    def __repr__(self) -> str:
        cls = type(self).__name__
        return (
            f"{cls}(cache_size={self._cache_size}, "
            f"cached_results={len(self._result_cache)}, "
            f"step_axes={len(self._step_axis_cache)}, "
            f"id_axes={len(self._id_axis_cache)})"
        )

    # ------------------------------------------------------------------ #
    # Subclass contract
    # ------------------------------------------------------------------ #
    @abstractmethod
    def fetch(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Concrete engines implement this — return a DataFrame with a
        MultiIndex on ``(step, id)`` and columns for result components.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chunk_sorted_take(
        h5_dataset: Any,
        row_indices: np.ndarray,
    ) -> np.ndarray:
        """Fancy-index ``h5_dataset`` at ``row_indices`` with chunk-aware
        ordering, restoring the original row order on return.

        HDF5 fancy indexing is much faster when indices are monotonic and
        chunk-aligned. Sort the request, do one fancy take, then invert
        the permutation to match the caller's order.

        Parameters
        ----------
        h5_dataset:
            An ``h5py.Dataset`` (or any object supporting ``ds[sorted_idx]``).
        row_indices:
            1-D numpy array of row indices to read.

        Returns
        -------
        numpy.ndarray
            The read slab, rows in the order specified by ``row_indices``.
        """
        idx = np.asarray(row_indices)
        if idx.ndim != 1:
            raise ValueError(f"row_indices must be 1-D (got shape {idx.shape!r})")
        if idx.size == 0:
            return np.empty((0,) + tuple(h5_dataset.shape[1:]), dtype=h5_dataset.dtype)
        if idx.size == 1:
            # scalar fancy index is already cheap; skip the sort cost
            return h5_dataset[idx]

        order = np.argsort(idx, kind="stable")
        sorted_idx = idx[order]
        # h5py requires sorted, unique indices for direct fancy-indexing
        # with a list; if the request has duplicates we expand post-read.
        uniq, inverse = np.unique(sorted_idx, return_inverse=True)
        slab = h5_dataset[uniq]
        expanded = slab[inverse]
        restored = np.empty_like(expanded)
        restored[order] = expanded
        return restored

    def _step_axis(self, stage: str, step_keys: Sequence[str]) -> pd.Index:
        """Return a cached ``pd.Index`` of step labels for ``stage``.

        The engine keys on stage alone — a stage's step set is immutable
        once written. If the passed ``step_keys`` disagree with the
        cache, the cache entry wins (callers should not truncate).
        """
        cached = self._step_axis_cache.get(stage)
        if cached is not None and len(cached) == len(step_keys):
            return cached
        idx = pd.Index(list(step_keys), name="step")
        self._step_axis_cache[stage] = idx
        return idx

    def _id_axis(self, key: Hashable, ids: np.ndarray, name: str = "id") -> pd.Index:
        """Return a cached ``pd.Index`` of IDs keyed by a hashable ``key``.

        Typical ``key`` is a tuple of sorted IDs or a selection-set id;
        callers are responsible for a stable key.
        """
        cached = self._id_axis_cache.get(key)
        if cached is not None:
            return cached
        idx = pd.Index(np.asarray(ids, dtype=np.int64), name=name)
        self._id_axis_cache[key] = idx
        return idx

    # ------------------------------------------------------------------ #
    # LRU cache for fetch results
    # ------------------------------------------------------------------ #
    def _cache_get(self, key: Hashable) -> Optional[pd.DataFrame]:
        if self._cache_size == 0:
            return None
        df = self._result_cache.get(key)
        if df is not None:
            self._result_cache.move_to_end(key)
        return df

    def _cache_put(self, key: Hashable, df: pd.DataFrame) -> None:
        if self._cache_size == 0:
            return
        self._result_cache[key] = df
        self._result_cache.move_to_end(key)
        while len(self._result_cache) > self._cache_size:
            self._result_cache.popitem(last=False)

    def clear_caches(self) -> None:
        """Drop all cached results, step axes, and id axes."""
        self._result_cache.clear()
        self._step_axis_cache.clear()
        self._id_axis_cache.clear()

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @property
    def cached_result_count(self) -> int:
        return len(self._result_cache)
