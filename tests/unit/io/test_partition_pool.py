"""Unit tests for ``Hdf5PartitionPool``.

These tests build tiny real HDF5 files on-disk via ``h5py`` so we exercise
the pool with actual file handles. This is much cheaper than mocking h5py
and proves the class is correct against the real library.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import pytest

from STKO_to_python.io.partition_pool import Hdf5PartitionPool


def _make_partition(path: Path, partition_idx: int) -> None:
    """Write a minimal HDF5 file containing one dataset tagged with its index."""
    with h5py.File(path, "w") as h5:
        h5.create_dataset("partition_idx", data=partition_idx)


@pytest.fixture
def partitions(tmp_path: Path) -> dict[int, Path]:
    """Build three tiny HDF5 files and return a partition_idx → path mapping."""
    paths: dict[int, Path] = {}
    for idx in (0, 1, 2):
        p = tmp_path / f"results.part-{idx}.mpco"
        _make_partition(p, idx)
        paths[idx] = p
    return paths


# ---------------------------------------------------------------------- #
# Construction and repr                                                  #
# ---------------------------------------------------------------------- #

def test_negative_pool_size_rejected(partitions: dict[int, Path]) -> None:
    with pytest.raises(ValueError, match="pool_size must be >= 0"):
        Hdf5PartitionPool(partitions, pool_size=-1)


def test_len_and_contains(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions)
    assert len(pool) == 3
    assert 0 in pool
    assert 2 in pool
    assert 99 not in pool


def test_repr_shape(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=4)
    r = repr(pool)
    assert "Hdf5PartitionPool" in r
    assert "n_partitions=3" in r
    assert "pool_size=4" in r
    assert "n_open=0" in r


def test_partition_indices_sorted(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool({2: partitions[2], 0: partitions[0], 1: partitions[1]})
    assert pool.partition_indices == [0, 1, 2]


def test_path_for_raises_for_unknown_idx(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions)
    with pytest.raises(KeyError, match="Unknown partition index"):
        pool.path_for(99)


# ---------------------------------------------------------------------- #
# pool_size=0: open-per-call (legacy behavior)                           #
# ---------------------------------------------------------------------- #

def test_pool_size_zero_opens_fresh_handle_each_time(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=0)
    h1 = pool.open(0)
    h2 = pool.open(0)
    try:
        assert h1 is not h2
        assert pool.n_open == 0  # LRU stays empty
    finally:
        h1.close()
        h2.close()


def test_with_partition_closes_handle_on_exit(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=0)
    with pool.with_partition(1) as h5:
        assert h5["partition_idx"][()] == 1
        held = h5
    # After context exit the handle must be closed.
    with pytest.raises(Exception):
        _ = held["partition_idx"][()]


def test_close_all_is_noop_when_pool_size_zero(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=0)
    pool.close_all()  # should not raise
    assert pool.n_open == 0


# ---------------------------------------------------------------------- #
# pool_size>0: LRU caching                                                #
# ---------------------------------------------------------------------- #

def test_pool_size_positive_reuses_handle(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=2)
    try:
        h1 = pool.open(0)
        h2 = pool.open(0)
        assert h1 is h2
        assert pool.n_open == 1
    finally:
        pool.close_all()


def test_with_partition_does_not_close_pooled_handle(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=2)
    try:
        with pool.with_partition(0) as h5:
            pass
        # Handle stays open in the LRU.
        assert pool.n_open == 1
        reused = pool.open(0)
        assert reused["partition_idx"][()] == 0
    finally:
        pool.close_all()


def test_lru_evicts_oldest_on_overflow(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=2)
    try:
        pool.open(0)
        pool.open(1)
        assert pool.n_open == 2
        # Opening a third partition must evict partition 0 (LRU).
        pool.open(2)
        assert pool.n_open == 2
        assert 0 not in pool._lru  # type: ignore[attr-defined]
        assert 1 in pool._lru  # type: ignore[attr-defined]
        assert 2 in pool._lru  # type: ignore[attr-defined]
    finally:
        pool.close_all()


def test_access_promotes_entry_to_mru(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=2)
    try:
        pool.open(0)
        pool.open(1)
        # Touching 0 makes it the MRU; opening 2 should now evict 1.
        pool.open(0)
        pool.open(2)
        assert 0 in pool._lru  # type: ignore[attr-defined]
        assert 1 not in pool._lru  # type: ignore[attr-defined]
        assert 2 in pool._lru  # type: ignore[attr-defined]
    finally:
        pool.close_all()


def test_close_all_releases_every_handle(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=3)
    pool.open(0)
    pool.open(1)
    pool.open(2)
    pool.close_all()
    assert pool.n_open == 0
    # Safe to call twice.
    pool.close_all()


def test_unknown_idx_raises_keyerror(partitions: dict[int, Path]) -> None:
    pool = Hdf5PartitionPool(partitions, pool_size=2)
    with pytest.raises(KeyError, match="Unknown partition index"):
        pool.open(99)
