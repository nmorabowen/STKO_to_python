"""Phase 1 integration contract: ``MPCODataSet`` exposes a pool and policy.

These tests don't construct a full dataset (that would need a real .mpco).
They verify the surface contract at the class level plus a minimal
exercise using a synthetic instance built with ``__new__``.
"""
from __future__ import annotations

import inspect

import pytest

from STKO_to_python.core.dataset import MPCODataSet
from STKO_to_python.io.partition_pool import Hdf5PartitionPool
from STKO_to_python.io.format_policy import MpcoFormatPolicy


def test_dataset_accepts_pool_size_parameter() -> None:
    """Phase 1 adds ``pool_size``; Phase 1.3 flips the default from 0 to
    None. ``None`` means "performance-first auto" (min(16, n_partitions));
    ``0`` is still valid and preserves legacy open-per-call behavior.
    """
    sig = inspect.signature(MPCODataSet.__init__)
    assert "pool_size" in sig.parameters
    assert sig.parameters["pool_size"].default is None


def test_dataset_init_signature_backcompat() -> None:
    """All pre-refactor parameters must still exist with the same defaults."""
    sig = inspect.signature(MPCODataSet.__init__)
    for name in ("hdf5_directory", "recorder_name"):
        assert name in sig.parameters

    expected_defaults = {
        "name": None,
        "file_extension": "*.mpco",
        "verbose": False,
        "plot_settings": None,
        "pool_size": None,
    }
    for name, default in expected_defaults.items():
        assert name in sig.parameters, f"missing param {name!r}"
        assert sig.parameters[name].default == default, (
            f"{name}: default changed to {sig.parameters[name].default!r}"
        )


def test_exit_closes_pool() -> None:
    """``__exit__`` must call ``close_all`` on the pool if one exists."""
    ds = MPCODataSet.__new__(MPCODataSet)

    class _CountingPool:
        calls = 0

        def close_all(self) -> None:
            _CountingPool.calls += 1

    ds._pool = _CountingPool()  # type: ignore[attr-defined]
    ds.__exit__(None, None, None)
    assert _CountingPool.calls == 1


def test_exit_tolerates_missing_pool() -> None:
    """If the pool never got constructed (e.g. __init__ raised), __exit__
    must not crash trying to close it."""
    ds = MPCODataSet.__new__(MPCODataSet)
    ds._pool = None  # type: ignore[attr-defined]
    # Must not raise.
    ds.__exit__(None, None, None)


def test_format_policy_class_is_importable_from_io() -> None:
    """Public import path — ``from STKO_to_python.io import MpcoFormatPolicy``."""
    from STKO_to_python.io import MpcoFormatPolicy as Reexport
    assert Reexport is MpcoFormatPolicy


def test_partition_pool_class_is_importable_from_io() -> None:
    """Public import path — ``from STKO_to_python.io import Hdf5PartitionPool``."""
    from STKO_to_python.io import Hdf5PartitionPool as Reexport
    assert Reexport is Hdf5PartitionPool
