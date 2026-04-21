"""Unit tests for ``BaseResultsQueryEngine``.

Uses a minimal concrete subclass because the base is abstract. No real
``.mpco`` fixture — everything is exercised through fake HDF5 datasets
(numpy arrays quacking like ``h5py.Dataset``) and a ``SelectionSetResolver``
built from a plain dict.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.query import BaseResultsQueryEngine
from STKO_to_python.selection import SelectionSetResolver


class _ConcreteEngine(BaseResultsQueryEngine):
    """Smallest concrete subclass to satisfy the abstract contract."""

    __slots__ = ()

    def fetch(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        return pd.DataFrame()


def _make_engine(cache_size: int = 32) -> _ConcreteEngine:
    resolver = SelectionSetResolver({})
    return _ConcreteEngine(
        dataset=object(),          # engine only holds the ref
        pool=object(),             # not exercised in these tests
        policy=object(),
        resolver=resolver,
        cache_size=cache_size,
    )


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #
def test_abstract_base_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseResultsQueryEngine(  # type: ignore[abstract]
            dataset=None, pool=None, policy=None, resolver=None,
        )


def test_cache_size_negative_rejected():
    with pytest.raises(ValueError, match="cache_size must be >= 0"):
        _ConcreteEngine(
            dataset=object(), pool=object(), policy=object(),
            resolver=SelectionSetResolver({}), cache_size=-1,
        )


def test_repr_shape():
    eng = _make_engine(cache_size=8)
    s = repr(eng)
    assert "_ConcreteEngine" in s
    assert "cache_size=8" in s
    assert "cached_results=0" in s


def test_slots_prevent_stray_attributes():
    eng = _make_engine()
    # __weakref__ must be present on the base (cache needs it)
    assert "__weakref__" in BaseResultsQueryEngine.__slots__
    with pytest.raises(AttributeError):
        eng.random_attr = 1  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- #
# _chunk_sorted_take
# ---------------------------------------------------------------------- #
def test_chunk_sorted_take_preserves_order_and_values():
    ds = np.arange(100).reshape(100, 1) * 10.0  # mimic (rows, cols)
    # Unordered request
    req = np.array([7, 2, 90, 15, 2])  # duplicate allowed
    out = BaseResultsQueryEngine._chunk_sorted_take(ds, req)
    assert out.shape == (5, 1)
    assert out[:, 0].tolist() == [70.0, 20.0, 900.0, 150.0, 20.0]


def test_chunk_sorted_take_single_row_skips_sort():
    ds = np.arange(20).reshape(10, 2)
    out = BaseResultsQueryEngine._chunk_sorted_take(ds, np.array([4]))
    assert out.tolist() == [[8, 9]]


def test_chunk_sorted_take_empty_returns_empty():
    ds = np.zeros((10, 3), dtype=np.float64)
    out = BaseResultsQueryEngine._chunk_sorted_take(ds, np.array([], dtype=np.int64))
    assert out.shape == (0, 3)
    assert out.dtype == np.float64


def test_chunk_sorted_take_rejects_non_1d():
    ds = np.arange(10)
    with pytest.raises(ValueError, match="1-D"):
        BaseResultsQueryEngine._chunk_sorted_take(ds, np.array([[1, 2]]))


# ---------------------------------------------------------------------- #
# MultiIndex axis caches
# ---------------------------------------------------------------------- #
def test_step_axis_cached_per_stage():
    eng = _make_engine()
    keys = ["STEP_0", "STEP_1", "STEP_2"]
    idx_a = eng._step_axis("MODEL_STAGE[0]", keys)
    idx_b = eng._step_axis("MODEL_STAGE[0]", keys)
    assert idx_a is idx_b  # identity: cache hit
    assert idx_a.name == "step"
    assert idx_a.tolist() == keys


def test_step_axis_differs_per_stage():
    eng = _make_engine()
    a = eng._step_axis("MODEL_STAGE[0]", ["S0", "S1"])
    b = eng._step_axis("MODEL_STAGE[1]", ["S0", "S1"])
    assert a is not b


def test_id_axis_cache_reuses_by_key():
    eng = _make_engine()
    ids = np.array([5, 10, 15], dtype=np.int64)
    idx = eng._id_axis("selection:1", ids)
    assert eng._id_axis("selection:1", ids) is idx
    # different key gives a new index
    idx2 = eng._id_axis("selection:2", ids)
    assert idx2 is not idx


# ---------------------------------------------------------------------- #
# LRU cache
# ---------------------------------------------------------------------- #
def test_cache_put_and_get_roundtrip():
    eng = _make_engine(cache_size=3)
    df = pd.DataFrame({"x": [1]})
    eng._cache_put(("a",), df)
    assert eng._cache_get(("a",)) is df


def test_cache_size_zero_disables():
    eng = _make_engine(cache_size=0)
    eng._cache_put(("a",), pd.DataFrame({"x": [1]}))
    assert eng._cache_get(("a",)) is None
    assert eng.cached_result_count == 0


def test_cache_evicts_oldest_when_full():
    eng = _make_engine(cache_size=2)
    eng._cache_put("a", pd.DataFrame({"v": [1]}))
    eng._cache_put("b", pd.DataFrame({"v": [2]}))
    eng._cache_put("c", pd.DataFrame({"v": [3]}))  # evicts "a"
    assert eng._cache_get("a") is None
    assert eng._cache_get("b") is not None
    assert eng._cache_get("c") is not None


def test_cache_get_promotes_to_mru():
    eng = _make_engine(cache_size=2)
    eng._cache_put("a", pd.DataFrame({"v": [1]}))
    eng._cache_put("b", pd.DataFrame({"v": [2]}))
    # Touch "a" so it becomes MRU
    eng._cache_get("a")
    eng._cache_put("c", pd.DataFrame({"v": [3]}))  # should evict "b"
    assert eng._cache_get("a") is not None
    assert eng._cache_get("b") is None
    assert eng._cache_get("c") is not None


def test_no_iterrows_in_chunk_sorted_take():
    """Policy: no per-row Python loops in hot path.

    Patches ``pd.DataFrame.iterrows`` to raise, then exercises the base
    helper. The helper doesn't use iterrows, but this locks in the
    policy so future changes can't regress.
    """
    ds = np.arange(50).reshape(25, 2)
    original = pd.DataFrame.iterrows

    def _forbidden(self, *a, **kw):
        raise AssertionError("iterrows is forbidden in query hot path")

    pd.DataFrame.iterrows = _forbidden  # type: ignore[assignment]
    try:
        out = BaseResultsQueryEngine._chunk_sorted_take(ds, np.array([5, 1, 10]))
        assert out.shape == (3, 2)
    finally:
        pd.DataFrame.iterrows = original  # type: ignore[assignment]


def test_clear_caches_wipes_everything():
    eng = _make_engine()
    eng._cache_put("a", pd.DataFrame({"v": [1]}))
    eng._step_axis("s", ["STEP_0"])
    eng._id_axis("k", np.array([1], dtype=np.int64))
    eng.clear_caches()
    assert eng.cached_result_count == 0
    assert len(eng._step_axis_cache) == 0
    assert len(eng._id_axis_cache) == 0
