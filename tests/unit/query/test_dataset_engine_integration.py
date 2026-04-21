"""Phase 2.8: verify MPCODataSet attaches the query engines during __init__.

No .mpco fixture — construct a synthetic instance via ``__new__`` and
simulate the engine-wiring block.
"""
from __future__ import annotations

from STKO_to_python.core.dataset import MPCODataSet
from STKO_to_python.query import ElementResultsQueryEngine, NodalResultsQueryEngine


def test_dataset_exposes_engine_attributes_after_wiring():
    """Pin the attribute contract — MPCODataSet.__init__ must set both
    ``_nodal_query_engine`` and ``_element_query_engine``.
    """
    # Sanity: both are present in the class body by searching source.
    # A real construction is exercised in tests/integration/.
    import inspect
    src = inspect.getsource(MPCODataSet)
    assert "_nodal_query_engine" in src
    assert "_element_query_engine" in src
    assert "NodalResultsQueryEngine(" in src
    assert "ElementResultsQueryEngine(" in src


def test_clear_result_caches_iterates_both_engines():
    """``clear_result_caches`` must touch both engine attributes and
    tolerate a partially-constructed instance.
    """
    calls: list[str] = []

    class _FakeEngine:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def clear_caches(self) -> None:
            calls.append(self.tag)

    ds = MPCODataSet.__new__(MPCODataSet)
    ds._nodal_query_engine = _FakeEngine("nodal")  # type: ignore[attr-defined]
    ds._element_query_engine = _FakeEngine("element")  # type: ignore[attr-defined]
    ds.clear_result_caches()
    assert sorted(calls) == ["element", "nodal"]


def test_clear_result_caches_tolerates_missing_engines():
    ds = MPCODataSet.__new__(MPCODataSet)
    # no engines attached at all
    ds.clear_result_caches()  # must not raise


def test_exit_clears_engine_caches():
    """Existing __exit__ closes the pool; also drop engine caches."""
    calls: list[str] = []

    class _FakeEngine:
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def clear_caches(self) -> None:
            calls.append(self.tag)

    class _FakePool:
        def close_all(self) -> None:
            calls.append("pool")

    ds = MPCODataSet.__new__(MPCODataSet)
    ds._pool = _FakePool()  # type: ignore[attr-defined]
    ds._nodal_query_engine = _FakeEngine("nodal")  # type: ignore[attr-defined]
    ds._element_query_engine = _FakeEngine("element")  # type: ignore[attr-defined]
    ds.__exit__(None, None, None)
    assert set(calls) == {"pool", "nodal", "element"}
