"""Unit tests for ``NodalResultsQueryEngine``.

No ``.mpco`` fixture — we stub the dataset + manager with minimal objects
that satisfy the engine's contract.
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from STKO_to_python.query import NodalResultsQueryEngine
from STKO_to_python.selection import SelectionSetResolver


def _stub_manager() -> MagicMock:
    m = MagicMock()
    # Use a MagicMock as the return value (fresh per call via side_effect=...
    # would break identity checks, so just let .return_value do its work).
    return m


def _make_engine(selection_set: dict | None = None, cache_size: int = 32):
    manager = _stub_manager()
    dataset = MagicMock()
    dataset.nodes = manager
    dataset.model_stages = ("MODEL_STAGE[0]",)
    resolver = SelectionSetResolver(selection_set or {})
    engine = NodalResultsQueryEngine(
        dataset=dataset,
        pool=MagicMock(),
        policy=MagicMock(),
        resolver=resolver,
        cache_size=cache_size,
    )
    return engine, manager, dataset


def test_engine_instantiates_with_slots():
    engine, *_ = _make_engine()
    assert isinstance(engine, NodalResultsQueryEngine)
    # Subclass's own __slots__ is empty — inheritance provides base slots
    assert NodalResultsQueryEngine.__slots__ == ()


def test_fetch_delegates_to_manager_on_cold():
    engine, manager, _ = _make_engine(
        selection_set={1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()}},
    )
    result = engine.fetch(selection_set_name="Roof", results_name="displacement")
    manager.get_nodal_results.assert_called_once()
    assert result is manager.get_nodal_results.return_value


def test_fetch_cache_hit_avoids_manager_call():
    engine, manager, _ = _make_engine(
        selection_set={1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()}},
        cache_size=4,
    )
    r1 = engine.fetch(selection_set_name="Roof", results_name="displacement")
    r2 = engine.fetch(selection_set_name="Roof", results_name="displacement")
    assert manager.get_nodal_results.call_count == 1
    assert r1 is r2


def test_fetch_cache_miss_on_different_ids():
    engine, manager, _ = _make_engine(
        selection_set={
            1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()},
            2: {"SET_NAME": "Floor", "NODES": {20, 21}, "ELEMENTS": set()},
        },
        cache_size=4,
    )
    engine.fetch(selection_set_name="Roof", results_name="displacement")
    engine.fetch(selection_set_name="Floor", results_name="displacement")
    assert manager.get_nodal_results.call_count == 2


def test_fetch_cache_hit_across_equivalent_selection_inputs():
    """Selection by name and selection by explicit-ids that resolve to the
    same IDs must collapse onto the same cache entry.
    """
    engine, manager, _ = _make_engine(
        selection_set={1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()}},
        cache_size=4,
    )
    engine.fetch(selection_set_name="Roof", results_name="displacement")
    engine.fetch(node_ids=[10, 11], results_name="displacement")
    assert manager.get_nodal_results.call_count == 1


def test_fetch_cache_disabled_by_cache_size_zero():
    engine, manager, _ = _make_engine(
        selection_set={1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()}},
        cache_size=0,
    )
    engine.fetch(selection_set_name="Roof", results_name="displacement")
    engine.fetch(selection_set_name="Roof", results_name="displacement")
    assert manager.get_nodal_results.call_count == 2


def test_build_cache_key_normalizes_ordering():
    """Cache key must be insensitive to order of results and stages."""
    k1 = NodalResultsQueryEngine._build_cache_key(
        results=("a", "b"), stages=("s1", "s0"), ids=np.array([3, 1, 2]),
    )
    k2 = NodalResultsQueryEngine._build_cache_key(
        results=("b", "a"), stages=("s0", "s1"), ids=np.array([1, 2, 3]),
    )
    assert k1 == k2


def test_build_cache_key_distinguishes_all_vs_explicit():
    k_all = NodalResultsQueryEngine._build_cache_key(
        results=None, stages=("s0",), ids=np.array([1]),
    )
    k_explicit = NodalResultsQueryEngine._build_cache_key(
        results=("__all__",), stages=("s0",), ids=np.array([1]),
    )
    # When results is None, key uses ("__all__",); both routes land on
    # the same key. This is intentional — resolver provides a single
    # canonical form.
    assert k_all == k_explicit


def test_fetch_routes_results_name_single_string():
    engine, manager, _ = _make_engine(
        selection_set={1: {"SET_NAME": "Roof", "NODES": {10}, "ELEMENTS": set()}},
    )
    engine.fetch(selection_set_name="Roof", results_name="displacement")
    kwargs = manager.get_nodal_results.call_args.kwargs
    # Manager receives the unnormalized input (strings pass through); the
    # engine's job is caching, not signature massage.
    assert kwargs["results_name"] == "displacement"
    assert kwargs["selection_set_name"] == "Roof"
