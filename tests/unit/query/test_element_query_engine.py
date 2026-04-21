"""Unit tests for ``ElementResultsQueryEngine``."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from STKO_to_python.query import ElementResultsQueryEngine
from STKO_to_python.selection import SelectionSetResolver


def _make_engine(selection_set: dict | None = None, cache_size: int = 32):
    manager = MagicMock()
    dataset = MagicMock()
    dataset.elements = manager
    dataset.model_stages = ("MODEL_STAGE[0]",)
    resolver = SelectionSetResolver(selection_set or {})
    engine = ElementResultsQueryEngine(
        dataset=dataset,
        pool=MagicMock(),
        policy=MagicMock(),
        resolver=resolver,
        cache_size=cache_size,
    )
    return engine, manager


def test_fetch_delegates_to_manager():
    eng, manager = _make_engine()
    result = eng.fetch("force", "5-ElasticBeam3d")
    manager.get_element_results.assert_called_once()
    assert result is manager.get_element_results.return_value


def test_fetch_cache_hit_avoids_manager_call():
    eng, manager = _make_engine(
        selection_set={1: {"SET_NAME": "Cols", "NODES": set(), "ELEMENTS": {100, 101}}},
    )
    eng.fetch("force", "5-ElasticBeam3d", selection_set_name="Cols")
    eng.fetch("force", "5-ElasticBeam3d", selection_set_name="Cols")
    assert manager.get_element_results.call_count == 1


def test_fetch_cache_miss_on_different_element_type():
    eng, manager = _make_engine()
    eng.fetch("force", "5-ElasticBeam3d")
    eng.fetch("force", "203-ASDShellQ4")
    assert manager.get_element_results.call_count == 2


def test_fetch_cache_miss_on_different_results_name():
    eng, manager = _make_engine()
    eng.fetch("force", "5-ElasticBeam3d")
    eng.fetch("localForce", "5-ElasticBeam3d")
    assert manager.get_element_results.call_count == 2


def test_fetch_cache_miss_on_different_stage():
    eng, manager = _make_engine()
    eng.fetch("force", "5-ElasticBeam3d", model_stage="MODEL_STAGE[0]")
    eng.fetch("force", "5-ElasticBeam3d", model_stage="MODEL_STAGE[1]")
    assert manager.get_element_results.call_count == 2


def test_fetch_cache_collapses_all_elements_requests():
    """Calls that don't specify any selection input must collapse onto a
    single cache entry regardless of whether stage was explicit or not.
    """
    eng, manager = _make_engine()
    eng.fetch("force", "5-ElasticBeam3d")
    eng.fetch("force", "5-ElasticBeam3d")  # same stage auto-resolved
    assert manager.get_element_results.call_count == 1


def test_fetch_cache_hit_across_equivalent_selection_inputs():
    eng, manager = _make_engine(
        selection_set={1: {"SET_NAME": "Cols", "NODES": set(), "ELEMENTS": {100, 101}}},
    )
    eng.fetch("force", "5-ElasticBeam3d", selection_set_name="Cols")
    eng.fetch("force", "5-ElasticBeam3d", element_ids=[100, 101])
    assert manager.get_element_results.call_count == 1


def test_cache_disabled_by_cache_size_zero():
    eng, manager = _make_engine(cache_size=0)
    eng.fetch("force", "5-ElasticBeam3d")
    eng.fetch("force", "5-ElasticBeam3d")
    assert manager.get_element_results.call_count == 2


def test_build_cache_key_sorts_ids():
    k1 = ElementResultsQueryEngine._build_cache_key(
        results_name="force",
        element_type="5-ElasticBeam3d",
        stage="s0",
        ids=np.array([3, 1, 2]),
    )
    k2 = ElementResultsQueryEngine._build_cache_key(
        results_name="force",
        element_type="5-ElasticBeam3d",
        stage="s0",
        ids=np.array([1, 2, 3]),
    )
    assert k1 == k2


def test_build_cache_key_all_vs_ids():
    k_all = ElementResultsQueryEngine._build_cache_key(
        results_name="force",
        element_type="5-ElasticBeam3d",
        stage="s0",
        ids=None,
    )
    k_ids = ElementResultsQueryEngine._build_cache_key(
        results_name="force",
        element_type="5-ElasticBeam3d",
        stage="s0",
        ids=np.array([1]),
    )
    assert k_all != k_ids


def test_element_engine_slots_empty_tuple():
    assert ElementResultsQueryEngine.__slots__ == ()
