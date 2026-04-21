"""Smoke tests for ``NodalResultsInfo`` selection-set helpers after the
Phase 2 routing through ``SelectionSetResolver``.
"""
from __future__ import annotations

import pytest

from STKO_to_python.results.nodal_results_info import NodalResultsInfo


def _info_with_sets():
    return NodalResultsInfo(
        nodes_ids=(10, 11, 12, 500),
        selection_set={
            1: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()},
            3: {"SET_NAME": "ControlPoint", "NODES": {500}, "ELEMENTS": set()},
            5: {"SET_NAME": "", "NODES": {999}, "ELEMENTS": set()},
        },
    )


def test_ids_from_names_basic():
    info = _info_with_sets()
    assert info.selection_set_ids_from_names("Roof") == (1,)
    assert info.selection_set_ids_from_names("controlpoint") == (3,)


def test_ids_from_names_comma_separated_string():
    """Legacy convenience: 'A, B' splits."""
    info = _info_with_sets()
    assert set(info.selection_set_ids_from_names("Roof, ControlPoint")) == {1, 3}


def test_ids_from_names_unknown_raises():
    info = _info_with_sets()
    with pytest.raises(ValueError, match="Selection set name not found"):
        info.selection_set_ids_from_names("Nope")


def test_ids_from_names_empty_raises():
    info = _info_with_sets()
    with pytest.raises(ValueError, match="selection_set_name is empty"):
        info.selection_set_ids_from_names([])


def test_selection_set_name_for():
    info = _info_with_sets()
    assert info._selection_set_name_for(1) == "Roof"
    assert info._selection_set_name_for(99) == ""


def test_selection_set_name_for_with_no_sets_returns_empty():
    info = NodalResultsInfo(nodes_ids=(1,), selection_set=None)
    assert info._selection_set_name_for(1) == ""


def test_ids_from_names_raises_when_selection_set_is_none():
    info = NodalResultsInfo(nodes_ids=(1,), selection_set=None)
    with pytest.raises(ValueError, match="selection_set is None"):
        info.selection_set_ids_from_names("Anything")
