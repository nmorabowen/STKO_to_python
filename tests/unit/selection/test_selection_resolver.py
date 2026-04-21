"""Unit tests for ``SelectionSetResolver``.

No HDF5 / .mpco fixture required — the resolver is pure-data and reads
from a plain selection-set dict (same shape produced by
``CData._get_selection_set``).
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.selection import SelectionSetResolver


def _fake_selection_set():
    """Representative selection_set dict (mirrors CData output)."""
    return {
        1: {
            "SET_NAME": "Roof_Nodes",
            "NODES": {101, 102, 103},
            "ELEMENTS": set(),
        },
        2: {
            "SET_NAME": "Columns_L1",
            "NODES": set(),
            "ELEMENTS": {1001, 1002},
        },
        3: {
            "SET_NAME": "ControlPoint",
            "NODES": {500},
            "ELEMENTS": set(),
        },
        4: {
            # duplicate name — case-insensitively identical to set 1
            "SET_NAME": "roof_nodes",
            "NODES": {104, 105},
            "ELEMENTS": set(),
        },
        5: {
            # unnamed set — resolvable only by id
            "SET_NAME": "",
            "NODES": {999},
            "ELEMENTS": {9999},
        },
    }


# ------------------------------------------------------------------ #
# Construction
# ------------------------------------------------------------------ #
def test_construction_indexes_all_sets():
    r = SelectionSetResolver(_fake_selection_set())
    assert len(r) == 5


def test_repr_shape():
    r = SelectionSetResolver(_fake_selection_set())
    s = repr(r)
    assert "SelectionSetResolver" in s
    assert "n_sets=5" in s
    assert "n_node_sets=4" in s  # sets 1, 3, 4, 5 have NODES payload
    assert "n_element_sets=2" in s  # sets 2, 5 have ELEMENTS payload


def test_slots_prevent_stray_attributes():
    r = SelectionSetResolver({})
    assert r.__slots__ == ("_by_name", "_by_id", "_node_ids", "_element_ids")
    with pytest.raises(AttributeError):
        r.extra = 1  # type: ignore[attr-defined]


def test_empty_selection_set_is_valid():
    r = SelectionSetResolver({})
    assert len(r) == 0
    assert r.list_node_sets() == []
    assert r.list_element_sets() == []


def test_malformed_payloads_are_skipped():
    r = SelectionSetResolver({"not_an_int": {"SET_NAME": "x"}, 7: "not_a_mapping"})
    assert len(r) == 0


# ------------------------------------------------------------------ #
# resolve_nodes
# ------------------------------------------------------------------ #
def test_resolve_nodes_by_explicit_ids():
    r = SelectionSetResolver(_fake_selection_set())
    out = r.resolve_nodes(explicit_ids=[101, 102, 101])
    assert out.dtype == np.int64
    assert out.tolist() == [101, 102]


def test_resolve_nodes_by_set_id():
    r = SelectionSetResolver(_fake_selection_set())
    out = r.resolve_nodes(ids=1)
    assert out.tolist() == [101, 102, 103]


def test_resolve_nodes_by_name_is_case_insensitive():
    r = SelectionSetResolver(_fake_selection_set())
    # "ControlPoint" is the only unambiguous name with NODES
    assert r.resolve_nodes(names="ControlPoint").tolist() == [500]
    assert r.resolve_nodes(names="controlpoint").tolist() == [500]
    assert r.resolve_nodes(names="  CONTROLPOINT  ").tolist() == [500]


def test_resolve_nodes_union_semantics():
    r = SelectionSetResolver(_fake_selection_set())
    out = r.resolve_nodes(
        ids=1,
        names="ControlPoint",
        explicit_ids=[9999],
    )
    assert out.tolist() == [101, 102, 103, 500, 9999]


def test_resolve_nodes_ambiguous_name_raises():
    r = SelectionSetResolver(_fake_selection_set())
    with pytest.raises(ValueError, match="Ambiguous"):
        r.resolve_nodes(names="Roof_Nodes")


def test_resolve_nodes_unknown_name_raises():
    r = SelectionSetResolver(_fake_selection_set())
    with pytest.raises(ValueError, match="Selection set name not found"):
        r.resolve_nodes(names="does_not_exist")


def test_resolve_nodes_empty_payload_raises():
    r = SelectionSetResolver(_fake_selection_set())
    # set 2 has NODES=empty set
    with pytest.raises(ValueError, match="empty or missing NODES"):
        r.resolve_nodes(ids=2)


def test_resolve_nodes_requires_at_least_one_input():
    r = SelectionSetResolver(_fake_selection_set())
    with pytest.raises(ValueError, match="Provide names, ids, and/or explicit_ids"):
        r.resolve_nodes()


def test_resolve_nodes_accepts_numpy_array_of_ids():
    r = SelectionSetResolver(_fake_selection_set())
    out = r.resolve_nodes(explicit_ids=np.array([101, 102, 103], dtype=np.int32))
    assert out.dtype == np.int64
    assert out.tolist() == [101, 102, 103]


# ------------------------------------------------------------------ #
# resolve_elements
# ------------------------------------------------------------------ #
def test_resolve_elements_by_name():
    r = SelectionSetResolver(_fake_selection_set())
    out = r.resolve_elements(names="Columns_L1")
    assert out.tolist() == [1001, 1002]


def test_resolve_elements_unknown_name_raises():
    r = SelectionSetResolver(_fake_selection_set())
    with pytest.raises(ValueError, match="Selection set name not found"):
        r.resolve_elements(names="no_such_set")


def test_resolve_elements_empty_payload_raises():
    r = SelectionSetResolver(_fake_selection_set())
    # set 1 has ELEMENTS=empty set
    with pytest.raises(ValueError, match="empty or missing ELEMENTS"):
        r.resolve_elements(ids=1)


# ------------------------------------------------------------------ #
# list_*
# ------------------------------------------------------------------ #
def test_list_node_sets_skips_empty_payload_and_unnamed():
    r = SelectionSetResolver(_fake_selection_set())
    # sets 1, 3, 4 have non-empty NODES+name; set 5 has no name; set 2 no NODES
    names = r.list_node_sets()
    # "Roof_Nodes" (sid=1) and "roof_nodes" (sid=4) both appear — different
    # case-preserved strings, both are legitimate names.
    assert "ControlPoint" in names
    assert "Roof_Nodes" in names
    assert "roof_nodes" in names
    assert "Columns_L1" not in names


def test_list_element_sets():
    r = SelectionSetResolver(_fake_selection_set())
    assert r.list_element_sets() == ["Columns_L1"]


# ------------------------------------------------------------------ #
# Name / id helpers
# ------------------------------------------------------------------ #
def test_name_for_known_id():
    r = SelectionSetResolver(_fake_selection_set())
    assert r.name_for(3) == "ControlPoint"


def test_name_for_unknown_id():
    r = SelectionSetResolver(_fake_selection_set())
    assert r.name_for(99) == ""


def test_ids_for_name_handles_duplicates():
    r = SelectionSetResolver(_fake_selection_set())
    assert sorted(r.ids_for_name("Roof_Nodes")) == [1, 4]
    assert r.ids_for_name("ControlPoint") == (3,)
    assert r.ids_for_name("no_such") == ()
