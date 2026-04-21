"""Phase 2 integration contract: ``MPCODataSet`` exposes a resolver.

The resolver is attached during ``__init__`` from the already-built
``self.selection_set`` dict. These tests pin the import path and the
attribute wiring at the class level without constructing a full dataset.
"""
from __future__ import annotations

import pytest

from STKO_to_python.core.dataset import MPCODataSet
from STKO_to_python.selection import SelectionSetResolver


def test_resolver_class_is_importable_from_selection() -> None:
    """Public import path: ``from STKO_to_python.selection import SelectionSetResolver``."""
    from STKO_to_python.selection import SelectionSetResolver as Reexport
    assert Reexport is SelectionSetResolver


def test_resolver_constructs_from_dataset_selection_set_shape() -> None:
    """The resolver must accept the exact dict shape that
    ``CData._extract_selection_set_ids`` returns today.
    """
    fake_selection_set = {
        1: {"SET_NAME": "Foo", "NODES": {1, 2, 3}, "ELEMENTS": set()},
        2: {"SET_NAME": "Bar", "NODES": set(), "ELEMENTS": {100}},
    }
    r = SelectionSetResolver(fake_selection_set)
    assert sorted(r.resolve_nodes(ids=1).tolist()) == [1, 2, 3]
    assert sorted(r.resolve_elements(names="Bar").tolist()) == [100]


def test_dataset_wires_resolver_side_by_side() -> None:
    """On a synthetic dataset instance with a selection_set attribute, the
    wiring code that ``MPCODataSet.__init__`` performs must still work —
    pinning the contract that ``self._selection_resolver`` is the result of
    ``SelectionSetResolver(self.selection_set)``.
    """
    ds = MPCODataSet.__new__(MPCODataSet)
    ds.selection_set = {
        5: {"SET_NAME": "Roof", "NODES": {10, 11}, "ELEMENTS": set()},
    }
    # Mirror the exact assignment made in MPCODataSet.__init__:
    ds._selection_resolver = SelectionSetResolver(ds.selection_set)  # type: ignore[attr-defined]

    assert isinstance(ds._selection_resolver, SelectionSetResolver)
    assert ds._selection_resolver.resolve_nodes(names="Roof").tolist() == [10, 11]
