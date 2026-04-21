"""Pins the nested-sequence flattening behavior that ``Nodes._resolve_node_ids``
historically supported. After routing through SelectionSetResolver, this
quirk is preserved by the module-level ``_flatten_node_ids`` helper.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.nodes.nodes import _flatten_node_ids


def test_scalar_int():
    assert _flatten_node_ids(7).tolist() == [7]


def test_flat_list():
    assert _flatten_node_ids([1, 2, 3]).tolist() == [1, 2, 3]


def test_nested_lists_flatten():
    assert _flatten_node_ids([[1, 2], [3]]).tolist() == [1, 2, 3]


def test_mixed_scalars_and_sequences():
    assert _flatten_node_ids([[1, 2], 3, [4]]).tolist() == [1, 2, 3, 4]


def test_numpy_array_passthrough():
    out = _flatten_node_ids(np.array([10, 11], dtype=np.int32))
    assert out.dtype == np.int64
    assert out.tolist() == [10, 11]


def test_returns_int64():
    assert _flatten_node_ids([1, 2, 3]).dtype == np.int64
