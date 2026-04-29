"""Unit tests for ``ModelMetadata`` and its back-compat ``MetaData`` alias."""
from __future__ import annotations

import copy
import pickle
from datetime import datetime

import pytest

from STKO_to_python.core.metadata import ModelMetadata


# ---------------------------------------------------------------------- #
# Alias identity
# ---------------------------------------------------------------------- #
def test_metadata_alias_is_model_metadata_and_emits_deprecation():
    """Importing ``MetaData`` from the deprecated deep path must emit a
    ``DeprecationWarning`` and resolve to ``ModelMetadata``."""
    with pytest.warns(DeprecationWarning, match="MetaData.*deprecated"):
        from STKO_to_python.core.dataclasses import MetaData
    assert MetaData is ModelMetadata


def test_top_level_core_exports_both_names_quietly():
    """The package-surface alias (``STKO_to_python.core.MetaData``) is
    quiet — only the deep ``core.dataclasses`` path emits the warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.core import MetaData as M, ModelMetadata as MM
    assert M is MM


# ---------------------------------------------------------------------- #
# Basic contract
# ---------------------------------------------------------------------- #
def test_default_construction_populates_date_created():
    m = ModelMetadata()
    assert isinstance(m.date_created, datetime)


def test_extras_are_stored_as_attributes():
    m = ModelMetadata(run_id=42, case_name="rocking")
    assert m.run_id == 42
    assert m.case_name == "rocking"


def test_setattr_routes_to_extras():
    m = ModelMetadata()
    m.run_id = 7
    assert m.get("run_id") == 7


def test_getattr_missing_raises_attribute_error():
    m = ModelMetadata()
    with pytest.raises(AttributeError):
        _ = m.does_not_exist


def test_delattr_removes_key():
    m = ModelMetadata(run_id=7)
    del m.run_id
    with pytest.raises(AttributeError):
        _ = m.run_id


def test_delattr_unknown_raises_attribute_error():
    m = ModelMetadata()
    with pytest.raises(AttributeError):
        del m.nope


def test_contains_and_len_and_iter():
    m = ModelMetadata(a=1, b=2)
    assert "a" in m
    assert "missing" not in m
    # len counts all extras including auto-populated date_created
    assert len(m) == 3
    assert set(iter(m)) == {"a", "b", "date_created"}


def test_slots_prevent_stray_slot_attributes():
    m = ModelMetadata()
    assert ModelMetadata.__slots__ == ("_extras",)
    # Any unknown attribute write lands in _extras, not on a new slot.
    m.new_thing = 99
    assert m._extras["new_thing"] == 99


def test_repr_contains_fields():
    m = ModelMetadata(case="demo")
    s = repr(m)
    assert "ModelMetadata(" in s
    assert "case='demo'" in s


def test_equality_by_extras():
    a = ModelMetadata(case="x")
    b = ModelMetadata(case="x")
    # date_created is injected per-instance; equality should account for it
    b._extras["date_created"] = a._extras["date_created"]
    assert a == b


# ---------------------------------------------------------------------- #
# Dict-like surface
# ---------------------------------------------------------------------- #
def test_set_get_has():
    m = ModelMetadata()
    m.set("run_id", 42)
    assert m.has("run_id")
    assert m.get("run_id") == 42
    assert m.get("missing", "def") == "def"


def test_keys_values_items():
    m = ModelMetadata(a=1, b=2)
    assert set(m.keys()) >= {"a", "b", "date_created"}
    assert 1 in list(m.values())
    items = dict(m.items())
    assert items["a"] == 1 and items["b"] == 2


def test_to_dict_includes_date_by_default():
    m = ModelMetadata(a=1)
    d = m.to_dict()
    assert "date_created" in d
    assert d["a"] == 1


def test_to_dict_can_omit_date():
    m = ModelMetadata(a=1)
    d = m.to_dict(include_date=False)
    assert "date_created" not in d
    assert d == {"a": 1}


def test_as_dict_alias():
    m = ModelMetadata(a=1)
    assert m.as_dict() == m.to_dict()


# ---------------------------------------------------------------------- #
# Pickle + copy
# ---------------------------------------------------------------------- #
def test_pickle_roundtrip_preserves_extras():
    m = ModelMetadata(run_id=7, case="x")
    restored = pickle.loads(pickle.dumps(m))
    assert isinstance(restored, ModelMetadata)
    assert restored.run_id == 7
    assert restored.case == "x"


def test_deepcopy_roundtrip():
    m = ModelMetadata(run_id=7)
    dup = copy.deepcopy(m)
    assert dup is not m
    assert dup.run_id == 7
    dup.run_id = 99
    assert m.run_id == 7  # deepcopy detached the _extras dict


def test_setstate_tolerates_missing_extras():
    m = ModelMetadata.__new__(ModelMetadata)
    m.__setstate__({})  # tolerant — empty state must not raise
    assert m.to_dict(include_date=False) == {}
