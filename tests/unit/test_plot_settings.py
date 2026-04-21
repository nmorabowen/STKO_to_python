"""Unit tests for ``PlotSettings`` and its ``ModelPlotSettings`` alias."""
from __future__ import annotations

import copy
import pickle

import pytest

from STKO_to_python.plotting.plot_settings import PlotSettings
from STKO_to_python.plotting.plot_dataclasses import ModelPlotSettings


# ---------------------------------------------------------------------- #
# Alias identity
# ---------------------------------------------------------------------- #
def test_alias_is_plot_settings():
    assert ModelPlotSettings is PlotSettings


def test_top_level_plotting_exports_both_names():
    from STKO_to_python.plotting import ModelPlotSettings as M, PlotSettings as P
    assert M is P


# ---------------------------------------------------------------------- #
# Construction
# ---------------------------------------------------------------------- #
def test_default_construction_all_none():
    s = PlotSettings()
    for name in PlotSettings.__slots__:
        assert getattr(s, name) is None


def test_construction_sets_fields():
    s = PlotSettings(color="blue", linewidth=2.0, label_base="Roof")
    assert s.color == "blue"
    assert s.linewidth == 2.0
    assert s.label_base == "Roof"


def test_constructor_is_kwargs_only():
    """Per ``__slots__`` discipline and explicit OOP style, only kwargs."""
    with pytest.raises(TypeError):
        PlotSettings("blue")  # type: ignore[misc]


def test_slots_discipline():
    s = PlotSettings()
    assert PlotSettings.__slots__ == (
        "color", "linewidth", "linestyle", "label_base", "marker", "alpha",
    )
    with pytest.raises(AttributeError):
        s.extra_attr = 1  # type: ignore[attr-defined]


def test_repr_omits_none_fields():
    s = PlotSettings(color="red", linewidth=1.5)
    r = repr(s)
    assert "color='red'" in r
    assert "linewidth=1.5" in r
    assert "marker" not in r  # None → omitted
    assert "alpha" not in r


def test_equality():
    a = PlotSettings(color="red", linewidth=2)
    b = PlotSettings(color="red", linewidth=2)
    c = PlotSettings(color="red", linewidth=3)
    assert a == b
    assert a != c
    assert a != "not-a-PlotSettings"


# ---------------------------------------------------------------------- #
# to_mpl_kwargs
# ---------------------------------------------------------------------- #
def test_to_mpl_kwargs_skips_none():
    s = PlotSettings(color="red")
    assert s.to_mpl_kwargs() == {"color": "red"}


def test_to_mpl_kwargs_overrides_win():
    s = PlotSettings(color="red", linewidth=1.0)
    assert s.to_mpl_kwargs(color="blue") == {"color": "blue", "linewidth": 1.0}


def test_to_mpl_kwargs_empty_when_all_none():
    assert PlotSettings().to_mpl_kwargs() == {}


# ---------------------------------------------------------------------- #
# make_label
# ---------------------------------------------------------------------- #
def test_make_label_base_only():
    assert PlotSettings(label_base="Roof").make_label() == "Roof"


def test_make_label_base_and_suffix():
    assert PlotSettings(label_base="Roof").make_label(suffix="X") == "Roof X"


def test_make_label_no_base_uses_suffix():
    assert PlotSettings().make_label(suffix="X") == "X"


def test_make_label_no_base_no_suffix_uses_default():
    assert PlotSettings().make_label(default="fallback") == "fallback"


def test_make_label_all_none_returns_none():
    assert PlotSettings().make_label() is None


# ---------------------------------------------------------------------- #
# Pickle + copy
# ---------------------------------------------------------------------- #
def test_pickle_roundtrip():
    s = PlotSettings(color="red", linewidth=2.0, label_base="Demo")
    restored = pickle.loads(pickle.dumps(s))
    assert isinstance(restored, PlotSettings)
    assert restored == s


def test_deepcopy_roundtrip():
    s = PlotSettings(color="red")
    dup = copy.deepcopy(s)
    assert dup == s
    assert dup is not s


def test_setstate_tolerates_missing_keys():
    s = PlotSettings.__new__(PlotSettings)
    s.__setstate__({"color": "red"})  # other keys missing → default to None
    assert s.color == "red"
    assert s.linewidth is None
    assert s.marker is None
