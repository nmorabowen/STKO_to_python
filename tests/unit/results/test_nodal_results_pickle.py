"""Phase 4.3.3 — tolerant ``__setstate__`` for ``NodalResults``.

Covers:
  * A fresh NodalResults survives a full pickle round-trip.
  * An "old-layout" pickle (a dict with extra fields that the current
    class no longer stores) loads cleanly via ``__setstate__``, drops
    the unknown keys with a DEBUG log, and leaves the engine attached
    lazily as a class attribute.
  * A state dict missing optional fields does not raise.
"""
from __future__ import annotations

import logging
import pickle

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.dataprocess import AggregationEngine
from STKO_to_python.results.nodal_results_dataclass import NodalResults


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _tiny_nr() -> NodalResults:
    """Minimal NodalResults with a (node_id, step) index and a
    (result_name, component) columns MultiIndex. No reliance on a
    real dataset; just enough shape for fetch() / list_* / views."""
    idx = pd.MultiIndex.from_product(
        [[1, 2], [0, 1, 2]], names=("node_id", "step")
    )
    cols = pd.MultiIndex.from_tuples(
        [("DISPLACEMENT", "1"), ("DISPLACEMENT", "2")],
        names=("result", "component"),
    )
    df = pd.DataFrame(
        np.arange(idx.size * 2, dtype=float).reshape(idx.size, 2),
        index=idx,
        columns=cols,
    )
    return NodalResults(
        df=df,
        time=np.array([0.0, 0.1, 0.2]),
        name="tiny",
        nodes_ids=(1, 2),
        results_components=("DISPLACEMENT",),
    )


# ---------------------------------------------------------------------- #
# Round-trip
# ---------------------------------------------------------------------- #
def test_fresh_pickle_roundtrip_preserves_df_and_views():
    nr = _tiny_nr()
    blob = pickle.dumps(nr)
    nr2 = pickle.loads(blob)
    assert isinstance(nr2, NodalResults)
    pd.testing.assert_frame_equal(nr2.df, nr.df)
    # views rebuilt post-load
    assert "DISPLACEMENT" in nr2._views
    # engine still resolves via the class attribute
    assert isinstance(nr2._aggregation_engine, AggregationEngine)
    assert nr2._aggregation_engine is NodalResults._aggregation_engine


def test_getstate_drops_views():
    """_views is transient; it should not survive pickling."""
    nr = _tiny_nr()
    state = nr.__getstate__()
    assert state["_views"] is None


# ---------------------------------------------------------------------- #
# Tolerant __setstate__ — unknown keys
# ---------------------------------------------------------------------- #
def test_setstate_drops_unknown_keys_with_debug_log(caplog):
    """A pickle saved under an older class layout can carry fields
    (e.g. inline drift-profile caches) that the current class no longer
    stores. ``__setstate__`` drops them silently and emits a DEBUG
    record so the drop is observable under the right log level."""
    nr = _tiny_nr()
    state = nr.__getstate__()
    # Inject extra fields that the current class never writes.
    state["drift_profile_cache"] = {"ignored": 42}
    state["residual_cache"] = "stale"

    target = NodalResults.__new__(NodalResults)
    with caplog.at_level(logging.DEBUG, logger="STKO_to_python.results.nodal_results_dataclass"):
        target.__setstate__(state)

    # Known fields land on the instance.
    pd.testing.assert_frame_equal(target.df, nr.df)
    # Unknown fields did not leak in.
    assert "drift_profile_cache" not in target.__dict__
    assert "residual_cache" not in target.__dict__
    # DEBUG log mentions the dropped keys by name.
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    joined = "\n".join(messages)
    assert "drift_profile_cache" in joined
    assert "residual_cache" in joined


def test_setstate_engine_available_after_old_layout_load():
    """_aggregation_engine lives on the class, so even a pickle state
    that predates its introduction resolves the attribute after load."""
    nr = _tiny_nr()
    state = nr.__getstate__()
    # Remove the engine-if-ever-included (it never is, but be defensive).
    state.pop("_aggregation_engine", None)

    target = NodalResults.__new__(NodalResults)
    target.__setstate__(state)
    assert isinstance(target._aggregation_engine, AggregationEngine)


# ---------------------------------------------------------------------- #
# Tolerant __setstate__ — missing keys
# ---------------------------------------------------------------------- #
def test_setstate_missing_optional_fields_does_not_raise():
    """Missing optional fields (time, name, plot_settings) should not
    raise on load; accessing them later is the caller's problem."""
    nr = _tiny_nr()
    state = nr.__getstate__()
    # drop every optional field
    for k in ("time", "name", "plot_settings"):
        state.pop(k, None)

    target = NodalResults.__new__(NodalResults)
    target.__setstate__(state)  # must not raise
    # df remains intact; views rebuild
    assert hasattr(target, "df")
    assert target._views  # non-empty after rebuild


def test_setstate_with_only_df_still_builds_views():
    """Bare-minimum state (only df) still rebuilds _views."""
    nr = _tiny_nr()
    state = {"df": nr.df}

    target = NodalResults.__new__(NodalResults)
    target.__setstate__(state)
    assert "DISPLACEMENT" in target._views


def test_setstate_without_df_skips_view_build():
    """An even sparser state (no df) does not raise; _views is empty."""
    target = NodalResults.__new__(NodalResults)
    target.__setstate__({})  # no df, nothing else
    assert target._views == {}


# ---------------------------------------------------------------------- #
# Class-attribute contract
# ---------------------------------------------------------------------- #
def test_pickle_field_list_is_stable():
    """If this list changes, pickle loaders from older releases may
    need updating. Guard against accidental edits."""
    assert NodalResults._PICKLE_FIELDS == (
        "df", "time", "name", "info", "plot_settings",
    )
