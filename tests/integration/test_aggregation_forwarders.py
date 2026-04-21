"""Phase 4.3.2 regression tests: each NodalResults engineering method
is a thin forwarder to ``AggregationEngine``. These tests exercise the
forwarder against the single-partition ``elasticFrame`` example and
compare the forwarded result to a direct call on the engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.dataprocess import AggregationEngine


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
@pytest.fixture
def nodal_displacement(elastic_frame_dir: Path):
    """Build a displacement NodalResults over every node of the
    single-partition elasticFrame model, MODEL_STAGE[1]."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    return nr


# ---------------------------------------------------------------------- #
# delta_u
# ---------------------------------------------------------------------- #
def test_delta_u_forwarder_matches_engine_series(nodal_displacement):
    nr = nodal_displacement
    # component 1 (index 1) between nodes 1 and 2
    via_nr = nr.delta_u(top=1, bottom=2, component=1)
    via_eng = nr._aggregation_engine.delta_u(nr, top=1, bottom=2, component=1)

    assert isinstance(via_nr, pd.Series)
    pd.testing.assert_series_equal(via_nr, via_eng)


def test_delta_u_forwarder_abs_max_reduce(nodal_displacement):
    nr = nodal_displacement
    v = nr.delta_u(top=1, bottom=2, component=1, reduce="abs_max")
    assert isinstance(v, float)
    # Must equal nanmax(abs(series))
    s = nr.delta_u(top=1, bottom=2, component=1, reduce="series")
    assert v == pytest.approx(float(np.nanmax(np.abs(s.to_numpy(dtype=float)))))


def test_delta_u_signed_false_equals_abs_of_signed(nodal_displacement):
    nr = nodal_displacement
    signed = nr.delta_u(top=1, bottom=2, component=1, signed=True)
    unsigned = nr.delta_u(top=1, bottom=2, component=1, signed=False)
    pd.testing.assert_series_equal(unsigned, signed.abs().rename(unsigned.name))


def test_delta_u_unknown_reduce_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="Unknown reduce"):
        nr.delta_u(top=1, bottom=2, component=1, reduce="nope")


# ---------------------------------------------------------------------- #
# drift
# ---------------------------------------------------------------------- #
def test_drift_forwarder_matches_engine_series(nodal_displacement):
    nr = nodal_displacement
    via_nr = nr.drift(top=1, bottom=2, component=1)
    via_eng = nr._aggregation_engine.drift(nr, top=1, bottom=2, component=1)

    assert isinstance(via_nr, pd.Series)
    pd.testing.assert_series_equal(via_nr, via_eng)


def test_drift_equals_delta_u_divided_by_dz(nodal_displacement):
    """drift(t) = delta_u(t) / (z_top - z_bot)."""
    nr = nodal_displacement
    ni = nr.info.nodes_info
    zcol = nr.info._resolve_column(ni, "z", required=True)
    nid_col = nr.info._resolve_column(ni, "node_id", required=False)

    def _z(nid: int) -> float:
        if nid_col is not None:
            return float(ni.loc[ni[nid_col].to_numpy() == nid].iloc[0][zcol])
        return float(ni.loc[nid, zcol])

    dz = _z(1) - _z(2)
    if dz == 0.0:
        pytest.skip("elasticFrame nodes 1 and 2 share z; pick different nodes if this ever changes.")

    du = nr.delta_u(top=1, bottom=2, component=1)
    dr = nr.drift(top=1, bottom=2, component=1)
    pd.testing.assert_series_equal(dr, (du / dz).rename(dr.name))


def test_drift_abs_max_reduce_matches_nanmax_of_series(nodal_displacement):
    nr = nodal_displacement
    s = nr.drift(top=1, bottom=2, component=1)
    v = nr.drift(top=1, bottom=2, component=1, reduce="abs_max")
    assert v == pytest.approx(float(np.nanmax(np.abs(s.to_numpy(dtype=float)))))


def test_drift_zero_dz_raises(nodal_displacement):
    """Two nodes at the same z should fail with the documented error."""
    nr = nodal_displacement
    ni = nr.info.nodes_info
    zcol = nr.info._resolve_column(ni, "z", required=True)
    nid_col = nr.info._resolve_column(ni, "node_id", required=False)

    # find two distinct nodes sharing the same z
    if nid_col is not None:
        by_z = ni.groupby(zcol)[nid_col].apply(list)
    else:
        by_z = ni.groupby(zcol).apply(lambda s: list(s.index))
    pair = None
    for nids in by_z:
        if len(nids) >= 2:
            pair = (int(nids[0]), int(nids[1]))
            break
    if pair is None:
        pytest.skip("No pair of nodes at identical z in elasticFrame fixture.")

    with pytest.raises(ValueError, match="dz"):
        nr.drift(top=pair[0], bottom=pair[1], component=1)


def test_drift_unknown_reduce_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="Unknown reduce"):
        nr.drift(top=1, bottom=2, component=1, reduce="nope")


# ---------------------------------------------------------------------- #
# residual_drift
# ---------------------------------------------------------------------- #
def test_residual_drift_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    via_nr = nr.residual_drift(top=1, bottom=2, component=1)
    via_eng = nr._aggregation_engine.residual_drift(nr, top=1, bottom=2, component=1)
    assert isinstance(via_nr, float)
    assert via_nr == pytest.approx(via_eng)


def test_residual_drift_tail_one_equals_last_drift_sample(nodal_displacement):
    """With tail=1 the residual equals the last sample of drift(top, bottom)."""
    nr = nodal_displacement
    series = nr.drift(top=1, bottom=2, component=1)
    r = nr.residual_drift(top=1, bottom=2, component=1, tail=1, agg="mean")
    assert r == pytest.approx(float(series.iloc[-1]))


def test_residual_drift_tail_median_matches_manual(nodal_displacement):
    nr = nodal_displacement
    series = nr.drift(top=1, bottom=2, component=1)
    a = series.to_numpy(dtype=float)
    tail = min(3, a.size)
    r = nr.residual_drift(top=1, bottom=2, component=1, tail=tail, agg="median")
    assert r == pytest.approx(float(np.nanmedian(a[-tail:])))


def test_residual_drift_tail_saturates_to_series_length(nodal_displacement):
    """Asking for a tail > series length is clipped silently to the whole series."""
    nr = nodal_displacement
    series = nr.drift(top=1, bottom=2, component=1)
    a = series.to_numpy(dtype=float)
    huge = a.size + 100
    r = nr.residual_drift(top=1, bottom=2, component=1, tail=huge, agg="mean")
    assert r == pytest.approx(float(np.nanmean(a)))


def test_residual_drift_rejects_bad_tail_and_agg(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="tail"):
        nr.residual_drift(top=1, bottom=2, component=1, tail=0)
    with pytest.raises(ValueError, match="agg"):
        nr.residual_drift(top=1, bottom=2, component=1, agg="nope")


# ---------------------------------------------------------------------- #
# _resolve_story_nodes_by_z_tol
# ---------------------------------------------------------------------- #
def test_resolve_stories_forwarder_matches_engine(nodal_displacement):
    """Forwarder == engine call; output is a sorted list of (z, [node_ids])."""
    nr = nodal_displacement
    via_nr = nr._resolve_story_nodes_by_z_tol(
        selection_set_id=None,
        selection_set_name=None,
        node_ids=[1, 2, 3, 4],
        coordinates=None,
        dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine._resolve_story_nodes_by_z_tol(
        nr,
        selection_set_id=None,
        selection_set_name=None,
        node_ids=[1, 2, 3, 4],
        coordinates=None,
        dz_tol=1e-6,
    )
    assert via_nr == via_eng

    # Sorted by z, each entry is (z, list[int])
    zs = [z for z, _ in via_nr]
    assert zs == sorted(zs)
    for _, nids in via_nr:
        assert all(isinstance(n, int) for n in nids)


def test_resolve_stories_large_tol_merges_all_into_one_level(nodal_displacement):
    nr = nodal_displacement
    stories = nr._resolve_story_nodes_by_z_tol(
        selection_set_id=None,
        selection_set_name=None,
        node_ids=[1, 2, 3, 4],
        coordinates=None,
        dz_tol=1e9,
    )
    # Every node clusters to the first story.
    assert len(stories) == 1
    assert sorted(stories[0][1]) == [1, 2, 3, 4]


def test_resolve_stories_requires_exactly_one_selector(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="exactly ONE"):
        nr._resolve_story_nodes_by_z_tol(
            selection_set_id=None,
            selection_set_name=None,
            node_ids=[1, 2],
            coordinates=[(0.0, 0.0, 0.0)],
            dz_tol=1e-3,
        )
    with pytest.raises(ValueError, match="exactly ONE"):
        nr._resolve_story_nodes_by_z_tol(
            selection_set_id=None,
            selection_set_name=None,
            node_ids=None,
            coordinates=None,
            dz_tol=1e-3,
        )


# ---------------------------------------------------------------------- #
# roof_torsion
# ---------------------------------------------------------------------- #
def _two_nodes_with_distinct_xy(nr) -> tuple[int, int]:
    """Pick any two node ids from nodes_info that have different (x, y)."""
    ni = nr.info.nodes_info
    xcol = nr.info._resolve_column(ni, "x", required=True)
    ycol = nr.info._resolve_column(ni, "y", required=True)
    nid_col = nr.info._resolve_column(ni, "node_id", required=False)

    if nid_col is not None:
        rows = list(zip(ni[nid_col], ni[xcol], ni[ycol]))
    else:
        rows = list(zip(ni.index, ni[xcol], ni[ycol]))

    for i, (ni_a, xa, ya) in enumerate(rows):
        for ni_b, xb, yb in rows[i + 1:]:
            if (float(xa), float(ya)) != (float(xb), float(yb)):
                return int(ni_a), int(ni_b)
    pytest.skip("No pair of nodes with distinct (x, y) in fixture.")


def test_roof_torsion_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    a, b = _two_nodes_with_distinct_xy(nr)
    via_nr = nr.roof_torsion(node_a_id=a, node_b_id=b)
    via_eng = nr._aggregation_engine.roof_torsion(nr, node_a_id=a, node_b_id=b)
    assert isinstance(via_nr, pd.Series)
    pd.testing.assert_series_equal(via_nr, via_eng)


def test_roof_torsion_same_node_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="same node id"):
        nr.roof_torsion(node_a_id=1, node_b_id=1)


def test_roof_torsion_requires_exactly_one_id_or_coord(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="exactly one"):
        nr.roof_torsion(node_a_id=1)  # missing node_b


def test_roof_torsion_abs_max_matches_nanmax(nodal_displacement):
    nr = nodal_displacement
    a, b = _two_nodes_with_distinct_xy(nr)
    s = nr.roof_torsion(node_a_id=a, node_b_id=b, reduce="series")
    v = nr.roof_torsion(node_a_id=a, node_b_id=b, reduce="abs_max")
    assert v == pytest.approx(float(np.nanmax(np.abs(s.to_numpy(dtype=float)))))


def test_roof_torsion_return_residual_tuple_and_columns(nodal_displacement):
    nr = nodal_displacement
    a, b = _two_nodes_with_distinct_xy(nr)
    out, debug = nr.roof_torsion(node_a_id=a, node_b_id=b, return_residual=True)
    assert isinstance(out, pd.Series)
    assert isinstance(debug, pd.DataFrame)
    assert {"du", "dv", "du_rot", "dv_rot", "ru", "rv"}.issubset(debug.columns)


def test_roof_torsion_return_quality_adds_rigidity_columns(nodal_displacement):
    nr = nodal_displacement
    a, b = _two_nodes_with_distinct_xy(nr)
    _, debug = nr.roof_torsion(node_a_id=a, node_b_id=b, return_quality=True)
    assert {"rel_norm", "res_norm", "rigidity_ratio"}.issubset(debug.columns)


def test_roof_torsion_unknown_reduce_raises(nodal_displacement):
    nr = nodal_displacement
    a, b = _two_nodes_with_distinct_xy(nr)
    with pytest.raises(ValueError, match="reduce must be"):
        nr.roof_torsion(node_a_id=a, node_b_id=b, reduce="nope")


# ---------------------------------------------------------------------- #
# Engine sanity
# ---------------------------------------------------------------------- #
def test_class_level_engine_is_shared_singleton(nodal_displacement):
    nr = nodal_displacement
    # Same engine instance across all NodalResults objects — it is stateless
    # and lives as a class attribute.
    assert isinstance(nr._aggregation_engine, AggregationEngine)
    from STKO_to_python.results.nodal_results_dataclass import NodalResults
    assert nr._aggregation_engine is NodalResults._aggregation_engine
