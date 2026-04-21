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
# base_rocking
# ---------------------------------------------------------------------- #
def _three_base_points(nr) -> tuple[list[tuple[float, float]], float]:
    """Return ([(x,y)] * 3, z_coord). Uses nodes_info to pick any three
    distinct-(x,y) base nodes at the minimum-z level; if none exist with
    distinct (x,y), returns three points from the fixture (possibly
    collinear — still useful for exercising the singular fallback)."""
    ni = nr.info.nodes_info
    xcol = nr.info._resolve_column(ni, "x", required=True)
    ycol = nr.info._resolve_column(ni, "y", required=True)
    zcol = nr.info._resolve_column(ni, "z", required=True)

    rows = list(zip(ni[xcol], ni[ycol], ni[zcol]))
    z_min = float(min(r[2] for r in rows))
    base = [(float(x), float(y)) for x, y, z in rows if z == z_min]
    if len(base) >= 3:
        return base[:3], z_min
    # fall back to any three nodes' xy (collinear → singular path)
    return [(float(x), float(y)) for x, y, _ in rows[:3]], z_min


def test_base_rocking_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    pts, z = _three_base_points(nr)
    via_nr = nr.base_rocking(node_coords_xy=pts, z_coord=z, uz_component=3)
    via_eng = nr._aggregation_engine.base_rocking(
        nr, node_coords_xy=pts, z_coord=z, uz_component=3
    )
    # Either both dicts (abs_max) or both DataFrames (series). Default is series.
    assert isinstance(via_nr, pd.DataFrame)
    pd.testing.assert_frame_equal(via_nr, via_eng)


def test_base_rocking_output_columns(nodal_displacement):
    nr = nodal_displacement
    pts, z = _three_base_points(nr)
    out = nr.base_rocking(node_coords_xy=pts, z_coord=z, uz_component=3)
    assert {"w0", "theta_x_rad", "theta_y_rad", "theta_mag_rad", "is_singular"}.issubset(
        out.columns
    )


def test_base_rocking_requires_three_points(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="exactly 3"):
        nr.base_rocking(node_coords_xy=[(0.0, 0.0), (1.0, 0.0)], z_coord=0.0)


def test_base_rocking_singular_geometry_fallback(nodal_displacement):
    """Three duplicate points → singular geometry → zero-rocking fallback,
    is_singular=True, no exception raised."""
    nr = nodal_displacement
    ni = nr.info.nodes_info
    xcol = nr.info._resolve_column(ni, "x", required=True)
    ycol = nr.info._resolve_column(ni, "y", required=True)
    zcol = nr.info._resolve_column(ni, "z", required=True)
    # any node's coords, repeated
    row0 = ni.iloc[0]
    pt = (float(row0[xcol]), float(row0[ycol]))
    z = float(row0[zcol])
    out = nr.base_rocking(node_coords_xy=[pt, pt, pt], z_coord=z, uz_component=3)
    assert isinstance(out, pd.DataFrame)
    assert bool(out["is_singular"].iloc[0]) is True
    assert (out["theta_x_rad"] == 0.0).all()
    assert (out["theta_y_rad"] == 0.0).all()


def test_base_rocking_abs_max_reduce_shape(nodal_displacement):
    nr = nodal_displacement
    pts, z = _three_base_points(nr)
    out = nr.base_rocking(
        node_coords_xy=pts, z_coord=z, uz_component=3, reduce="abs_max"
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {"theta_x_abs_max", "theta_y_abs_max", "theta_mag_abs_max"}


def test_base_rocking_unknown_reduce_raises(nodal_displacement):
    nr = nodal_displacement
    pts, z = _three_base_points(nr)
    with pytest.raises(ValueError, match="reduce must be"):
        nr.base_rocking(node_coords_xy=pts, z_coord=z, uz_component=3, reduce="nope")


# ---------------------------------------------------------------------- #
# asce_torsional_irregularity
# ---------------------------------------------------------------------- #
def _pick_two_sides(nr) -> tuple[tuple, tuple, tuple, tuple]:
    """Pick (A_top, A_bot, B_top, B_bot) as (x,y,z) tuples from nodes_info,
    such that A_top/A_bot share the same (x,y) with different z, and
    similarly for B; sides A and B differ in (x,y). Returns the coords of
    real nodes so nearest-node resolution is deterministic."""
    ni = nr.info.nodes_info
    xcol = nr.info._resolve_column(ni, "x", required=True)
    ycol = nr.info._resolve_column(ni, "y", required=True)
    zcol = nr.info._resolve_column(ni, "z", required=True)

    rows = [(float(r[xcol]), float(r[ycol]), float(r[zcol])) for _, r in ni.iterrows()]

    # group by (x, y)
    by_xy: dict[tuple[float, float], list[float]] = {}
    for x, y, z in rows:
        by_xy.setdefault((x, y), []).append(z)

    # find two (x, y) columns that each have >= 2 z levels
    cols = [(xy, sorted(zs)) for xy, zs in by_xy.items() if len(zs) >= 2]
    if len(cols) < 2:
        pytest.skip("fixture lacks two distinct (x,y) columns each with >= 2 z levels.")

    (xa, ya), zs_a = cols[0]
    (xb, yb), zs_b = cols[1]
    return (
        (xa, ya, zs_a[-1]),   # A top (highest z)
        (xa, ya, zs_a[0]),    # A bottom (lowest z)
        (xb, yb, zs_b[-1]),
        (xb, yb, zs_b[0]),
    )


def test_asce_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    A_top, A_bot, B_top, B_bot = _pick_two_sides(nr)
    via_nr = nr.asce_torsional_irregularity(
        component=1,
        side_a_top=A_top, side_a_bottom=A_bot,
        side_b_top=B_top, side_b_bottom=B_bot,
    )
    via_eng = nr._aggregation_engine.asce_torsional_irregularity(
        nr, component=1,
        side_a_top=A_top, side_a_bottom=A_bot,
        side_b_top=B_top, side_b_bottom=B_bot,
    )
    assert isinstance(via_nr, dict)
    # Exclude keys that might be comparison-sensitive; check the structural keys equal.
    for k in ("drift_A", "drift_B", "drift_avg", "drift_max", "ratio", "ctrl_side"):
        assert via_nr[k] == via_eng[k]
    assert via_nr["node_ids"] == via_eng["node_ids"]
    assert via_nr["metadata"] == via_eng["metadata"]


def test_asce_result_keys_and_metadata(nodal_displacement):
    nr = nodal_displacement
    A_top, A_bot, B_top, B_bot = _pick_two_sides(nr)
    out = nr.asce_torsional_irregularity(
        component=1,
        side_a_top=A_top, side_a_bottom=A_bot,
        side_b_top=B_top, side_b_bottom=B_bot,
    )
    assert set(out.keys()) == {
        "drift_A", "drift_B", "drift_avg", "drift_max",
        "ratio", "ctrl_side", "node_ids", "metadata",
    }
    assert set(out["node_ids"].keys()) == {"A_top", "A_bottom", "B_top", "B_bottom"}
    assert out["metadata"]["definition"] == "max_over_avg"
    assert out["metadata"]["reduce_time"] == "abs_max"


def test_asce_ordering_invariants(nodal_displacement):
    """Structural invariants: drift_max >= drift_avg, magnitudes are
    non-negative, and ctrl_side is whichever side's magnitude is larger.
    Avoids a ratio-magnitude check because elasticFrame drifts can be
    below the eps guard for this fixture geometry."""
    nr = nodal_displacement
    A_top, A_bot, B_top, B_bot = _pick_two_sides(nr)
    out = nr.asce_torsional_irregularity(
        component=1,
        side_a_top=A_top, side_a_bottom=A_bot,
        side_b_top=B_top, side_b_bottom=B_bot,
    )
    assert out["drift_A"] >= 0.0
    assert out["drift_B"] >= 0.0
    assert out["drift_max"] >= out["drift_avg"]
    expected_side = "A" if out["drift_A"] >= out["drift_B"] else "B"
    assert out["ctrl_side"] == expected_side


def test_asce_rejects_non_tuple_coord(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(TypeError, match="must be a tuple"):
        nr.asce_torsional_irregularity(
            component=1,
            side_a_top=[0.0, 0.0, 0.0],   # list, not tuple
            side_a_bottom=(0.0, 0.0, 0.0),
            side_b_top=(0.0, 0.0, 0.0),
            side_b_bottom=(0.0, 0.0, 0.0),
        )


def test_asce_bad_definition_raises(nodal_displacement):
    nr = nodal_displacement
    A_top, A_bot, B_top, B_bot = _pick_two_sides(nr)
    with pytest.raises(ValueError, match="definition"):
        nr.asce_torsional_irregularity(
            component=1,
            side_a_top=A_top, side_a_bottom=A_bot,
            side_b_top=B_top, side_b_bottom=B_bot,
            definition="nope",
        )


def test_asce_bad_reduce_time_raises(nodal_displacement):
    nr = nodal_displacement
    A_top, A_bot, B_top, B_bot = _pick_two_sides(nr)
    with pytest.raises(ValueError, match="reduce_time"):
        nr.asce_torsional_irregularity(
            component=1,
            side_a_top=A_top, side_a_bottom=A_bot,
            side_b_top=B_top, side_b_bottom=B_bot,
            reduce_time="nope",
        )


# ---------------------------------------------------------------------- #
# interstory_drift_envelope
# ---------------------------------------------------------------------- #
def _distinct_z_levels_exist(nr) -> bool:
    ni = nr.info.nodes_info
    zcol = nr.info._resolve_column(ni, "z", required=True)
    return ni[zcol].nunique() >= 2


def test_interstory_drift_envelope_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    via_nr = nr.interstory_drift_envelope(
        component=1,
        node_ids=[1, 2, 3, 4],
        dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine.interstory_drift_envelope(
        nr,
        component=1,
        node_ids=[1, 2, 3, 4],
        dz_tol=1e-6,
    )
    assert isinstance(via_nr, pd.DataFrame)
    pd.testing.assert_frame_equal(via_nr, via_eng)


def test_interstory_drift_envelope_columns_and_index(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.interstory_drift_envelope(
        component=1,
        node_ids=[1, 2, 3, 4],
        dz_tol=1e-6,
    )
    assert out.index.names == ["z_lower", "z_upper"]
    assert {
        "z_lower", "z_upper",
        "lower_node", "upper_node", "dz",
        "max_drift", "min_drift", "max_abs_drift",
    }.issubset(out.columns)
    # envelope ordering: max >= min, max_abs >= 0
    assert (out["max_drift"] >= out["min_drift"]).all()
    assert (out["max_abs_drift"] >= 0).all()


def test_interstory_drift_envelope_too_few_stories_raises(nodal_displacement):
    """With a huge tolerance every node merges into one cluster."""
    nr = nodal_displacement
    with pytest.raises(ValueError, match="at least 2 story levels"):
        nr.interstory_drift_envelope(
            component=1,
            node_ids=[1, 2, 3, 4],
            dz_tol=1e9,
        )


def test_interstory_drift_envelope_max_abs_peak_representative(nodal_displacement):
    """`representative='max_abs_peak'` uses a different per-story node picker;
    run both representatives and verify the result schema matches."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.interstory_drift_envelope(
        component=1,
        node_ids=[1, 2, 3, 4],
        dz_tol=1e-6,
        representative="max_abs_peak",
    )
    assert {"lower_node", "upper_node", "max_drift", "min_drift"}.issubset(out.columns)


def test_interstory_drift_envelope_unknown_representative_raises(nodal_displacement):
    """Unknown representative should raise once _pick_node is called.
    Skip when the fixture's geometry clusters into a single story (no
    inter-story pairs → _pick_node never fires)."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    with pytest.raises(ValueError, match="Unknown representative"):
        nr.interstory_drift_envelope(
            component=1,
            node_ids=[1, 2, 3, 4],
            dz_tol=1e-6,
            representative="nope",
        )


# ---------------------------------------------------------------------- #
# story_pga_envelope
# ---------------------------------------------------------------------- #
def test_story_pga_envelope_forwarder_matches_engine(nodal_displacement):
    """result_name is pinned to DISPLACEMENT (what the fixture exposes);
    the envelope logic is identical to what it would do for ACCELERATION."""
    nr = nodal_displacement
    via_nr = nr.story_pga_envelope(
        component=1,
        node_ids=[1, 2, 3, 4],
        result_name="DISPLACEMENT",
        dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine.story_pga_envelope(
        nr,
        component=1,
        node_ids=[1, 2, 3, 4],
        result_name="DISPLACEMENT",
        dz_tol=1e-6,
    )
    assert isinstance(via_nr, pd.DataFrame)
    pd.testing.assert_frame_equal(via_nr, via_eng)


def test_story_pga_envelope_columns_and_ordering(nodal_displacement):
    nr = nodal_displacement
    out = nr.story_pga_envelope(
        component=1,
        node_ids=[1, 2, 3, 4],
        result_name="DISPLACEMENT",
        dz_tol=1e-6,
    )
    assert out.index.name == "story_z"
    assert {
        "n_nodes", "n_nodes_present", "max_acc", "min_acc", "pga",
        "ctrl_node_max", "ctrl_node_min", "ctrl_node_pga",
    }.issubset(out.columns)
    # story_z sorted ascending
    zs = out.index.to_numpy()
    assert (zs[:-1] <= zs[1:]).all()
    # envelope ordering + pga non-negative
    assert (out["max_acc"] >= out["min_acc"]).all()
    assert (out["pga"] >= 0).all()


def test_story_pga_envelope_to_g_divides(nodal_displacement):
    """to_g=True scales results by 1/g_value."""
    nr = nodal_displacement
    out_raw = nr.story_pga_envelope(
        component=1, node_ids=[1, 2, 3, 4], result_name="DISPLACEMENT",
        dz_tol=1e-6,
    )
    out_g = nr.story_pga_envelope(
        component=1, node_ids=[1, 2, 3, 4], result_name="DISPLACEMENT",
        dz_tol=1e-6, to_g=True, g_value=2.0,
    )
    pd.testing.assert_series_equal(
        out_g["pga"], out_raw["pga"] / 2.0, check_names=False,
    )


def test_story_pga_envelope_unknown_reduce_nodes_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="reduce_nodes"):
        nr.story_pga_envelope(
            component=1, node_ids=[1, 2, 3, 4], result_name="DISPLACEMENT",
            dz_tol=1e-6, reduce_nodes="nope",
        )


# ---------------------------------------------------------------------- #
# residual_interstory_drift_profile
# ---------------------------------------------------------------------- #
def test_residual_interstory_profile_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    via_nr = nr.residual_interstory_drift_profile(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine.residual_interstory_drift_profile(
        nr, component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert isinstance(via_nr, pd.DataFrame)
    pd.testing.assert_frame_equal(via_nr, via_eng)


def test_residual_interstory_profile_columns(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.residual_interstory_drift_profile(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert out.index.names == ["z_lower", "z_upper"]
    assert {"z_lower", "z_upper", "lower_node", "upper_node", "dz", "residual_drift"}.issubset(
        out.columns
    )


def test_residual_interstory_profile_values_match_pairwise_residual_drift(nodal_displacement):
    """Each row's residual_drift equals calling residual_drift on its
    (lower_node, upper_node) pair with the same tail/agg."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.residual_interstory_drift_profile(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
        tail=1, agg="mean",
    )
    for _, row in out.iterrows():
        r_pair = nr.residual_drift(
            top=int(row["upper_node"]),
            bottom=int(row["lower_node"]),
            component=1,
            tail=1, agg="mean",
        )
        assert float(row["residual_drift"]) == pytest.approx(r_pair)


def test_residual_interstory_profile_too_few_stories_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="at least 2 story levels"):
        nr.residual_interstory_drift_profile(
            component=1, node_ids=[1, 2, 3, 4], dz_tol=1e9,
        )


# ---------------------------------------------------------------------- #
# residual_drift_envelope
# ---------------------------------------------------------------------- #
def test_residual_drift_envelope_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    via_nr = nr.residual_drift_envelope(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine.residual_drift_envelope(
        nr, component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert via_nr == via_eng


def test_residual_drift_envelope_keys_and_invariants(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.residual_drift_envelope(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert set(out.keys()) == {
        "max_abs_residual_story_drift",
        "max_pos_residual_story_drift",
        "max_neg_residual_story_drift",
    }
    # max_abs >= max(|max_pos|, |max_neg|) by construction
    assert out["max_abs_residual_story_drift"] >= abs(out["max_pos_residual_story_drift"])
    assert out["max_abs_residual_story_drift"] >= abs(out["max_neg_residual_story_drift"])


def test_residual_drift_envelope_matches_profile_reduction(nodal_displacement):
    """The envelope values equal the reductions over the profile."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    prof = nr.residual_interstory_drift_profile(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
        signed=True, tail=1, agg="mean",
    )
    env = nr.residual_drift_envelope(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
        tail=1, agg="mean",
    )
    r = prof["residual_drift"].to_numpy(dtype=float)
    assert env["max_abs_residual_story_drift"] == pytest.approx(float(np.nanmax(np.abs(r))))
    assert env["max_pos_residual_story_drift"] == pytest.approx(float(np.nanmax(r)))
    assert env["max_neg_residual_story_drift"] == pytest.approx(float(np.nanmin(r)))


# ---------------------------------------------------------------------- #
# interstory_drift_envelope_pd
# ---------------------------------------------------------------------- #
def test_interstory_drift_envelope_pd_forwarder_matches_engine(nodal_displacement):
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    via_nr = nr.interstory_drift_envelope_pd(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    via_eng = nr._aggregation_engine.interstory_drift_envelope_pd(
        nr, component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert isinstance(via_nr, pd.DataFrame)
    pd.testing.assert_frame_equal(via_nr, via_eng)


def test_interstory_drift_envelope_pd_flat_columns_and_sort(nodal_displacement):
    """Flat RangeIndex (no MultiIndex), z_lower sorted ascending."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    out = nr.interstory_drift_envelope_pd(
        component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
    )
    assert isinstance(out.index, pd.RangeIndex)
    assert {
        "z_lower", "z_upper", "dz", "max_drift", "min_drift",
        "max_abs_drift", "representative_drift", "lower_node", "upper_node",
    } == set(out.columns)
    zs = out["z_lower"].to_numpy()
    assert (zs[:-1] <= zs[1:]).all()


def test_interstory_drift_envelope_pd_representative_picks_correct_column(nodal_displacement):
    """representative='max' picks max_drift; 'min' picks min_drift; 'max_abs' picks max_abs_drift."""
    nr = nodal_displacement
    if not _distinct_z_levels_exist(nr):
        pytest.skip("fixture has < 2 distinct z levels.")
    kwargs = dict(component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6)
    out_max_abs = nr.interstory_drift_envelope_pd(representative="max_abs", **kwargs)
    out_max = nr.interstory_drift_envelope_pd(representative="max", **kwargs)
    out_min = nr.interstory_drift_envelope_pd(representative="min", **kwargs)
    pd.testing.assert_series_equal(
        out_max_abs["representative_drift"], out_max_abs["max_abs_drift"], check_names=False
    )
    pd.testing.assert_series_equal(
        out_max["representative_drift"], out_max["max_drift"], check_names=False
    )
    pd.testing.assert_series_equal(
        out_min["representative_drift"], out_min["min_drift"], check_names=False
    )


def test_interstory_drift_envelope_pd_unknown_representative_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="representative"):
        nr.interstory_drift_envelope_pd(
            component=1, node_ids=[1, 2, 3, 4], dz_tol=1e-6,
            representative="nope",
        )


# ---------------------------------------------------------------------- #
# orbit
# ---------------------------------------------------------------------- #
def test_orbit_forwarder_matches_engine_single_node(nodal_displacement):
    nr = nodal_displacement
    via_nr = nr.orbit(node_ids=1, x_component=1, y_component=2)
    via_eng = nr._aggregation_engine.orbit(nr, node_ids=1, x_component=1, y_component=2)
    assert isinstance(via_nr, tuple) and len(via_nr) == 2
    pd.testing.assert_series_equal(via_nr[0], via_eng[0])
    pd.testing.assert_series_equal(via_nr[1], via_eng[1])


def test_orbit_return_nodes_adds_ids_tuple(nodal_displacement):
    nr = nodal_displacement
    out = nr.orbit(node_ids=[1, 2], x_component=1, y_component=2, return_nodes=True)
    assert isinstance(out, tuple) and len(out) == 3
    sx, sy, ids = out
    assert isinstance(sx, pd.Series)
    assert isinstance(sy, pd.Series)
    assert sorted(ids) == [1, 2]


def test_orbit_series_names_encode_components(nodal_displacement):
    nr = nodal_displacement
    sx, sy = nr.orbit(node_ids=1, x_component=1, y_component=2)
    assert sx.name == "DISPLACEMENT[1]"
    assert sy.name == "DISPLACEMENT[2]"


def test_orbit_reduce_nodes_mean_matches_pandas_groupby(nodal_displacement):
    """reduce_nodes='mean' collapses per-step across nodes equals groupby on step."""
    nr = nodal_displacement
    # Raw two-node result (MultiIndex node_id, step) then manual mean per step
    raw_sx, raw_sy = nr.orbit(node_ids=[1, 2], x_component=1, y_component=2)
    expected_x = raw_sx.groupby(level=-1).mean()
    expected_y = raw_sy.groupby(level=-1).mean()

    sx, sy = nr.orbit(
        node_ids=[1, 2], x_component=1, y_component=2, reduce_nodes="mean",
    )
    pd.testing.assert_series_equal(
        sx.reset_index(drop=True),
        expected_x.reset_index(drop=True),
        check_names=False,
    )
    pd.testing.assert_series_equal(
        sy.reset_index(drop=True),
        expected_y.reset_index(drop=True),
        check_names=False,
    )


def test_orbit_requires_exactly_one_selector(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="exactly ONE"):
        nr.orbit(node_ids=1, selection_set_id=1)
    with pytest.raises(ValueError, match="exactly ONE"):
        nr.orbit()  # no selector


def test_orbit_unknown_reduce_nodes_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="reduce_nodes"):
        nr.orbit(node_ids=[1, 2], reduce_nodes="nope")


def test_orbit_signed_false_abs(nodal_displacement):
    nr = nodal_displacement
    sx_signed, sy_signed = nr.orbit(node_ids=1, x_component=1, y_component=2)
    sx_abs, sy_abs = nr.orbit(
        node_ids=1, x_component=1, y_component=2, signed=False,
    )
    pd.testing.assert_series_equal(sx_abs, sx_signed.abs().rename(sx_abs.name))
    pd.testing.assert_series_equal(sy_abs, sy_signed.abs().rename(sy_abs.name))


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
