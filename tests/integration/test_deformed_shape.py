"""Integration tests for ``ds.plot.deformed_shape`` / ``undeformed_shape``.

Exercises the real fixtures (single- and multi-partition) and verifies:

* deformed positions equal ``node_coords + scale * displacement_at_step``
  cell-by-cell;
* edge counts per element class match the expected per-class topology
  (1 for beams, 3 for tris, 4 for quads, 12 for bricks);
* the ``(ax, meta)`` contract holds and ``ax`` accepts overlays.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Single-partition (line elements only)                                   #
# ---------------------------------------------------------------------- #
def test_deformed_shape_elastic_frame_returns_ax_and_meta(
    elastic_frame_dir: Path,
) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.deformed_shape(
        model_stage="MODEL_STAGE[1]", step=5, scale=10.0
    )
    try:
        assert ax is not None
        assert isinstance(meta, dict)
        assert "deformed_coords" in meta
        assert "original_coords" in meta
        assert "edges_per_class" in meta

        # 3 ElasticBeam3d elements, 1 edge each -> 3 segments.
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["segment_count"] == 3
        assert meta["skipped_classes"] == []
        # Beam frame in (x, z) is laid out vertically — _decide_3d picks
        # 3D when z varies.
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


def test_deformed_coords_match_disp_cell_by_cell(
    elastic_frame_dir: Path,
) -> None:
    """deformed_coords[node] == original_coords[node] + scale * disp."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    scale = 250.0
    step = 7
    stage = "MODEL_STAGE[1]"

    ax, meta = ds.plot.deformed_shape(
        model_stage=stage, step=step, scale=scale, show_undeformed=False
    )
    try:
        # Independent fetch: get DISPLACEMENT for every node, snapshot at step.
        all_ids = ds.nodes_info["dataframe"]["node_id"].tolist()
        nr = ds.nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage=stage,
            node_ids=all_ids,
        )
        disp_snap = nr.df.xs(step, level="step")
        # MultiIndex columns; pull DISPLACEMENT block.
        disp_snap = disp_snap.loc[:, [c for c in disp_snap.columns if c[0] == "DISPLACEMENT"]]
        disp_snap.columns = [c[1] for c in disp_snap.columns]

        for nid in all_ids:
            orig = meta["original_coords"][int(nid)]
            d_meta = meta["deformed_coords"][int(nid)]
            d_row = disp_snap.loc[int(nid)].to_numpy(dtype=float)
            # Pad to length 3 if needed.
            if d_row.size < 3:
                d_row = np.concatenate(
                    [d_row, np.zeros(3 - d_row.size, dtype=float)]
                )
            expected = orig + scale * d_row[:3]
            np.testing.assert_allclose(d_meta, expected, rtol=0.0, atol=1e-9)
    finally:
        plt.close("all")


def test_undeformed_shape_elastic_frame(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.undeformed_shape()
    try:
        assert ax is not None
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["segment_count"] == 3
        # Coordinates exactly match the dataset's nodes_info.
        df_nodes = ds.nodes_info["dataframe"]
        for row in df_nodes.itertuples(index=False):
            np.testing.assert_array_equal(
                meta["original_coords"][int(row.node_id)],
                np.array([row.x, row.y, row.z], dtype=float),
            )
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Multi-partition (shell quads + beams)                                   #
# ---------------------------------------------------------------------- #
def test_deformed_shape_quad_frame_topology(quad_frame_dir: Path) -> None:
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.deformed_shape(
        model_stage="MODEL_STAGE[1]", step=4, scale=5.0
    )
    try:
        # Counts in QuadFrame fixture: 75 beams + 625 ASDShellQ4.
        # Beams contribute 75 segments, quads 625 * 4 = 2500.
        assert meta["edges_per_class"] == {
            "5-ElasticBeam3d(2n)": 1,
            "203-ASDShellQ4(4n)": 4,
        }
        assert meta["segment_count"] == 75 * 1 + 625 * 4
        assert meta["skipped_classes"] == []
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


def test_undeformed_segment_count_matches_deformed(
    quad_frame_dir: Path,
) -> None:
    """Topology is geometry-independent, so segment counts must match."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    _, meta_def = ds.plot.deformed_shape(
        model_stage="MODEL_STAGE[1]", step=2, scale=1.0
    )
    plt.close("all")
    _, meta_und = ds.plot.undeformed_shape()
    plt.close("all")
    assert meta_def["segment_count"] == meta_und["segment_count"]
    assert meta_def["edges_per_class"] == meta_und["edges_per_class"]


# ---------------------------------------------------------------------- #
# Composition with user axes
# ---------------------------------------------------------------------- #
def test_deformed_shape_accepts_user_axes(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    fig = plt.figure()
    user_ax = fig.add_subplot(111, projection="3d")
    try:
        ax, meta = ds.plot.deformed_shape(
            model_stage="MODEL_STAGE[1]",
            step=3,
            scale=50.0,
            ax=user_ax,
        )
        assert ax is user_ax
        # User axes must still receive the segments.
        assert meta["segment_count"] > 0
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Edge cases                                                              #
# ---------------------------------------------------------------------- #
def test_deformed_shape_scale_zero_matches_original(
    elastic_frame_dir: Path,
) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.deformed_shape(
        model_stage="MODEL_STAGE[1]", step=5, scale=0.0, show_undeformed=False
    )
    try:
        for nid, orig in meta["original_coords"].items():
            np.testing.assert_array_equal(meta["deformed_coords"][nid], orig)
    finally:
        plt.close("all")


def test_deformed_shape_invalid_step(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    with pytest.raises(ValueError, match="step="):
        ds.plot.deformed_shape(
            model_stage="MODEL_STAGE[1]", step=99999, scale=1.0
        )
    plt.close("all")
