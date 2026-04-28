"""Integration tests for ``ds.plot.mesh`` and ``ds.plot.mesh_with_contour``.

Verifies:

* ``mesh()`` returns ``(ax, meta)`` with ``n_edges`` / ``n_elements_drawn``
  matching the per-class topology (1 for beams, 4 for quads, 12 for
  bricks);
* ``element_type`` and ``element_ids`` filters narrow the rendered set;
* ``er.plot.scatter`` overlays cleanly when handed an ``ax`` returned by
  ``mesh()`` (composition contract);
* ``mesh_with_contour`` is sugar that produces the same composition in
  one call.
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
# Single-partition (line elements only)
# ---------------------------------------------------------------------- #
def test_mesh_elastic_frame_basic_contract(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.mesh()
    try:
        assert ax is not None
        assert isinstance(meta, dict)
        # 3 ElasticBeam3d elements, 1 edge each.
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["n_edges"] == 3
        assert meta["n_elements_drawn"] == 3
        assert meta["skipped_classes"] == []
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Multi-partition (shell quads + beams)
# ---------------------------------------------------------------------- #
def test_mesh_quad_frame_topology(quad_frame_dir: Path) -> None:
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.mesh()
    try:
        # 75 beams × 1 + 625 quads × 4 = 2575 edges total.
        assert meta["edges_per_class"] == {
            "5-ElasticBeam3d(2n)": 1,
            "203-ASDShellQ4(4n)": 4,
        }
        assert meta["n_edges"] == 75 * 1 + 625 * 4
        assert meta["n_elements_drawn"] == 75 + 625
        assert meta["skipped_classes"] == []
    finally:
        plt.close("all")


def test_mesh_filter_by_element_type(quad_frame_dir: Path) -> None:
    """Selecting ASDShellQ4 alone should yield exactly 50 quads → 200 edges
    sample (50 picked) — well, the fixture has 625; spec asks for the
    50→200 expectation phrased in narrow terms. Verify by element_ids
    instead so the count is reproducible."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    df = ds.elements_info["dataframe"]
    quad_ids = df.loc[
        df["element_type"] == "203-ASDShellQ4", "element_id"
    ].head(50).tolist()
    ax, meta = ds.plot.mesh(
        element_type="203-ASDShellQ4", element_ids=quad_ids
    )
    try:
        # 50 quad shells → 200 edges, single class.
        assert meta["edges_per_class"] == {"203-ASDShellQ4(4n)": 4}
        assert meta["n_edges"] == 200
        assert meta["n_elements_drawn"] == 50
    finally:
        plt.close("all")


def test_mesh_filter_only_beams(quad_frame_dir: Path) -> None:
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.mesh(element_type="5-ElasticBeam3d")
    try:
        assert meta["edges_per_class"] == {"5-ElasticBeam3d(2n)": 1}
        assert meta["n_edges"] == 75
        assert meta["n_elements_drawn"] == 75
    finally:
        plt.close("all")


def test_mesh_empty_filter_raises(quad_frame_dir: Path) -> None:
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    with pytest.raises(ValueError, match="No elements remain"):
        ds.plot.mesh(element_ids=[999_999_999])
    plt.close("all")


# ---------------------------------------------------------------------- #
# Composition with er.plot.scatter
# ---------------------------------------------------------------------- #
def test_mesh_then_scatter_compose(quad_frame_dir: Path) -> None:
    """mesh() returns ax; scatter(..., ax=ax) overlays onto same axes."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_ids = (
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .head(20)
        .tolist()
    )
    try:
        ax, mesh_meta = ds.plot.mesh(
            element_type="203-ASDShellQ4", element_ids=shell_ids
        )
        n_lines_before = len(ax.collections)
        assert n_lines_before >= 1  # mesh produced a LineCollection

        er = ds.elements.get_element_results(
            results_name="section.force",
            element_type="203-ASDShellQ4",
            element_ids=shell_ids,
            model_stage="MODEL_STAGE[1]",
        )
        _, scatter_meta = er.plot.scatter("membrane_xx", step=2, ax=ax)
        # Scatter adds a PathCollection on the same axes.
        assert len(ax.collections) > n_lines_before
        assert scatter_meta["x"].size == 20 * er.n_ip
    finally:
        plt.close("all")


def test_mesh_with_contour_sugar(quad_frame_dir: Path) -> None:
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_ids = (
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .head(10)
        .tolist()
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=shell_ids,
        model_stage="MODEL_STAGE[1]",
    )
    try:
        ax, meta = ds.plot.mesh_with_contour(
            er,
            "membrane_xx",
            step=2,
            element_type="203-ASDShellQ4",
            element_ids=shell_ids,
        )
        assert "mesh" in meta and "scatter" in meta
        assert meta["mesh"]["n_edges"] == 10 * 4
        assert meta["scatter"]["x"].size == 10 * er.n_ip
        # Single ax holds both the mesh wireframe and the scatter.
        assert len(ax.collections) >= 2
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# Solid example (skipped if absent)
# ---------------------------------------------------------------------- #
def test_mesh_solid_partition_brick_topology(solid_partition_dir: Path) -> None:
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    df = ds.elements_info["dataframe"]
    n_brick = int((df["element_type"] == "56-Brick").sum())
    if n_brick == 0:
        pytest.skip("No bricks in solid_partition fixture")
    try:
        ax, meta = ds.plot.mesh(element_type="56-Brick")
        # 12 edges per brick, single class.
        assert meta["edges_per_class"] == {"56-Brick(8n)": 12}
        assert meta["n_edges"] == n_brick * 12
        assert meta["n_elements_drawn"] == n_brick
        assert meta["is_3d"] is True
    finally:
        plt.close("all")


# ---------------------------------------------------------------------- #
# User-supplied axes
# ---------------------------------------------------------------------- #
def test_mesh_accepts_user_axes(elastic_frame_dir: Path) -> None:
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    fig = plt.figure()
    user_ax = fig.add_subplot(111, projection="3d")
    try:
        ax, meta = ds.plot.mesh(ax=user_ax)
        assert ax is user_ax
        assert meta["n_edges"] == 3
    finally:
        plt.close("all")
