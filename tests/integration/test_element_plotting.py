"""End-to-end tests for ``ElementResults.plot.*``.

Uses the ``Agg`` matplotlib backend so the tests run headlessly. Each
test asserts the meta dict the plotter returns, since pixel-level
checks are fragile and unnecessary — the meta carries the same x/y
arrays that drive the plot.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # before any other matplotlib import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from STKO_to_python import MPCODataSet


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ---------------------------------------------------------------------- #
# history()                                                               #
# ---------------------------------------------------------------------- #


def test_history_single_element(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.history("N_1", element_ids=1)
    assert ax is not None
    assert "x" in meta and "y_per_element" in meta
    assert list(meta["y_per_element"].keys()) == [1]
    assert meta["y_per_element"][1].shape == (er.n_steps,)


def test_history_multi_element(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.history("N_1")
    assert sorted(meta["y_per_element"].keys()) == [1, 2, 3]


def test_history_x_axis_step_works_without_time(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.history("N_1", x_axis="step")
    np.testing.assert_array_equal(meta["x"], np.arange(er.n_steps))


def test_history_unknown_component_raises(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="not in this result"):
        er.plot.history("nonexistent")


# ---------------------------------------------------------------------- #
# diagram()  — line elements                                              #
# ---------------------------------------------------------------------- #


def test_diagram_line_element_physical_x():
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip("disp-based fixture missing")
    ds = MPCODataSet(str(p), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.diagram("axial_force", element_id=1, step=5)
    # Values are one per IP.
    assert meta["y"].shape == (er.n_ip,)
    # x-axis spans the beam length.
    assert meta["x"][0] == pytest.approx(0.0, abs=1e-9)
    assert meta["x"][-1] > meta["x"][0]


def test_diagram_line_element_natural_x():
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip("disp-based fixture missing")
    ds = MPCODataSet(str(p), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.diagram(
        "bending_moment_z", element_id=1, step=5, x_in_natural=True
    )
    np.testing.assert_allclose(meta["x"], er.gp_xi)


def test_diagram_raises_on_shell(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=[shell_id],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="line elements"):
        er.plot.diagram("membrane_xx", element_id=shell_id, step=0)


def test_diagram_raises_on_unknown_canonical(elastic_frame_dir: Path):
    """A canonical that doesn't apply to the bucket should raise even
    on a line element."""
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip("disp-based fixture missing")
    ds = MPCODataSet(str(p), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="doesn't match"):
        er.plot.diagram("membrane_xx", element_id=1, step=0)


# ---------------------------------------------------------------------- #
# scatter()  — shells / solids                                            #
# ---------------------------------------------------------------------- #


def test_scatter_shell(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_ids = [
        int(i)
        for i in ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .head(20)
    ]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=shell_ids,
        model_stage="MODEL_STAGE[1]",
    )
    ax, meta = er.plot.scatter("membrane_xx", step=2)
    # 20 elements × 4 IPs = 80 points
    assert meta["x"].shape == (len(shell_ids) * er.n_ip,)
    assert meta["y"].shape == meta["x"].shape
    assert meta["values"].shape == meta["x"].shape


def test_scatter_brick_with_axis_choice(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_ids = [
        int(i)
        for i in ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .head(10)
    ]
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=brick_ids,
        model_stage="MODEL_STAGE[1]",
    )
    # x/z elevation view
    ax, meta = er.plot.scatter("stress_11", step=1, axes=("x", "z"))
    assert meta["x"].shape == (len(brick_ids) * er.n_ip,)


def test_scatter_invalid_axis_raises(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=[shell_id],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="axes must be"):
        er.plot.scatter("membrane_xx", step=0, axes=("u", "v"))


def test_scatter_raises_on_closed_form(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="physical_coords"):
        er.plot.scatter("axial_force", step=0)


# ---------------------------------------------------------------------- #
# Plotter survives pickle roundtrip                                       #
# ---------------------------------------------------------------------- #


def test_plotter_rebuilds_after_pickle(elastic_frame_dir: Path, tmp_path: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2],
        model_stage="MODEL_STAGE[1]",
    )
    pkl = tmp_path / "er.pkl"
    er.save_pickle(pkl)

    from STKO_to_python.elements.element_results import ElementResults

    er2 = ElementResults.load_pickle(pkl)
    # plot attribute exists and works
    assert er2.plot is not None
    ax, meta = er2.plot.history("N_1")
    assert ax is not None
