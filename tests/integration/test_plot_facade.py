"""Phase 4.4.4 — ``Plot.xy`` dataset-level convenience wrapper.

Verifies that ``ds.plot.xy(...)`` fetches a NodalResults and delegates
to ``NodalResultsPlotter.xy`` producing the same output as the manual
two-step flow (fetch → plot).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.plotting.plot import Plot


# ---------------------------------------------------------------------- #
# Facade shape
# ---------------------------------------------------------------------- #
def test_plot_facade_on_dataset(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    assert isinstance(ds.plot, Plot)
    assert "Plot facade" in repr(ds.plot)


def test_plot_facade_has_xy(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    assert callable(ds.plot.xy)


# ---------------------------------------------------------------------- #
# ds.plot.xy == ds.nodes.get_nodal_results(...) then nr.plot.xy(...)
# ---------------------------------------------------------------------- #
def test_plot_xy_one_shot_matches_two_step(elastic_frame_dir: Path):
    """The convenience wrapper should produce numerically identical
    output to the manual fetch→plot path."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)

    # Manual path
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    _, meta_manual = nr.plot.xy(
        y_results_name="DISPLACEMENT",
        y_direction=1,
        y_operation="Max",
        x_results_name="TIME",
    )
    plt.close("all")

    # Convenience path
    _, meta_shortcut = ds.plot.xy(
        model_stage="MODEL_STAGE[1]",
        results_name="DISPLACEMENT",
        node_ids=[1, 2, 3, 4],
        y_direction=1,
        y_operation="Max",
        x_results_name="TIME",
    )
    plt.close("all")

    # Both should produce the same x / y arrays (NodalResultsPlotter.xy
    # keys them as "x" and "y", not "x_array"/"y_array").
    assert "x" in meta_manual and "y" in meta_manual
    assert "x" in meta_shortcut and "y" in meta_shortcut
    np.testing.assert_array_equal(
        np.asarray(meta_manual["x"]), np.asarray(meta_shortcut["x"])
    )
    np.testing.assert_array_equal(
        np.asarray(meta_manual["y"]), np.asarray(meta_shortcut["y"])
    )


def test_plot_xy_returns_axes_and_meta(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ax, meta = ds.plot.xy(
        model_stage="MODEL_STAGE[1]",
        results_name="DISPLACEMENT",
        node_ids=[1, 2, 3, 4],
        y_direction=1,
        y_operation="Max",
        x_results_name="TIME",
    )
    assert ax is not None
    assert isinstance(meta, dict)
    assert "x" in meta
    assert "y" in meta
    plt.close("all")
