"""Unit tests for :mod:`STKO_to_python.utilities.coords`."""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.utilities.coords import (
    x_physical_to_natural,
    xi_natural_to_physical,
)


# ----- xi_natural_to_physical --------------------------------------------- #


def test_xi_natural_to_physical_endpoints():
    """ξ=-1 → 0, ξ=+1 → L."""
    assert xi_natural_to_physical(-1.0, 10.0) == pytest.approx(0.0)
    assert xi_natural_to_physical(1.0, 10.0) == pytest.approx(10.0)


def test_xi_natural_to_physical_midpoint():
    """ξ=0 → L/2."""
    assert xi_natural_to_physical(0.0, 8.0) == pytest.approx(4.0)


def test_xi_natural_to_physical_array():
    """Array of 5 Lobatto points → physical positions on a 4 m beam."""
    xi = np.array([-1.0, -0.65465367, 0.0, 0.65465367, 1.0])
    x = xi_natural_to_physical(xi, 4.0)
    assert x.shape == (5,)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(4.0)
    assert x[2] == pytest.approx(2.0)


def test_xi_natural_to_physical_zero_length_raises():
    with pytest.raises(ValueError, match="positive"):
        xi_natural_to_physical(0.0, 0.0)


def test_xi_natural_to_physical_negative_length_raises():
    with pytest.raises(ValueError, match="positive"):
        xi_natural_to_physical(0.0, -2.0)


# ----- x_physical_to_natural ---------------------------------------------- #


def test_x_physical_to_natural_endpoints():
    """0 → -1, L → +1."""
    assert x_physical_to_natural(0.0, 10.0) == pytest.approx(-1.0)
    assert x_physical_to_natural(10.0, 10.0) == pytest.approx(1.0)


def test_x_physical_to_natural_midpoint():
    """L/2 → 0."""
    assert x_physical_to_natural(5.0, 10.0) == pytest.approx(0.0)


def test_round_trip_through_both_directions():
    """xi → x → xi recovers the input."""
    L = 7.5
    xi = np.array([-0.9, -0.3, 0.0, 0.4, 0.95])
    x = xi_natural_to_physical(xi, L)
    xi2 = x_physical_to_natural(x, L)
    np.testing.assert_array_almost_equal(xi2, xi)


# ----- ElementResults.physical_x integration ------------------------------ #


def test_element_results_physical_x_integration(elastic_frame_dir):
    """End-to-end: a 5-IP Lobatto beam's gp_xi → physical."""
    from STKO_to_python import MPCODataSet

    p = (
        elastic_frame_dir.parent
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip("displacement-based fixture not available")

    ds = MPCODataSet(str(p), "results", verbose=False)
    beam_ids = ds.elements_info["dataframe"]["element_id"].tolist()[:1]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=beam_ids,
    )
    x = er.physical_x(2.0)  # 2 m beam
    assert x.shape == (5,)
    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(2.0)
    assert x[2] == pytest.approx(1.0)


def test_element_results_physical_x_closed_form_raises(elastic_frame_dir):
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1],
    )
    with pytest.raises(ValueError, match="closed-form"):
        er.physical_x(1.0)
