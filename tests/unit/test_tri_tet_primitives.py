"""Unit tests for the tri/tet shape-function and Gauss-point primitives.

These primitives are exported as building blocks; users register them
under their own element-class keys when those classes appear in
fixtures (e.g. ``SHAPE_FUNCTIONS["204-ASDShellT3"] = (tri3_N, tri3_dN,
"shell")``). The tests verify the math is correct on the canonical
parent domains (unit triangle and unit tetrahedron); end-to-end
integration tests will follow once a real fixture lands.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.format.gauss_points import (
    gauss_tetrahedron,
    gauss_triangle,
)
from STKO_to_python.format.shape_functions import (
    compute_jacobian_dets,
    compute_physical_coords,
    tet4_N,
    tet4_dN,
    tri3_N,
    tri3_dN,
)


# ---------------------------------------------------------------------- #
# gauss_triangle                                                          #
# ---------------------------------------------------------------------- #


def test_gauss_triangle_1pt_centroid():
    coords, weights = gauss_triangle(1)
    assert coords.shape == (1, 2)
    np.testing.assert_allclose(coords, [[1 / 3, 1 / 3]])
    np.testing.assert_allclose(weights, [0.5])


def test_gauss_triangle_3pt_weights_sum_to_half():
    coords, weights = gauss_triangle(3)
    assert coords.shape == (3, 2)
    assert weights.sum() == pytest.approx(0.5)


def test_gauss_triangle_unsupported_n_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        gauss_triangle(7)


# ---------------------------------------------------------------------- #
# gauss_tetrahedron                                                       #
# ---------------------------------------------------------------------- #


def test_gauss_tetrahedron_1pt_centroid():
    coords, weights = gauss_tetrahedron(1)
    np.testing.assert_allclose(coords, [[0.25, 0.25, 0.25]])
    np.testing.assert_allclose(weights, [1.0 / 6.0])


def test_gauss_tetrahedron_4pt_weights_sum_to_sixth():
    coords, weights = gauss_tetrahedron(4)
    assert coords.shape == (4, 3)
    assert weights.sum() == pytest.approx(1.0 / 6.0)


# ---------------------------------------------------------------------- #
# tri3 shape functions                                                    #
# ---------------------------------------------------------------------- #


def test_tri3_partition_of_unity():
    coords, _ = gauss_triangle(3)
    N = tri3_N(coords)
    assert np.allclose(N.sum(axis=1), 1.0)


def test_tri3_recovers_node_at_corner():
    """At node 1 (0, 0): N_1=1, others=0."""
    nat = np.array([[0.0, 0.0]])
    N = tri3_N(nat)
    np.testing.assert_allclose(N[0], [1.0, 0.0, 0.0])

    # Node 2 (1, 0)
    np.testing.assert_allclose(tri3_N(np.array([[1.0, 0.0]]))[0], [0.0, 1.0, 0.0])
    # Node 3 (0, 1)
    np.testing.assert_allclose(tri3_N(np.array([[0.0, 1.0]]))[0], [0.0, 0.0, 1.0])


def test_tri3_area_recovery_unit_triangle_in_xy():
    """Map a triangle with vertices at (0,0,0), (1,0,0), (0,1,0) — area
    1/2. Integrate 1 → 1/2."""
    nodes = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    coords, weights = gauss_triangle(3)
    detJ = compute_jacobian_dets(coords, nodes, tri3_dN, "shell")
    area = (1.0 * weights * detJ[0]).sum()
    assert area == pytest.approx(0.5)


def test_tri3_area_scaled_triangle():
    """Vertices (0,0,0), (3,0,0), (0,4,0) → area = 6."""
    nodes = np.array([[[0, 0, 0], [3, 0, 0], [0, 4, 0]]], dtype=np.float64)
    coords, weights = gauss_triangle(3)
    detJ = compute_jacobian_dets(coords, nodes, tri3_dN, "shell")
    area = (1.0 * weights * detJ[0]).sum()
    assert area == pytest.approx(6.0)


def test_tri3_area_invariant_under_rotation():
    """Same triangle rotated into the xz plane gives the same area."""
    nodes = np.array([[[0, 0, 0], [3, 0, 0], [0, 0, 4]]], dtype=np.float64)
    coords, weights = gauss_triangle(3)
    detJ = compute_jacobian_dets(coords, nodes, tri3_dN, "shell")
    area = (1.0 * weights * detJ[0]).sum()
    assert area == pytest.approx(6.0)


# ---------------------------------------------------------------------- #
# tet4 shape functions                                                    #
# ---------------------------------------------------------------------- #


def test_tet4_partition_of_unity():
    coords, _ = gauss_tetrahedron(4)
    N = tet4_N(coords)
    assert np.allclose(N.sum(axis=1), 1.0)


def test_tet4_recovers_nodes_at_corners():
    np.testing.assert_allclose(
        tet4_N(np.array([[0.0, 0.0, 0.0]]))[0], [1.0, 0.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        tet4_N(np.array([[1.0, 0.0, 0.0]]))[0], [0.0, 1.0, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        tet4_N(np.array([[0.0, 1.0, 0.0]]))[0], [0.0, 0.0, 1.0, 0.0]
    )
    np.testing.assert_allclose(
        tet4_N(np.array([[0.0, 0.0, 1.0]]))[0], [0.0, 0.0, 0.0, 1.0]
    )


def test_tet4_volume_unit_tet():
    """Reference unit tet: volume = 1/6."""
    nodes = np.array(
        [[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]], dtype=np.float64
    )
    coords, weights = gauss_tetrahedron(4)
    detJ = compute_jacobian_dets(coords, nodes, tet4_dN, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(1.0 / 6.0)


def test_tet4_volume_scaled_tet():
    """Tet with edges 3, 4, 5 along axes: volume = (3*4*5)/6 = 10."""
    nodes = np.array(
        [[[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]]], dtype=np.float64
    )
    coords, weights = gauss_tetrahedron(4)
    detJ = compute_jacobian_dets(coords, nodes, tet4_dN, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(10.0)


def test_tet4_volume_works_with_1pt_rule():
    """1-pt centroid rule integrates degree-1 exactly — including 1."""
    nodes = np.array(
        [[[0, 0, 0], [3, 0, 0], [0, 4, 0], [0, 0, 5]]], dtype=np.float64
    )
    coords, weights = gauss_tetrahedron(1)
    detJ = compute_jacobian_dets(coords, nodes, tet4_dN, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(10.0)


# ---------------------------------------------------------------------- #
# Registration pattern                                                    #
# ---------------------------------------------------------------------- #


def test_user_registration_pattern_works():
    """Document the API contract: user adds a class to the catalog by
    assigning to the SHAPE_FUNCTIONS dict."""
    from STKO_to_python.format.shape_functions import (
        SHAPE_FUNCTIONS,
        get_shape_functions,
    )

    key = "999-MyTestShellT3"
    SHAPE_FUNCTIONS[key] = (tri3_N, tri3_dN, "shell")
    try:
        fns = get_shape_functions(key)
        assert fns is not None
        N_fn, dN_fn, kind = fns
        assert N_fn is tri3_N
        assert dN_fn is tri3_dN
        assert kind == "shell"
    finally:
        del SHAPE_FUNCTIONS[key]
