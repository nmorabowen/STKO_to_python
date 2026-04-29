"""Unit tests for :mod:`STKO_to_python.utilities.shape_functions`.

Validates shape-function evaluations and Jacobian determinants on
simple analytic geometries:

* Unit cube and scaled brick — square Jacobian.
* Flat shell rectangle — surface Jacobian (area measure).
* Line element — line Jacobian (length measure).

Plus interpolation sanity (shape functions sum to 1 at every IP and
exactly recover node coords at parametric corners).
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.format.gauss_points import (
    gauss_legendre_1d,
    tensor_product_2d,
    tensor_product_3d,
)
from STKO_to_python.format.shape_functions import (
    SHAPE_FUNCTIONS,
    compute_jacobian_dets,
    compute_physical_coords,
    get_shape_functions,
)


# ---------------------------------------------------------------------- #
# Catalog                                                                  #
# ---------------------------------------------------------------------- #


def test_catalog_has_known_classes():
    assert "203-ASDShellQ4" in SHAPE_FUNCTIONS
    assert "56-Brick" in SHAPE_FUNCTIONS
    assert "5-ElasticBeam3d" in SHAPE_FUNCTIONS
    assert "64-DispBeamColumn3d" in SHAPE_FUNCTIONS


def test_get_shape_functions_unknown_returns_none():
    assert get_shape_functions("99-MysteryElement") is None


def test_get_shape_functions_geom_kinds():
    assert get_shape_functions("56-Brick")[2] == "solid"
    assert get_shape_functions("203-ASDShellQ4")[2] == "shell"
    assert get_shape_functions("5-ElasticBeam3d")[2] == "line"


# ---------------------------------------------------------------------- #
# Brick (8-node trilinear hex)                                             #
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def unit_brick_nodes():
    """Cube centered at origin, sides length 2 — node coords match
    natural coords exactly. Used so that physical_coords should equal
    the natural coords for any IP."""
    return np.array(
        [[
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ]],
        dtype=np.float64,
    )


def test_brick_shape_functions_partition_of_unity():
    """Σ N_i = 1 at every IP (consistency test)."""
    N_fn, _, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat, _ = tensor_product_3d(2)
    N = N_fn(nat)
    assert np.allclose(N.sum(axis=1), 1.0)


def test_brick_shape_functions_recover_node_at_corner():
    """At the parametric corner (-1,-1,-1), only N_1 should be 1."""
    N_fn, _, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat = np.array([[-1, -1, -1]], dtype=np.float64)
    N = N_fn(nat)
    assert N[0, 0] == pytest.approx(1.0)
    for j in range(1, 8):
        assert N[0, j] == pytest.approx(0.0)


def test_brick_physical_coords_unit_cube_match_natural(unit_brick_nodes):
    """Identity mapping: physical coords equal natural coords."""
    N_fn, _, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat, _ = tensor_product_3d(2)
    phys = compute_physical_coords(nat, unit_brick_nodes, N_fn)
    np.testing.assert_allclose(phys[0], nat)


def test_brick_jacobian_unit_cube_is_identity(unit_brick_nodes):
    _, dN_fn, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat, _ = tensor_product_3d(2)
    detJ = compute_jacobian_dets(nat, unit_brick_nodes, dN_fn, "solid")
    assert np.allclose(detJ[0], 1.0)


def test_brick_volume_recovery_unit_cube(unit_brick_nodes):
    """∫ 1 dV over [-1,1]^3 = 8."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat, weights = tensor_product_3d(2)
    detJ = compute_jacobian_dets(nat, unit_brick_nodes, dN_fn, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(8.0)


def test_brick_volume_recovery_scaled():
    """Scaled brick of size 4×6×10 → volume 240."""
    N_fn, dN_fn, _ = SHAPE_FUNCTIONS["56-Brick"]
    nodes = np.array(
        [[
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ]],
        dtype=np.float64,
    ) * np.array([2.0, 3.0, 5.0])  # half-widths
    nat, weights = tensor_product_3d(2)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(240.0)


def test_brick_volume_invariant_under_shear():
    """Shearing the top face doesn't change the volume."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["56-Brick"]
    nodes = np.array(
        [[
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ]],
        dtype=np.float64,
    )
    nodes[0, 4:, 0] += 1.0  # translate top face +x
    nat, weights = tensor_product_3d(2)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "solid")
    volume = (1.0 * weights * detJ[0]).sum()
    assert volume == pytest.approx(8.0)


# ---------------------------------------------------------------------- #
# ASDShellQ4 (4-node bilinear)                                             #
# ---------------------------------------------------------------------- #


def test_shell_shape_functions_partition_of_unity():
    N_fn, _, _ = SHAPE_FUNCTIONS["203-ASDShellQ4"]
    nat, _ = tensor_product_2d(2)
    N = N_fn(nat)
    assert np.allclose(N.sum(axis=1), 1.0)


def test_shell_area_recovery_xy_rectangle():
    """4×6 rectangle in the xy plane → area 24."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["203-ASDShellQ4"]
    nodes = np.array(
        [[[-2, -3, 0], [+2, -3, 0], [+2, +3, 0], [-2, +3, 0]]],
        dtype=np.float64,
    )
    nat, weights = tensor_product_2d(2)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "shell")
    area = (1.0 * weights * detJ[0]).sum()
    assert area == pytest.approx(24.0)


def test_shell_area_independent_of_orientation():
    """Same 4×6 rectangle rotated to lie in the xz plane should give
    the same area."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["203-ASDShellQ4"]
    nodes = np.array(
        [[[-2, 0, -3], [+2, 0, -3], [+2, 0, +3], [-2, 0, +3]]],
        dtype=np.float64,
    )
    nat, weights = tensor_product_2d(2)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "shell")
    area = (1.0 * weights * detJ[0]).sum()
    assert area == pytest.approx(24.0)


# ---------------------------------------------------------------------- #
# Line element (2-node)                                                    #
# ---------------------------------------------------------------------- #


def test_line_jacobian_equals_half_length():
    """For a 2-node line of length L, |J| = L/2 at every IP."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["5-ElasticBeam3d"]
    L = 7.5
    nodes = np.array([[[0, 0, 0], [L, 0, 0]]], dtype=np.float64)
    nat_pts, _ = gauss_legendre_1d(3)
    nat = nat_pts.reshape(-1, 1)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "line")
    assert np.allclose(detJ[0], L / 2)


def test_line_length_recovery():
    _, dN_fn, _ = SHAPE_FUNCTIONS["5-ElasticBeam3d"]
    L = 12.0
    nodes = np.array([[[0, 0, 0], [L, 0, 0]]], dtype=np.float64)
    nat_pts, weights = gauss_legendre_1d(3)
    nat = nat_pts.reshape(-1, 1)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "line")
    length = (1.0 * weights * detJ[0]).sum()
    assert length == pytest.approx(L)


def test_line_length_for_arbitrary_orientation():
    """Length is invariant under direction in 3-D space."""
    _, dN_fn, _ = SHAPE_FUNCTIONS["5-ElasticBeam3d"]
    end = np.array([3.0, 4.0, 12.0])  # length √(9+16+144) = √169 = 13
    nodes = np.array([[[0, 0, 0], end]], dtype=np.float64)
    nat_pts, weights = gauss_legendre_1d(3)
    nat = nat_pts.reshape(-1, 1)
    detJ = compute_jacobian_dets(nat, nodes, dN_fn, "line")
    length = (1.0 * weights * detJ[0]).sum()
    assert length == pytest.approx(13.0)


# ---------------------------------------------------------------------- #
# Vectorization across multiple elements                                  #
# ---------------------------------------------------------------------- #


def test_compute_physical_coords_vectorizes_over_elements():
    N_fn, _, _ = SHAPE_FUNCTIONS["56-Brick"]
    nat, _ = tensor_product_3d(2)
    # Two unit bricks centered at different points
    base = np.array(
        [
            [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
            [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
        ],
        dtype=np.float64,
    )
    nodes = np.stack([base + 10.0, base - 5.0], axis=0)  # (2, 8, 3)
    phys = compute_physical_coords(nat, nodes, N_fn)
    assert phys.shape == (2, 8, 3)
    # Element 0 IP positions are natural + 10
    np.testing.assert_allclose(phys[0], nat + 10.0)
    np.testing.assert_allclose(phys[1], nat - 5.0)


def test_compute_jacobian_dets_unknown_geom_kind_raises():
    _, dN_fn, _ = SHAPE_FUNCTIONS["56-Brick"]
    nodes = np.zeros((1, 8, 3))
    nat, _ = tensor_product_3d(2)
    with pytest.raises(ValueError, match="Unknown geom_kind"):
        compute_jacobian_dets(nat, nodes, dN_fn, "fishtail")
