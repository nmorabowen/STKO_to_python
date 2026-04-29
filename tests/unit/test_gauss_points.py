"""Unit tests for :mod:`STKO_to_python.utilities.gauss_points`.

Validates the 1-D / 2-D / 3-D Gauss-Legendre primitives and the
catalog lookups used to fill in ``ElementResults.gp_natural`` for
fixed-class elements (shells, plane elements, solids).

Ordering convention (per the module docstring): ξ varies fastest, then
η, then ζ. Tests pin this so a future refactor that flips the
iteration order breaks loudly.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.format.gauss_points import (
    ELEMENT_IP_CATALOG,
    gauss_legendre_1d,
    get_ip_layout,
    tensor_product_2d,
    tensor_product_3d,
)


# ---------------------------------------------------------------------- #
# 1-D Gauss-Legendre primitives                                           #
# ---------------------------------------------------------------------- #


def test_gl1d_n2_points_and_weights():
    pts, wts = gauss_legendre_1d(2)
    assert np.allclose(pts, [-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)])
    assert np.allclose(wts, [1.0, 1.0])


def test_gl1d_n3_points_and_weights():
    pts, wts = gauss_legendre_1d(3)
    assert np.allclose(pts, [-np.sqrt(3 / 5), 0.0, np.sqrt(3 / 5)])
    assert np.allclose(wts, [5 / 9, 8 / 9, 5 / 9])


def test_gl1d_weights_sum_to_2():
    """Sum of weights on the parent interval [-1, +1] is 2."""
    for n in range(1, 6):
        _, wts = gauss_legendre_1d(n)
        assert wts.sum() == pytest.approx(2.0), n


def test_gl1d_invalid_n_raises():
    with pytest.raises(ValueError, match="≥ 1"):
        gauss_legendre_1d(0)


def test_gl1d_is_cached():
    """lru_cache on the public function — same array object returned."""
    a, _ = gauss_legendre_1d(2)
    b, _ = gauss_legendre_1d(2)
    assert a is b


# ---------------------------------------------------------------------- #
# 2-D tensor product                                                       #
# ---------------------------------------------------------------------- #


def test_tensor_product_2d_n2_shape_and_ordering():
    coords, weights = tensor_product_2d(2)
    a = 1.0 / np.sqrt(3.0)
    expected = np.array(
        [
            [-a, -a],
            [+a, -a],
            [-a, +a],
            [+a, +a],
        ]
    )
    assert coords.shape == (4, 2)
    assert weights.shape == (4,)
    np.testing.assert_allclose(coords, expected, rtol=1e-12)
    assert weights.sum() == pytest.approx(4.0)  # area of [-1,1]^2


def test_tensor_product_2d_n3_count_and_weights_sum():
    coords, weights = tensor_product_2d(3)
    assert coords.shape == (9, 2)
    assert weights.sum() == pytest.approx(4.0)


# ---------------------------------------------------------------------- #
# 3-D tensor product                                                       #
# ---------------------------------------------------------------------- #


def test_tensor_product_3d_n2_shape_and_ordering():
    coords, weights = tensor_product_3d(2)
    a = 1.0 / np.sqrt(3.0)
    expected = np.array(
        [
            [-a, -a, -a],
            [+a, -a, -a],
            [-a, +a, -a],
            [+a, +a, -a],
            [-a, -a, +a],
            [+a, -a, +a],
            [-a, +a, +a],
            [+a, +a, +a],
        ]
    )
    assert coords.shape == (8, 3)
    assert weights.shape == (8,)
    np.testing.assert_allclose(coords, expected, rtol=1e-12)
    assert weights.sum() == pytest.approx(8.0)  # volume of [-1,1]^3


def test_tensor_product_3d_n3_count():
    coords, weights = tensor_product_3d(3)
    assert coords.shape == (27, 3)
    assert weights.sum() == pytest.approx(8.0)


# ---------------------------------------------------------------------- #
# Numerical-integration sanity                                             #
# ---------------------------------------------------------------------- #


def test_2d_gauss_legendre_integrates_polynomials_exactly():
    """Gauss-Legendre n=2 integrates polynomials of degree ≤ 3 exactly."""
    coords, weights = tensor_product_2d(2)
    # ∫∫ (xi^2 + eta^2) dxi deta over [-1,1]^2
    #   = ∫(2 xi^2) dxi + ∫(2 eta^2) deta
    #   = 2*(2/3) + 2*(2/3) = 8/3
    f = coords[:, 0] ** 2 + coords[:, 1] ** 2
    integral = (f * weights).sum()
    assert integral == pytest.approx(8 / 3)


def test_3d_gauss_legendre_integrates_polynomials_exactly():
    coords, weights = tensor_product_3d(2)
    # ∫∫∫ (xi*eta*zeta + 1) over [-1,1]^3 = 0 + 8 = 8
    f = coords[:, 0] * coords[:, 1] * coords[:, 2] + 1.0
    integral = (f * weights).sum()
    assert integral == pytest.approx(8.0)


# ---------------------------------------------------------------------- #
# Catalog lookups                                                          #
# ---------------------------------------------------------------------- #


def test_catalog_lookup_asd_shell_q4():
    layout = get_ip_layout("203-ASDShellQ4", 4)
    assert layout is not None
    coords, weights = layout
    assert coords.shape == (4, 2)
    assert weights.shape == (4,)
    assert weights.sum() == pytest.approx(4.0)


def test_catalog_lookup_brick_8ip():
    coords, weights = get_ip_layout("56-Brick", 8)
    assert coords.shape == (8, 3)
    assert weights.sum() == pytest.approx(8.0)


def test_catalog_lookup_brick_27ip():
    coords, weights = get_ip_layout("56-Brick", 27)
    assert coords.shape == (27, 3)
    assert weights.sum() == pytest.approx(8.0)


def test_catalog_unknown_class_returns_none():
    assert get_ip_layout("99-MysteryElement", 4) is None


def test_catalog_unknown_n_ip_returns_none():
    """Class is in the catalog but the requested IP count isn't."""
    assert get_ip_layout("56-Brick", 12) is None


def test_catalog_consistency_n_ip_matches_coords_length():
    """Every catalog entry advertises the right n_ip in its key."""
    for class_name, schemes in ELEMENT_IP_CATALOG.items():
        for n_ip, (coords, weights) in schemes.items():
            assert coords.shape[0] == n_ip, (class_name, n_ip)
            assert weights.shape[0] == n_ip, (class_name, n_ip)


def test_catalog_dim_per_class():
    """Spot-check that 2-D classes return 2-vectors and 3-D return 3-vectors."""
    coords, _ = get_ip_layout("203-ASDShellQ4", 4)
    assert coords.shape[1] == 2
    coords, _ = get_ip_layout("56-Brick", 8)
    assert coords.shape[1] == 3
