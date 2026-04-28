"""Standard Gauss-point coordinates for fixed-class elements.

For force/disp-based beam-columns the integration-point coordinates
live on the connectivity dataset's ``@GP_X`` attribute (custom rule —
see docs/mpco_format_conventions.md §1, §4). For continuum solids,
shells, and plane elements, the coordinates are *fixed by the
element class and integration-point count* and are **not** written to
disk. This module is the catalog the read path consults to fill in
``ElementResults.gp_natural`` for those classes.

Values are in **natural coordinates** (parent-element parametric
space): ξ ∈ [-1, +1] for line elements, (ξ, η) ∈ [-1, +1]² for 2-D
shells / plane elements, (ξ, η, ζ) ∈ [-1, +1]³ for 3-D solids.

Ordering convention
-------------------
Tensor-product integration schemes enumerate points with **ξ varying
fastest, then η, then ζ** — a left-to-right, bottom-to-top, front-to-
back walk. This is the standard convention used by most FEM textbooks
and by OpenSees's natural-coordinate quadrature loops. If a particular
element class turns out to enumerate differently, override its catalog
entry rather than changing the global helpers.

The catalog is intentionally small in v1 — entries are added per
element class as fixtures or user reports appear. Lookup falls back
to ``None`` for unknown (class, n_ip) pairs; the read path then leaves
``gp_natural`` unset rather than guessing.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np


__all__ = [
    "gauss_legendre_1d",
    "tensor_product_2d",
    "tensor_product_3d",
    "gauss_triangle",
    "gauss_tetrahedron",
    "ELEMENT_IP_CATALOG",
    "get_ip_layout",
]


# --------------------------------------------------------------------- #
# Gauss-Legendre primitives                                             #
# --------------------------------------------------------------------- #


@lru_cache(maxsize=16)
def gauss_legendre_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the 1-D Gauss-Legendre points and weights for ``n``-point
    quadrature.

    Backed by :func:`numpy.polynomial.legendre.leggauss`. Cached so a
    repeat lookup is free.

    Returns
    -------
    points, weights : np.ndarray of shape (n,)
        Sorted ascending in ξ ∈ (-1, +1).
    """
    if n < 1:
        raise ValueError(f"n must be ≥ 1, got {n!r}")
    pts, wts = np.polynomial.legendre.leggauss(int(n))
    return pts.astype(np.float64), wts.astype(np.float64)


def tensor_product_2d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 2-D tensor-product Gauss-Legendre rule with ``n`` points
    per direction.

    Ordering: ξ varies fastest, then η.

    Returns
    -------
    coords : np.ndarray of shape (n*n, 2)
        Each row is (ξ_i, η_i).
    weights : np.ndarray of shape (n*n,)
        Product weights.
    """
    pts, wts = gauss_legendre_1d(n)
    # ξ-fastest: outer loop over η.
    xi_grid, eta_grid = np.meshgrid(pts, pts, indexing="xy")
    wxi, weta = np.meshgrid(wts, wts, indexing="xy")
    coords = np.stack([xi_grid.ravel(), eta_grid.ravel()], axis=1)
    weights = (wxi * weta).ravel()
    return coords.astype(np.float64), weights.astype(np.float64)


def tensor_product_3d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Build the 3-D tensor-product Gauss-Legendre rule with ``n`` points
    per direction.

    Ordering: ξ varies fastest, then η, then ζ.

    Returns
    -------
    coords : np.ndarray of shape (n^3, 3)
        Each row is (ξ_i, η_i, ζ_i).
    weights : np.ndarray of shape (n^3,)
        Product weights.
    """
    pts, wts = gauss_legendre_1d(n)
    # ξ-fastest, then η, then ζ.
    xi_grid, eta_grid, zeta_grid = np.meshgrid(pts, pts, pts, indexing="ij")
    # 'ij' indexing makes the FIRST axis vary slowest. We want xi-fastest
    # which means xi should be the LAST varying axis when raveled. Re-
    # arrange axes accordingly.
    coords = np.stack(
        [
            xi_grid.transpose(2, 1, 0).ravel(),
            eta_grid.transpose(2, 1, 0).ravel(),
            zeta_grid.transpose(2, 1, 0).ravel(),
        ],
        axis=1,
    )
    wxi, weta, wzeta = np.meshgrid(wts, wts, wts, indexing="ij")
    weights = (wxi * weta * wzeta).transpose(2, 1, 0).ravel()
    return coords.astype(np.float64), weights.astype(np.float64)


# --------------------------------------------------------------------- #
# Triangle / tetrahedron quadrature                                     #
# --------------------------------------------------------------------- #
#
# Triangles and tets use barycentric-style natural coords on the
# **unit** triangle / tetrahedron (vertices at the origin and on the
# axes). This is a different parent domain than tensor-product
# rules — the parent measure is 1/2 (triangle) or 1/6 (tet), not 4 / 8
# — and the corresponding shape functions in
# :mod:`STKO_to_python.utilities.shape_functions` use the same domain.

# Standard 1-, 3-, 7-point rules on the unit triangle.
_TRI_RULES: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    1: (
        np.array([[1.0 / 3.0, 1.0 / 3.0]]),
        np.array([0.5]),
    ),
    3: (
        np.array(
            [
                [1.0 / 6.0, 1.0 / 6.0],
                [2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0],
            ]
        ),
        np.array([1.0 / 6.0] * 3),
    ),
}

# Standard 1- and 4-point rules on the unit tetrahedron.
_TET_A = (5.0 + 3.0 * np.sqrt(5.0)) / 20.0
_TET_B = (5.0 - np.sqrt(5.0)) / 20.0
_TET_RULES: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
    1: (
        np.array([[0.25, 0.25, 0.25]]),
        np.array([1.0 / 6.0]),
    ),
    4: (
        np.array(
            [
                [_TET_A, _TET_B, _TET_B],
                [_TET_B, _TET_A, _TET_B],
                [_TET_B, _TET_B, _TET_A],
                [_TET_B, _TET_B, _TET_B],
            ]
        ),
        np.array([1.0 / 24.0] * 4),
    ),
}


def gauss_triangle(n_ip: int) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Gauss rule on the unit reference triangle.

    Parent domain: vertices at (0,0), (1,0), (0,1); area 1/2.
    Sum of weights equals the parent area.

    Parameters
    ----------
    n_ip : int
        ``1`` (degree-1, centroid) or ``3`` (degree-2, edge midpoints
        of the medial triangle). Other rules can be added if needed.

    Returns
    -------
    (coords, weights) : (ndarray (n_ip, 2), ndarray (n_ip,))

    Raises
    ------
    ValueError if ``n_ip`` is not in the supported set.
    """
    if n_ip not in _TRI_RULES:
        raise ValueError(
            f"Unsupported n_ip={n_ip!r} for triangle rule. "
            f"Available: {sorted(_TRI_RULES)}"
        )
    coords, weights = _TRI_RULES[n_ip]
    return coords.copy(), weights.copy()


def gauss_tetrahedron(n_ip: int) -> Tuple[np.ndarray, np.ndarray]:
    """Standard Gauss rule on the unit reference tetrahedron.

    Parent domain: vertices at the origin and the three unit axis
    points; volume 1/6. Sum of weights equals the parent volume.

    Parameters
    ----------
    n_ip : int
        ``1`` (degree-1, centroid) or ``4`` (degree-2). Other rules
        (e.g. 5-point degree-3) can be added if needed.

    Returns
    -------
    (coords, weights) : (ndarray (n_ip, 3), ndarray (n_ip,))
    """
    if n_ip not in _TET_RULES:
        raise ValueError(
            f"Unsupported n_ip={n_ip!r} for tetrahedron rule. "
            f"Available: {sorted(_TET_RULES)}"
        )
    coords, weights = _TET_RULES[n_ip]
    return coords.copy(), weights.copy()


# --------------------------------------------------------------------- #
# Element catalog                                                       #
# --------------------------------------------------------------------- #
#
# Keys are the *base* element name carried in MPCO connectivity datasets
# — i.e. ``"<classTag>-<className>"`` with the bracket suffix stripped.
# Inner keys are the integration-point count expected for the bucket.
# Values are ``(coords, weights)`` tuples.
#
# Add new entries as fixtures or user models surface them. Unknown
# (class, n_ip) pairs return ``None`` from :func:`get_ip_layout` and
# the read path leaves ``gp_natural`` empty.

ELEMENT_IP_CATALOG: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {
    # --- Shells (in-plane 2-D Gauss-Legendre) ---
    "203-ASDShellQ4": {
        4: tensor_product_2d(2),  # standard 2×2 quadrature
    },
    # --- 3-D continuum solids ---
    "56-Brick": {
        8: tensor_product_3d(2),   # standard 2×2×2
        27: tensor_product_3d(3),  # full 3×3×3
    },
    # --- Plane elements (2-D quads) ---
    # Common 4-node quad classes — names need verification against
    # actual fixtures before relying on these in production.
    "FourNodeQuad": {
        4: tensor_product_2d(2),
        9: tensor_product_2d(3),
    },
}


def get_ip_layout(
    element_class_base: str, n_ip: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Look up natural coords + weights for a ``(class, n_ip)`` pair.

    Parameters
    ----------
    element_class_base : str
        Base class name as it appears in the MPCO connectivity bracket
        — e.g. ``"56-Brick"`` (not ``"56-Brick[401:0]"``).
    n_ip : int
        Number of integration points actually present in the bucket
        (from META/GAUSS_IDS).

    Returns
    -------
    (coords, weights) or None
        ``coords`` shape ``(n_ip, dim)``; ``weights`` shape ``(n_ip,)``.
        ``None`` if the class or the requested IP count is not in the
        catalog — the read path then leaves ``gp_natural`` unset and
        the user keeps named columns but no spatial coordinates.
    """
    entry = ELEMENT_IP_CATALOG.get(element_class_base)
    if entry is None:
        return None
    layout = entry.get(int(n_ip))
    if layout is None:
        return None
    coords, weights = layout
    if coords.shape[0] != int(n_ip):
        # Defensive: catch a miscataloged entry early.
        raise RuntimeError(
            f"Catalog inconsistency for {element_class_base!r} n_ip={n_ip}: "
            f"got {coords.shape[0]} coords."
        )
    return coords, weights
