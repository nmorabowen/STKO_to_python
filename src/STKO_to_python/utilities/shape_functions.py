"""Shape functions and derivatives for fixed-class elements.

Maps natural coordinates (ξ, η, ζ) from
:mod:`STKO_to_python.utilities.gauss_points` to physical (x, y, z) on
the actual element via standard FE shape functions, and provides the
Jacobian determinants needed for physical-volume / physical-area
integration.

For each supported element class the catalog returns:

* ``N(nat_coords)`` → ``(n_ip, n_nodes)`` shape-function values
* ``dN_dnat(nat_coords)`` → ``(n_ip, n_nodes, parent_dim)`` derivatives
  in the parametric directions
* ``geom_kind`` ∈ ``{"line", "shell", "solid"}`` — determines how the
  Jacobian determinant is computed:

  - **solid** (parent_dim=3, physical_dim=3): square Jacobian, det.
  - **shell** (parent_dim=2, physical_dim=3): rectangular Jacobian; the
    surface measure is ``||∂x/∂ξ × ∂x/∂η||``.
  - **line** (parent_dim=1, physical_dim=3): single-row Jacobian; the
    line measure is the norm of ``∂x/∂ξ``.

Node-ordering convention follows OpenSees's standard for each class:
ASDShellQ4 corners CCW (1:(-1,-1) → 2:(+1,-1) → 3:(+1,+1) → 4:(-1,+1));
Brick bottom-face CCW then top-face CCW (1..4 at ζ=-1, 5..8 at ζ=+1).
Override these per-class if a particular model uses a different node
order.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


__all__ = [
    "SHAPE_FUNCTIONS",
    "get_shape_functions",
    "compute_physical_coords",
    "compute_jacobian_dets",
    # Reusable shape-function primitives — exported so users can
    # register them under their own element-class keys when those
    # surface in fixtures (e.g. ``SHAPE_FUNCTIONS["204-ASDShellT3"] =
    # (tri3_N, tri3_dN, "shell")``).
    "tri3_N",
    "tri3_dN",
    "tet4_N",
    "tet4_dN",
]


ShapeFn = Callable[[np.ndarray], np.ndarray]
GeomKind = str  # "line" | "shell" | "solid"


# --------------------------------------------------------------------- #
# ASDShellQ4 — 4-node bilinear quad in 3-D space (shell)                #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: (-1, -1)
#   2: (+1, -1)
#   3: (+1, +1)
#   4: (-1, +1)


def _asd_shell_q4_N(nat: np.ndarray) -> np.ndarray:
    """N_i(ξ, η) for i=1..4 at each row of ``nat`` shape (n_ip, 2)."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    return 0.25 * np.stack(
        [
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ],
        axis=1,
    )


def _asd_shell_q4_dN(nat: np.ndarray) -> np.ndarray:
    """∂N_i/∂(ξ, η) — shape (n_ip, 4, 2)."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 4, 2), dtype=np.float64)
    # ∂N/∂ξ
    out[:, 0, 0] = -0.25 * (1 - eta)
    out[:, 1, 0] = +0.25 * (1 - eta)
    out[:, 2, 0] = +0.25 * (1 + eta)
    out[:, 3, 0] = -0.25 * (1 + eta)
    # ∂N/∂η
    out[:, 0, 1] = -0.25 * (1 - xi)
    out[:, 1, 1] = -0.25 * (1 + xi)
    out[:, 2, 1] = +0.25 * (1 + xi)
    out[:, 3, 1] = +0.25 * (1 - xi)
    return out


# --------------------------------------------------------------------- #
# Brick — 8-node trilinear hex (solid)                                  #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: (-1, -1, -1)        5: (-1, -1, +1)
#   2: (+1, -1, -1)        6: (+1, -1, +1)
#   3: (+1, +1, -1)        7: (+1, +1, +1)
#   4: (-1, +1, -1)        8: (-1, +1, +1)


_BRICK_NODE_SIGNS = np.array(
    [
        [-1, -1, -1],
        [+1, -1, -1],
        [+1, +1, -1],
        [-1, +1, -1],
        [-1, -1, +1],
        [+1, -1, +1],
        [+1, +1, +1],
        [-1, +1, +1],
    ],
    dtype=np.float64,
)


def _brick_N(nat: np.ndarray) -> np.ndarray:
    """N_i(ξ, η, ζ) for i=1..8 — shape (n_ip, 8)."""
    # nat: (n_ip, 3); _BRICK_NODE_SIGNS: (8, 3)
    # Broadcast: (n_ip, 1, 3) * (1, 8, 3) → (n_ip, 8, 3)
    factors = 1.0 + _BRICK_NODE_SIGNS[None, :, :] * nat[:, None, :]
    return 0.125 * np.prod(factors, axis=2)


def _brick_dN(nat: np.ndarray) -> np.ndarray:
    """∂N_i/∂(ξ, η, ζ) — shape (n_ip, 8, 3)."""
    # ∂N_i/∂x_k = (sign_ik / 8) * ∏_{j != k} (1 + sign_ij * x_j)
    factors = 1.0 + _BRICK_NODE_SIGNS[None, :, :] * nat[:, None, :]
    # full_prod: (n_ip, 8) — product over the 3 axes
    full_prod = np.prod(factors, axis=2)
    # ∂/∂x_k pulls out factor (1 + sign_ik * x_k); divide it out.
    # Avoid division by zero when (1 + sign_ik * x_k) == 0 (only at the
    # opposite face) by recomputing those rows manually.
    out = np.empty((nat.shape[0], 8, 3), dtype=np.float64)
    for k in range(3):
        # Other two axes' factors product
        other = np.delete(factors, k, axis=2).prod(axis=2)  # (n_ip, 8)
        out[:, :, k] = 0.125 * _BRICK_NODE_SIGNS[None, :, k] * other
    return out


# --------------------------------------------------------------------- #
# Linear line element — 2-node (line)                                   #
# --------------------------------------------------------------------- #
#
# Node ordering (natural):
#   1: (-1)
#   2: (+1)


def _line2_N(nat: np.ndarray) -> np.ndarray:
    """N_i(ξ) for i=1..2 — shape (n_ip, 2)."""
    xi = nat[:, 0]
    return 0.5 * np.stack([1 - xi, 1 + xi], axis=1)


def _line2_dN(nat: np.ndarray) -> np.ndarray:
    """∂N_i/∂ξ — shape (n_ip, 2, 1)."""
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 2, 1), dtype=np.float64)
    out[:, 0, 0] = -0.5
    out[:, 1, 0] = +0.5
    return out


# --------------------------------------------------------------------- #
# Tri3 — 3-node linear triangle (shell or plane)                        #
# --------------------------------------------------------------------- #
#
# Parent domain: unit triangle with vertices at (0,0), (1,0), (0,1).
# Pair with ``gauss_triangle()`` from :mod:`gauss_points`.
#
# Node ordering (natural):
#   1: (0, 0)
#   2: (1, 0)
#   3: (0, 1)


def tri3_N(nat: np.ndarray) -> np.ndarray:
    """Linear-triangle shape functions — shape ``(n_ip, 3)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    return np.stack([1.0 - xi - eta, xi, eta], axis=1)


def tri3_dN(nat: np.ndarray) -> np.ndarray:
    """Linear-triangle derivatives — shape ``(n_ip, 3, 2)``.

    Derivatives are constant on the parent triangle; the per-IP
    repetition just makes the array shape consistent with the rest of
    the catalog.
    """
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 3, 2), dtype=np.float64)
    out[:, 0, 0] = -1.0; out[:, 0, 1] = -1.0  # ∂N1/∂(ξ, η)
    out[:, 1, 0] = +1.0; out[:, 1, 1] = +0.0  # ∂N2/∂(ξ, η)
    out[:, 2, 0] = +0.0; out[:, 2, 1] = +1.0  # ∂N3/∂(ξ, η)
    return out


# --------------------------------------------------------------------- #
# Tet4 — 4-node linear tetrahedron (solid)                              #
# --------------------------------------------------------------------- #
#
# Parent domain: unit tetrahedron with vertices at the origin and the
# three unit-axis points. Pair with ``gauss_tetrahedron()`` from
# :mod:`gauss_points`.
#
# Node ordering (natural):
#   1: (0, 0, 0)
#   2: (1, 0, 0)
#   3: (0, 1, 0)
#   4: (0, 0, 1)


def tet4_N(nat: np.ndarray) -> np.ndarray:
    """Linear-tet shape functions — shape ``(n_ip, 4)``."""
    xi = nat[:, 0]
    eta = nat[:, 1]
    zeta = nat[:, 2]
    return np.stack(
        [1.0 - xi - eta - zeta, xi, eta, zeta], axis=1
    )


def tet4_dN(nat: np.ndarray) -> np.ndarray:
    """Linear-tet derivatives — shape ``(n_ip, 4, 3)`` (constant)."""
    n_ip = nat.shape[0]
    out = np.empty((n_ip, 4, 3), dtype=np.float64)
    out[:, 0, :] = [-1.0, -1.0, -1.0]
    out[:, 1, :] = [+1.0,  0.0,  0.0]
    out[:, 2, :] = [ 0.0, +1.0,  0.0]
    out[:, 3, :] = [ 0.0,  0.0, +1.0]
    return out


# --------------------------------------------------------------------- #
# Catalog                                                               #
# --------------------------------------------------------------------- #
#
# Keys are the *base* element name carried in MPCO connectivity datasets
# — i.e. ``"<classTag>-<className>"`` with the bracket suffix stripped.

SHAPE_FUNCTIONS: Dict[str, Tuple[ShapeFn, ShapeFn, GeomKind]] = {
    "203-ASDShellQ4": (_asd_shell_q4_N, _asd_shell_q4_dN, "shell"),
    "204-ASDShellT3": (tri3_N, tri3_dN, "shell"),
    "56-Brick":       (_brick_N, _brick_dN, "solid"),
    # Beam/disp-based-beam line elements use 2-node linear interpolation
    # for the geometric mapping (the integration scheme is independent;
    # custom rules carry their own GP_X). Only used here when the user
    # asks for physical coordinates of beam IPs.
    "5-ElasticBeam3d":      (_line2_N, _line2_dN, "line"),
    "64-DispBeamColumn3d":  (_line2_N, _line2_dN, "line"),
}


def get_shape_functions(
    element_class_base: str,
) -> Optional[Tuple[ShapeFn, ShapeFn, GeomKind]]:
    """Look up shape functions for a base element class.

    Returns
    -------
    (N, dN_dnat, geom_kind) or None
        ``None`` if the class is not in the catalog. The caller should
        leave physical coords / Jacobians unset and surface a clear
        message rather than guessing.
    """
    return SHAPE_FUNCTIONS.get(element_class_base)


# --------------------------------------------------------------------- #
# Vectorized mapping                                                    #
# --------------------------------------------------------------------- #


def compute_physical_coords(
    natural_coords: np.ndarray,
    element_node_coords: np.ndarray,
    N_fn: ShapeFn,
) -> np.ndarray:
    """Map natural-coord IP positions to physical (x, y, z).

    Parameters
    ----------
    natural_coords : np.ndarray, shape (n_ip, parent_dim)
        IP positions in the parent domain (output of
        :mod:`STKO_to_python.utilities.gauss_points`).
    element_node_coords : np.ndarray, shape (n_elements, n_nodes_per, 3)
        Physical coordinates of each element's nodes, in node-ordering
        consistent with the shape function.
    N_fn : callable
        Shape-function evaluator returning ``(n_ip, n_nodes_per)``.

    Returns
    -------
    np.ndarray, shape (n_elements, n_ip, 3)
        Physical position of each IP for each element.
    """
    N = N_fn(natural_coords)  # (n_ip, n_nodes)
    # einsum: (i,n) @ (e,n,3) → (e,i,3)
    return np.einsum("in,enj->eij", N, element_node_coords)


def compute_jacobian_dets(
    natural_coords: np.ndarray,
    element_node_coords: np.ndarray,
    dN_fn: ShapeFn,
    geom_kind: GeomKind,
) -> np.ndarray:
    """Jacobian determinants at each IP for each element.

    Returns
    -------
    np.ndarray, shape (n_elements, n_ip)
        - ``"solid"``: ``det(J)`` where ``J`` is the 3x3 ∂x/∂ξ matrix.
        - ``"shell"``: surface measure ``||∂x/∂ξ × ∂x/∂η||``.
        - ``"line"``: line measure ``||∂x/∂ξ||``.

    All three are non-negative scalars; multiplying a Gauss-point value
    by ``weight * |J|`` produces a contribution to the integral over
    the physical element.
    """
    dN = dN_fn(natural_coords)  # (n_ip, n_nodes, parent_dim)
    # J: ∂x_a/∂ξ_k for a ∈ physical (3), k ∈ parent.
    # einsum: (i,n,k) @ (e,n,a) → (e,i,k,a)
    J = np.einsum("ink,ena->eika", dN, element_node_coords)
    # J shape: (n_elements, n_ip, parent_dim, 3)

    if geom_kind == "solid":
        # J is (e, i, 3, 3) — square matrix per IP; take det.
        return np.abs(np.linalg.det(J))
    if geom_kind == "shell":
        # J is (e, i, 2, 3); cross product of the two parent rows.
        cross = np.cross(J[..., 0, :], J[..., 1, :])
        return np.linalg.norm(cross, axis=-1)
    if geom_kind == "line":
        # J is (e, i, 1, 3); norm of the single row.
        return np.linalg.norm(J[..., 0, :], axis=-1)
    raise ValueError(
        f"Unknown geom_kind {geom_kind!r}; expected 'solid', 'shell', or 'line'."
    )
