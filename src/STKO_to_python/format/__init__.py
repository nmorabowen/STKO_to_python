"""MPCO format quirks — Gauss-point catalog, shape functions, and other
per-element-class conventions live here.

This package is the single home for "how does an MPCO file say
something?" knowledge: integration-point natural coordinates, shape
functions, Jacobian-determinant kinds, and (in :mod:`STKO_to_python.io.format_policy`)
the format policy that decides shell-vs-beam keywords and similar
class-tag-driven branching. Code that needs to interpret raw MPCO
data goes through this package; nothing else.

The legacy import paths
``STKO_to_python.utilities.gauss_points`` and
``STKO_to_python.utilities.shape_functions`` continue to work as
``DeprecationWarning``-emitting shims; new code should import from
this package directly.
"""
from __future__ import annotations

from .gauss_points import (
    ELEMENT_IP_CATALOG,
    gauss_legendre_1d,
    gauss_tetrahedron,
    gauss_triangle,
    get_ip_layout,
    tensor_product_2d,
    tensor_product_3d,
)
from .shape_functions import (
    SHAPE_FUNCTIONS,
    compute_jacobian_dets,
    compute_physical_coords,
    get_shape_functions,
    tet4_N,
    tet4_dN,
    tri3_N,
    tri3_dN,
)

__all__ = [
    # gauss_points
    "ELEMENT_IP_CATALOG",
    "gauss_legendre_1d",
    "gauss_tetrahedron",
    "gauss_triangle",
    "get_ip_layout",
    "tensor_product_2d",
    "tensor_product_3d",
    # shape_functions
    "SHAPE_FUNCTIONS",
    "compute_jacobian_dets",
    "compute_physical_coords",
    "get_shape_functions",
    "tet4_N",
    "tet4_dN",
    "tri3_N",
    "tri3_dN",
]
