"""Coordinate conversions for integration-point positions.

OpenSees and MPCO use **two different conventions** for integration-
point coordinates along a beam:

* MPCO writes ``GP_X`` to disk in **natural** ξ ∈ [-1, +1]. (The
  parent-element parametric coordinate.)
* ``ops.eleResponse(eid, "integrationPoints")`` returns positions in
  **physical** length ``[0, L]`` (per ``ForceBeamColumn3d.cpp:3338``).

When integrating these two surfaces (e.g. cross-checking MPCO output
against a live openseespy model), one of the two has to be converted.
This module is the canonical place for those conversions.

See ``docs/mpco_format_conventions.md`` §7.
"""
from __future__ import annotations

from typing import Union

import numpy as np


__all__ = [
    "xi_natural_to_physical",
    "x_physical_to_natural",
]


ArrayLike = Union[float, np.ndarray]


def xi_natural_to_physical(xi: ArrayLike, length: float) -> np.ndarray:
    """Convert natural ξ ∈ [-1, +1] to physical x ∈ [0, L].

    Formula: ``x = (xi + 1) * L / 2``.

    Parameters
    ----------
    xi : float or array_like
        Natural-coordinate position(s).
    length : float
        Element length ``L`` (positive).

    Returns
    -------
    np.ndarray
        Physical position(s) along the element. ``xi=-1`` maps to
        ``0``, ``xi=+1`` maps to ``L``, ``xi=0`` to ``L/2``.

    Raises
    ------
    ValueError
        If ``length`` is non-positive.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length!r}")
    return (np.asarray(xi, dtype=np.float64) + 1.0) * (length / 2.0)


def x_physical_to_natural(x: ArrayLike, length: float) -> np.ndarray:
    """Convert physical x ∈ [0, L] to natural ξ ∈ [-1, +1].

    Formula: ``xi = 2 * x / L - 1``. This is the conversion required
    when comparing MPCO ``GP_X`` (natural) against openseespy's
    ``eleResponse("integrationPoints")`` (physical).

    Parameters
    ----------
    x : float or array_like
        Physical position(s) along the element.
    length : float
        Element length ``L`` (positive).

    Returns
    -------
    np.ndarray
        Natural-coordinate position(s).

    Raises
    ------
    ValueError
        If ``length`` is non-positive.
    """
    if length <= 0:
        raise ValueError(f"length must be positive, got {length!r}")
    return 2.0 * np.asarray(x, dtype=np.float64) / length - 1.0
