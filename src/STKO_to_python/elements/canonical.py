"""Canonical (engineering-friendly) names for MPCO result components.

Users pulling element results today must know each result-bucket's
on-disk shortnames: ``P_ip0`` for axial force at the first IP of a
line-station beam, ``N_1`` for axial force at node 1 of a closed-form
``localForce`` bucket, ``Fxx_ip0`` for shell membrane Fxx, and so on.
Every element family carries the same engineering quantity under a
different MPCO shortname, by historical convention rather than design.

This module provides a tiny canonical → shortname map plus helpers
that filter the columns of an :class:`ElementResults` by canonical
name. With it a user can write::

    axial = er.canonical("axial_force")  # P_* or N_* depending on bucket
    moments = er.canonical("bending_moment_z")  # Mz_*

regardless of element family or whether the bucket is closed-form,
line-station, or compressed-fiber.

The mapping is intentionally **non-exhaustive**: only the engineering
quantities that have a stable, unambiguous meaning across element
classes are exposed. Disambiguating identifiers like the My/Mz
collision between ``globalForce`` (global axes) and ``localForce``
(element-local axes) is left to the user — they pick the bucket via
the existing ``get_element_results(results_name, ...)`` call. See
docs/mpco_format_conventions.md §9.
"""
from __future__ import annotations

import re
from typing import Iterable, Tuple


__all__ = [
    "CANONICAL_TO_MPCO",
    "available_canonicals",
    "match_canonical_columns",
    "list_canonical_for_columns",
    "shortname_of",
]


# ---------------------------------------------------------------------- #
# Canonical map                                                           #
# ---------------------------------------------------------------------- #
#
# Each canonical name maps to a tuple of MPCO shortnames that may
# appear in the column-name prefix once IP/node/fiber suffixes are
# stripped. Multiple shortnames per canonical are expected: e.g.
# ``axial_force`` is ``P`` in line-station ``section.force`` buckets
# but ``N`` in element-local ``localForce`` buckets.

CANONICAL_TO_MPCO: dict[str, Tuple[str, ...]] = {
    # ---- Beam element actions ------------------------------------------
    "axial_force":          ("P", "N"),
    "bending_moment_z":     ("Mz",),
    "bending_moment_y":     ("My",),
    "torsion":              ("T",),
    "shear_y":              ("Vy",),
    "shear_z":              ("Vz",),

    # ---- Beam section deformations -------------------------------------
    "axial_strain":         ("eps",),
    "curvature_z":          ("kappaZ",),
    "curvature_y":          ("kappaY",),
    "twist":                ("theta",),

    # ---- Shell section forces (resultants) -----------------------------
    "membrane_xx":          ("Fxx",),
    "membrane_yy":          ("Fyy",),
    "membrane_xy":          ("Fxy",),
    "bending_moment_xx":    ("Mxx",),
    "bending_moment_yy":    ("Myy",),
    "bending_moment_xy":    ("Mxy",),
    "transverse_shear_xz":  ("Vxz",),
    "transverse_shear_yz":  ("Vyz",),

    # ---- Shell section deformations ------------------------------------
    "membrane_strain_xx":   ("epsXX",),
    "membrane_strain_yy":   ("epsYY",),
    "membrane_strain_xy":   ("epsXY",),
    "curvature_xx":         ("kappaXX",),
    "curvature_yy":         ("kappaYY",),
    "curvature_xy":         ("kappaXY",),
    "transverse_shear_strain_xz": ("gammaXZ",),
    "transverse_shear_strain_yz": ("gammaYZ",),

    # ---- Continuum stresses --------------------------------------------
    "stress_11":            ("sigma11",),
    "stress_22":            ("sigma22",),
    "stress_33":            ("sigma33",),
    "stress_12":            ("sigma12",),
    "stress_23":            ("sigma23",),
    "stress_13":            ("sigma13",),

    # ---- Continuum strains ---------------------------------------------
    "strain_11":            ("eps11",),
    "strain_22":            ("eps22",),
    "strain_33":            ("eps33",),
    "strain_12":            ("eps12",),
    "strain_23":            ("eps23",),
    "strain_13":            ("eps13",),

    # ---- Damage / plasticity (continuum) -------------------------------
    "damage_pos":           ("d+",),
    "damage_neg":           ("d-",),
    "plastic_strain_pos":   ("PLE+",),
    "plastic_strain_neg":   ("PLE-",),
    "crack_width":          ("cw",),

    # ---- Closed-form global axes (per-node DOFs) -----------------------
    "force_x_global":       ("Px",),
    "force_y_global":       ("Py",),
    "force_z_global":       ("Pz",),
    "moment_x_global":      ("Mx",),
    # Note: moment_y_global / moment_z_global would collide with the
    # local-axis "bending_moment_y" / "bending_moment_z". Users wanting
    # global-axis moments should fetch the column directly by its
    # shortname (My_1, Mz_2, ...).
}


# ---------------------------------------------------------------------- #
# Suffix stripper                                                         #
# ---------------------------------------------------------------------- #
#
# Column names produced by io/meta_parser.py end in one of:
#   * ``_<int>``                       — closed-form node-suffixed (Px_1, N_2, P3_8)
#   * ``_ip<int>``                     — line-stations / gauss-level (P_ip0, sigma11_ip7)
#   * ``_f<int>_ip<int>``              — compressed fiber (sigma11_f0_ip0)
#   * ``_l<int>_ip<int>``              — layered shell, no fibers (d+_l2_ip3)
#   * ``_f<int>_l<int>_ip<int>``       — layered shell with fibers
# Stripping the suffix yields the MPCO shortname.
#
# Order in the regex alternation is longest-first so the most-specific
# suffix wins; otherwise ``_l0_ip0`` would shadow under ``_ip0``.

_SUFFIX_RE = re.compile(
    r"_(?:"
    r"f\d+_l\d+_ip\d+"   # fiber + layer + ip (layered shell with fibers)
    r"|f\d+_ip\d+"        # fiber + ip
    r"|l\d+_ip\d+"        # layer + ip (layered shell, no fibers)
    r"|ip\d+"             # plain ip
    r"|\d+"               # closed-form node
    r")$"
)


def shortname_of(column: str) -> str:
    """Return the MPCO shortname carried by a flat column name.

    Strips one of the suffixes introduced by
    :func:`STKO_to_python.io.meta_parser.parse_bucket_meta`:
    ``_<int>``, ``_ip<int>``, ``_f<int>_ip<int>``, ``_l<int>_ip<int>``,
    or ``_f<int>_l<int>_ip<int>``.

    Examples
    --------
    >>> shortname_of("Px_1")
    'Px'
    >>> shortname_of("P_ip3")
    'P'
    >>> shortname_of("sigma11_f5_ip0")
    'sigma11'
    >>> shortname_of("d+_l2_ip3")
    'd+'
    >>> shortname_of("sigma11_f4_l2_ip0")
    'sigma11'
    >>> shortname_of("d+")            # already a shortname
    'd+'
    """
    return _SUFFIX_RE.sub("", str(column))


# ---------------------------------------------------------------------- #
# Public helpers                                                          #
# ---------------------------------------------------------------------- #


def available_canonicals() -> Tuple[str, ...]:
    """Tuple of all known canonical names, sorted."""
    return tuple(sorted(CANONICAL_TO_MPCO))


def match_canonical_columns(
    canonical: str, columns: Iterable[str]
) -> list[str]:
    """Return the subset of ``columns`` whose MPCO shortname matches
    the canonical name's mapped shortnames.

    Parameters
    ----------
    canonical : str
        A name from :data:`CANONICAL_TO_MPCO`.
    columns : iterable of str
        Flat column names from an :class:`ElementResults.df`.

    Returns
    -------
    list of str
        Original column names (preserving order) that match.

    Raises
    ------
    ValueError
        If ``canonical`` is unknown.
    """
    if canonical not in CANONICAL_TO_MPCO:
        raise ValueError(
            f"Unknown canonical name {canonical!r}. "
            f"Available: {available_canonicals()}"
        )
    targets = set(CANONICAL_TO_MPCO[canonical])
    return [c for c in columns if shortname_of(c) in targets]


def list_canonical_for_columns(columns: Iterable[str]) -> Tuple[str, ...]:
    """Return canonical names whose shortnames are present in the
    given columns. Useful for ``ElementResults.list_canonicals()``.
    """
    cols = list(columns)
    if not cols:
        return ()
    present_shortnames = {shortname_of(c) for c in cols}
    hits = []
    for canon, shortnames in CANONICAL_TO_MPCO.items():
        if any(s in present_shortnames for s in shortnames):
            hits.append(canon)
    return tuple(sorted(hits))
