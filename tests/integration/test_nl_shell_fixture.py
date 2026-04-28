"""End-to-end tests against the Test_NLShell fixture.

Covers the new capabilities introduced by registering
``204-ASDShellT3`` in the catalogs and relaxing the META parser to
accept layered-shell buckets:

* Triangular shells get ``gp_natural`` / ``gp_weights`` /
  ``physical_coords`` / ``jacobian_dets`` from the static catalog,
  matching analytical triangle areas.
* Layered ``section.fiber.*`` buckets parse cleanly with column
  names like ``d+_l0_ip0``, ``d+_l2_ip3``, etc.
* Canonical-name resolution still works on the layered buckets.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# ASDShellT3 — new catalog entry                                          #
# ---------------------------------------------------------------------- #


def test_asdshell_t3_section_force_basic(nl_shell_dir: Path):
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    df = ds.elements_info["dataframe"]
    tri_ids = [
        int(i)
        for i in df.query("element_type == '204-ASDShellT3'")["element_id"].head(3)
    ]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="204-ASDShellT3",
        element_ids=tri_ids,
        model_stage="MODEL_STAGE[1]",
    )
    assert er.n_ip == 3
    assert er.gp_dim == 2
    assert er.gp_natural.shape == (3, 2)
    # Gauss-3 rule on the unit triangle: weights all 1/6, sum = 1/2.
    assert er.gp_weights.shape == (3,)
    assert er.gp_weights.sum() == pytest.approx(0.5)
    assert np.allclose(er.gp_weights, 1 / 6)


def test_asdshell_t3_areas_match_analytical(nl_shell_dir: Path):
    """For each triangular element, the integrated 1 dA equals the
    analytical cross-product area of its three nodes."""
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    df = ds.elements_info["dataframe"]
    tri_ids = [
        int(i)
        for i in df.query("element_type == '204-ASDShellT3'")["element_id"].head(5)
    ]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="204-ASDShellT3",
        element_ids=tri_ids,
        model_stage="MODEL_STAGE[1]",
    )
    dets = er.jacobian_dets()
    integrated_areas = (er.gp_weights[None, :] * dets).sum(axis=1)

    # Analytical area = 0.5 * ||(p1-p0) × (p2-p0)||
    analytical = []
    for nc in er.element_node_coords:
        v1 = nc[1] - nc[0]
        v2 = nc[2] - nc[0]
        analytical.append(0.5 * np.linalg.norm(np.cross(v1, v2)))

    np.testing.assert_allclose(integrated_areas, analytical, rtol=1e-12)


def test_asdshell_t3_physical_coords_lie_on_triangle_plane(nl_shell_dir: Path):
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    df = ds.elements_info["dataframe"]
    tri_id = int(
        df.query("element_type == '204-ASDShellT3'")["element_id"].iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="204-ASDShellT3",
        element_ids=[tri_id],
        model_stage="MODEL_STAGE[1]",
    )
    phys = er.physical_coords()[0]            # (3, 3)
    nc = er.element_node_coords[0]            # (3, 3)
    n = np.cross(nc[1] - nc[0], nc[2] - nc[0])
    n /= np.linalg.norm(n)
    for ip in phys:
        # Each IP physical position has zero out-of-plane component.
        assert abs(np.dot(ip - nc[0], n)) < 1e-6


def test_asdshell_t3_integrate_canonical(nl_shell_dir: Path):
    """``integrate_canonical`` works on triangular shells the same
    way as on quad shells — surface integral over each element."""
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    df = ds.elements_info["dataframe"]
    tri_ids = [
        int(i)
        for i in df.query("element_type == '204-ASDShellT3'")["element_id"].head(2)
    ]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="204-ASDShellT3",
        element_ids=tri_ids,
        model_stage="MODEL_STAGE[1]",
    )
    s = er.integrate_canonical("membrane_xx")
    assert s.shape == (len(tri_ids) * er.n_steps,)
    assert np.isfinite(s.to_numpy()).all()


# ---------------------------------------------------------------------- #
# Layered fiber buckets — column naming                                  #
# ---------------------------------------------------------------------- #


def test_layered_fiber_damage_column_names(nl_shell_dir: Path):
    """``section.fiber.damage`` on the layered ASDShellQ4 produces
    columns named like ``d+_l<layer>_ip<gid>`` for every (layer, gp)
    pair where the layer carries d+/d- components."""
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    quad_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.fiber.damage",
        element_type="203-ASDShellQ4",
        element_ids=[quad_id],
        model_stage="MODEL_STAGE[1]",
    )
    assert er.n_ip == 4
    cols = list(er.df.columns)
    # In the fixture only layers 0, 2, 4 carry d+/d- (layers 1, 3 had
    # NUM_COMPONENTS=0 and were skipped).
    assert "d+_l0_ip0" in cols
    assert "d+_l2_ip0" in cols
    assert "d+_l4_ip0" in cols
    assert "d-_l4_ip3" in cols
    # Every column ends with _ipN for N in 0..3.
    for c in cols:
        assert any(c.endswith(f"_ip{i}") for i in range(4)), c


def test_canonical_resolves_on_layered_bucket(nl_shell_dir: Path):
    """``canonical("damage_pos")`` finds all ``d+_l<L>_ip<I>`` columns
    across every (layer, gauss-point) — the suffix-stripping regex
    handles the new ``_l<int>_ip<int>`` pattern."""
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    quad_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.fiber.damage",
        element_type="203-ASDShellQ4",
        element_ids=[quad_id],
        model_stage="MODEL_STAGE[1]",
    )
    canons = er.list_canonicals()
    assert "damage_pos" in canons
    assert "damage_neg" in canons

    pos_cols = er.canonical_columns("damage_pos")
    # 3 layers × 4 gauss-points = 12 d+ columns
    assert len(pos_cols) == 12
    assert all(c.startswith("d+") for c in pos_cols)


def test_quad_shell_section_force_unchanged(nl_shell_dir: Path):
    """Backward-compat sanity: non-layered ASDShellQ4 ``section.force``
    still uses the original ``_ip<idx>`` suffix (no ``_l`` infix)."""
    ds = MPCODataSet(str(nl_shell_dir), "Results", verbose=False)
    quad_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=[quad_id],
        model_stage="MODEL_STAGE[1]",
    )
    cols = list(er.df.columns)
    # No layer suffix on a non-layered bucket.
    assert all("_l" not in c for c in cols)
    assert "Fxx_ip0" in cols
    assert "Vyz_ip3" in cols
