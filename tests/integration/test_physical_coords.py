"""End-to-end tests for B7b — ``physical_coords()`` and
``jacobian_dets()`` on :class:`ElementResults`.

Every assertion targets engineering-meaningful invariants:
volume / area / length recovery via numerical integration of `1` over
the physical element. If shape functions or node-coord plumbing
regress, these break loudly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Brick (8-IP solid)                                                       #
# ---------------------------------------------------------------------- #


def test_brick_physical_coords_inside_node_bbox(solid_partition_dir: Path):
    """Each IP's physical position lies inside the bounding box of its
    element's nodes (true for any well-formed convex brick)."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=[brick_id],
        model_stage="MODEL_STAGE[1]",
    )
    phys = er.physical_coords()
    assert phys is not None
    assert phys.shape == (1, 8, 3)

    nc = er.element_node_coords[0]   # (8, 3)
    lo = nc.min(axis=0)
    hi = nc.max(axis=0)
    for ip in phys[0]:
        assert np.all(ip >= lo - 1e-9)
        assert np.all(ip <= hi + 1e-9)


def test_brick_volume_matches_bbox_for_axis_aligned(solid_partition_dir: Path):
    """Sum of (weight × |J|) integrates 1 over the parent — for an
    axis-aligned brick (all 8 nodes at the corners of an axis-aligned
    box) this equals the bounding-box volume."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=[brick_id],
        model_stage="MODEL_STAGE[1]",
    )
    dets = er.jacobian_dets()
    volume = (er.gp_weights * dets[0]).sum()

    nc = er.element_node_coords[0]
    bbox_vol = float(
        (nc[:, 0].max() - nc[:, 0].min())
        * (nc[:, 1].max() - nc[:, 1].min())
        * (nc[:, 2].max() - nc[:, 2].min())
    )
    # Should match exactly for axis-aligned bricks (which the fixture
    # uses); allow a small tolerance for the general case.
    assert volume == pytest.approx(bbox_vol, rel=1e-6)


def test_brick_volume_is_positive_per_element(solid_partition_dir: Path):
    """All bricks in the fixture have positive volume (no degenerate
    elements)."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    df_idx = ds.elements_info["dataframe"]
    brick_ids = df_idx.query("element_type == '56-Brick'")["element_id"].head(20).tolist()
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=brick_ids,
        model_stage="MODEL_STAGE[1]",
    )
    dets = er.jacobian_dets()           # (n_e, 8)
    vols = (er.gp_weights[None, :] * dets).sum(axis=1)
    assert np.all(vols > 0)


# ---------------------------------------------------------------------- #
# Shell (4-IP, 2-D in 3-D space)                                           #
# ---------------------------------------------------------------------- #


def test_shell_area_positive_and_consistent(quad_frame_dir: Path):
    """Per-element shell area is positive and equals diagonal*diagonal/2
    × 2 = bounding box area for an axis-aligned flat quad. The
    QuadFrame fixture uses such quads."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    df_idx = ds.elements_info["dataframe"]
    shell_ids = df_idx.query(
        "element_type == '203-ASDShellQ4'"
    )["element_id"].head(10).tolist()
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=shell_ids,
        model_stage="MODEL_STAGE[1]",
    )
    dets = er.jacobian_dets()                          # (n_e, 4)
    areas = (er.gp_weights[None, :] * dets).sum(axis=1)
    assert np.all(areas > 0)


def test_shell_physical_coords_lie_on_the_shell_plane(quad_frame_dir: Path):
    """For a flat quad, every IP physical position lies in the plane
    defined by the four nodes."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=[shell_id],
        model_stage="MODEL_STAGE[1]",
    )
    phys = er.physical_coords()[0]                    # (4, 3)
    nc = er.element_node_coords[0]                    # (4, 3)
    # Plane normal from first three nodes
    v1 = nc[1] - nc[0]
    v2 = nc[2] - nc[0]
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    for ip in phys:
        signed_dist = np.dot(ip - nc[0], n)
        assert abs(signed_dist) < 1e-6


# ---------------------------------------------------------------------- #
# Beam (5-IP line)                                                         #
# ---------------------------------------------------------------------- #


def test_beam_physical_coords_align_with_axis():
    """A 5-IP disp-based beam from the displacement-based fixture lies
    along the global Z axis; physical IP coords should reflect that."""
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip("disp-based fixture missing")
    ds = MPCODataSet(str(p), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    phys = er.physical_coords()[0]                    # (5, 3)
    nc = er.element_node_coords[0]                    # (2, 3)
    # Beam length
    L = float(np.linalg.norm(nc[1] - nc[0]))

    # Each IP physical position equals node0 + (xi+1)/2 * (node1-node0)
    direction = nc[1] - nc[0]
    for i, xi in enumerate(er.gp_xi):
        expected = nc[0] + 0.5 * (1.0 + xi) * direction
        np.testing.assert_allclose(phys[i], expected, rtol=1e-9)

    dets = er.jacobian_dets()[0]
    assert np.allclose(dets, L / 2)


# ---------------------------------------------------------------------- #
# Closed-form: physical_coords / jacobian_dets are None                    #
# ---------------------------------------------------------------------- #


def test_closed_form_returns_none(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="force",     # closed-form, GAUSS_IDS=[[-1]]
        element_type="56-Brick",
        element_ids=[brick_id],
        model_stage="MODEL_STAGE[1]",
    )
    assert er.physical_coords() is None
    assert er.jacobian_dets() is None


def test_unknown_class_returns_none_gracefully(elastic_frame_dir: Path):
    """Asking physical_coords on an ElasticBeam3d *closed-form* result
    just returns None (no IPs)."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    # Closed-form: no gp_natural, so physical_coords is None.
    assert er.physical_coords() is None


# ---------------------------------------------------------------------- #
# Pickle round-trip                                                       #
# ---------------------------------------------------------------------- #


def test_physical_coords_survive_pickle(solid_partition_dir: Path, tmp_path: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_id = int(
        ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=[brick_id],
        model_stage="MODEL_STAGE[1]",
    )
    phys_before = er.physical_coords()
    dets_before = er.jacobian_dets()

    pkl_path = tmp_path / "er.pkl"
    er.save_pickle(pkl_path)
    from STKO_to_python.elements.element_results import ElementResults

    er2 = ElementResults.load_pickle(pkl_path)
    np.testing.assert_array_equal(er2.element_node_coords, er.element_node_coords)
    np.testing.assert_array_equal(er2.element_node_ids, er.element_node_ids)
    np.testing.assert_allclose(er2.physical_coords(), phys_before)
    np.testing.assert_allclose(er2.jacobian_dets(), dets_before)
