"""End-to-end tests for B2 — ``GP_X`` (integration-point coordinates)
and the ``at_ip()`` per-IP slicer on :class:`ElementResults`.

Per docs/mpco_format_conventions.md §1, custom-rule beam-columns write
their integration-point coordinates as a ``GP_X`` attribute on the
connectivity dataset (in natural ξ ∈ [-1, +1]). The library now reads
that attribute and exposes it as ``ElementResults.gp_xi`` for line-
station / gauss-level buckets. Closed-form buckets and continuum
classes (which lack ``GP_X`` on connectivity) get ``gp_xi=None``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ----- helpers ------------------------------------------------------------ #


def _disp_based_dir() -> Path | None:
    """Directory of the displacement-based 5-IP fixture (5 Lobatto IPs)."""
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    return p if (p / "results.mpco").exists() else None


@pytest.fixture(scope="module")
def disp_based_ds():
    p = _disp_based_dir()
    if p is None:
        pytest.skip("displacement-based fixture not available")
    return MPCODataSet(str(p), "results", verbose=False)


@pytest.fixture(scope="module")
def disp_based_beam_ids(disp_based_ds):
    return disp_based_ds.elements_info["dataframe"]["element_id"].tolist()[:3]


# ----- closed-form: gp_xi is None ---------------------------------------- #


def test_closed_form_has_no_gp_xi(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="localForce",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    assert er.gp_xi is None
    assert er.n_ip == 0


def test_closed_form_at_ip_raises(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    with pytest.raises(ValueError, match="closed-form"):
        er.at_ip(0)


# ----- 5-IP line-stations ------------------------------------------------ #


def test_line_station_gp_xi_is_5_lobatto_points(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    assert er.n_ip == 5
    assert er.gp_xi is not None
    assert er.gp_xi.shape == (5,)
    # Endpoints are exactly ±1 (Lobatto includes endpoints)
    assert er.gp_xi[0] == pytest.approx(-1.0)
    assert er.gp_xi[-1] == pytest.approx(1.0)
    # Mid-station is at the geometric center
    assert er.gp_xi[2] == pytest.approx(0.0)
    # Strictly increasing
    assert np.all(np.diff(er.gp_xi) > 0)


def test_at_ip_returns_only_that_ip_columns(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    sub = er.at_ip(0)
    assert list(sub.columns) == ["P_ip0", "Mz_ip0", "My_ip0", "T_ip0"]
    sub4 = er.at_ip(4)
    assert list(sub4.columns) == ["P_ip4", "Mz_ip4", "My_ip4", "T_ip4"]
    # Same MultiIndex preserved
    assert list(sub.index.names) == ["element_id", "step"]


def test_at_ip_out_of_range_raises(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    with pytest.raises(ValueError, match=r"out of range"):
        er.at_ip(10)
    with pytest.raises(ValueError, match=r"out of range"):
        er.at_ip(-1)


def test_repr_includes_n_ip_when_present(disp_based_ds, disp_based_beam_ids):
    er = disp_based_ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    assert "n_ip=5" in repr(er)

    er_cf = disp_based_ds.elements.get_element_results(
        results_name="force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    # Closed-form: n_ip section omitted from repr
    assert "n_ip" not in repr(er_cf)


# ----- compressed fibers + continuum gauss --------------------------------- #


def test_compressed_fiber_at_ip_returns_all_fibers_for_ip(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    beam_ids = ds.elements_info["dataframe"].query(
        "element_type == '64-DispBeamColumn3d'"
    )["element_id"].head(3).tolist()

    er = ds.elements.get_element_results(
        results_name="section.fiber.stress",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=beam_ids,
    )
    assert er.n_ip == 2
    assert er.gp_xi is not None
    assert er.gp_xi.tolist() == [-1.0, 1.0]

    # at_ip(0) returns the 6 fibers at IP 0
    sub = er.at_ip(0)
    assert sub.shape[1] == 6
    assert all(c.endswith("_ip0") for c in sub.columns)
    assert all(c.startswith("sigma11_f") for c in sub.columns)


def test_continuum_class_uses_catalog_gauss_points(solid_partition_dir: Path):
    """Brick continuum has 8 Gauss IPs and no ``GP_X`` attribute on its
    connectivity. The natural coordinates of those IPs are *fixed by
    the element class* and come from the static catalog at
    :mod:`STKO_to_python.utilities.gauss_points` (B7a). The 1-D ``gp_xi``
    stays ``None`` (it's line-element-only); the multi-D ``gp_natural``
    carries the (8, 3) array of (ξ, η, ζ) Gauss-Legendre points.
    """
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_ids = ds.elements_info["dataframe"].query(
        "element_type == '56-Brick'"
    )["element_id"].head(2).tolist()

    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        model_stage="MODEL_STAGE[1]",
        element_ids=brick_ids,
    )
    # 1-D ξ stays None (multi-D bucket).
    assert er.gp_xi is None
    # 3-D natural coords from catalog.
    assert er.n_ip == 8
    assert er.gp_dim == 3
    assert er.gp_natural is not None
    assert er.gp_natural.shape == (8, 3)
    # Standard 2x2x2 Gauss-Legendre points are at ±1/sqrt(3) per axis.
    assert np.allclose(np.abs(er.gp_natural), 1.0 / np.sqrt(3.0))
    # Weights are uniform 1.0 on a [-1,1]^3 cube → sum = 8.
    assert er.gp_weights is not None
    assert er.gp_weights.shape == (8,)
    assert er.gp_weights.sum() == pytest.approx(8.0)
    # at_ip now works for continuum gauss-level buckets too.
    sub = er.at_ip(0)
    assert sub.shape[1] == 6  # 6 stress components per Gauss point
    assert all(c.endswith("_ip0") for c in sub.columns)


# ----- pickle round-trip preserves gp_xi --------------------------------- #


def test_gp_xi_survives_pickle(disp_based_ds, disp_based_beam_ids, tmp_path: Path):
    er = disp_based_ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=disp_based_beam_ids,
    )
    p = tmp_path / "er.pkl"
    er.save_pickle(p)

    from STKO_to_python.elements.element_results import ElementResults

    er2 = ElementResults.load_pickle(p)
    assert er2.n_ip == 5
    assert er2.gp_xi is not None
    np.testing.assert_array_equal(er2.gp_xi, er.gp_xi)
    # at_ip still works after reload
    assert list(er2.at_ip(2).columns) == ["P_ip2", "Mz_ip2", "My_ip2", "T_ip2"]
