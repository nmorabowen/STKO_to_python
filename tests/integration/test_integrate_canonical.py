"""End-to-end tests for ``ElementResults.integrate_canonical()``.

The helper applies the standard quadrature pattern
``Σ value * weight * |J|`` across all elements and steps in one call.
These tests exercise it on the real fixtures the library already
covers (Brick, ASDShellQ4) and verify it matches a manual numpy
computation, plus error behavior on closed-form / fiber / unknown-
canonical inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Brick (8-IP solid) — volume integral                                    #
# ---------------------------------------------------------------------- #


def test_brick_integrate_stress_11_matches_manual(solid_partition_dir: Path):
    """Helper output equals the manual ``Σ σ * w * |J|`` over IPs."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_ids = (
        ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .head(3)
        .tolist()
    )
    brick_ids = [int(i) for i in brick_ids]
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=brick_ids,
        model_stage="MODEL_STAGE[1]",
    )

    s = er.integrate_canonical("stress_11")
    # Index aligns with self.df.
    assert isinstance(s, pd.Series)
    assert list(s.index.names) == ["element_id", "step"]
    assert s.shape == (len(brick_ids) * er.n_steps,)
    assert s.name == "integral_stress_11"

    # Manual verification: pick a single (element_id, step) and recompute.
    cols = list(er.canonical_columns("stress_11"))
    dets = er.jacobian_dets()
    eid_to_row = {int(e): i for i, e in enumerate(er.element_ids)}

    eid = brick_ids[0]
    step = 1
    sigma = er.df.xs((eid, step))[cols].to_numpy(dtype=np.float64)
    manual = float((sigma * er.gp_weights * dets[eid_to_row[eid]]).sum())
    helper = float(s.loc[eid, step])
    assert manual == pytest.approx(helper, rel=1e-12, abs=1e-12)


def test_brick_integrate_unstack_to_step_x_element_matrix(solid_partition_dir: Path):
    """``.unstack("element_id")`` gives a tidy step × element matrix."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    brick_ids = [
        int(i)
        for i in ds.elements_info["dataframe"]
        .query("element_type == '56-Brick'")["element_id"]
        .head(2)
    ]
    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        element_ids=brick_ids,
        model_stage="MODEL_STAGE[1]",
    )
    s = er.integrate_canonical("stress_11")
    mtx = s.unstack("element_id")
    assert mtx.shape == (er.n_steps, len(brick_ids))
    assert sorted(mtx.columns.tolist()) == sorted(brick_ids)


# ---------------------------------------------------------------------- #
# Shell (4-IP) — surface integral                                         #
# ---------------------------------------------------------------------- #


def test_shell_integrate_membrane_xx(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    shell_ids = [
        int(i)
        for i in ds.elements_info["dataframe"]
        .query("element_type == '203-ASDShellQ4'")["element_id"]
        .head(3)
    ]
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="203-ASDShellQ4",
        element_ids=shell_ids,
        model_stage="MODEL_STAGE[1]",
    )
    s = er.integrate_canonical("membrane_xx")
    assert s.shape == (len(shell_ids) * er.n_steps,)
    # Result is finite (no NaN from missing JC or weights)
    assert np.isfinite(s.to_numpy()).all()


# ---------------------------------------------------------------------- #
# Error paths                                                             #
# ---------------------------------------------------------------------- #


def test_closed_form_integrate_raises(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="no gp_weights"):
        er.integrate_canonical("axial_force")


def test_unknown_canonical_raises(solid_partition_dir: Path):
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
    with pytest.raises(ValueError, match="Unknown canonical name"):
        er.integrate_canonical("nonexistent_quantity")


def test_canonical_present_but_no_match_raises(solid_partition_dir: Path):
    """Asking for ``membrane_xx`` (shell) on a Brick result — the
    canonical exists in the global map but no columns in this result
    match. integrate_canonical inherits from canonical_columns which
    returns empty tuple; the helper raises with a clear message."""
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
    with pytest.raises(ValueError, match="no columns matching"):
        er.integrate_canonical("membrane_xx")


def test_fiber_bucket_raises_clearly(solid_partition_dir: Path):
    """``section.fiber.stress`` has cols of the form ``sigma11_f<j>_ip<i>``
    with multiplicity > 1 — len(cols) != n_ip. Helper raises with a
    pointer to manual integration."""
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    beam_ids = [
        int(i)
        for i in ds.elements_info["dataframe"]
        .query("element_type == '64-DispBeamColumn3d'")["element_id"]
        .head(2)
    ]
    er = ds.elements.get_element_results(
        results_name="section.fiber.stress",
        element_type="64-DispBeamColumn3d",
        element_ids=beam_ids,
        model_stage="MODEL_STAGE[1]",
    )
    # gp_weights is None for line elements, so the gp_weights branch
    # raises first. Either error path is acceptable here — both are
    # informative; this just locks in the contract.
    with pytest.raises(ValueError):
        er.integrate_canonical("stress_11")


# ---------------------------------------------------------------------- #
# Pickle round-trip                                                       #
# ---------------------------------------------------------------------- #


def test_integrate_after_pickle_roundtrip(solid_partition_dir: Path, tmp_path: Path):
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
    s_before = er.integrate_canonical("stress_11")

    pkl = tmp_path / "er.pkl"
    er.save_pickle(pkl)
    from STKO_to_python.elements.element_results import ElementResults

    er2 = ElementResults.load_pickle(pkl)
    s_after = er2.integrate_canonical("stress_11")
    pd.testing.assert_series_equal(s_before, s_after, check_names=True)
