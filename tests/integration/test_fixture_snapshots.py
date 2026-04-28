"""Snapshot tests against checked-in real fixtures.

Per docs/mpco_format_conventions.md §16, the strongest signal that
"convention bugs are absent" is identical numeric output across
independent paths to the same data. We don't run a live OpenSees model
in CI, so this file is the lightweight version: pin the expected
``(n_cols, n_ip, gp_xi, first_3_cols, last_3_cols)`` tuple for each
result bucket the library is known to handle. If MPCO format drifts,
or if the META parser regresses on naming/IP-coord conventions, these
tests fail loudly.

Numeric sanity (no NaN in fetched data, time array lengths align with
step count) is also asserted on a representative subset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Snapshot table                                                          #
# ---------------------------------------------------------------------- #
#
# Each row pins what the library should currently produce for the
# corresponding (fixture, recorder, results_name, element_type) tuple.
# Update ONLY when a deliberate format/library change occurs.
#
# ``element_id_picker`` chooses a single element_id from the cached
# index so each parametrized case loads the smallest meaningful slab.
# ``gp_xi_expected`` is None for closed-form / continuum-without-GP_X
# buckets.

SNAPSHOTS = [
    # ---- elasticFrame (single partition, ElasticBeam3d only) -----------
    {
        "id": "elasticFrame::force/5-ElasticBeam3d",
        "fixture_attr": "elastic_frame_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "force",
        "element_type": "5-ElasticBeam3d",
        "n_cols": 12,
        "n_ip": 0,
        "gp_xi": None,
        "first3": ("Px_1", "Py_1", "Pz_1"),
        "last3": ("Mx_2", "My_2", "Mz_2"),
    },
    {
        "id": "elasticFrame::localForce/5-ElasticBeam3d",
        "fixture_attr": "elastic_frame_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "localForce",
        "element_type": "5-ElasticBeam3d",
        "n_cols": 12,
        "n_ip": 0,
        "gp_xi": None,
        "first3": ("N_1", "Vy_1", "Vz_1"),
        "last3": ("T_2", "My_2", "Mz_2"),
    },
    # ---- QuadFrame (multi-partition: ElasticBeam3d + ASDShellQ4) -------
    {
        "id": "QuadFrame::force/203-ASDShellQ4",
        "fixture_attr": "quad_frame_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "force",
        "element_type": "203-ASDShellQ4",
        "n_cols": 24,
        "n_ip": 0,
        "gp_xi": None,
        "first3": ("P1", "P2", "P3"),
        "last3": ("P22", "P23", "P24"),
    },
    {
        "id": "QuadFrame::section.force/203-ASDShellQ4",
        "fixture_attr": "quad_frame_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "section.force",
        "element_type": "203-ASDShellQ4",
        "n_cols": 32,
        "n_ip": 4,                     # 2x2 Gauss-Legendre, from catalog
        "gp_xi": None,                 # multi-D — gp_xi is line-element only
        "gp_natural_shape": (4, 2),    # in-plane (ξ, η)
        "first3": ("Fxx_ip0", "Fyy_ip0", "Fxy_ip0"),
        "last3": ("Mxy_ip3", "Vxz_ip3", "Vyz_ip3"),
    },
    {
        "id": "QuadFrame::section.deformation/203-ASDShellQ4",
        "fixture_attr": "quad_frame_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "section.deformation",
        "element_type": "203-ASDShellQ4",
        "n_cols": 32,
        "n_ip": 4,
        "gp_xi": None,
        "gp_natural_shape": (4, 2),
        "first3": ("epsXX_ip0", "epsYY_ip0", "epsXY_ip0"),
        "last3": ("kappaXY_ip3", "gammaXZ_ip3", "gammaYZ_ip3"),
    },
    # ---- solid_partition (DispBeamColumn3d + Brick, multi-partition) ---
    {
        "id": "solid::force/56-Brick",
        "fixture_attr": "solid_partition_dir",
        "recorder": "Recorder",
        "stage": "MODEL_STAGE[1]",
        "results_name": "force",
        "element_type": "56-Brick",
        "n_cols": 24,
        "n_ip": 0,
        "gp_xi": None,
        "first3": ("P1_1", "P2_1", "P3_1"),
        "last3": ("P1_8", "P2_8", "P3_8"),
    },
    {
        "id": "solid::section.force/64-DispBeamColumn3d",
        "fixture_attr": "solid_partition_dir",
        "recorder": "Recorder",
        "stage": "MODEL_STAGE[1]",
        "results_name": "section.force",
        "element_type": "64-DispBeamColumn3d",
        "n_cols": 8,
        "n_ip": 2,
        "gp_xi": (-1.0, 1.0),
        "first3": ("P_ip0", "Mz_ip0", "My_ip0"),
        "last3": ("Mz_ip1", "My_ip1", "T_ip1"),
    },
    {
        "id": "solid::material.stress/56-Brick",
        "fixture_attr": "solid_partition_dir",
        "recorder": "Recorder",
        "stage": "MODEL_STAGE[1]",
        "results_name": "material.stress",
        "element_type": "56-Brick",
        "n_cols": 48,
        "n_ip": 8,                     # 2x2x2 Gauss-Legendre, from catalog
        "gp_xi": None,                 # multi-D — gp_xi is line-element only
        "gp_natural_shape": (8, 3),    # 3-D (ξ, η, ζ)
        "first3": ("sigma11_ip0", "sigma22_ip0", "sigma33_ip0"),
        "last3": ("sigma12_ip7", "sigma23_ip7", "sigma13_ip7"),
    },
    {
        "id": "solid::section.fiber.stress/64-DispBeamColumn3d",
        "fixture_attr": "solid_partition_dir",
        "recorder": "Recorder",
        "stage": "MODEL_STAGE[1]",
        "results_name": "section.fiber.stress",
        "element_type": "64-DispBeamColumn3d",
        "n_cols": 12,
        "n_ip": 2,
        "gp_xi": (-1.0, 1.0),
        "first3": ("sigma11_f0_ip0", "sigma11_f1_ip0", "sigma11_f2_ip0"),
        "last3": ("sigma11_f3_ip1", "sigma11_f4_ip1", "sigma11_f5_ip1"),
    },
    # ---- 5-IP Lobatto: separate disp-based fixture ---------------------
    {
        "id": "dispBased5IP::section.force/64-DispBeamColumn3d",
        "fixture_attr": "_disp_based_dir",
        "recorder": "results",
        "stage": "MODEL_STAGE[1]",
        "results_name": "section.force",
        "element_type": "64-DispBeamColumn3d",
        "n_cols": 20,
        "n_ip": 5,
        "gp_xi": (-1.0, -0.65465367, 0.0, 0.65465367, 1.0),
        "first3": ("P_ip0", "Mz_ip0", "My_ip0"),
        "last3": ("Mz_ip4", "My_ip4", "T_ip4"),
    },
]


# ---------------------------------------------------------------------- #
# Helper to find the disp-based 5-IP fixture (not in conftest)            #
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _disp_based_dir() -> Path:
    p = (
        Path(__file__).resolve().parents[2]
        / "stko_results_examples"
        / "elasticFrame"
        / "elasticFrame_mesh_displacementBased_results"
    )
    if not (p / "results.mpco").exists():
        pytest.skip(f"disp-based fixture missing at {p}")
    return p


# ---------------------------------------------------------------------- #
# Parametrized snapshot test                                              #
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def _ds_cache():
    """Cache MPCODataSet per (fixture_path, recorder) so we don't
    re-open large multi-partition fixtures for every snapshot row."""
    return {}


@pytest.mark.parametrize("snap", SNAPSHOTS, ids=[s["id"] for s in SNAPSHOTS])
def test_bucket_snapshot(snap, request, _ds_cache):
    fixture_dir: Path = request.getfixturevalue(snap["fixture_attr"])

    cache_key = (str(fixture_dir), snap["recorder"])
    if cache_key in _ds_cache:
        ds = _ds_cache[cache_key]
    else:
        ds = MPCODataSet(str(fixture_dir), snap["recorder"], verbose=False)
        _ds_cache[cache_key] = ds

    # Pick the first element of the requested type from the index.
    df_idx = ds.elements_info["dataframe"]
    matched = df_idx[df_idx["element_type"].str.startswith(snap["element_type"])]
    assert not matched.empty, (
        f"no elements of type {snap['element_type']!r} in fixture"
    )
    eid = int(matched["element_id"].iloc[0])

    er = ds.elements.get_element_results(
        results_name=snap["results_name"],
        element_type=snap["element_type"],
        model_stage=snap["stage"],
        element_ids=[eid],
    )

    cols = list(er.df.columns)
    assert er.df.shape[1] == snap["n_cols"], (
        f"{snap['id']}: n_cols got {er.df.shape[1]}, expected {snap['n_cols']}"
    )
    assert tuple(cols[:3]) == snap["first3"], (
        f"{snap['id']}: first3 got {tuple(cols[:3])}, expected {snap['first3']}"
    )
    assert tuple(cols[-3:]) == snap["last3"], (
        f"{snap['id']}: last3 got {tuple(cols[-3:])}, expected {snap['last3']}"
    )
    assert er.n_ip == snap["n_ip"], (
        f"{snap['id']}: n_ip got {er.n_ip}, expected {snap['n_ip']}"
    )

    if snap["gp_xi"] is None:
        assert er.gp_xi is None, (
            f"{snap['id']}: expected gp_xi=None, got {er.gp_xi!r}"
        )
    else:
        assert er.gp_xi is not None, (
            f"{snap['id']}: expected gp_xi shape {len(snap['gp_xi'])}, got None"
        )
        np.testing.assert_allclose(
            er.gp_xi,
            np.asarray(snap["gp_xi"], dtype=np.float64),
            rtol=1e-6,
            atol=1e-7,
            err_msg=f"{snap['id']}: gp_xi snapshot mismatch",
        )

    expected_nat_shape = snap.get("gp_natural_shape")
    if expected_nat_shape is not None:
        assert er.gp_natural is not None, (
            f"{snap['id']}: expected gp_natural shape {expected_nat_shape}, "
            f"got None"
        )
        assert er.gp_natural.shape == expected_nat_shape, (
            f"{snap['id']}: gp_natural shape "
            f"{er.gp_natural.shape} != {expected_nat_shape}"
        )


# ---------------------------------------------------------------------- #
# Numeric sanity                                                          #
# ---------------------------------------------------------------------- #


def test_no_nans_in_section_force(elastic_frame_dir: Path):
    """Result data is finite — catches silently-misaligned fancy
    indexing or empty-step bugs that would surface as NaN."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    assert np.isfinite(er.df.to_numpy()).all()


def test_time_array_length_matches_steps(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1],
    )
    assert er.time.size == er.n_steps


def test_step_index_is_dense(quad_frame_dir: Path):
    """The (element_id, step) MultiIndex covers every step from 0..n-1
    for each element — no missing rows that would produce NaN under
    common pandas operations."""
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    df_idx = ds.elements_info["dataframe"]
    eid = int(
        df_idx[df_idx["element_type"].str.startswith("5-ElasticBeam3d")]
        ["element_id"].iloc[0]
    )
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[eid],
    )
    steps = sorted(er.df.xs(eid, level="element_id").index.unique().tolist())
    assert steps == list(range(len(steps)))
