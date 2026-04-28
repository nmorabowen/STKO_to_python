"""End-to-end tests that the META parser produces real component names
through the public API.

These tests exercise every bucket shape codified in
``docs/mpco_format_conventions.md`` against real fixtures:

* Closed-form beams (``GAUSS_IDS=[[-1]]``) — globalForce, localForce
* Closed-form continuum (8-node brick) — globalForce
* Line-stations (sequential GAUSS_IDS) — section.force, section.deformation
* Gauss-level continuum — material.stress, material.strain
* Compressed META (``MULTIPLICITY > 1``) — section.fiber.stress

The previous behaviour (``val_1, val_2, ...``) is explicitly negated so
any regression in the meta-parser would cause these to fail loudly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Closed-form: elasticFrame / 5-ElasticBeam3d
# ---------------------------------------------------------------------- #

def test_global_force_columns_are_named_per_node(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    assert list(er.df.columns) == [
        "Px_1", "Py_1", "Pz_1", "Mx_1", "My_1", "Mz_1",
        "Px_2", "Py_2", "Pz_2", "Mx_2", "My_2", "Mz_2",
    ]
    # Real names, not val_*
    assert "val_1" not in er.df.columns


def test_local_force_columns_use_local_axis_names(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    # localForce uses N/Vy/Vz/T/My/Mz — disambiguated from globalForce
    assert er.df.columns[0] == "N_1"
    assert er.df.columns[3] == "T_1"
    assert er.df.columns[6] == "N_2"


def test_attribute_style_access_uses_real_names(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    # User can now reach into result by physical name, not val_3
    series = er.N_1[1]
    assert len(series) == er.n_steps
    assert er.fetch(component="Mz_2", element_ids=[1]) is not None


# ---------------------------------------------------------------------- #
# Line-stations + compressed fibers + gauss continuum: solid_partition_example
# ---------------------------------------------------------------------- #

def test_line_station_section_force_has_ip_suffix(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    beams = ds.elements_info["dataframe"]
    beam_ids = beams[
        beams["element_type"].str.startswith("64-DispBeamColumn3d")
    ]["element_id"].head(3).tolist()

    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=beam_ids,
    )
    assert list(er.df.columns) == [
        "P_ip0", "Mz_ip0", "My_ip0", "T_ip0",
        "P_ip1", "Mz_ip1", "My_ip1", "T_ip1",
    ]


def test_compressed_fiber_columns_expand(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    beams = ds.elements_info["dataframe"]
    beam_ids = beams[
        beams["element_type"].str.startswith("64-DispBeamColumn3d")
    ]["element_id"].head(3).tolist()

    er = ds.elements.get_element_results(
        results_name="section.fiber.stress",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=beam_ids,
    )
    # MULTIPLICITY=[6,6] with 1 component each → 12 named fiber columns
    assert er.df.shape[1] == 12
    assert er.df.columns[0] == "sigma11_f0_ip0"
    assert er.df.columns[5] == "sigma11_f5_ip0"
    assert er.df.columns[6] == "sigma11_f0_ip1"
    assert er.df.columns[-1] == "sigma11_f5_ip1"


def test_gauss_continuum_brick_stress(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    bricks = ds.elements_info["dataframe"]
    brick_ids = bricks[
        bricks["element_type"].str.startswith("56-Brick")
    ]["element_id"].head(2).tolist()

    er = ds.elements.get_element_results(
        results_name="material.stress",
        element_type="56-Brick",
        model_stage="MODEL_STAGE[1]",
        element_ids=brick_ids,
    )
    # 8 Gauss × 6 stress components
    assert er.df.shape[1] == 48
    assert er.df.columns.tolist()[:6] == [
        "sigma11_ip0", "sigma22_ip0", "sigma33_ip0",
        "sigma12_ip0", "sigma23_ip0", "sigma13_ip0",
    ]
    assert er.df.columns.tolist()[-1] == "sigma13_ip7"


def test_brick_global_force_uses_per_node_dof_names(solid_partition_dir: Path):
    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    bricks = ds.elements_info["dataframe"]
    brick_ids = bricks[
        bricks["element_type"].str.startswith("56-Brick")
    ]["element_id"].head(2).tolist()

    er = ds.elements.get_element_results(
        results_name="force",
        element_type="56-Brick",
        model_stage="MODEL_STAGE[1]",
        element_ids=brick_ids,
    )
    # 8-node brick × 3 force DOFs = 24 closed-form columns
    assert er.df.shape[1] == 24
    assert er.df.columns[0] == "P1_1"
    assert er.df.columns[-1] == "P3_8"
