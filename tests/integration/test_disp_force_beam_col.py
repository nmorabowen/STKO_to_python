"""Integration tests for the dispBeamCol and forceBeamCol fixtures.

The two fixtures isolate the displacement-based vs force-based
beam-column distinction on a clean, single-element-type model. Each
uses Lobatto integration with 5 integration points (transformation
tag 3 in the Tcl).

The fixtures are developer-local (gitignored — see ``.gitignore``);
each test skips cleanly via the ``disp_beam_col_dir`` /
``force_beam_col_dir`` conftest fixtures when the data is absent. This
keeps the suite green on a clean clone and exercised on a workstation
that has the recorder outputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #

def _open_dataset(directory: Path) -> MPCODataSet:
    return MPCODataSet(str(directory), "results", verbose=False)


def _all_element_ids(ds: MPCODataSet, element_type: str) -> list[int]:
    """Return every element ID for the given base type in load order.

    The fetch APIs require an explicit selection; this helper avoids
    hard-coding fixture-specific ID counts in every test.
    """
    df = ds.elements_info["dataframe"]
    base = element_type.split("[", 1)[0]
    rows = df[df["element_type"].str.startswith(base)]
    return [int(eid) for eid in rows["element_id"].to_numpy()]


# ---------------------------------------------------------------------- #
# Disp beam-column                                                        #
# ---------------------------------------------------------------------- #

def test_disp_beam_col_metadata(disp_beam_col_dir: Path):
    """``dispBeamCol`` reports the expected stage and class tag."""
    ds = _open_dataset(disp_beam_col_dir)
    assert ds.model_stages == ["MODEL_STAGE[1]"]
    decorated = sorted(ds.element_types["unique_element_types"])
    bases = sorted({d.split("[", 1)[0] for d in decorated})
    assert bases == ["64-DispBeamColumn3d"]


def test_disp_beam_col_closed_form_force(disp_beam_col_dir: Path):
    """Closed-form ``force`` columns parse as ``Px_1, Py_1, ..., Mz_2``
    on a displacement-based beam-column, same as on any other 3-D
    beam-column class."""
    ds = _open_dataset(disp_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "64-DispBeamColumn3d"),
    )
    cols = list(er.df.columns)
    # 12 globalForce DOFs at the two end nodes.
    assert cols[:12] == [
        "Px_1", "Py_1", "Pz_1", "Mx_1", "My_1", "Mz_1",
        "Px_2", "Py_2", "Pz_2", "Mx_2", "My_2", "Mz_2",
    ]
    assert "val_1" not in cols


def test_disp_beam_col_section_force_is_line_stationed(disp_beam_col_dir: Path):
    """``section.force`` on Lobatto-5 dispBeamColumn produces five
    integration-point bands. The line-station META suffixes the
    component names with ``_ip0`` ... ``_ip4``."""
    ds = _open_dataset(disp_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "64-DispBeamColumn3d"),
    )
    cols = list(er.df.columns)
    # Five line stations expected from Lobatto-5 integration.
    ip_suffixes = {c.rsplit("_ip", 1)[-1] for c in cols if "_ip" in c}
    assert ip_suffixes == {"0", "1", "2", "3", "4"}, (
        f"Expected five Lobatto stations; got suffixes {sorted(ip_suffixes)}"
    )


def test_disp_beam_col_gp_layout(disp_beam_col_dir: Path):
    """The Lobatto-5 integration rule is custom (not in the standard
    Gauss-point catalog); ``gp_xi`` should be present from the
    connectivity ``GP_X`` attribute and have five points in
    ``[-1, +1]``."""
    ds = _open_dataset(disp_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "64-DispBeamColumn3d"),
    )
    assert er.gp_xi is not None, (
        "dispBeamCol should ship gp_xi from its connectivity GP_X attribute"
    )
    assert er.gp_xi.shape[0] == 5
    assert np.all(er.gp_xi >= -1.0 - 1e-9)
    assert np.all(er.gp_xi <= 1.0 + 1e-9)


# ---------------------------------------------------------------------- #
# Force beam-column                                                       #
# ---------------------------------------------------------------------- #

def test_force_beam_col_metadata(force_beam_col_dir: Path):
    """``forceBeamCol`` reports the expected stage and class tag."""
    ds = _open_dataset(force_beam_col_dir)
    assert ds.model_stages == ["MODEL_STAGE[1]"]
    decorated = sorted(ds.element_types["unique_element_types"])
    bases = sorted({d.split("[", 1)[0] for d in decorated})
    assert bases == ["74-ForceBeamColumn3d"]


def test_force_beam_col_closed_form_force(force_beam_col_dir: Path):
    """Closed-form ``force`` columns parse as ``Px_1, Py_1, ..., Mz_2``
    on a force-based beam-column."""
    ds = _open_dataset(force_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="74-ForceBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "74-ForceBeamColumn3d"),
    )
    cols = list(er.df.columns)
    assert cols[:12] == [
        "Px_1", "Py_1", "Pz_1", "Mx_1", "My_1", "Mz_1",
        "Px_2", "Py_2", "Pz_2", "Mx_2", "My_2", "Mz_2",
    ]
    assert "val_1" not in cols


def test_force_beam_col_local_force_uses_local_axis_names(force_beam_col_dir: Path):
    """``localForce`` parses to local-axis names (N / Vy / Vz / T / My /
    Mz) — confirms the META parser's beam-vs-shell keyword swap holds
    on a forceBeamColumn class tag."""
    ds = _open_dataset(force_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="74-ForceBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "74-ForceBeamColumn3d"),
    )
    cols = list(er.df.columns)
    assert cols[0] == "N_1"
    assert cols[3] == "T_1"
    assert cols[6] == "N_2"


def test_force_beam_col_section_force_is_line_stationed(force_beam_col_dir: Path):
    """``section.force`` on Lobatto-5 forceBeamColumn produces five
    integration-point bands."""
    ds = _open_dataset(force_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="74-ForceBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "74-ForceBeamColumn3d"),
    )
    cols = list(er.df.columns)
    ip_suffixes = {c.rsplit("_ip", 1)[-1] for c in cols if "_ip" in c}
    assert ip_suffixes == {"0", "1", "2", "3", "4"}


def test_force_beam_col_gp_layout(force_beam_col_dir: Path):
    """``gp_xi`` should be populated from the connectivity ``GP_X``
    attribute — same pattern as dispBeamCol; both classes are
    line-station based."""
    ds = _open_dataset(force_beam_col_dir)
    er = ds.elements.get_element_results(
        results_name="section.force",
        element_type="74-ForceBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds, "74-ForceBeamColumn3d"),
    )
    assert er.gp_xi is not None
    assert er.gp_xi.shape[0] == 5


# ---------------------------------------------------------------------- #
# Cross-fixture: disp vs force on the same recorder name parse cleanly    #
# ---------------------------------------------------------------------- #

def test_disp_and_force_section_force_have_same_columns(
    disp_beam_col_dir: Path,
    force_beam_col_dir: Path,
):
    """The two fixtures use the same integration rule (Lobatto-5);
    their ``section.force`` MultiIndex/columns should match in shape
    and component-name set, even though the underlying solver
    formulation differs."""
    ds_d = _open_dataset(disp_beam_col_dir)
    ds_f = _open_dataset(force_beam_col_dir)
    er_d = ds_d.elements.get_element_results(
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds_d, "64-DispBeamColumn3d"),
    )
    er_f = ds_f.elements.get_element_results(
        results_name="section.force",
        element_type="74-ForceBeamColumn3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=_all_element_ids(ds_f, "74-ForceBeamColumn3d"),
    )
    assert set(er_d.df.columns) == set(er_f.df.columns), (
        "Lobatto-5 section.force columns should agree across "
        "displacement-based and force-based formulations on identical "
        "section + integration rule."
    )
