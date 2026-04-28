"""Unit tests for :mod:`STKO_to_python.elements.canonical`."""
from __future__ import annotations

import pytest

from STKO_to_python.elements.canonical import (
    CANONICAL_TO_MPCO,
    available_canonicals,
    list_canonical_for_columns,
    match_canonical_columns,
    shortname_of,
)


# ---------------------------------------------------------------------- #
# shortname_of                                                            #
# ---------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "column,expected",
    [
        # Closed-form node-suffixed
        ("Px_1", "Px"),
        ("Mz_2", "Mz"),
        ("N_1", "N"),
        ("T_2", "T"),
        ("Vy_1", "Vy"),
        # Brick closed-form (Pi_j naming)
        ("P1_1", "P1"),
        ("P3_8", "P3"),
        # Line-stations / gauss-level
        ("P_ip0", "P"),
        ("Mz_ip4", "Mz"),
        ("sigma11_ip7", "sigma11"),
        ("Fxx_ip3", "Fxx"),
        ("kappaXY_ip0", "kappaXY"),
        # Compressed fibers
        ("sigma11_f0_ip0", "sigma11"),
        ("sigma11_f5_ip1", "sigma11"),
        ("eps11_f3_ip0", "eps11"),
        # Special characters preserved
        ("d+_ip0", "d+"),
        ("d-_ip3", "d-"),
        ("PLE+_ip2", "PLE+"),
        # Fallback: nothing to strip
        ("cw", "cw"),
    ],
)
def test_shortname_of(column, expected):
    assert shortname_of(column) == expected


# ---------------------------------------------------------------------- #
# match_canonical_columns                                                 #
# ---------------------------------------------------------------------- #


def test_axial_force_matches_P_in_line_stations():
    cols = ("P_ip0", "Mz_ip0", "My_ip0", "T_ip0", "P_ip1", "Mz_ip1", "My_ip1", "T_ip1")
    assert match_canonical_columns("axial_force", cols) == ["P_ip0", "P_ip1"]


def test_axial_force_matches_N_in_local_force():
    cols = ("N_1", "Vy_1", "Vz_1", "T_1", "My_1", "Mz_1",
            "N_2", "Vy_2", "Vz_2", "T_2", "My_2", "Mz_2")
    assert match_canonical_columns("axial_force", cols) == ["N_1", "N_2"]


def test_bending_moment_matches_compressed_fiber_layout():
    """For section.fiber.* the shortnames are sigma11/eps11 not Mz —
    bending_moment_z should return [] there, not raise."""
    cols = ("sigma11_f0_ip0", "sigma11_f1_ip0", "sigma11_f0_ip1", "sigma11_f1_ip1")
    assert match_canonical_columns("bending_moment_z", cols) == []


def test_membrane_xx_only_matches_shell_columns():
    cols = ("Fxx_ip0", "Fyy_ip0", "Fxy_ip0", "Mxx_ip0", "Myy_ip0", "Mxy_ip0",
            "Fxx_ip1", "Fyy_ip1")
    assert match_canonical_columns("membrane_xx", cols) == ["Fxx_ip0", "Fxx_ip1"]


def test_stress_11_matches_continuum_gauss():
    cols = tuple(f"{c}_ip{i}" for i in range(3) for c in ("sigma11", "sigma22", "sigma33"))
    got = match_canonical_columns("stress_11", cols)
    assert got == ["sigma11_ip0", "sigma11_ip1", "sigma11_ip2"]


def test_damage_with_plus_minus_in_shortname():
    cols = ("d+_ip0", "d-_ip0", "d+_ip1", "d-_ip1")
    assert match_canonical_columns("damage_pos", cols) == ["d+_ip0", "d+_ip1"]
    assert match_canonical_columns("damage_neg", cols) == ["d-_ip0", "d-_ip1"]


def test_unknown_canonical_raises():
    with pytest.raises(ValueError, match="Unknown canonical name"):
        match_canonical_columns("made_up_thing", ["P_ip0"])


def test_no_match_returns_empty_list():
    """Asking for shell membrane_xx on a beam returns [] (silent miss
    — the higher-level ``ElementResults.canonical()`` is the layer
    that raises on no-match)."""
    cols = ("Px_1", "Py_1", "Pz_1")
    assert match_canonical_columns("membrane_xx", cols) == []


# ---------------------------------------------------------------------- #
# list_canonical_for_columns                                              #
# ---------------------------------------------------------------------- #


def test_list_canonicals_for_local_force_beam():
    cols = ("N_1", "Vy_1", "Vz_1", "T_1", "My_1", "Mz_1",
            "N_2", "Vy_2", "Vz_2", "T_2", "My_2", "Mz_2")
    got = set(list_canonical_for_columns(cols))
    assert got == {
        "axial_force", "shear_y", "shear_z", "torsion",
        "bending_moment_y", "bending_moment_z",
    }


def test_list_canonicals_for_shell_section_force():
    cols = tuple(
        f"{c}_ip{i}"
        for i in range(4)
        for c in ("Fxx", "Fyy", "Fxy", "Mxx", "Myy", "Mxy", "Vxz", "Vyz")
    )
    got = set(list_canonical_for_columns(cols))
    assert "membrane_xx" in got
    assert "bending_moment_xx" in got
    assert "transverse_shear_xz" in got
    # Beam canonicals not present
    assert "axial_force" not in got
    assert "torsion" not in got


def test_list_canonicals_for_empty_columns():
    assert list_canonical_for_columns(()) == ()


# ---------------------------------------------------------------------- #
# Map sanity                                                              #
# ---------------------------------------------------------------------- #


def test_canonical_names_lowercase_underscore():
    """Naming convention — every canonical key is snake_case."""
    for name in CANONICAL_TO_MPCO:
        assert name == name.lower(), name
        assert " " not in name, name


def test_available_canonicals_sorted():
    avail = available_canonicals()
    assert list(avail) == sorted(avail)


def test_no_duplicate_shortname_targets_per_canonical():
    for name, targets in CANONICAL_TO_MPCO.items():
        assert len(targets) == len(set(targets)), name


# ---------------------------------------------------------------------- #
# ElementResults integration (uses real fixture)                           #
# ---------------------------------------------------------------------- #


def test_element_results_canonical_methods(elastic_frame_dir):
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1, 2, 3],
    )
    assert "axial_force" in er.list_canonicals()
    assert er.canonical_columns("axial_force") == ("N_1", "N_2")
    assert list(er.canonical("bending_moment_z").columns) == ["Mz_1", "Mz_2"]


def test_element_results_canonical_raises_on_no_match(elastic_frame_dir):
    """The DataFrame-returning ``canonical()`` raises if no columns
    match (vs ``canonical_columns()`` which returns the empty tuple)."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        model_stage="MODEL_STAGE[1]",
        element_ids=[1],
    )
    with pytest.raises(ValueError, match="No columns matching"):
        er.canonical("membrane_xx")  # shell quantity asked of a beam
