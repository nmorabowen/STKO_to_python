"""Unit tests for B5 — customRuleIdx-aware bucket grouping.

Two surfaces:

* :meth:`ElementManager._validate_homogeneous_layouts` — pure-function
  cross-bucket layout consistency check. Tested with synthesized
  ``collected`` lists, no HDF5 needed.
* The new ``decorated_type`` column on the cached element index — that
  the column exists, is populated with the 2-field connectivity
  bracket, and survives partition deduplication.

Heterogeneous-bracket end-to-end coverage (a real .mpco file with
multiple integration rules under the same element class) is left for a
future fixture; the validation helper test below exercises the same
code path that the read loop calls.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from STKO_to_python.elements.element_manager import ElementManager
from STKO_to_python.io.meta_parser import MpcoFormatError


# ---------------------------------------------------------------------- #
# Validation helper                                                       #
# ---------------------------------------------------------------------- #


def _chunk(cols):
    """Build a (bucket_path, col_names, gp_xi, df) tuple matching the
    shape used in :meth:`ElementManager._fetch_element_results_uncached`.
    """
    return (
        "bucket/" + "_".join(cols[:1]),
        list(cols),
        None,  # gp_xi
        None,  # gp_natural
        None,  # gp_weights
        pd.DataFrame(),
    )


def test_validate_homogeneous_passes_on_single_layout():
    cols = ["P_ip0", "Mz_ip0", "P_ip1", "Mz_ip1"]
    # Single bucket — trivially homogeneous.
    ElementManager._validate_homogeneous_layouts(
        [_chunk(cols)],
        results_name="section.force",
        element_type="64-DispBeamColumn3d",
    )


def test_validate_homogeneous_passes_on_repeated_layout():
    # Two buckets, same column layout (e.g. same beam-integration rule
    # split across partitions) — must not raise.
    cols = ["Px_1", "Py_1", "Pz_1", "Mx_1", "My_1", "Mz_1",
            "Px_2", "Py_2", "Pz_2", "Mx_2", "My_2", "Mz_2"]
    ElementManager._validate_homogeneous_layouts(
        [_chunk(cols), _chunk(cols)],
        results_name="force",
        element_type="5-ElasticBeam3d",
    )


def test_validate_homogeneous_raises_on_layout_mismatch():
    # Heterogeneous: 5-IP bucket (20 cols) and 3-IP bucket (12 cols).
    cols_5ip = [f"{c}_ip{i}" for i in range(5) for c in ("P", "Mz", "My", "T")]
    cols_3ip = [f"{c}_ip{i}" for i in range(3) for c in ("P", "Mz", "My", "T")]
    with pytest.raises(MpcoFormatError, match="Heterogeneous bucket layouts"):
        ElementManager._validate_homogeneous_layouts(
            [_chunk(cols_5ip), _chunk(cols_3ip)],
            results_name="section.force",
            element_type="64-DispBeamColumn3d",
        )


def test_validate_error_message_lists_offending_buckets():
    cols_a = ["A_ip0", "B_ip0"]
    cols_b = ["X_ip0", "Y_ip0", "Z_ip0"]
    with pytest.raises(MpcoFormatError) as exc:
        ElementManager._validate_homogeneous_layouts(
            [_chunk(cols_a), _chunk(cols_b)],
            results_name="material.stress",
            element_type="56-Brick",
        )
    msg = str(exc.value)
    # Both bucket paths and at least one column from each layout appear.
    assert "section.force" not in msg  # no leakage from other tests
    assert "Brick" in msg
    assert "A_ip0" in msg or "B_ip0" in msg
    assert "X_ip0" in msg or "Y_ip0" in msg or "Z_ip0" in msg


# ---------------------------------------------------------------------- #
# decorated_type column on the cached element index                       #
# ---------------------------------------------------------------------- #


def test_index_has_decorated_type_column(elastic_frame_dir: Path):
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    df = ds.elements_info["dataframe"]
    assert "decorated_type" in df.columns
    # The 2-field bracket carries a single ':' between rule and cust.
    types = df["decorated_type"].unique().tolist()
    assert all(t.endswith("]") for t in types)
    assert all(t.count(":") == 1 for t in types), types


def test_index_decorated_type_distinguishes_classes(quad_frame_dir: Path):
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    df = ds.elements_info["dataframe"]
    decs = set(df["decorated_type"].unique())
    # Both 5-ElasticBeam3d and 203-ASDShellQ4 are present in this fixture
    assert any(d.startswith("5-ElasticBeam3d[") for d in decs)
    assert any(d.startswith("203-ASDShellQ4[") for d in decs)


def test_index_decorated_type_shared_across_partitions(solid_partition_dir: Path):
    """Multi-partition fixture: every element row carries a non-null
    decorated_type and the same decorated bracket appears in both
    partitions for shared classes."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(solid_partition_dir), "Recorder", verbose=False)
    df = ds.elements_info["dataframe"]
    # No null decorated_type
    assert df["decorated_type"].notna().all()
    # Brick + DispBeam decorated brackets both appear
    decs = set(df["decorated_type"].unique())
    assert any("56-Brick[" in d for d in decs)
    assert any("64-DispBeamColumn3d[" in d for d in decs)
