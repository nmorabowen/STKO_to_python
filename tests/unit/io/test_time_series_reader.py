"""Unit tests for ``TimeSeriesReader``.

Exercises the STEP/TIME attr extraction against real on-disk HDF5 groups
built in ``tmp_path``. No .mpco fixture needed — we hand-build the
attrs in the shape MPCO uses (length-1 numpy arrays) and the shape
pre-numpy-1.25 writers used (python scalars).
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from STKO_to_python.io import TimeSeriesReader


def _build_data_group(path: Path, step_attrs: list[tuple]) -> None:
    """Create an MPCO-shaped DATA/STEP_k group layout.

    Parameters
    ----------
    step_attrs:
        Sequence of ``(step_name, step_value, time_value)`` tuples. The
        ``*_value`` entries are stored verbatim (caller chooses array
        vs scalar form).
    """
    with h5py.File(path, "w") as f:
        data = f.create_group("DATA")
        for step_name, step_value, time_value in step_attrs:
            g = data.create_group(step_name)
            if step_value is not None:
                g.attrs["STEP"] = step_value
            if time_value is not None:
                g.attrs["TIME"] = time_value


def test_reader_has_empty_slots():
    assert TimeSeriesReader.__slots__ == ()


def test_reader_repr_is_stable():
    assert repr(TimeSeriesReader()) == "TimeSeriesReader()"


def test_read_none_group_returns_empty_dict():
    r = TimeSeriesReader()
    assert r.read_step_time_pairs(None) == {}


def test_read_handles_length_1_array_attrs(tmp_path: Path):
    """MPCO format: STEP/TIME are length-1 numpy arrays."""
    p = tmp_path / "mpco_like.h5"
    _build_data_group(p, [
        ("STEP_0", np.array([0], dtype=np.int32), np.array([0.1])),
        ("STEP_1", np.array([1], dtype=np.int32), np.array([0.2])),
        ("STEP_2", np.array([2], dtype=np.int32), np.array([0.3])),
    ])
    r = TimeSeriesReader()
    with h5py.File(p, "r") as f:
        out = r.read_step_time_pairs(f["DATA"])
    assert out == {0: 0.1, 1: 0.2, 2: 0.3}


def test_read_handles_scalar_attrs(tmp_path: Path):
    """Legacy format: scalars (as older numpy / older STKO would write)."""
    p = tmp_path / "scalar.h5"
    _build_data_group(p, [
        ("STEP_0", np.int32(0), np.float64(0.5)),
        ("STEP_1", np.int32(1), np.float64(1.0)),
    ])
    r = TimeSeriesReader()
    with h5py.File(p, "r") as f:
        out = r.read_step_time_pairs(f["DATA"])
    assert out == {0: 0.5, 1: 1.0}


def test_read_skips_steps_missing_attrs(tmp_path: Path):
    p = tmp_path / "partial.h5"
    _build_data_group(p, [
        ("STEP_0", np.array([0]), np.array([0.1])),
        ("STEP_BAD", None, np.array([99.9])),      # no STEP
        ("STEP_NOTIME", np.array([5]), None),       # no TIME
        ("STEP_1", np.array([1]), np.array([0.2])),
    ])
    r = TimeSeriesReader()
    with h5py.File(p, "r") as f:
        out = r.read_step_time_pairs(f["DATA"])
    assert out == {0: 0.1, 1: 0.2}


def test_read_empty_data_group(tmp_path: Path):
    p = tmp_path / "empty.h5"
    _build_data_group(p, [])
    r = TimeSeriesReader()
    with h5py.File(p, "r") as f:
        out = r.read_step_time_pairs(f["DATA"])
    assert out == {}


def test_read_multi_unions_across_partitions(tmp_path: Path):
    """read_step_time_pairs_multi unions dicts; later partitions win on
    duplicate keys (they should agree in practice).
    """
    p1 = tmp_path / "p1.h5"
    p2 = tmp_path / "p2.h5"
    _build_data_group(p1, [
        ("STEP_0", np.array([0]), np.array([0.1])),
        ("STEP_1", np.array([1]), np.array([0.2])),
    ])
    _build_data_group(p2, [
        ("STEP_1", np.array([1]), np.array([0.2])),  # agreeing duplicate
        ("STEP_2", np.array([2]), np.array([0.3])),
    ])
    r = TimeSeriesReader()
    with h5py.File(p1, "r") as f1, h5py.File(p2, "r") as f2:
        out = r.read_step_time_pairs_multi({0: f1["DATA"], 1: f2["DATA"]})
    assert out == {0: 0.1, 1: 0.2, 2: 0.3}


def test_read_multi_tolerates_none_group():
    r = TimeSeriesReader()
    # One "partition" has no DATA group — must not raise
    out = r.read_step_time_pairs_multi({0: None, 1: None})
    assert out == {}
