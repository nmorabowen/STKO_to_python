"""Unit tests for the MPCO META parser.

Mix of in-memory mock buckets (for invariant + error coverage) and
real-fixture exercises (for end-to-end shape coverage across every
bucket-type the example folder covers).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from STKO_to_python.io.meta_parser import (
    BucketLayout,
    MpcoFormatError,
    parse_bucket_meta,
    validate_data_shape,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "stko_results_examples"


# ----- helpers ------------------------------------------------------------ #


class _FakeMeta(dict):
    """Map-like META that returns numpy arrays via ``[k][()]`` semantics."""

    class _Holder:
        def __init__(self, value):
            self._value = value

        def __getitem__(self, idx):
            assert idx == ()
            return self._value

    def __getitem__(self, k):
        return self._Holder(super().__getitem__(k))


class _FakeBucket:
    """Minimal h5py-Group lookalike for parser tests."""

    def __init__(self, *, components, multiplicity, gauss_ids,
                 num_components, num_columns, name="<fake>"):
        self._meta = _FakeMeta(
            COMPONENTS=np.array([components.encode("utf-8")]),
            MULTIPLICITY=np.asarray(multiplicity, dtype=np.int32),
            GAUSS_IDS=np.asarray(gauss_ids, dtype=np.int32),
            NUM_COMPONENTS=np.asarray(num_components, dtype=np.int32),
        )
        self.attrs = {"NUM_COLUMNS": np.asarray([num_columns], dtype=np.int32)}
        self.name = name

    def __contains__(self, key):
        return key == "META"

    def __getitem__(self, key):
        if key == "META":
            return self._meta
        raise KeyError(key)


# ----- closed-form (GAUSS_IDS=[[-1]]) ------------------------------------- #


def test_closed_form_global_force_3d():
    bucket = _FakeBucket(
        components="0.Px_1,Py_1,Pz_1,Mx_1,My_1,Mz_1,Px_2,Py_2,Pz_2,Mx_2,My_2,Mz_2",
        multiplicity=[[1]],
        gauss_ids=[[-1]],
        num_components=[[12]],
        num_columns=12,
    )
    layout = parse_bucket_meta(bucket)

    assert layout.closed_form is True
    assert layout.n_ip == 0
    assert layout.gauss_ids == (-1,)
    assert layout.num_columns == 12
    assert layout.flat_columns[0] == "Px_1"
    assert layout.flat_columns[-1] == "Mz_2"
    assert len(layout.flat_columns) == 12


def test_closed_form_local_force_3d():
    bucket = _FakeBucket(
        components="0.N_1,Vy_1,Vz_1,T_1,My_1,Mz_1,N_2,Vy_2,Vz_2,T_2,My_2,Mz_2",
        multiplicity=[[1]],
        gauss_ids=[[-1]],
        num_components=[[12]],
        num_columns=12,
    )
    layout = parse_bucket_meta(bucket)

    assert layout.closed_form is True
    # N/Vy/Vz/T/My/Mz pattern — disambiguates from globalForce's Px/Py/Pz/Mx/My/Mz
    assert layout.flat_columns[0] == "N_1"
    assert layout.flat_columns[3] == "T_1"


# ----- line-stations (sequential GAUSS_IDS) ------------------------------- #


def test_line_station_5_ip_section_force():
    seg = "0.1.2.P,Mz,My,T"
    bucket = _FakeBucket(
        components=";".join([seg] * 5),
        multiplicity=[[1], [1], [1], [1], [1]],
        gauss_ids=[[0], [1], [2], [3], [4]],
        num_components=[[4], [4], [4], [4], [4]],
        num_columns=20,
    )
    layout = parse_bucket_meta(bucket)

    assert layout.closed_form is False
    assert layout.n_ip == 5
    assert layout.gauss_ids == (0, 1, 2, 3, 4)
    assert layout.num_columns == 20
    # IP suffix included
    assert layout.flat_columns[0] == "P_ip0"
    assert layout.flat_columns[3] == "T_ip0"
    assert layout.flat_columns[4] == "P_ip1"
    assert layout.flat_columns[-1] == "T_ip4"


# ----- compressed META (MULTIPLICITY > 1, fibers) ------------------------- #


def test_compressed_fiber_bucket_expands_with_fiber_suffix():
    bucket = _FakeBucket(
        components="0.1.2.3.4.sigma11;0.1.2.3.4.sigma11",
        multiplicity=[[6], [6]],
        gauss_ids=[[0], [1]],
        num_components=[[1], [1]],
        num_columns=12,
    )
    layout = parse_bucket_meta(bucket)

    assert layout.closed_form is False
    assert layout.num_columns == 12
    assert len(layout.flat_columns) == 12
    assert layout.flat_columns[0] == "sigma11_f0_ip0"
    assert layout.flat_columns[5] == "sigma11_f5_ip0"
    assert layout.flat_columns[6] == "sigma11_f0_ip1"
    assert layout.flat_columns[-1] == "sigma11_f5_ip1"


# ----- invariant violations (fail loud) ----------------------------------- #


def test_num_columns_invariant_violation_raises():
    bucket = _FakeBucket(
        components="0.Px_1,Py_1",
        multiplicity=[[1]],
        gauss_ids=[[-1]],
        num_components=[[2]],
        num_columns=99,  # disagrees with sum(MULT * NUM_COMP) = 2
    )
    with pytest.raises(MpcoFormatError, match="NUM_COLUMNS invariant violated"):
        parse_bucket_meta(bucket)


def test_segment_count_mismatch_raises():
    bucket = _FakeBucket(
        components="0.1.2.P,Mz,My,T",  # 1 segment
        multiplicity=[[1], [1], [1]],   # 3 blocks
        gauss_ids=[[0], [1], [2]],
        num_components=[[4], [4], [4]],
        num_columns=12,
    )
    with pytest.raises(MpcoFormatError, match="META segment count"):
        parse_bucket_meta(bucket)


def test_per_block_component_count_mismatch_raises():
    bucket = _FakeBucket(
        components="0.1.2.P,Mz,My",  # 3 names...
        multiplicity=[[1]],
        gauss_ids=[[0]],
        num_components=[[4]],         # ...but block expects 4
        num_columns=4,
    )
    with pytest.raises(MpcoFormatError, match="block 0 META segment has 3 names"):
        parse_bucket_meta(bucket)


def test_non_sequential_gauss_ids_raises():
    bucket = _FakeBucket(
        components="0.1.2.P;0.1.2.P",
        multiplicity=[[1], [1]],
        gauss_ids=[[0], [3]],         # not sequential
        num_components=[[1], [1]],
        num_columns=2,
    )
    with pytest.raises(MpcoFormatError, match="sequential 0..n-1"):
        parse_bucket_meta(bucket)


def test_closed_form_with_multiblock_raises():
    # Closed-form sentinel but >1 block → format violation.
    bucket = _FakeBucket(
        components="0.A;0.B",
        multiplicity=[[1], [1]],
        gauss_ids=[[-1]],             # sentinel for closed-form
        num_components=[[1], [1]],
        num_columns=2,
    )
    # Sentinel detected on flat gauss_ids; segment_count(2) != n_blocks(2) ok,
    # but n_blocks(2) != 1 for closed-form → raises.
    with pytest.raises(MpcoFormatError, match="closed-form bucket"):
        parse_bucket_meta(bucket)


# ----- validate_data_shape -------------------------------------------------- #


def test_validate_data_shape_passes_on_match():
    layout = BucketLayout(
        closed_form=True, n_ip=0, gauss_ids=(-1,),
        ip_components=(("a", "b", "c"),),
        flat_columns=("a", "b", "c"), num_columns=3,
    )
    # Should not raise.
    validate_data_shape(layout, (10, 3))


def test_validate_data_shape_raises_on_mismatch():
    layout = BucketLayout(
        closed_form=True, n_ip=0, gauss_ids=(-1,),
        ip_components=(("a", "b"),),
        flat_columns=("a", "b"), num_columns=2,
    )
    with pytest.raises(MpcoFormatError, match="disagrees with META"):
        validate_data_shape(layout, (10, 5))


# ----- real-fixture coverage ---------------------------------------------- #


def _real_buckets():
    """Iterate over (path, fixture_label, bucket_path, bucket_grp) for every
    bucket with a META under stko_results_examples/."""
    if not EXAMPLES.exists():
        return
    for mpco in sorted(EXAMPLES.rglob("*.mpco")):
        try:
            f = h5py.File(mpco, "r")
        except OSError:
            continue
        try:
            for stage in (k for k in f.keys() if k.startswith("MODEL_STAGE")):
                base = f"{stage}/RESULTS/ON_ELEMENTS"
                if base not in f:
                    continue
                for rname in f[base].keys():
                    rgrp = f[base][rname]
                    for ekey in rgrp.keys():
                        eg = rgrp[ekey]
                        if "META" in eg:
                            yield mpco, f"{base}/{rname}/{ekey}", eg
        finally:
            # Don't close — h5py.File needs to outlive the layout.flat_columns
            # snapshot (parameterize closes it). We close eagerly.
            pass
        f.close()


@pytest.mark.parametrize(
    "mpco_path,bucket_path",
    [
        pytest.param(p, b, id=f"{p.parent.name}::{b.split('/RESULTS/')[1]}")
        for p, b, _ in _real_buckets()
    ],
)
def test_parse_real_fixture_bucket(mpco_path: Path, bucket_path: str) -> None:
    """Every checked-in fixture bucket parses cleanly and passes the
    NUM_COLUMNS invariant. Catches format drift in real-world files."""
    with h5py.File(mpco_path, "r") as f:
        layout = parse_bucket_meta(f[bucket_path], bucket_path=bucket_path)

    assert isinstance(layout, BucketLayout)
    assert layout.num_columns > 0
    assert len(layout.flat_columns) == layout.num_columns
    # Names are non-empty and unique
    assert all(isinstance(c, str) and c for c in layout.flat_columns)
    assert len(set(layout.flat_columns)) == len(layout.flat_columns), (
        f"duplicate flat column names in {bucket_path}: {layout.flat_columns}"
    )
