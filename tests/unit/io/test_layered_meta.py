"""Unit tests for layered-shell META parsing.

Layered fiber buckets (e.g. ``section.fiber.damage`` on
``203-ASDShellQ4`` from the Test_NLShell fixture) extend the META
format documented in §2 with two complications:

* GAUSS_IDS repeats: the same gauss-id appears once per thickness
  layer, so the array looks like ``[0,0,0,0,0, 1,1,1,1,1, ...]``.
* COMPONENTS includes empty segments paired with NUM_COMPONENTS=0 for
  layers that don't carry the requested quantity.

These cases were previously rejected by the parser. The relaxed
validator allows non-decreasing GAUSS_IDS with unique values forming
0..n_unique-1 and skips empty blocks during column-name generation.
The geometric ``n_ip`` (used by the catalog and physical-coord
mapping) reflects the unique gauss-point count, not the raw block
count.
"""
from __future__ import annotations

import numpy as np
import pytest

from STKO_to_python.io.meta_parser import (
    BucketLayout,
    MpcoFormatError,
    parse_bucket_meta,
)


class _FakeMeta(dict):
    class _Holder:
        def __init__(self, value):
            self._value = value

        def __getitem__(self, idx):
            assert idx == ()
            return self._value

    def __getitem__(self, k):
        return self._Holder(super().__getitem__(k))


class _FakeBucket:
    def __init__(self, *, components, multiplicity, gauss_ids, num_components,
                 num_columns, name="<fake>"):
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


# ---------------------------------------------------------------------- #
# Layered shell — minimal synthetic                                       #
# ---------------------------------------------------------------------- #


def test_layered_shell_two_layers_simple():
    """4 IPs × 2 layers, every layer has 1 component (sigma11).

    GAUSS_IDS = [0, 0, 1, 1, 2, 2, 3, 3] (each repeated per layer).
    """
    bucket = _FakeBucket(
        components=";".join(["0.1.2.3.4.sigma11"] * 8),
        multiplicity=[[1]] * 8,
        gauss_ids=[[0], [0], [1], [1], [2], [2], [3], [3]],
        num_components=[[1]] * 8,
        num_columns=8,
    )
    layout = parse_bucket_meta(bucket)
    assert layout.closed_form is False
    # Geometric n_ip is the unique count (4), not the block count (8).
    assert layout.n_ip == 4
    assert len(layout.flat_columns) == 8
    # Layer index resets per gauss-point.
    assert layout.flat_columns[0] == "sigma11_l0_ip0"
    assert layout.flat_columns[1] == "sigma11_l1_ip0"
    assert layout.flat_columns[2] == "sigma11_l0_ip1"
    assert layout.flat_columns[7] == "sigma11_l1_ip3"


def test_layered_shell_skips_empty_segments():
    """Like the real ``section.fiber.damage`` bucket: 4 IPs × 5
    layers, but only layers 0, 2, 4 carry the components. Layers 1, 3
    have NUM_COMPONENTS=0 and an empty segment after ``0.1.2.3.4.``.
    """
    # Per gauss-point: 5 segments. Layers 0/2/4 carry "d+,d-" (2
    # components each), layers 1/3 are empty.
    seg_filled = "0.1.2.3.4.d+,d-"
    seg_empty = "0.1.2.3.4."
    pattern_per_gid = [seg_filled, seg_empty, seg_filled, seg_empty, seg_filled]

    components = ";".join(pattern_per_gid * 4)  # 4 gauss-points
    multiplicity = [[1]] * 20
    num_components = [[2], [0], [2], [0], [2]] * 4
    gauss_ids = [[g] for g in (
        [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5
    )]
    # Total NUM_COLUMNS = 4 IPs * 3 filled layers * 2 components = 24
    bucket = _FakeBucket(
        components=components,
        multiplicity=multiplicity,
        gauss_ids=gauss_ids,
        num_components=num_components,
        num_columns=24,
    )
    layout = parse_bucket_meta(bucket)
    assert layout.n_ip == 4
    assert len(layout.flat_columns) == 24
    # Layer index keeps incrementing across empty layers (so the
    # layer count reflects position in the layer stack).
    assert layout.flat_columns[:6] == (
        "d+_l0_ip0", "d-_l0_ip0",
        "d+_l2_ip0", "d-_l2_ip0",
        "d+_l4_ip0", "d-_l4_ip0",
    )
    assert layout.flat_columns[-6:] == (
        "d+_l0_ip3", "d-_l0_ip3",
        "d+_l2_ip3", "d-_l2_ip3",
        "d+_l4_ip3", "d-_l4_ip3",
    )


def test_layered_with_fibers():
    """Both fibers (MULT>1) and layers — naming is
    ``<comp>_f<j>_l<layer>_ip<gid>``."""
    # 1 IP × 2 layers, each layer has 3 fibers carrying sigma11.
    bucket = _FakeBucket(
        components=";".join(["0.1.2.3.4.sigma11"] * 2),
        multiplicity=[[3], [3]],
        gauss_ids=[[0], [0]],
        num_components=[[1], [1]],
        num_columns=6,
    )
    layout = parse_bucket_meta(bucket)
    assert layout.n_ip == 1
    assert layout.flat_columns[:3] == (
        "sigma11_f0_l0_ip0",
        "sigma11_f1_l0_ip0",
        "sigma11_f2_l0_ip0",
    )
    assert layout.flat_columns[3:] == (
        "sigma11_f0_l1_ip0",
        "sigma11_f1_l1_ip0",
        "sigma11_f2_l1_ip0",
    )


def test_non_decreasing_gauss_ids_required():
    """Out-of-order gauss-ids (e.g. ``[0, 1, 0, 1]``) are still
    rejected — only sorted-with-repeats is allowed."""
    bucket = _FakeBucket(
        components="0.1.2.A;0.1.2.B;0.1.2.A;0.1.2.B",
        multiplicity=[[1]] * 4,
        gauss_ids=[[0], [1], [0], [1]],
        num_components=[[1]] * 4,
        num_columns=4,
    )
    with pytest.raises(MpcoFormatError, match="non-decreasing"):
        parse_bucket_meta(bucket)


def test_unique_gauss_ids_must_start_at_zero():
    bucket = _FakeBucket(
        components="0.1.2.A;0.1.2.B;0.1.2.A",
        multiplicity=[[1]] * 3,
        gauss_ids=[[1], [2], [3]],   # missing 0
        num_components=[[1]] * 3,
        num_columns=3,
    )
    with pytest.raises(MpcoFormatError, match="0..n-1"):
        parse_bucket_meta(bucket)


def test_non_layered_bucket_unchanged_naming():
    """For a non-layered bucket (each gauss-id appears exactly once),
    the column suffix stays at ``_ip<gid>`` — no ``_l`` infix.
    Locks in backward compat with everything from B1/B7a."""
    bucket = _FakeBucket(
        components=";".join(["0.1.2.P,Mz,My,T"] * 5),
        multiplicity=[[1]] * 5,
        gauss_ids=[[0], [1], [2], [3], [4]],
        num_components=[[4]] * 5,
        num_columns=20,
    )
    layout = parse_bucket_meta(bucket)
    # No "_l" anywhere when there's only one layer per IP.
    assert all("_l" not in c for c in layout.flat_columns)
    assert layout.flat_columns[0] == "P_ip0"
    assert layout.flat_columns[-1] == "T_ip4"


# ---------------------------------------------------------------------- #
# Real fixture — Test_NLShell layered fiber buckets                       #
# ---------------------------------------------------------------------- #


def test_real_nlshell_fixture_section_fiber_damage(nl_shell_dir):
    """Parse the actual ``section.fiber.damage`` bucket from the
    NL_layered_shell fixture and verify the column-name pattern."""
    import h5py

    p = nl_shell_dir / "Results.part-0.mpco"
    with h5py.File(p, "r") as f:
        bucket = f[
            "MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/section.fiber.damage/"
            "203-ASDShellQ4[201:0:0]"
        ]
        layout = parse_bucket_meta(bucket)

    assert layout.n_ip == 4  # 4 in-plane Gauss points
    assert layout.num_columns == 24
    cols = layout.flat_columns
    # First 2 components at IP 0 are layer 0
    assert cols[0] == "d+_l0_ip0"
    assert cols[1] == "d-_l0_ip0"
    # Layer indices in the file are 0, 2, 4 (1 and 3 are empty)
    layers_at_ip0 = sorted({
        int(c.split("_l")[1].split("_ip")[0]) for c in cols if c.endswith("_ip0")
    })
    assert layers_at_ip0 == [0, 2, 4]
