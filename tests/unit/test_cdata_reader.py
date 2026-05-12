"""Unit tests for the CDataReader text parser.

The parser is exercised directly against synthetic .cdata files on disk;
``MPCODataSet`` is bypassed because the reader only needs
``dataset.cdata_partitions``.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

import numpy as np

from STKO_to_python.model.cdata_format import CDataFormatPolicy
from STKO_to_python.model.cdata_reader import (
    BeamProfile,
    CDataReader,
    ElementInfo,
    _consume_ids,
    _read_length_prefixed,
)


def _ids_block(ids: list[int]) -> str:
    """Render an id list as the cdata wraps it: ten ids per line."""
    chunks = [
        " ".join(str(x) for x in ids[start : start + 10])
        for start in range(0, len(ids), 10)
    ]
    body = "\n".join(chunks)
    return body + "\n" if ids else ""


def _make_reader(tmp_path, *cdata_texts: str) -> CDataReader:
    partitions = {}
    for idx, text in enumerate(cdata_texts):
        path = tmp_path / f"results.part-{idx}.mpco.cdata"
        path.write_text(text, encoding="utf-8")
        partitions[idx] = str(path)
    return CDataReader(SimpleNamespace(cdata_partitions=partitions))


def test_single_partition_simple_set(tmp_path) -> None:
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "12 ControlPoint\n"
        "1\n"
        "0\n"
        "624\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert set(sets.keys()) == {1}
    assert sets[1]["SET_NAME"] == "ControlPoint"
    assert sets[1]["NODES"] == [624]
    assert sets[1]["ELEMENTS"] == []


def test_wrapped_multiline_node_list(tmp_path) -> None:
    """25 nodes wrap to 3 lines (10/10/5) — pins the (n+9)//10 math."""
    nodes = list(range(1, 26))
    text = (
        "*SELECTION_SET\n"
        "7\n"
        "10 Many_Nodes\n"
        f"{len(nodes)}\n"
        "0\n"
        f"{_ids_block(nodes)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[7]["NODES"] == nodes


def test_nodes_and_elements(tmp_path) -> None:
    nodes = [10, 20, 30]
    elements = list(range(100, 115))  # 15 elements -> 2 lines
    text = (
        "*SELECTION_SET\n"
        "9\n"
        "4 Both\n"
        f"{len(nodes)}\n"
        f"{len(elements)}\n"
        f"{_ids_block(nodes)}"
        f"{_ids_block(elements)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[9]["NODES"] == nodes
    assert sets[9]["ELEMENTS"] == elements


def test_zero_nodes_zero_elements(tmp_path) -> None:
    """A set with no members keeps its name (matches real AbsorbingBoundary)."""
    text = (
        "*SELECTION_SET\n"
        "21\n"
        "17 AbsorbingBoundary\n"
        "0\n"
        "0\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[21]["SET_NAME"] == "AbsorbingBoundary"
    assert sets[21]["NODES"] == []
    assert sets[21]["ELEMENTS"] == []


def test_zero_nodes_with_elements(tmp_path) -> None:
    """Regression: NNODES=0 + NELEMENTS>0 previously referenced an
    unbound ``nodes_end_line`` and raised UnboundLocalError.
    """
    elements = [100, 200, 300]
    text = (
        "*SELECTION_SET\n"
        "11\n"
        "8 ElemOnly\n"
        "0\n"
        f"{len(elements)}\n"
        f"{_ids_block(elements)}"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[11]["NODES"] == []
    assert sets[11]["ELEMENTS"] == elements


def test_no_selection_set_section(tmp_path) -> None:
    """A .cdata file without any *SELECTION_SET (matches elasticFrame example)
    yields an empty dict, not an error.
    """
    text = (
        "#Begin LOCAL AXES definitions.\n"
        "*LOCAL_AXES\n"
        "1 0 0.707107 0 0.707107\n"
    )
    reader = _make_reader(tmp_path, text)
    assert reader._extract_selection_set_ids() == {}


def test_multi_partition_aggregation(tmp_path) -> None:
    """Same SET_ID across partitions merges members and dedupes."""
    part0 = (
        "*SELECTION_SET\n"
        "1\n"
        "4 Roof\n"
        "3\n"
        "0\n"
        "1 2 3\n"
    )
    part1 = (
        "*SELECTION_SET\n"
        "1\n"
        "4 Roof\n"
        "2\n"
        "0\n"
        "3 4\n"  # 3 is a duplicate across partitions
    )
    reader = _make_reader(tmp_path, part0, part1)
    sets = reader._extract_selection_set_ids()
    assert sets[1]["NODES"] == [1, 2, 3, 4]
    assert sets[1]["SET_NAME"] == "Roof"


def test_filter_by_selection_set_ids(tmp_path) -> None:
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        "1\n"
        "0\n"
        "100\n"
        "*SELECTION_SET\n"
        "2\n"
        "1 B\n"
        "1\n"
        "0\n"
        "200\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids(selection_set_ids=[2])
    assert set(sets.keys()) == {2}


def test_malformed_file_raises_not_silent(tmp_path) -> None:
    """Truncated SELECTION_SET block bubbles up instead of silently
    returning {}; otherwise downstream queries fail far from the cause.
    """
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        # missing NNODES, NELEMENTS, and data
    )
    reader = _make_reader(tmp_path, text)
    with pytest.raises(Exception):
        reader._extract_selection_set_ids()


def test_parse_errors_logged_not_printed(tmp_path, caplog) -> None:
    """Error path uses ``logger.exception``, not ``print``."""
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "1 A\n"
        # truncated
    )
    reader = _make_reader(tmp_path, text)
    with caplog.at_level(logging.ERROR, logger="STKO_to_python.model.cdata_reader"):
        with pytest.raises(Exception):
            reader._extract_selection_set_ids()
    assert any(
        rec.name == "STKO_to_python.model.cdata_reader" and rec.levelno >= logging.ERROR
        for rec in caplog.records
    )


# ---------------------------------------------------------------------- #
# *LOCAL_AXES
# ---------------------------------------------------------------------- #

def test_local_axes_parses(tmp_path) -> None:
    text = (
        "*LOCAL_AXES\n"
        "1 0 0.707107 0 0.707107\n"
        "2 1 0 0 0\n"
    )
    reader = _make_reader(tmp_path, text)
    axes = reader.local_axes
    assert set(axes.keys()) == {1, 2}
    np.testing.assert_allclose(axes[1], [0, 0.707107, 0, 0.707107])
    np.testing.assert_allclose(axes[2], [1, 0, 0, 0])


def test_local_axes_multi_partition_merge(tmp_path) -> None:
    """Each partition owns disjoint elements; the merged dict covers both."""
    part0 = "*LOCAL_AXES\n1 1 0 0 0\n"
    part1 = "*LOCAL_AXES\n2 0 1 0 0\n"
    reader = _make_reader(tmp_path, part0, part1)
    axes = reader.local_axes
    assert set(axes.keys()) == {1, 2}


def test_local_axes_terminator_blank_line(tmp_path) -> None:
    """The blank line that separates sections must end the LOCAL_AXES block."""
    text = (
        "*LOCAL_AXES\n"
        "1 1 0 0 0\n"
        "\n"
        "#Begin SECTION OFFSET definitions.\n"
        "*SECTION_OFFSET\n"
    )
    reader = _make_reader(tmp_path, text)
    assert set(reader.local_axes.keys()) == {1}


# ---------------------------------------------------------------------- #
# *SECTION_OFFSET
# ---------------------------------------------------------------------- #

def test_section_offset_parses(tmp_path) -> None:
    text = (
        "*SECTION_OFFSET\n"
        "5 1.5 -2.0\n"
        "6 0 0\n"
    )
    reader = _make_reader(tmp_path, text)
    offsets = reader.section_offsets
    np.testing.assert_allclose(offsets[5], [1.5, -2.0])
    np.testing.assert_allclose(offsets[6], [0, 0])


def test_section_offset_empty_section(tmp_path) -> None:
    """STKO emits *SECTION_OFFSET with no data when nothing is offset
    (matches the elasticFrame and QuadFrame examples).
    """
    text = (
        "*SECTION_OFFSET\n"
        "\n"
        "#Begin BEAM PROFILE definitions.\n"
    )
    reader = _make_reader(tmp_path, text)
    assert reader.section_offsets == {}


# ---------------------------------------------------------------------- #
# *ELEMENT_INFO + length-prefixed name helper
# ---------------------------------------------------------------------- #

def test_read_length_prefixed_simple() -> None:
    name, rest = _read_length_prefixed("5 Merge 0 Edge")
    assert name == "Merge"
    assert rest == "0 Edge"


def test_read_length_prefixed_name_with_spaces() -> None:
    """STKO encodes 'LENGTH NAME' with LENGTH = char count of NAME,
    so embedded spaces in the name are preserved.
    """
    name, rest = _read_length_prefixed("21 New physical property 4 2 Q4")
    assert name == "New physical property"
    assert rest == "4 2 Q4"


def test_read_length_prefixed_at_end_of_string() -> None:
    name, rest = _read_length_prefixed("4 None")
    assert name == "None"
    assert rest == ""


def test_element_info_parses(tmp_path) -> None:
    """Mirrors the elasticFrame example row format."""
    text = (
        "*ELEMENT_INFO\n"
        "1 7 5 Merge 0 Edge 1 7 elastic 2 14 elasticBeamCol\n"
    )
    reader = _make_reader(tmp_path, text)
    info = reader.element_info
    assert set(info.keys()) == {1}
    ei = info[1]
    assert ei.element_id == 1
    assert ei.geom_id == 7
    assert ei.geom_name == "Merge"
    assert ei.sub_geom_idx == 0
    assert ei.sub_geom_type == "Edge"
    assert ei.physical_property_id == 1
    assert ei.physical_property_name == "elastic"
    assert ei.element_property_id == 2
    assert ei.element_property_name == "elasticBeamCol"


def test_element_info_property_name_with_spaces(tmp_path) -> None:
    """Property names can contain spaces (e.g. 'New physical property')."""
    text = (
        "*ELEMENT_INFO\n"
        "498 9 5 Merge 0 Face 4 21 New physical property 4 2 Q4\n"
    )
    reader = _make_reader(tmp_path, text)
    ei = reader.element_info[498]
    assert ei.geom_name == "Merge"
    assert ei.sub_geom_type == "Face"
    assert ei.physical_property_id == 4
    assert ei.physical_property_name == "New physical property"
    assert ei.element_property_id == 4
    assert ei.element_property_name == "Q4"


def test_element_info_returns_dataclass(tmp_path) -> None:
    text = (
        "*ELEMENT_INFO\n"
        "1 7 5 Merge 0 Edge 0 4 None 0 4 None\n"
    )
    reader = _make_reader(tmp_path, text)
    assert isinstance(reader.element_info[1], ElementInfo)


# ---------------------------------------------------------------------- #
# Multi-section single-pass
# ---------------------------------------------------------------------- #

def test_single_pass_parses_all_sections_one_file(tmp_path) -> None:
    """The driver dispatches markers in a single pass over the file."""
    text = (
        "#Begin LOCAL AXES.\n"
        "*LOCAL_AXES\n"
        "1 1 0 0 0\n"
        "\n"
        "*SECTION_OFFSET\n"
        "1 10 20\n"
        "\n"
        "*ELEMENT_INFO\n"
        "1 7 5 Merge 0 Edge 0 4 None 0 4 None\n"
        "\n"
        "*SELECTION_SET\n"
        "1\n"
        "4 Roof\n"
        "2\n"
        "0\n"
        "10 20\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()  # triggers eager parse
    assert sets[1]["NODES"] == [10, 20]
    np.testing.assert_allclose(reader.local_axes[1], [1, 0, 0, 0])
    np.testing.assert_allclose(reader.section_offsets[1], [10, 20])
    assert reader.element_info[1].geom_name == "Merge"


def test_real_quadframe_partitions(tmp_path) -> None:
    """Smoke test against the committed multi-partition example: every
    section that exists in the file must populate without error.
    """
    import os.path
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base = os.path.join(here, "stko_results_examples", "elasticFrame", "QuadFrame_results")
    p0 = os.path.join(base, "results.part-0.mpco.cdata")
    p1 = os.path.join(base, "results.part-1.mpco.cdata")
    if not (os.path.exists(p0) and os.path.exists(p1)):
        pytest.skip("QuadFrame example not present in this checkout")

    reader = CDataReader(SimpleNamespace(cdata_partitions={0: p0, 1: p1}))
    # Each row in *LOCAL_AXES is one element; partitions should not
    # overlap, so the merged dict has at least as many entries as either.
    assert len(reader.local_axes) > 0
    assert all(arr.shape == (4,) for arr in reader.local_axes.values())
    # SECTION_OFFSET is empty in the example.
    assert reader.section_offsets == {}
    # ELEMENT_INFO covers every element with *LOCAL_AXES.
    assert len(reader.element_info) >= len(reader.local_axes)
    # Beam profile 1 (the rectangular section) exists in both partitions
    # — dedup should produce exactly one entry.
    assert 1 in reader.beam_profiles
    p1 = reader.beam_profiles[1]
    assert p1.points.shape == (4, 2)
    # Every beam element in the file has a profile assignment.
    assert len(reader.beam_profile_assignments) > 0


# ---------------------------------------------------------------------- #
# *BEAM_PROFILE
# ---------------------------------------------------------------------- #

# Rectangular 4-point profile with 2 triangles, 4 straight edges, 4 sweep
# points. Mirrors the bundled elasticFrame example exactly.
_RECT_PROFILE = (
    "*BEAM_PROFILE\n"
    "1\n"
    "4 2 4 4\n"
    "-150 -200\n"
    "150 -200\n"
    "150 200\n"
    "-150 200\n"
    "0 1 2\n"
    "0 2 3\n"
    "2 0 1\n"
    "2 1 2\n"
    "2 2 3\n"
    "2 3 0\n"
    "0\n"
    "1\n"
    "2\n"
    "3\n"
)


def test_beam_profile_rectangle_parses(tmp_path) -> None:
    reader = _make_reader(tmp_path, _RECT_PROFILE)
    profiles = reader.beam_profiles
    assert set(profiles.keys()) == {1}

    p = profiles[1]
    assert isinstance(p, BeamProfile)
    assert p.profile_id == 1
    np.testing.assert_allclose(
        p.points, [[-150, -200], [150, -200], [150, 200], [-150, 200]]
    )
    np.testing.assert_array_equal(p.triangles, [[0, 1, 2], [0, 2, 3]])
    # All 4 edges are 2-point edges around the rectangle outline.
    assert len(p.edges) == 4
    np.testing.assert_array_equal(p.edges[0], [0, 1])
    np.testing.assert_array_equal(p.edges[3], [3, 0])
    np.testing.assert_array_equal(p.sweeps, [0, 1, 2, 3])


def test_beam_profile_variable_length_edges(tmp_path) -> None:
    """An edge with more than 2 points (curved outline)."""
    text = (
        "*BEAM_PROFILE\n"
        "5\n"
        "3 1 2 1\n"
        "0 0\n"
        "1 0\n"
        "0 1\n"
        "0 1 2\n"        # 1 triangle
        "3 0 1 2\n"      # edge with 3 points
        "2 2 0\n"        # edge with 2 points
        "0\n"            # 1 sweep
    )
    reader = _make_reader(tmp_path, text)
    p = reader.beam_profiles[5]
    assert len(p.edges) == 2
    np.testing.assert_array_equal(p.edges[0], [0, 1, 2])
    np.testing.assert_array_equal(p.edges[1], [2, 0])


def test_beam_profile_multiple_profiles_in_one_section(tmp_path) -> None:
    """A *BEAM_PROFILE section can carry several profile blocks back-to-back."""
    text = (
        "*BEAM_PROFILE\n"
        # profile 1: 2 points, 0 tris, 1 edge (2-point), 0 sweeps
        "1\n"
        "2 0 1 0\n"
        "0 0\n"
        "1 0\n"
        "2 0 1\n"
        # profile 2: 3 points, 1 tri, 0 edges, 1 sweep
        "2\n"
        "3 1 0 1\n"
        "0 0\n"
        "1 0\n"
        "0 1\n"
        "0 1 2\n"
        "0\n"
    )
    reader = _make_reader(tmp_path, text)
    assert set(reader.beam_profiles.keys()) == {1, 2}
    assert reader.beam_profiles[1].points.shape == (2, 2)
    assert reader.beam_profiles[2].triangles.shape == (1, 3)


def test_beam_profile_multi_partition_dedup_first_wins(tmp_path) -> None:
    """Profile defs are duplicated identically across MP partitions; keep one."""
    reader = _make_reader(tmp_path, _RECT_PROFILE, _RECT_PROFILE)
    assert set(reader.beam_profiles.keys()) == {1}


# ---------------------------------------------------------------------- #
# *BEAM_PROFILE_ASSIGNMENT
# ---------------------------------------------------------------------- #

def test_beam_profile_assignment_simple(tmp_path) -> None:
    text = (
        "*BEAM_PROFILE_ASSIGNMENT\n"
        "1 1 1 1\n"
        "2 1 1 1\n"
        "3 1 2 0.5\n"
    )
    reader = _make_reader(tmp_path, text)
    a = reader.beam_profile_assignments
    assert a[1] == [(1, 1.0)]
    assert a[2] == [(1, 1.0)]
    assert a[3] == [(2, 0.5)]


def test_beam_profile_assignment_multiple_profiles_per_element(tmp_path) -> None:
    """An element can carry multiple profiles, modelling a tapered section."""
    text = (
        "*BEAM_PROFILE_ASSIGNMENT\n"
        "1 2 1 0.3 2 0.7\n"
    )
    reader = _make_reader(tmp_path, text)
    assert reader.beam_profile_assignments[1] == [(1, 0.3), (2, 0.7)]


def test_beam_profile_assignment_multi_partition(tmp_path) -> None:
    """Each element is owned by exactly one partition; merging is dict-union."""
    part0 = "*BEAM_PROFILE_ASSIGNMENT\n1 1 1 1\n2 1 1 1\n"
    part1 = "*BEAM_PROFILE_ASSIGNMENT\n3 1 1 1\n4 1 1 1\n"
    reader = _make_reader(tmp_path, part0, part1)
    a = reader.beam_profile_assignments
    assert set(a.keys()) == {1, 2, 3, 4}


# ---------------------------------------------------------------------- #
# CDataFormatPolicy — section marker constants and queries
# ---------------------------------------------------------------------- #

def test_format_policy_known_markers_covers_every_section() -> None:
    """Every section the parser dispatches on must be in known_markers().

    Pins the contract: if a new *MARKER is added to the parser, it must
    also be added to CDataFormatPolicy.
    """
    expected = {
        "*SELECTION_SET",
        "*LOCAL_AXES",
        "*SECTION_OFFSET",
        "*BEAM_PROFILE",
        "*BEAM_PROFILE_ASSIGNMENT",
        "*ELEMENT_INFO",
    }
    assert CDataFormatPolicy.known_markers() == expected


def test_format_policy_is_section_marker_recognizes_known() -> None:
    assert CDataFormatPolicy.is_section_marker("*SELECTION_SET")
    assert CDataFormatPolicy.is_section_marker("  *LOCAL_AXES  ")
    assert not CDataFormatPolicy.is_section_marker("*UNKNOWN")
    assert not CDataFormatPolicy.is_section_marker("# comment")
    assert not CDataFormatPolicy.is_section_marker("")


def test_format_policy_is_any_marker_flags_arbitrary_star_lines() -> None:
    """Boundary detection: any '*' line ends a section, recognized or not."""
    assert CDataFormatPolicy.is_any_marker("*UNKNOWN")
    assert CDataFormatPolicy.is_any_marker("*ELEMENT_INFO")
    assert not CDataFormatPolicy.is_any_marker("# comment")
    assert not CDataFormatPolicy.is_any_marker("")


def test_format_policy_is_stateless_and_hashable() -> None:
    """The policy uses __slots__ = () so instances are interchangeable."""
    a = CDataFormatPolicy()
    b = CDataFormatPolicy()
    assert repr(a) == repr(b) == "CDataFormatPolicy()"


# ---------------------------------------------------------------------- #
# _consume_ids — width-agnostic id list reader
# ---------------------------------------------------------------------- #

def test_consume_ids_zero_count_is_noop() -> None:
    lines = ["10 20 30\n"]
    arr, end = _consume_ids(lines, 0, 0)
    assert arr.shape == (0,)
    assert end == 0


def test_consume_ids_single_line_default_wrap() -> None:
    lines = ["1 2 3 4 5 6 7 8 9 10\n", "11 12\n"]
    arr, end = _consume_ids(lines, 0, 12)
    assert arr.tolist() == list(range(1, 13))
    assert end == 2


def test_consume_ids_one_per_line() -> None:
    """Width-agnostic: a file wrapped at 1 id per line still parses."""
    lines = [f"{x}\n" for x in range(1, 11)]
    arr, end = _consume_ids(lines, 0, 10)
    assert arr.tolist() == list(range(1, 11))
    assert end == 10


def test_consume_ids_all_on_one_line() -> None:
    """Width-agnostic: a file with no wrap (all on one line) still parses."""
    lines = ["1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n"]
    arr, end = _consume_ids(lines, 0, 15)
    assert arr.tolist() == list(range(1, 16))
    assert end == 1


def test_consume_ids_stops_when_count_reached_in_middle_of_line() -> None:
    """If the wrap width over-emits on the last line, take only what's needed."""
    lines = ["1 2 3 4 5\n"]
    arr, end = _consume_ids(lines, 0, 3)
    assert arr.tolist() == [1, 2, 3]
    assert end == 1


def test_consume_ids_raises_on_truncation_by_marker() -> None:
    lines = ["1 2 3\n", "*SELECTION_SET\n"]
    with pytest.raises(ValueError, match="truncated"):
        _consume_ids(lines, 0, 10)


def test_consume_ids_raises_on_truncation_by_blank() -> None:
    lines = ["1 2 3\n", "\n", "next section\n"]
    with pytest.raises(ValueError, match="truncated"):
        _consume_ids(lines, 0, 10)


def test_consume_ids_raises_on_eof() -> None:
    lines = ["1 2 3\n"]
    with pytest.raises(ValueError, match="EOF"):
        _consume_ids(lines, 0, 10)


# ---------------------------------------------------------------------- #
# Selection-set parser is now width-agnostic end-to-end
# ---------------------------------------------------------------------- #

def test_selection_set_parses_with_one_id_per_line_wrap(tmp_path) -> None:
    """A file wrapping ids at width=1 should parse identically to width=10."""
    text = (
        "*SELECTION_SET\n"
        "1\n"
        "4 Test\n"
        "5\n"           # NNODES = 5
        "0\n"           # NELEMENTS = 0
        # 5 nodes, one per line — atypical but legal under the new parser
        "100\n"
        "200\n"
        "300\n"
        "400\n"
        "500\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[1]["NODES"] == [100, 200, 300, 400, 500]


def test_selection_set_parses_with_all_ids_on_one_line(tmp_path) -> None:
    """A file with no wrap at all should still parse."""
    text = (
        "*SELECTION_SET\n"
        "2\n"
        "3 Big\n"
        "15\n"
        "0\n"
        # 15 nodes all on one line — outside the conventional wrap
        "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\n"
    )
    reader = _make_reader(tmp_path, text)
    sets = reader._extract_selection_set_ids()
    assert sets[2]["NODES"] == list(range(1, 16))
