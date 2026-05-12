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

from STKO_to_python.model.cdata_reader import (
    CDataReader,
    ElementInfo,
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
