from __future__ import annotations

import pandas as pd
import pytest

from STKO_to_python.nodes.nodes import Nodes
from STKO_to_python.results.nodal_results_info import NodalResultsInfo


def _info_with_selection_set(selection_set: dict) -> NodalResultsInfo:
    return NodalResultsInfo(
        selection_set=selection_set,
        nodes_ids=(1, 2, 3, 4),
    )


class _DummyDataset:
    def __init__(self) -> None:
        self.model_stages = ["MODEL_STAGE[1]"]
        self.node_results_names = ["DISPLACEMENT"]
        self.selection_set = {
            33: {"SET_NAME": "TopNode", "NODES": []},
            41: {"SET_NAME": "topNode", "NODES": [2]},
        }
        self.nodes_info = {
            "dataframe": pd.DataFrame(
                {
                    "node_id": [1, 2, 3, 4],
                    "file_id": [0, 0, 0, 0],
                    "index": [0, 1, 2, 3],
                    "x": [0.0, 1.0, 2.0, 3.0],
                    "y": [0.0, 0.0, 0.0, 0.0],
                    "z": [0.0, 0.0, 0.0, 0.0],
                }
            )
        }


def test_selection_set_name_uses_exact_case_first():
    info = _info_with_selection_set(
        {
            33: {"SET_NAME": "TopNode", "NODES": [1, 2]},
            41: {"SET_NAME": "topNode", "NODES": [3, 4]},
        }
    )

    assert info.selection_set_ids_from_names("TopNode") == (33,)
    assert info.selection_set_ids_from_names("topNode") == (41,)


def test_selection_set_name_uses_case_insensitive_match_when_unique():
    info = _info_with_selection_set(
        {
            7: {"SET_NAME": "RoofNodes", "NODES": [1, 2]},
        }
    )

    assert info.selection_set_ids_from_names("roofnodes") == (7,)


def test_selection_set_name_ambiguity_still_raises_when_no_exact_match():
    info = _info_with_selection_set(
        {
            33: {"SET_NAME": "TopNode", "NODES": [1]},
            41: {"SET_NAME": "topNode", "NODES": [2]},
        }
    )

    with pytest.raises(ValueError, match="case-insensitive matches IDs \\[33, 41\\]"):
        info.selection_set_ids_from_names("TOPNODE")


def test_empty_selection_set_raises_clear_error():
    info = _info_with_selection_set(
        {
            33: {"SET_NAME": "TopNode", "NODES": []},
            41: {"SET_NAME": "topNode", "NODES": [2]},
        }
    )

    with pytest.raises(ValueError, match="Selection set 'TopNode' \\(id=33\\) contains 0 nodes"):
        info.selection_set_node_ids_by_name("TopNode", only_available=False)


def test_example_stage_validation_happens_before_selection_lookup():
    model = _DummyDataset()
    nodes = Nodes(model)

    with pytest.raises(ValueError, match=r"Invalid model_stage value\(s\): \('MODEL_STAGE\[5\]',\)\. Available: \('MODEL_STAGE\[1\]',\)"):
        nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage="MODEL_STAGE[5]",
            selection_set_name="TopNode",
        )


def test_example_topnode_resolves_exact_case_then_fails_empty():
    model = _DummyDataset()
    nodes = Nodes(model)

    assert model.model_stages == ["MODEL_STAGE[1]"]
    assert model.selection_set[33]["SET_NAME"] == "TopNode"

    with pytest.raises(ValueError, match="Selection set 'TopNode' \\(id=33\\) contains 0 nodes"):
        nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage="MODEL_STAGE[1]",
            selection_set_name="TopNode",
        )


def test_resolve_node_ids_defaults_to_all_available_nodes():
    class DummyDataset:
        selection_set = {}

    nodes = Nodes(DummyDataset())
    nodes._node_index_df = pd.DataFrame(
        {
            "node_id": [10, 20, 30],
            "file_id": [0, 0, 1],
            "index": [0, 1, 0],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
        }
    )

    out = nodes._resolve_node_ids(
        node_ids=None,
        selection_set_id=None,
        selection_set_name=None,
    )

    assert out.tolist() == [10, 20, 30]
