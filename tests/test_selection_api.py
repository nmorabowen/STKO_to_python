from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest

from STKO_to_python.core.selection import Selection, SelectionBox
from STKO_to_python.nodes.nodes import Nodes
from STKO_to_python.results.nodal_results_dataclass import NodalResults
from STKO_to_python.results.nodal_results_info import NodalResultsInfo


class _DummyInfo:
    analysis_time = 0.0
    size = 0


class _DummyDataset:
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"

    def __init__(self, h5_path: Path):
        self.results_partitions = {0: str(h5_path)}
        self.model_stages = ["MODEL_STAGE[1]"]
        self.node_results_names = ["DISPLACEMENT"]
        self.selection_set = {
            7: {"SET_NAME": "Corner", "NODES": [20, 30]},
            8: {"SET_NAME": "Edge", "NODES": [10, 20]},
        }
        self.time = pd.DataFrame(
            {"TIME": [0.0, 1.0]},
            index=pd.MultiIndex.from_arrays(
                [["MODEL_STAGE[1]", "MODEL_STAGE[1]"], [0, 1]],
                names=["stage", "step"],
            ),
        )
        self.name = "dummy"
        self.plot_settings = None
        self.info = _DummyInfo()


@pytest.fixture()
def dummy_h5(tmp_path: Path) -> Path:
    h5_path = tmp_path / "results_nodes.mpco"
    with h5py.File(h5_path, "w") as h5:
        nodes = h5.create_group("/MODEL_STAGE[1]/MODEL/NODES")
        nodes.create_dataset("ID1", data=[10, 20, 30])
        nodes.create_dataset(
            "COORDINATES1",
            data=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0],
            ],
        )

        data = h5.create_group("/MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA")
        data.create_dataset(
            "STEP_0",
            data=[
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ],
        )
        data.create_dataset(
            "STEP_1",
            data=[
                [10.0, 10.0, 10.0],
                [11.0, 11.0, 11.0],
                [12.0, 12.0, 12.0],
            ],
        )
    return h5_path


@pytest.fixture()
def dummy_dataset(dummy_h5: Path) -> _DummyDataset:
    return _DummyDataset(dummy_h5)


@pytest.fixture()
def nodes_info_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "node_id": [10, 20, 30, 40],
            "x": [0.0, 1.0, 2.0, 5.0],
            "y": [0.0, 0.0, 0.0, 5.0],
            "z": [0.0, 0.0, 1.0, 5.0],
            "file_id": [0, 0, 1, 1],
        }
    ).set_index("node_id", drop=False)


@pytest.fixture()
def nodal_info(nodes_info_df: pd.DataFrame) -> NodalResultsInfo:
    return NodalResultsInfo(
        nodes_ids=(10, 20, 30, 40),
        nodes_info=nodes_info_df,
        selection_set={
            7: {"SET_NAME": "Corner", "NODES": [20, 30]},
            8: {"SET_NAME": "Edge", "NODES": [10, 20]},
        },
    )


@pytest.fixture()
def nodal_results(nodes_info_df: pd.DataFrame) -> NodalResults:
    columns = pd.MultiIndex.from_product(
        [["DISPLACEMENT"], [1, 2]],
        names=["result", "component"],
    )
    index = pd.MultiIndex.from_product(
        [[10, 20, 30, 40], [0, 1]],
        names=["node_id", "step"],
    )
    values = []
    for node_id in [10, 20, 30, 40]:
        for step in [0, 1]:
            values.append([node_id + step, node_id + 100 + step])

    df = pd.DataFrame(values, index=index, columns=columns)
    return NodalResults(
        df=df,
        time=[0.0, 1.0],
        name="synthetic",
        nodes_ids=(10, 20, 30, 40),
        nodes_info=nodes_info_df,
        results_components=("DISPLACEMENT|1", "DISPLACEMENT|2"),
        model_stages=("MODEL_STAGE[1]",),
        selection_set={
            7: {"SET_NAME": "Corner", "NODES": [20, 30]},
            8: {"SET_NAME": "Edge", "NODES": [10, 20]},
        },
    )


@pytest.fixture()
def story_pga_results() -> NodalResults:
    columns = pd.MultiIndex.from_product(
        [["ACCELERATION"], [1]],
        names=["result", "component"],
    )
    index = pd.MultiIndex.from_product(
        [[10, 20, 30, 40], [0, 1]],
        names=["node_id", "step"],
    )
    df = pd.DataFrame(
        [[-5.0], [-1.0], [3.0], [4.0], [2.0], [-7.0], [1.0], [6.0]],
        index=index,
        columns=columns,
    )
    nodes_info = pd.DataFrame(
        {
            "node_id": [10, 20, 30, 40],
            "x": [0.0, 1.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 3.0, 3.0],
            "file_id": [0, 0, 0, 0],
        }
    ).set_index("node_id", drop=False)
    return NodalResults(
        df=df,
        time=[0.0, 1.0],
        name="story",
        nodes_ids=(10, 20, 30, 40),
        nodes_info=nodes_info,
        results_components=("ACCELERATION|1",),
        model_stages=("MODEL_STAGE[1]",),
    )


@pytest.fixture()
def nodal_results_multi_stage(nodes_info_df: pd.DataFrame) -> NodalResults:
    columns = pd.MultiIndex.from_product(
        [["DISPLACEMENT"], [1, 2]],
        names=["result", "component"],
    )
    index = pd.MultiIndex.from_product(
        [["MODEL_STAGE[1]", "MODEL_STAGE[2]"], [10, 20], [0, 1]],
        names=["stage", "node_id", "step"],
    )
    values = []
    for stage_idx in [0, 1]:
        for node_id in [10, 20]:
            for step in [0, 1]:
                base = stage_idx * 100 + node_id + step
                values.append([base, base + 1000])

    df = pd.DataFrame(values, index=index, columns=columns)
    return NodalResults(
        df=df,
        time={"MODEL_STAGE[1]": [0.0, 1.0], "MODEL_STAGE[2]": [0.0, 1.0]},
        name="multi-stage",
        nodes_ids=(10, 20),
        nodes_info=nodes_info_df.loc[[10, 20]].copy(),
        results_components=("DISPLACEMENT|1", "DISPLACEMENT|2"),
        model_stages=("MODEL_STAGE[1]", "MODEL_STAGE[2]"),
        selection_set={
            7: {"SET_NAME": "Corner", "NODES": [20]},
            8: {"SET_NAME": "Edge", "NODES": [10, 20]},
        },
    )


def test_selection_normalizes_ids_and_box():
    selection = Selection(
        ids=[30, 10, 30],
        coordinates=[(0.0, 0.0), (1.0, 0.0)],
        box=SelectionBox((2.0, 1.0, 3.0), (0.0, -1.0, 1.0)),
    )

    assert selection.ids == (10, 30)
    assert selection.coordinates == ((0.0, 0.0), (1.0, 0.0))
    assert selection.box is not None
    assert selection.box.min_corner == (0.0, -1.0, 1.0)
    assert selection.box.max_corner == (2.0, 1.0, 3.0)


def test_selection_rejects_mixed_coordinate_dimensions():
    with pytest.raises(TypeError, match="cannot mix 2D and 3D"):
        Selection(coordinates=[(0.0, 0.0), (1.0, 0.0, 0.0)])


def test_selection_rejects_invalid_combine():
    with pytest.raises(ValueError, match="combine must be"):
        Selection(combine="xor")


def test_resolve_selection_supports_union_and_intersection(nodal_info: NodalResultsInfo):
    union_ids = nodal_info.resolve_selection(
        Selection(
            selection_set_name=("Corner",),
            coordinates=((0.0, 0.0, 0.0),),
            combine="union",
        )
    )
    intersection_ids = nodal_info.resolve_selection(
        Selection(
            selection_set_name=("Corner",),
            box=SelectionBox((1.5, -1.0, -1.0), (3.0, 1.0, 2.0)),
            combine="intersection",
        )
    )

    assert union_ids == [10, 20, 30]
    assert intersection_ids == [30]


def test_resolve_selection_supports_file_id_restricted_nearest(nodal_info: NodalResultsInfo):
    out = nodal_info.resolve_selection(
        Selection(
            coordinates=((1.1, 0.0, 0.0),),
            file_id=1,
        )
    )
    assert out == [30]


def test_get_nodal_results_accepts_coordinates(dummy_dataset: _DummyDataset):
    nodes = Nodes(dummy_dataset)
    result = nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        coordinates=[(1.1, 0.0, 0.0)],
    )

    assert result.info.nodes_ids == (20,)
    assert list(result.df.index.get_level_values("node_id").unique()) == [20]


def test_get_nodal_results_accepts_selection_box(dummy_dataset: _DummyDataset):
    nodes = Nodes(dummy_dataset)
    result = nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        selection_box=SelectionBox((0.5, -1.0, -1.0), (2.1, 1.0, 2.0)),
    )

    assert result.info.nodes_ids == (20, 30)
    assert list(result.df.index.get_level_values("node_id").unique()) == [20, 30]


def test_get_nodal_results_without_selector_returns_all_nodes(dummy_dataset: _DummyDataset):
    nodes = Nodes(dummy_dataset)
    result = nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
    )

    assert result.info.nodes_ids == (10, 20, 30)


def test_get_nodal_results_rejects_invalid_results_name(dummy_dataset: _DummyDataset):
    nodes = Nodes(dummy_dataset)

    with pytest.raises(ValueError, match="Invalid results_name value\\(s\\): \\('BOGUS',\\)"):
        nodes.get_nodal_results(
            results_name="BOGUS",
            model_stage="MODEL_STAGE[1]",
        )


def test_fetch_accepts_selection_and_matches_legacy_kwargs(nodal_results: NodalResults):
    legacy, legacy_nodes = nodal_results.fetch(
        result_name="DISPLACEMENT",
        component=1,
        coordinates=[(1.1, 0.0, 0.0)],
        return_nodes=True,
    )
    modern, modern_nodes = nodal_results.fetch(
        result_name="DISPLACEMENT",
        component=1,
        selection=Selection(coordinates=((1.1, 0.0, 0.0),)),
        return_nodes=True,
    )

    pd.testing.assert_series_equal(legacy, modern)
    assert legacy_nodes == modern_nodes == [20]


def test_resolve_node_ids_wrapper_supports_selection_and_legacy_kwargs(
    nodal_results: NodalResults,
):
    legacy = nodal_results.resolve_node_ids(coordinates=[(1.1, 0.0, 0.0)])
    modern = nodal_results.resolve_node_ids(
        selection=Selection(coordinates=((1.1, 0.0, 0.0),))
    )

    assert legacy == modern == [20]


def test_select_returns_new_filtered_results_and_preserves_original(
    nodal_results: NodalResults,
    tmp_path: Path,
):
    selected = nodal_results.select(
        selection=Selection(
            selection_set_name=("Corner",),
            box=SelectionBox((1.5, -1.0, -1.0), (3.0, 1.0, 2.0)),
            combine="intersection",
        )
    )

    assert selected is not nodal_results
    assert selected.info.nodes_ids == (30,)
    assert nodal_results.info.nodes_ids == (10, 20, 30, 40)
    assert list(selected.df.index.get_level_values("node_id").unique()) == [30]
    assert list(nodal_results.df.index.get_level_values("node_id").unique()) == [10, 20, 30, 40]

    path = tmp_path / "selected.pkl.gz"
    selected.save_pickle(path)
    loaded = NodalResults.load_pickle(path)
    assert loaded.info.nodes_ids == (30,)


def test_story_pga_envelope_reduce_nodes_exposes_reduced_fields(
    story_pga_results: NodalResults,
):
    outputs = {
        mode: story_pga_results.story_pga_envelope(
            result_name="ACCELERATION",
            component=1,
            node_ids=[10, 20, 30, 40],
            dz_tol=0.1,
            reduce_nodes=mode,
        )
        for mode in ("max_abs", "max", "min")
    }

    for col in ("max_acc", "min_acc", "pga", "ctrl_node_max", "ctrl_node_min", "ctrl_node_pga"):
        assert outputs["max_abs"][col].equals(outputs["max"][col])
        assert outputs["max_abs"][col].equals(outputs["min"][col])

    for out in outputs.values():
        assert "reduced_acc" in out.columns
        assert "ctrl_node_reduced" in out.columns

    assert outputs["max_abs"].loc[0.0, "reduced_acc"] == pytest.approx(5.0)
    assert outputs["max_abs"].loc[0.0, "ctrl_node_reduced"] == 10
    assert outputs["max"].loc[0.0, "reduced_acc"] == pytest.approx(4.0)
    assert outputs["max"].loc[0.0, "ctrl_node_reduced"] == 20
    assert outputs["min"].loc[0.0, "reduced_acc"] == pytest.approx(-5.0)
    assert outputs["min"].loc[0.0, "ctrl_node_reduced"] == 10


def test_orbit_accepts_selection(nodal_results: NodalResults):
    sx, sy, node_ids = nodal_results.orbit(
        result_name="DISPLACEMENT",
        x_component=1,
        y_component=2,
        selection=Selection(selection_set_name=("Corner",)),
        return_nodes=True,
    )

    assert node_ids == [20, 30]
    assert list(sx.index.get_level_values("node_id").unique()) == [20, 30]
    assert list(sy.index.get_level_values("node_id").unique()) == [20, 30]


def test_orbit_accepts_selection_box(nodal_results: NodalResults):
    sx, sy, node_ids = nodal_results.orbit(
        result_name="DISPLACEMENT",
        x_component=1,
        y_component=2,
        selection_box=SelectionBox((0.5, -1.0, -1.0), (2.1, 1.0, 2.0)),
        return_nodes=True,
    )

    assert node_ids == [20, 30]
    assert list(sx.index.get_level_values("node_id").unique()) == [20, 30]
    assert list(sy.index.get_level_values("node_id").unique()) == [20, 30]


def test_orbit_merges_selection_with_legacy_kwargs(nodal_results: NodalResults):
    _, _, node_ids = nodal_results.orbit(
        result_name="DISPLACEMENT",
        x_component=1,
        y_component=2,
        selection=Selection(selection_set_name=("Corner",)),
        node_ids=[10],
        return_nodes=True,
    )

    assert node_ids == [10, 20, 30]


def test_orbit_legacy_kwargs_still_work(nodal_results: NodalResults):
    sx, sy, node_ids = nodal_results.orbit(
        result_name="DISPLACEMENT",
        x_component=1,
        y_component=2,
        coordinates=[(1.1, 0.0, 0.0)],
        return_nodes=True,
    )

    assert node_ids == [20]
    assert list(sx.index.get_level_values("node_id").unique()) == [20]
    assert list(sy.index.get_level_values("node_id").unique()) == [20]


def test_orbit_requires_stage_for_multi_stage_results(
    nodal_results_multi_stage: NodalResults,
):
    with pytest.raises(ValueError, match="Multi-stage results detected. Provide stage=... Available:"):
        nodal_results_multi_stage.orbit(
            result_name="DISPLACEMENT",
            x_component=1,
            y_component=2,
            node_ids=[10],
        )
