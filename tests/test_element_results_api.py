from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest

from STKO_to_python.core.element_selection import ElementSelection
from STKO_to_python.core.selection import SelectionBox
from STKO_to_python.elements.elements import Elements
from STKO_to_python.results.element_results_dataclass import ElementResults


class _DummyInfo:
    analysis_time = 0.0
    size = 0


class _DummyElementDataset:
    MODEL_ELEMENTS_PATH = "/{model_stage}/MODEL/ELEMENTS"

    def __init__(self, h5_path: Path):
        self.results_partitions = {0: str(h5_path)}
        self.model_stages = ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]
        self.element_results_names = ["globalForces"]
        self.selection_set = {
            17: {"SET_NAME": "Core", "ELEMENTS": [100, 300]},
            18: {"SET_NAME": "BeamOnly", "ELEMENTS": [100, 200]},
        }
        self.time = pd.DataFrame(
            {"TIME": [0.0, 1.0, 0.0, 1.0]},
            index=pd.MultiIndex.from_arrays(
                [
                    ["MODEL_STAGE[1]", "MODEL_STAGE[1]", "MODEL_STAGE[2]", "MODEL_STAGE[2]"],
                    [0, 1, 0, 1],
                ],
                names=["stage", "step"],
            ),
        )
        self.name = "dummy-elements"
        self.plot_settings = None
        self.info = _DummyInfo()
        self.nodes_info = {
            "dataframe": pd.DataFrame(
                {
                    "node_id": [1, 2, 3, 4],
                    "x": [0.0, 1.0, 2.0, 0.0],
                    "y": [0.0, 0.0, 0.0, 2.0],
                    "z": [0.0, 0.0, 0.0, 0.0],
                }
            )
        }


@pytest.fixture()
def dummy_elements_h5(tmp_path: Path) -> Path:
    h5_path = tmp_path / "results_elements.mpco"
    beam = [[100, 1, 2], [200, 2, 3]]
    shell = [[300, 1, 3, 4]]

    with h5py.File(h5_path, "w") as h5:
        for stage_idx, stage in enumerate(["MODEL_STAGE[1]", "MODEL_STAGE[2]"]):
            model = h5.create_group(f"/{stage}/MODEL/ELEMENTS")
            model.create_dataset("Beam[0]", data=beam)
            model.create_dataset("Shell[0]", data=shell)

            beam_data = h5.create_group(f"/{stage}/RESULTS/ON_ELEMENTS/globalForces/Beam[0]/DATA")
            shell_data = h5.create_group(f"/{stage}/RESULTS/ON_ELEMENTS/globalForces/Shell[0]/DATA")

            offset = stage_idx * 100
            beam_data.create_dataset(
                "STEP_0",
                data=[
                    [1.0 + offset, 10.0 + offset],
                    [2.0 + offset, 20.0 + offset],
                ],
            )
            beam_data.create_dataset(
                "STEP_1",
                data=[
                    [11.0 + offset, 110.0 + offset],
                    [12.0 + offset, 120.0 + offset],
                ],
            )
            shell_data.create_dataset(
                "STEP_0",
                data=[[3.0 + offset, 30.0 + offset]],
            )
            shell_data.create_dataset(
                "STEP_1",
                data=[[13.0 + offset, 130.0 + offset]],
            )
    return h5_path


@pytest.fixture()
def dummy_element_dataset(dummy_elements_h5: Path) -> _DummyElementDataset:
    dataset = _DummyElementDataset(dummy_elements_h5)
    elements = Elements(dataset)
    dataset.elements_info = elements._get_all_element_index()
    return dataset


@pytest.fixture()
def elements(dummy_element_dataset: _DummyElementDataset) -> Elements:
    return Elements(dummy_element_dataset)


def test_get_element_results_without_selector_returns_all_elements(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
    )

    assert isinstance(results, ElementResults)
    assert results.info.elements_ids == (100, 200, 300)
    assert list(results.df.index.get_level_values("element_id").unique()) == [100, 200, 300]


def test_get_element_results_rejects_invalid_results_name(elements: Elements):
    with pytest.raises(ValueError, match="Invalid results_name value\\(s\\): \\('BOGUS',\\)"):
        elements.get_element_results(
            results_name="BOGUS",
            model_stage="MODEL_STAGE[1]",
        )


def test_get_element_results_validates_stage_before_selection_lookup(elements: Elements):
    with pytest.raises(ValueError, match="Invalid model_stage value\\(s\\): \\('MODEL_STAGE\\[9\\]',\\)"):
        elements.get_element_results(
            results_name="globalForces",
            model_stage="MODEL_STAGE[9]",
            selection_set_name="Core",
        )


def test_get_element_results_accepts_element_ids(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        element_ids=[200],
    )

    assert results.info.elements_ids == (200,)


def test_get_element_results_accepts_selection_set_id(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_set_id=17,
    )

    assert results.info.elements_ids == (100, 300)


def test_get_element_results_accepts_selection_set_name(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_set_name="BeamOnly",
    )

    assert results.info.elements_ids == (100, 200)


def test_get_element_results_accepts_selection_box(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_box=SelectionBox((1.0, -1.0), (2.0, 1.0)),
    )

    assert results.info.elements_ids == (200,)


def test_get_element_results_merges_selection_with_legacy_kwargs(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection=ElementSelection(selection_set_name=("BeamOnly",)),
        element_ids=[300],
    )

    assert results.info.elements_ids == (100, 200, 300)


def test_get_element_results_supports_mixed_decorated_types(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_set_id=17,
    )

    assert set(results.info.elements_info["element_type"].tolist()) == {"Beam[0]", "Shell[0]"}


def test_get_element_results_allows_optional_element_type_narrowing(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_set_id=17,
        element_type="Shell",
    )

    assert results.info.elements_ids == (300,)


def test_element_results_select_returns_new_filtered_results(elements: Elements):
    all_results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
    )
    selected = all_results.select(selection_set_name="BeamOnly")

    assert selected is not all_results
    assert selected.info.elements_ids == (100, 200)
    assert all_results.info.elements_ids == (100, 200, 300)


def test_element_results_pickle_round_trip(elements: Elements, tmp_path: Path):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
        selection_set_name="BeamOnly",
    )

    path = tmp_path / "element_results.pkl.gz"
    results.save_pickle(path)
    loaded = ElementResults.load_pickle(path)

    assert loaded.info.elements_ids == (100, 200)


def test_get_element_results_legacy_call_returns_dataframe(elements: Elements):
    out = elements.get_element_results(
        "globalForces",
        "Beam",
        [100],
        "MODEL_STAGE[1]",
    )

    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["val_1", "val_2"]
    assert list(out.index.get_level_values("element_id").unique()) == [100]


def test_get_element_results_multi_stage_uses_stage_index(elements: Elements):
    results = elements.get_element_results(
        results_name="globalForces",
        model_stage=None,
        element_ids=[100],
    )

    assert isinstance(results, ElementResults)
    assert results.df.index.names == ["stage", "element_id", "step"]
