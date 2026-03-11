from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from STKO_to_python.core.dataset import MPCODataSet
from STKO_to_python.model.model_info import ModelInfo


def test_print_summary_counts_dict_backed_metadata(capsys):
    dataset = object.__new__(MPCODataSet)
    dataset.recorder_name = "results_nodes"
    dataset.results_partitions = {0: "part-0.mpco", 1: "part-1.mpco"}
    dataset.model_stages = ["MODEL_STAGE[1]"]
    dataset.node_results_names = ["DISPLACEMENT"]
    dataset.element_results_names = ["FORCE"]
    dataset.unique_element_types = ["ASDShellQ4"]
    dataset.nodes_info = {
        "dataframe": pd.DataFrame({"node_id": [1, 2, 3]}),
        "array": np.array([1, 2, 3]),
    }
    dataset.elements_info = {
        "dataframe": pd.DataFrame({"element_id": [10, 20, 30, 40]}),
        "array": np.array([10, 20, 30, 40]),
    }
    dataset.number_of_steps = {"MODEL_STAGE[1]": 2}
    dataset.selection_set = {7: {"SET_NAME": "Top"}}

    dataset.print_summary()
    out = capsys.readouterr().out

    assert "Number of nodes: 3" in out
    assert "Number of elements: 4" in out


def test_time_series_readers_coerce_singleton_hdf5_attrs(tmp_path: Path):
    h5_path = tmp_path / "results_nodes.part-0.mpco"
    with h5py.File(h5_path, "w") as h5:
        node_data = h5.create_group("MODEL_STAGE[1]/RESULTS/ON_NODES/DISPLACEMENT/DATA")
        step0 = node_data.create_dataset("STEP_0", data=np.zeros((1, 1)))
        step0.attrs["STEP"] = np.array([0], dtype=np.int64)
        step0.attrs["TIME"] = np.array([0.25], dtype=np.float64)
        step1 = node_data.create_dataset("STEP_1", data=np.zeros((1, 1)))
        step1.attrs["STEP"] = np.array([1], dtype=np.int64)
        step1.attrs["TIME"] = np.array([0.5], dtype=np.float64)

        elem_data = h5.create_group("MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/FORCE/ASDShellQ4/DATA")
        estep0 = elem_data.create_dataset("STEP_0", data=np.zeros((1, 1)))
        estep0.attrs["STEP"] = np.array([0], dtype=np.int64)
        estep0.attrs["TIME"] = np.array([1.25], dtype=np.float64)
        estep1 = elem_data.create_dataset("STEP_1", data=np.zeros((1, 1)))
        estep1.attrs["STEP"] = np.array([1], dtype=np.int64)
        estep1.attrs["TIME"] = np.array([1.5], dtype=np.float64)

    class _DummyDataset:
        def __init__(self, path: Path) -> None:
            self.results_partitions = {0: str(path)}

    info = ModelInfo(_DummyDataset(h5_path))

    nodal_df = info._get_time_series_on_nodes_for_stage("MODEL_STAGE[1]", "DISPLACEMENT")
    element_df = info._get_time_series_on_elements_for_stage(
        "MODEL_STAGE[1]", "FORCE", "ASDShellQ4"
    )

    assert nodal_df["STEP"].tolist() == [0, 1]
    assert nodal_df["TIME"].tolist() == [0.25, 0.5]
    assert element_df["STEP"].tolist() == [0, 1]
    assert element_df["TIME"].tolist() == [1.25, 1.5]
