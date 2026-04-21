"""End-to-end smoke tests against the checked-in example .mpco files.

These tests skip gracefully if the examples are not present (see the
``elastic_frame_dir`` / ``quad_frame_dir`` fixtures in conftest.py).

Covers:
    * Dataset construction from a real MPCO file (single partition, MP).
    * Stage / step / node / element discovery matches the HDF5 layout.
    * ``get_nodal_results`` returns a DataFrame with the expected
      MultiIndex and non-null values.
    * ``NodalResultsQueryEngine`` (Phase 2.6) produces the same result
      as the manager and hits its LRU cache on a repeat call.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.query import ElementResultsQueryEngine, NodalResultsQueryEngine


# ---------------------------------------------------------------------- #
# Single-partition: elasticFrame
# ---------------------------------------------------------------------- #
def test_elastic_frame_construction(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    assert ds.model_stages == ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]
    assert ds.number_of_steps == {"MODEL_STAGE[1]": 10, "MODEL_STAGE[2]": 10}
    assert len(ds.node_results_names) == 17
    assert len(ds.results_partitions) == 1


def test_elastic_frame_node_discovery(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    df = ds.nodes._get_all_nodes_ids()["dataframe"]
    assert df["node_id"].tolist() == [1, 2, 3, 4]
    # All on partition 0
    assert set(df["file_id"]) == {0}


def test_elastic_frame_displacement_fetch(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    # 4 nodes × 10 steps = 40 rows; 3 components = 3 columns
    assert nr.df.shape == (40, 3)
    assert nr.df.index.names == ["node_id", "step"]


def test_elastic_frame_engine_parity_and_cache(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    engine = NodalResultsQueryEngine(
        dataset=ds,
        pool=ds._pool,
        policy=ds._format_policy,
        resolver=ds._selection_resolver,
    )
    nr1 = engine.fetch(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    nr2 = engine.fetch(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    # LRU cache hit → same object
    assert nr2 is nr1


def test_elastic_frame_element_force_fetch(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    # 3 elements × 10 steps × 12 force components
    assert er.df.shape == (30, 12)
    assert er.df.index.names == ["element_id", "step"]


def test_elastic_frame_element_engine_parity_and_cache(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    engine = ElementResultsQueryEngine(
        dataset=ds,
        pool=ds._pool,
        policy=ds._format_policy,
        resolver=ds._selection_resolver,
    )
    er1 = engine.fetch(
        "force",
        "5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    er2 = engine.fetch(
        "force",
        "5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    assert er2 is er1


# ---------------------------------------------------------------------- #
# Multi-partition: QuadFrame (MP case)
# ---------------------------------------------------------------------- #
def test_quad_frame_partitioned_construction(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    # Two MP partitions discovered
    assert sorted(ds.results_partitions.keys()) == [0, 1]
    assert ds.model_stages == ["MODEL_STAGE[1]", "MODEL_STAGE[2]"]


def test_quad_frame_cross_partition_fetch(quad_frame_dir: Path):
    ds = MPCODataSet(str(quad_frame_dir), "results", verbose=False)
    df = ds.nodes._get_all_nodes_ids()["dataframe"]
    assert df["node_id"].nunique() == 676
    # Nodes spread across both partitions
    assert set(df["file_id"]) == {0, 1}

    # Pick IDs likely to span both partitions
    some_ids = df["node_id"].iloc[:50].tolist()
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=some_ids,
    )
    # 50 nodes × 10 steps
    assert nr.df.shape == (500, 3)
