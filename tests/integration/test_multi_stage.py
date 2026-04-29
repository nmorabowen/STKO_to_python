"""Multi-stage fetch tests against the elasticFrame example.

elasticFrame/results carries 2 stages (MODEL_STAGE[1], MODEL_STAGE[2])
with 10 steps each. The single-stage path is the original behavior;
the multi-stage path concatenates with a *contiguous global* step axis
(stage 2's first step = stage 1's last step + 1) and stitches the time
arrays end-to-end into a monotonic ndarray.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.elements.element_results import ElementResults
from STKO_to_python.io.meta_parser import MpcoFormatError


# ---------------------------------------------------------------------- #
# Single-stage back-compat
# ---------------------------------------------------------------------- #
def test_single_stage_fetch_unchanged(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    assert er.df.shape == (30, 12)
    assert er.df.index.names == ["element_id", "step"]
    assert er.model_stage == "MODEL_STAGE[1]"
    assert er.model_stages == ("MODEL_STAGE[1]",)
    assert not er.is_multi_stage
    assert er.stage_step_ranges == {"MODEL_STAGE[1]": (0, 10)}
    assert isinstance(er.time, np.ndarray)
    assert er.time.size == 10


# ---------------------------------------------------------------------- #
# Element multi-stage fetch
# ---------------------------------------------------------------------- #
def test_element_multi_stage_shape_and_steps(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    # Shape doubles: 3 elements × (10 + 10) steps × 12 components
    assert er.df.shape == (60, 12)
    assert er.df.index.names == ["element_id", "step"]
    assert er.is_multi_stage
    assert er.model_stages == ("MODEL_STAGE[1]", "MODEL_STAGE[2]")
    # Contiguous step axis: 0..19, monotonic
    steps = (
        er.df.index.get_level_values("step")
        .to_numpy()
    )
    assert steps.min() == 0
    assert steps.max() == 19
    # Step axis is monotonic across stages (per element)
    for eid in [1, 2, 3]:
        elem_steps = er.df.xs(eid, level="element_id").index.to_numpy()
        assert np.all(np.diff(elem_steps) > 0)
    # Stage boundaries: stage 1 ends at 10 (exclusive), stage 2 starts at 10
    s1_end = er.stage_step_ranges["MODEL_STAGE[1]"][1]
    s2_start = er.stage_step_ranges["MODEL_STAGE[2]"][0]
    assert s2_start == s1_end
    assert s1_end == 10


def test_element_multi_stage_time_monotonic(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    assert isinstance(er.time, np.ndarray)
    assert er.time.size == 20
    # Time array is monotonic non-decreasing — each stage's TIME records
    # absolute time, so the concatenation is monotonic for the
    # gravity → pushover → dynamic workflow this targets.
    assert np.all(np.diff(er.time) >= 0)


def test_element_multi_stage_matches_manual_concat(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er1 = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2],
        model_stage="MODEL_STAGE[1]",
    )
    er2 = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2],
        model_stage="MODEL_STAGE[2]",
    )
    er_multi = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    # Stage-1 slice of multi-stage matches single-stage er1 element-wise.
    s1_start, s1_end = er_multi.stage_step_ranges["MODEL_STAGE[1]"]
    multi_s1 = er_multi.df.loc[
        (slice(None), slice(s1_start, s1_end - 1)), :
    ].sort_index()
    np.testing.assert_array_equal(
        multi_s1.to_numpy(), er1.df.sort_index().to_numpy()
    )
    # Stage-2 slice (with offset removed) matches single-stage er2.
    s2_start, s2_end = er_multi.stage_step_ranges["MODEL_STAGE[2]"]
    multi_s2 = er_multi.df.loc[
        (slice(None), slice(s2_start, s2_end - 1)), :
    ].sort_index()
    np.testing.assert_array_equal(
        multi_s2.to_numpy(), er2.df.sort_index().to_numpy()
    )


def test_element_multi_stage_pickle_roundtrip(
    elastic_frame_dir: Path, tmp_path: Path
):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    p = tmp_path / "er_multi.pkl"
    er.save_pickle(p)
    er2 = ElementResults.load_pickle(p)

    pd.testing.assert_frame_equal(er.df, er2.df)
    np.testing.assert_array_equal(er.time, er2.time)
    assert er2.model_stages == er.model_stages
    assert er2.stage_step_ranges == er.stage_step_ranges
    assert er2.is_multi_stage


# ---------------------------------------------------------------------- #
# Engine cache keys: list ordering matters; same args → cache hit
# ---------------------------------------------------------------------- #
def test_element_multi_stage_engine_cache_hit(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er1 = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    er2 = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    assert er2 is er1  # cache hit


def test_element_stage_order_distinct_cache_entries(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage=["MODEL_STAGE[2]", "MODEL_STAGE[1]"],
    )
    # Different stage order → different concatenation order on the step
    # axis → different cache entries.
    assert ds._element_query_engine.cached_result_count == 2


# ---------------------------------------------------------------------- #
# Cross-stage layout validation: synthesized via a stub stage_buckets
# ---------------------------------------------------------------------- #
def test_cross_stage_layout_mismatch_raises():
    """Synthesize the stage_buckets dict with mismatched columns and
    confirm the validator raises ``MpcoFormatError`` with bucket-by-
    bucket detail. We don't have a fixture that exhibits this on disk
    (it's the restart-style models the spec calls out), so exercise the
    validator directly.
    """
    from STKO_to_python.elements.elements import ElementManager

    # The validator keys on the stage-independent bucket key
    # (``<connectivity>/<results_decorated>``) so a real cross-stage
    # mismatch shares the same key but disagrees on column tuples.
    bucket_key = "5-ElasticBeam3d[1000:1]/0:1[1]"
    stage_buckets = {
        "MODEL_STAGE[1]": {
            bucket_key: ("Px", "Py", "Pz"),
        },
        "MODEL_STAGE[2]": {
            bucket_key: ("Px", "Py", "Pz", "Mx"),
        },
    }
    with pytest.raises(MpcoFormatError) as excinfo:
        ElementManager._validate_homogeneous_layouts_across_stages(
            stage_buckets,
            results_name="force",
            element_type="5-ElasticBeam3d",
        )
    msg = str(excinfo.value)
    assert "Heterogeneous" in msg
    # bucket-by-bucket detail
    assert "MODEL_STAGE[1]" in msg
    assert "MODEL_STAGE[2]" in msg


# ---------------------------------------------------------------------- #
# Nodal multi-stage: same shape promise — contiguous step axis,
# monotonic time, stitched layout.
# ---------------------------------------------------------------------- #
def test_nodal_multi_stage_shape_and_steps(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
        node_ids=[1, 2, 3, 4],
    )
    # 4 nodes × (10 + 10) = 80 rows
    assert nr.df.shape[0] == 80
    assert nr.df.index.names == ["node_id", "step"]
    # Contiguous step axis 0..19
    steps = nr.df.index.get_level_values("step").to_numpy()
    assert steps.min() == 0
    assert steps.max() == 19
    # Monotonic time
    assert isinstance(nr.time, np.ndarray)
    assert nr.time.size == 20
    assert np.all(np.diff(nr.time) >= 0)
    # Stage metadata available on info
    assert nr.info.model_stages == ("MODEL_STAGE[1]", "MODEL_STAGE[2]")
    s1_end = nr.info.stage_step_ranges["MODEL_STAGE[1]"][1]
    s2_start = nr.info.stage_step_ranges["MODEL_STAGE[2]"][0]
    assert s1_end == s2_start == 10


def test_nodal_multi_stage_pickle_roundtrip(
    elastic_frame_dir: Path, tmp_path: Path
):
    from STKO_to_python.results.nodal_results_dataclass import NodalResults

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
        node_ids=[1, 2, 3, 4],
    )
    p = tmp_path / "nr_multi.pkl"
    nr.save_pickle(p)
    nr2 = NodalResults.load_pickle(p)
    pd.testing.assert_frame_equal(nr.df, nr2.df)
    np.testing.assert_array_equal(nr.time, nr2.time)
    assert nr2.info.model_stages == nr.info.model_stages
    assert nr2.info.stage_step_ranges == nr.info.stage_step_ranges


# ---------------------------------------------------------------------- #
# Plot helpers: stage boundary annotation
# ---------------------------------------------------------------------- #
def test_history_stage_boundary_annotation(elastic_frame_dir: Path):
    import matplotlib
    matplotlib.use("Agg")

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    component = er.list_components()[0]
    ax, meta = er.plot.history(component, x_axis="step")
    boundaries = meta.get("stage_boundaries", [])
    # One transition between the two stages — at step 9 (last step of
    # stage 1, since x_axis="step" puts boundaries at end_step - 1).
    assert len(boundaries) == 1
    assert boundaries[0][0] == "MODEL_STAGE[1]"
    assert boundaries[0][1] == 9.0


def test_history_no_annotation_when_disabled(elastic_frame_dir: Path):
    import matplotlib
    matplotlib.use("Agg")

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage=["MODEL_STAGE[1]", "MODEL_STAGE[2]"],
    )
    component = er.list_components()[0]
    ax, meta = er.plot.history(
        component, x_axis="step", annotate_stage_boundaries=False,
    )
    assert "stage_boundaries" not in meta


def test_history_no_annotation_for_single_stage(elastic_frame_dir: Path):
    import matplotlib
    matplotlib.use("Agg")

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="force",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    component = er.list_components()[0]
    ax, meta = er.plot.history(component, x_axis="step")
    # Single-stage results don't get any boundary annotation regardless
    # of the flag setting.
    assert "stage_boundaries" not in meta
