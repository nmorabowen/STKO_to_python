"""Phase 5.1.b — AggregationEngine benchmarks.

Exercises the engineering-aggregation hot path: ``drift`` and the
story-envelope methods that chain through ``fetch`` +
``_resolve_story_nodes_by_z_tol`` + repeated ``drift`` calls. These
are the operations that drive multi-case drift profiles in the spec
§6 target, so their per-call cost sets the floor for that workflow.

Run with::

    pip install -e ".[bench]"
    pytest bench/test_aggregation_bench.py -q
"""
from __future__ import annotations

import pytest

from STKO_to_python import MPCODataSet


pytestmark = pytest.mark.bench


# ---------------------------------------------------------------------- #
# Pair-wise drift (forwarded → AggregationEngine.drift)
# ---------------------------------------------------------------------- #
def test_bench_drift_pairwise(benchmark, warm_dataset):
    """One drift time-history between two nodes. Cross-section of the
    full chain: NodalResults.drift forwarder → engine.drift → fetch
    (cache hit) → align + divide."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )

    def _drift():
        return nr.drift(top=1, bottom=2, component=1)

    series = benchmark(_drift)
    assert series.size > 0


# ---------------------------------------------------------------------- #
# Story clustering (private helper; no fetch)
# ---------------------------------------------------------------------- #
def test_bench_resolve_story_nodes(benchmark, warm_dataset):
    """Z-tolerance clustering is a prerequisite for every story-level
    method. Runs purely against nodes_info + a sort; no HDF5 traffic."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )

    def _cluster():
        return nr._resolve_story_nodes_by_z_tol(
            selection_set_id=None,
            selection_set_name=None,
            node_ids=[1, 2, 3, 4],
            coordinates=None,
            dz_tol=1e-6,
        )

    stories = benchmark(_cluster)
    assert isinstance(stories, list)


# ---------------------------------------------------------------------- #
# Interstory drift envelope — full chain
# ---------------------------------------------------------------------- #
def test_bench_interstory_drift_envelope(benchmark, warm_dataset):
    """End-to-end story-drift envelope. Touches story clustering,
    per-story representative selection, and N pair-wise drift calls
    (one per interstory pair). Scales with the story count — a proxy
    for the spec §6 "multi-case drift profile" target once a larger
    fixture is available."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )

    def _envelope():
        return nr.interstory_drift_envelope(
            component=1,
            node_ids=[1, 2, 3, 4],
            dz_tol=1e-6,
        )

    out = benchmark(_envelope)
    assert len(out) > 0


# ---------------------------------------------------------------------- #
# Orbit — dual fetch + stage/align
# ---------------------------------------------------------------------- #
def test_bench_orbit(benchmark, warm_dataset):
    """Two fetches (x-component and y-component) plus alignment. Cheap
    on cached results but the floor cost is mostly in the alignment
    and unstack rather than the fetches."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )

    def _orbit():
        return nr.orbit(node_ids=[1, 2], x_component=1, y_component=2)

    sx, sy = benchmark(_orbit)
    assert sx.size > 0
    assert sy.size > 0
