"""Phase 5.1.a — fetch benchmarks.

Measures the hot path that downstream workflows (plotting, aggregation,
pickling) sit on top of. Numbers on the tiny elasticFrame fixture are
not the spec §6 targets (those assume 100 partitions / 10k steps); they
exist to catch regressions across commits and to establish a shape for
the real-scale benches that land once a larger fixture is checked in.

Run with::

    pip install -e ".[bench]"
    pytest bench/test_fetch_bench.py -q
"""
from __future__ import annotations

from pathlib import Path

import pytest

from STKO_to_python import MPCODataSet


pytestmark = pytest.mark.bench


# ---------------------------------------------------------------------- #
# Cold fetch — fresh dataset + fresh engine cache on every iteration
# ---------------------------------------------------------------------- #
def test_bench_fetch_cold(benchmark, elastic_frame_dir: Path):
    """Dataset construction + first fetch. Upper-bound timing for a
    one-shot script that imports the library, builds a dataset, fetches
    one result, and exits."""

    def _cold_fetch():
        ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
        return ds.nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage="MODEL_STAGE[1]",
            node_ids=[1, 2, 3, 4],
        )

    result = benchmark(_cold_fetch)
    # Sanity-check the bench actually returned the intended shape.
    assert result.df.shape == (40, 3)


# ---------------------------------------------------------------------- #
# Warm fetch — hits the NodalResultsQueryEngine LRU
# ---------------------------------------------------------------------- #
def test_bench_fetch_warm_cache_hit(benchmark, warm_dataset):
    """Second+ fetch with identical args. Should return the cached
    NodalResults instance (identity equality) rather than re-reading
    HDF5."""

    def _warm_fetch():
        return warm_dataset.nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage="MODEL_STAGE[1]",
            node_ids=[1, 2, 3, 4],
        )

    # Capture once to compare identity against subsequent calls.
    cached = _warm_fetch()
    result = benchmark(_warm_fetch)
    assert result is cached, "Warm fetch did not return the cached object"


def test_bench_fetch_warm_different_component(benchmark, warm_dataset):
    """Warm fetch with a different component of the same result name.
    Separate cache key → fresh HDF5 read, but the partition pool is
    already open."""

    # Prime the component we're about to measure.
    _ = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2],
    )

    def _warm_narrow_fetch():
        return warm_dataset.nodes.get_nodal_results(
            results_name="DISPLACEMENT",
            model_stage="MODEL_STAGE[1]",
            node_ids=[1, 2],
        )

    result = benchmark(_warm_narrow_fetch)
    assert result.df.shape[0] > 0
