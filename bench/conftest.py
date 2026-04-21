"""Shared fixtures for the benchmark suite.

The benchmark suite lives outside ``tests/`` so that the default
``pytest tests/`` command ignores it (``testpaths = ["tests"]`` in
``pyproject.toml``). Run the benchmarks explicitly with::

    pip install -e ".[bench]"
    pytest bench/ -q

Fixtures here are session-scoped where sensible: the elasticFrame
``MPCODataSet`` construction cost is amortized across every bench that
re-fetches from it, so we don't pay it per iteration.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Fail loudly at collection if pytest-benchmark isn't installed — the
# tests in this tree use the ``benchmark`` fixture unconditionally.
pytest_benchmark = pytest.importorskip(
    "pytest_benchmark",
    reason="pytest-benchmark is not installed; install with 'pip install -e \".[bench]\"'.",
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "stko_results_examples"
ELASTIC_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "results"
QUAD_FRAME_DIR = EXAMPLES_DIR / "elasticFrame" / "QuadFrame_results"


@pytest.fixture(scope="session")
def elastic_frame_dir() -> Path:
    """Small single-partition fixture (4 nodes, 10 steps). Every bench
    uses this unless a larger fixture is explicitly added. Skips the
    collection if the folder is absent."""
    if not (ELASTIC_FRAME_DIR / "results.mpco").exists():
        pytest.skip(f"elasticFrame example not available at {ELASTIC_FRAME_DIR}")
    return ELASTIC_FRAME_DIR


@pytest.fixture(scope="session")
def quad_frame_dir() -> Path:
    """Small multi-partition fixture. Reserved for benches that want
    to exercise the partition-pool code path."""
    if not (QUAD_FRAME_DIR / "results.part-0.mpco").exists():
        pytest.skip(f"QuadFrame MP example not available at {QUAD_FRAME_DIR}")
    return QUAD_FRAME_DIR


@pytest.fixture(scope="session")
def warm_dataset(elastic_frame_dir: Path):
    """One MPCODataSet per session — primed with a first fetch so the
    LRU cache is warm for downstream benches."""
    from STKO_to_python import MPCODataSet

    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    # Prime the cache — benches that measure warm fetches can rely on
    # this one already being cached.
    _ = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    return ds
