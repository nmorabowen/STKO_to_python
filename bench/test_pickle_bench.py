"""Phase 5.1.c — NodalResults pickle roundtrip benchmarks.

Spec §6 targets a 500 MB NodalResults load at ~4 s. The elasticFrame
fixture produces a tiny pickle (<10 KB), so these numbers are only
useful as regression guards for the Phase 4.3.3 tolerant
``__setstate__`` + view-rebuild path, not for validating the spec
target. Larger-fixture benches land when that fixture is checked in.

Run with::

    pip install -e ".[bench]"
    pytest bench/test_pickle_bench.py -q
"""
from __future__ import annotations

import pickle

import pytest

from STKO_to_python.results.nodal_results_dataclass import NodalResults


pytestmark = pytest.mark.bench


# ---------------------------------------------------------------------- #
# Round-trip: dumps + loads
# ---------------------------------------------------------------------- #
def test_bench_pickle_roundtrip(benchmark, warm_dataset):
    """Dumps + loads a real NodalResults. Touches __getstate__,
    __setstate__, and the view-rebuild pass."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )

    def _roundtrip():
        blob = pickle.dumps(nr)
        return pickle.loads(blob)

    loaded = benchmark(_roundtrip)
    assert isinstance(loaded, NodalResults)
    assert loaded.df.shape == nr.df.shape


# ---------------------------------------------------------------------- #
# loads only — measures __setstate__ + view rebuild in isolation
# ---------------------------------------------------------------------- #
def test_bench_pickle_loads_only(benchmark, warm_dataset):
    """Pre-serialise once; the benchmark measures only the load path.
    Useful for isolating the tolerant __setstate__ + _build_views cost
    from the dumps cost."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    blob = pickle.dumps(nr)

    def _loads():
        return pickle.loads(blob)

    loaded = benchmark(_loads)
    assert isinstance(loaded, NodalResults)


# ---------------------------------------------------------------------- #
# Tolerant-__setstate__ with extra keys
# ---------------------------------------------------------------------- #
def test_bench_setstate_with_unknown_keys(benchmark, warm_dataset):
    """Simulates an old-layout pickle that carried fields the current
    class no longer stores. Verifies the Phase 4.3.3 tolerant
    __setstate__ doesn't pay a visible cost for dropping unknown keys."""
    nr = warm_dataset.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    state = nr.__getstate__()
    # Inject extra keys that __setstate__ must drop.
    state["drift_profile_cache"] = {"x": 1}
    state["residual_cache"] = [1, 2, 3]
    state["legacy_flag"] = True

    def _setstate():
        target = NodalResults.__new__(NodalResults)
        target.__setstate__(state)
        return target

    loaded = benchmark(_setstate)
    assert "drift_profile_cache" not in loaded.__dict__
    assert "residual_cache" not in loaded.__dict__
    assert "legacy_flag" not in loaded.__dict__
