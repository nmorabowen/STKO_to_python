"""Phase 4.3.2 regression tests: each NodalResults engineering method
is a thin forwarder to ``AggregationEngine``. These tests exercise the
forwarder against the single-partition ``elasticFrame`` example and
compare the forwarded result to a direct call on the engine.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet
from STKO_to_python.dataprocess import AggregationEngine


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
@pytest.fixture
def nodal_displacement(elastic_frame_dir: Path):
    """Build a displacement NodalResults over every node of the
    single-partition elasticFrame model, MODEL_STAGE[1]."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    nr = ds.nodes.get_nodal_results(
        results_name="DISPLACEMENT",
        model_stage="MODEL_STAGE[1]",
        node_ids=[1, 2, 3, 4],
    )
    return nr


# ---------------------------------------------------------------------- #
# delta_u
# ---------------------------------------------------------------------- #
def test_delta_u_forwarder_matches_engine_series(nodal_displacement):
    nr = nodal_displacement
    # component 1 (index 1) between nodes 1 and 2
    via_nr = nr.delta_u(top=1, bottom=2, component=1)
    via_eng = nr._aggregation_engine.delta_u(nr, top=1, bottom=2, component=1)

    assert isinstance(via_nr, pd.Series)
    pd.testing.assert_series_equal(via_nr, via_eng)


def test_delta_u_forwarder_abs_max_reduce(nodal_displacement):
    nr = nodal_displacement
    v = nr.delta_u(top=1, bottom=2, component=1, reduce="abs_max")
    assert isinstance(v, float)
    # Must equal nanmax(abs(series))
    s = nr.delta_u(top=1, bottom=2, component=1, reduce="series")
    assert v == pytest.approx(float(np.nanmax(np.abs(s.to_numpy(dtype=float)))))


def test_delta_u_signed_false_equals_abs_of_signed(nodal_displacement):
    nr = nodal_displacement
    signed = nr.delta_u(top=1, bottom=2, component=1, signed=True)
    unsigned = nr.delta_u(top=1, bottom=2, component=1, signed=False)
    pd.testing.assert_series_equal(unsigned, signed.abs().rename(unsigned.name))


def test_delta_u_unknown_reduce_raises(nodal_displacement):
    nr = nodal_displacement
    with pytest.raises(ValueError, match="Unknown reduce"):
        nr.delta_u(top=1, bottom=2, component=1, reduce="nope")


# ---------------------------------------------------------------------- #
# Engine sanity
# ---------------------------------------------------------------------- #
def test_class_level_engine_is_shared_singleton(nodal_displacement):
    nr = nodal_displacement
    # Same engine instance across all NodalResults objects — it is stateless
    # and lives as a class attribute.
    assert isinstance(nr._aggregation_engine, AggregationEngine)
    from STKO_to_python.results.nodal_results_dataclass import NodalResults
    assert nr._aggregation_engine is NodalResults._aggregation_engine
