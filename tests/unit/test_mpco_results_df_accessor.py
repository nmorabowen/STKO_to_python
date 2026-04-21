"""Phase 4.5 \u2014 ``MPCOResults.df`` accessor.

Per spec \u00a78: collapse ``MPCO_df`` into ``MPCOResults`` as a ``.df``
accessor. The existing ``.create_df`` attribute remains as an alias so
existing call sites (``mpco_results.create_df.drift_df(...)``) keep
working unchanged.
"""
from __future__ import annotations

import pytest

from STKO_to_python.MPCOList.MPCOResults import MPCOResults
from STKO_to_python.MPCOList.MPCOdf import MPCO_df


@pytest.fixture
def empty_results() -> MPCOResults:
    """MPCOResults doesn't need a real dataset to test the accessor shape."""
    return MPCOResults(data={})


def test_df_accessor_returns_mpco_df_instance(empty_results: MPCOResults):
    assert isinstance(empty_results.df, MPCO_df)


def test_df_and_create_df_are_the_same_object(empty_results: MPCOResults):
    """``.df`` and ``.create_df`` must reference the SAME instance so
    callers that used either spelling see identical state (any caches,
    style, etc. held on the MPCO_df object)."""
    assert empty_results.df is empty_results.create_df


def test_df_is_stable_across_calls(empty_results: MPCOResults):
    """The property must not rebuild MPCO_df on each access \u2014 returning
    a fresh instance would break any state the caller held on it."""
    assert empty_results.df is empty_results.df


def test_df_back_reference_points_to_owning_results(empty_results: MPCOResults):
    assert empty_results.df.results is empty_results


def test_df_exposes_mpco_specific_extractors(empty_results: MPCOResults):
    """Smoke check that the methods from spec \u00a78 are reachable via ``.df``."""
    for method in (
        "drift_df", "drift_df_long",
        "pga_df", "pga_df_long", "pga_df_mod", "pga_df_long_mod",
        "torsion_df", "torsion_df_long",
        "base_rocking_df", "base_rocking_df_long",
        "wide_to_long",
    ):
        assert callable(getattr(empty_results.df, method)), (
            f"MPCOResults.df is missing extractor {method!r}"
        )


def test_df_not_in_public_reexports_but_class_still_is():
    """Phase 4.5 adds the accessor without changing the module-level
    public API \u2014 ``MPCO_df`` remains importable from the top-level
    package per the public-API contract (tests/unit/test_public_api.py)."""
    import STKO_to_python as pkg
    assert "MPCO_df" in pkg.__all__
    assert pkg.MPCO_df is MPCO_df
