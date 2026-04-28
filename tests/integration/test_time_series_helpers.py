"""End-to-end tests for the per-element time-series helpers
(``peak_abs``, ``time_of_peak``, ``cumulative_envelope``, ``summary``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from STKO_to_python import MPCODataSet


# ---------------------------------------------------------------------- #
# peak_abs                                                                #
# ---------------------------------------------------------------------- #


def test_peak_abs_returns_per_element_max_abs(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    pa = er.peak_abs()
    # Every column gets a *_peak_abs.
    assert all(c.endswith("_peak_abs") for c in pa.columns)
    assert sorted(pa.index.tolist()) == [1, 2, 3]
    # peak_abs >= max(|envelope|).
    env = er.envelope()
    for col in er.df.columns:
        peak_col = f"{col}_peak_abs"
        env_max = env[[f"{col}_min", f"{col}_max"]].abs().max(axis=1)
        np.testing.assert_array_less(
            -1e-9 + env_max.values, pa[peak_col].values + 1e-9
        )


def test_peak_abs_single_component_subset(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    pa = er.peak_abs(component="N_1")
    assert list(pa.columns) == ["N_1_peak_abs"]


# ---------------------------------------------------------------------- #
# time_of_peak                                                            #
# ---------------------------------------------------------------------- #


def test_time_of_peak_indexes_step_axis(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    s = er.time_of_peak("N_1")
    assert sorted(s.index.tolist()) == [1, 2, 3]
    # Indices must be valid steps.
    assert s.between(0, er.n_steps - 1).all()
    # Cross-check against argmax of |N_1| per element.
    expected = er.df["N_1"].abs().groupby("element_id").idxmax().apply(lambda t: t[1])
    pd.testing.assert_series_equal(
        s.sort_index().rename(None), expected.sort_index().rename(None),
        check_dtype=False,
    )


def test_time_of_peak_signed_vs_abs_can_differ(elastic_frame_dir: Path):
    """Most fixtures will agree; this test just locks in that the
    code-path executes for both modes without error."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    s_abs = er.time_of_peak("N_1", abs=True)
    s_signed = er.time_of_peak("N_1", abs=False)
    assert s_abs.notna().all()
    assert s_signed.notna().all()


def test_time_of_peak_unknown_component_raises(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    with pytest.raises(ValueError, match="not in this result"):
        er.time_of_peak("nonexistent")


# ---------------------------------------------------------------------- #
# cumulative_envelope                                                     #
# ---------------------------------------------------------------------- #


def test_cumulative_envelope_has_running_columns(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    ce = er.cumulative_envelope("N_1")
    assert "N_1_running_max" in ce.columns
    assert "N_1_running_min" in ce.columns
    # Running max is monotonically non-decreasing per element.
    arr = ce["N_1_running_max"].xs(1, level="element_id").to_numpy()
    assert np.all(np.diff(arr) >= -1e-12)


def test_cumulative_envelope_last_step_equals_envelope(elastic_frame_dir: Path):
    """At the final step, running_min/max equals the regular envelope."""
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    ce = er.cumulative_envelope()
    env = er.envelope()
    last_step = er.n_steps - 1
    last = ce.xs(last_step, level="step").sort_index()
    env_sorted = env.sort_index()

    for col in er.df.columns:
        np.testing.assert_allclose(
            last[f"{col}_running_max"].values,
            env_sorted[f"{col}_max"].values,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            last[f"{col}_running_min"].values,
            env_sorted[f"{col}_min"].values,
            rtol=1e-12,
        )


# ---------------------------------------------------------------------- #
# summary                                                                 #
# ---------------------------------------------------------------------- #


def test_summary_has_expected_columns(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1, 2, 3],
        model_stage="MODEL_STAGE[1]",
    )
    s = er.summary()
    assert sorted(s.index.tolist()) == [1, 2, 3]
    suffixes = {"_max", "_min", "_peak_abs", "_residual", "_mean"}
    for col in er.df.columns:
        for suf in suffixes:
            assert f"{col}{suf}" in s.columns
    # Per-element peak_abs >= |max| and >= |min|.
    for col in er.df.columns:
        assert (s[f"{col}_peak_abs"] + 1e-12 >= s[f"{col}_max"].abs()).all()
        assert (s[f"{col}_peak_abs"] + 1e-12 >= s[f"{col}_min"].abs()).all()


def test_summary_residual_is_last_step_value(elastic_frame_dir: Path):
    ds = MPCODataSet(str(elastic_frame_dir), "results", verbose=False)
    er = ds.elements.get_element_results(
        results_name="localForce",
        element_type="5-ElasticBeam3d",
        element_ids=[1],
        model_stage="MODEL_STAGE[1]",
    )
    s = er.summary()
    last_step_data = er.df.xs(er.n_steps - 1, level="step")
    for col in er.df.columns:
        assert s.loc[1, f"{col}_residual"] == pytest.approx(
            last_step_data.loc[1, col]
        )


# ---------------------------------------------------------------------- #
# Empty result sanity                                                     #
# ---------------------------------------------------------------------- #


def test_helpers_on_empty_result_return_empty_frames():
    from STKO_to_python.elements.element_results import ElementResults

    er = ElementResults(df=pd.DataFrame(), time=np.array([]))
    assert er.peak_abs().empty
    assert er.cumulative_envelope().empty
    assert er.summary().empty
