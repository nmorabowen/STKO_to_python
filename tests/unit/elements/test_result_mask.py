"""Unit tests for ``ResultMask`` and the ``er.where(...)`` chain.

Builds a small synthetic :class:`ElementResults` (3 elements × 5 steps
× 2 components) with known values so each reduction's expected output
can be reasoned about by hand. No HDF5 fixture required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.elements.element_results import ElementResults
from STKO_to_python.elements.result_mask import (
    ResultMask,
    resolve_step_indices,
)


# ---------------------------------------------------------------------- #
# Fixture — synthetic ElementResults                                     #
# ---------------------------------------------------------------------- #

@pytest.fixture
def er() -> ElementResults:
    """3 elements × 5 steps; explicit hand-tuned values."""
    eids = (1, 2, 3)
    steps = list(range(5))
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # Hand-tuned values:
    mz = {
        1: [10.0, 20.0, -30.0, 5.0, 0.0],     # peak=20, trough=-30, |peak|=30, mean=1, last=0
        2: [-50.0, 60.0, -70.0, 80.0, 90.0],  # peak=90, trough=-70, |peak|=90, mean=22, last=90
        3: [1.0, 2.0, 3.0, 4.0, 5.0],         # peak=5, trough=1, |peak|=5, mean=3, last=5
    }
    n1 = {
        1: [100.0, 100.0, 100.0, 100.0, 100.0],
        2: [-50.0, -50.0, -50.0, -50.0, -50.0],
        3: [0.0, 25.0, 50.0, 75.0, 100.0],
    }
    rows: list[dict] = []
    for e in eids:
        for s in steps:
            rows.append(
                {
                    "element_id": e,
                    "step": s,
                    "Mz_ip0": mz[e][s],
                    "N_1": n1[e][s],
                }
            )
    df = (
        pd.DataFrame(rows)
        .set_index(["element_id", "step"])
        .sort_index()
    )
    return ElementResults(
        df=df,
        time=time,
        element_ids=eids,
        element_type="DispBeamColumn3d",
        results_name="globalForces",
        model_stage="MODEL_STAGE[1]",
    )


# ---------------------------------------------------------------------- #
# Time-spec resolver                                                     #
# ---------------------------------------------------------------------- #

class TestTimeResolver:
    time = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    def test_none_returns_all(self):
        out = resolve_step_indices(None, self.time)
        assert out.tolist() == [0, 1, 2, 3, 4]

    def test_int_returns_single(self):
        assert resolve_step_indices(2, self.time).tolist() == [2]

    def test_negative_int_wraps(self):
        assert resolve_step_indices(-1, self.time).tolist() == [4]

    def test_int_out_of_range_raises(self):
        with pytest.raises(IndexError):
            resolve_step_indices(99, self.time)

    def test_float_picks_nearest(self):
        assert resolve_step_indices(2.4, self.time).tolist() == [2]
        assert resolve_step_indices(2.6, self.time).tolist() == [3]

    def test_slice_is_time_range(self):
        # half-open: 1.0 included, 3.0 excluded
        out = resolve_step_indices(slice(1.0, 3.0), self.time)
        assert out.tolist() == [1, 2]

    def test_tuple_is_time_range(self):
        out = resolve_step_indices((1.0, 4.0), self.time)
        assert out.tolist() == [1, 2, 3]

    def test_slice_with_step_raises(self):
        with pytest.raises(ValueError, match="step is not supported"):
            resolve_step_indices(slice(0.0, 4.0, 1.0), self.time)

    def test_list_int_returns_steps(self):
        out = resolve_step_indices([0, 2, 4], self.time)
        assert out.tolist() == [0, 2, 4]

    def test_list_float_returns_nearest(self):
        out = resolve_step_indices([0.4, 2.6], self.time)
        assert out.tolist() == [0, 3]

    def test_ndarray_int(self):
        out = resolve_step_indices(np.array([1, 3]), self.time)
        assert out.tolist() == [1, 3]


# ---------------------------------------------------------------------- #
# Component reductions                                                    #
# ---------------------------------------------------------------------- #

class TestReductions:
    def test_at_step(self, er: ElementResults):
        s = er.where().component("Mz_ip0").at_step(2).values()
        # Mz at step 2: e1=-30, e2=-70, e3=3
        assert s.loc[1] == -30.0
        assert s.loc[2] == -70.0
        assert s.loc[3] == 3.0

    def test_at_time(self, er: ElementResults):
        # time=2.5 → nearest step 2 or 3; 0.5 distance both, argmin picks 2
        s = er.where().component("Mz_ip0").at_time(2.4).values()
        assert s.loc[2] == -70.0  # step 2

    def test_peak_no_window(self, er: ElementResults):
        s = er.where().component("Mz_ip0").peak().values()
        assert s.loc[1] == 20.0
        assert s.loc[2] == 90.0
        assert s.loc[3] == 5.0

    def test_trough_no_window(self, er: ElementResults):
        s = er.where().component("Mz_ip0").trough().values()
        assert s.loc[1] == -30.0
        assert s.loc[2] == -70.0
        assert s.loc[3] == 1.0

    def test_abs_peak_no_window(self, er: ElementResults):
        s = er.where().component("Mz_ip0").abs_peak().values()
        assert s.loc[1] == 30.0
        assert s.loc[2] == 90.0
        assert s.loc[3] == 5.0

    def test_mean_no_window(self, er: ElementResults):
        s = er.where().component("Mz_ip0").mean().values()
        assert pytest.approx(s.loc[1]) == 1.0  # (10+20-30+5+0)/5
        assert pytest.approx(s.loc[2]) == 22.0  # (-50+60-70+80+90)/5
        assert pytest.approx(s.loc[3]) == 3.0

    def test_residual_no_window(self, er: ElementResults):
        s = er.where().component("Mz_ip0").residual().values()
        assert s.loc[1] == 0.0
        assert s.loc[2] == 90.0
        assert s.loc[3] == 5.0

    def test_peak_with_explicit_window(self, er: ElementResults):
        # Window steps 0..2 (time 0..3 half-open) → look at first 3 steps
        s = (
            er.where()
            .component("Mz_ip0")
            .peak(time=(0.0, 3.0))
            .values()
        )
        # e1 first 3: max(10,20,-30) = 20
        # e2 first 3: max(-50,60,-70) = 60
        # e3 first 3: max(1,2,3) = 3
        assert s.loc[1] == 20.0
        assert s.loc[2] == 60.0
        assert s.loc[3] == 3.0

    def test_default_window_picked_up(self, er: ElementResults):
        s = (
            er.where(time=(0.0, 3.0))
            .component("Mz_ip0")
            .peak()
            .values()
        )
        assert s.loc[2] == 60.0  # 90 was at step 4, outside window

    def test_explicit_overrides_default(self, er: ElementResults):
        s = (
            er.where(time=(0.0, 3.0))
            .component("Mz_ip0")
            .peak(time=None)  # explicit override → all steps
            .values()
        )
        assert s.loc[2] == 90.0

    def test_over_threshold_returns_fraction(self, er: ElementResults):
        # Element 2 |Mz| values: 50,60,70,80,90 — fraction > 60: 3/5 = 0.6
        s = (
            er.where()
            .component("Mz_ip0")
            .over_threshold(60.0)
            .values()
        )
        # Mz raw (not |.|), so >60: e2 has steps 3 (80) and 4 (90) → 2/5 = 0.4
        assert pytest.approx(s.loc[2]) == 0.4
        assert s.loc[1] == 0.0  # never above 60

    def test_unknown_component_raises(self, er: ElementResults):
        with pytest.raises(ValueError, match="not in this ElementResults"):
            er.where().component("nonexistent")


# ---------------------------------------------------------------------- #
# Comparators                                                            #
# ---------------------------------------------------------------------- #

class TestComparators:
    def test_gt(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().gt(50.0)
        assert sorted(m.ids().tolist()) == [2]

    def test_lt(self, er: ElementResults):
        m = er.where().component("Mz_ip0").trough().lt(-40.0)
        assert sorted(m.ids().tolist()) == [2]

    def test_ge_le(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().ge(20.0)
        assert sorted(m.ids().tolist()) == [1, 2]
        m = er.where().component("Mz_ip0").peak().le(20.0)
        assert sorted(m.ids().tolist()) == [1, 3]

    def test_between_inclusive(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().between(5.0, 20.0)
        assert sorted(m.ids().tolist()) == [1, 3]

    def test_between_exclusive(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().between(
            5.0, 20.0, inclusive=False
        )
        # peak values: e1=20, e2=90, e3=5; strictly between 5 and 20 → none
        assert sorted(m.ids().tolist()) == []

    def test_outside(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().outside(0.0, 30.0)
        # peaks: 20, 90, 5; outside [0,30] strictly → e2=90
        assert sorted(m.ids().tolist()) == [2]

    def test_eq_exact(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().eq(20.0)
        assert sorted(m.ids().tolist()) == [1]

    def test_near_atol(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().near(21.0, atol=2.0)
        # e1 peak=20 → |20-21|=1 ≤ 2; others not
        assert sorted(m.ids().tolist()) == [1]


# ---------------------------------------------------------------------- #
# Boolean composition                                                    #
# ---------------------------------------------------------------------- #

class TestComposition:
    def test_and(self, er: ElementResults):
        m1 = er.where().component("Mz_ip0").peak().gt(10.0)  # {1,2}
        m2 = er.where().component("Mz_ip0").trough().gt(-40.0)  # {1, 3} (e1=-30, e3=1)
        # Wait: e1 trough = -30 > -40 yes. e3 trough = 1 > -40 yes. e2=-70 no.
        # But m1 = {1, 2}; m2 = {1, 3}; AND = {1}
        m = m1 & m2
        assert sorted(m.ids().tolist()) == [1]

    def test_or(self, er: ElementResults):
        m1 = er.where().component("Mz_ip0").peak().gt(50.0)  # {2}
        m2 = er.where().component("Mz_ip0").trough().lt(-25.0)  # {1, 2}
        m = m1 | m2
        assert sorted(m.ids().tolist()) == [1, 2]

    def test_invert(self, er: ElementResults):
        m1 = er.where().component("Mz_ip0").peak().gt(50.0)  # {2}
        m_not = ~m1
        assert sorted(m_not.ids().tolist()) == [1, 3]

    def test_cross_er_raises(self, er: ElementResults):
        # Build a second er instance with same shape; AND across them must raise.
        df = er.df.copy()
        er2 = ElementResults(
            df=df,
            time=er.time,
            element_ids=er.element_ids,
            element_type=er.element_type,
            results_name=er.results_name,
            model_stage=er.model_stage,
        )
        m1 = er.where().component("Mz_ip0").peak().gt(0.0)
        m2 = er2.where().component("Mz_ip0").peak().gt(0.0)
        with pytest.raises(ValueError, match="different ElementResults"):
            m1 & m2


# ---------------------------------------------------------------------- #
# Predicate escape hatch                                                  #
# ---------------------------------------------------------------------- #

class TestPredicate:
    def test_predicate_per_element(self, er: ElementResults):
        # element_ids are (1,2,3); pick e1 and e3
        m = er.where().predicate(lambda df: np.array([True, False, True]))
        assert sorted(m.ids().tolist()) == [1, 3]

    def test_predicate_full_index_any_reduction(self, er: ElementResults):
        # mask aligned with full (e_id, step) index; "any True step per element"
        df = er.df
        # True wherever Mz_ip0 > 60 — only e2 has such steps
        m = er.where().predicate(lambda d: d["Mz_ip0"] > 60.0)
        assert sorted(m.ids().tolist()) == [2]

    def test_predicate_bad_shape_raises(self, er: ElementResults):
        with pytest.raises(ValueError, match="returned shape"):
            er.where().predicate(lambda df: np.array([True, False]))


# ---------------------------------------------------------------------- #
# Apply / __getitem__                                                     #
# ---------------------------------------------------------------------- #

class TestApply:
    def test_er_getitem_returns_filtered(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().gt(50.0)
        out = er[m]
        assert isinstance(out, ElementResults)
        assert out.element_ids == (2,)
        # df is trimmed to one element × 5 steps
        assert len(out.df) == 5
        # time array preserved
        assert np.array_equal(out.time, er.time)
        # element_type, results_name, model_stage preserved
        assert out.element_type == er.element_type
        assert out.results_name == er.results_name

    def test_apply_empty_mask(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().gt(1e9)
        out = m.apply()
        assert out.element_ids == ()
        assert out.empty

    def test_er_getitem_rejects_non_mask(self, er: ElementResults):
        with pytest.raises(TypeError, match="ResultMask"):
            er[42]

    def test_apply_preserves_column_layout(self, er: ElementResults):
        m = er.where().component("Mz_ip0").peak().gt(0.0)  # all ids
        out = er[m]
        assert list(out.df.columns) == list(er.df.columns)


# ---------------------------------------------------------------------- #
# Canonical resolution                                                    #
# ---------------------------------------------------------------------- #

class TestCanonical:
    def test_canonical_single_column_works(self, er: ElementResults):
        # N_1 maps to a canonical (axial_force) on closed-form beams.
        # If canonical_columns returns one column we can use it.
        cols = er.canonical_columns("axial_force")
        if len(cols) == 1:
            m = er.where().canonical("axial_force").peak().gt(50.0)
            # e1 N_1 always 100 → True; e2 always -50; e3 max=100 → True
            assert sorted(m.ids().tolist()) == [1, 3]
        else:
            pytest.skip(
                "canonical 'axial_force' did not resolve to a single column "
                "in this synthetic fixture"
            )

    def test_canonical_unknown_raises(self, er: ElementResults):
        with pytest.raises(ValueError):
            er.where().canonical("totally_made_up_name_xyz")
