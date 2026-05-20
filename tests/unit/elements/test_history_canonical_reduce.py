"""Unit tests for the private ``_reduce_fiber_ip`` / ``_filter_canonical_columns``
helpers that back :meth:`ElementResultsPlotter.history_canonical`.

Pure-pandas — no HDF5 fixture, no plot rendering. Each reduction is
checked against hand-computed expected values so a regression in the
column-filter regex or the reduction dispatch fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from STKO_to_python.elements.element_results_plotting import (
    _filter_canonical_columns,
    _reduce_fiber_ip,
)


# ---------------------------------------------------------------------- #
# Synthetic frames
# ---------------------------------------------------------------------- #

def _make_fiber_df() -> pd.DataFrame:
    """2 elements × 3 steps; 2 fibers × 2 IPs (compressed-fiber pattern)."""
    idx = pd.MultiIndex.from_product(
        [[1, 2], [0, 1, 2]], names=["element_id", "step"]
    )
    cols = [f"sigma_f{f}_ip{ip}" for f in (0, 1) for ip in (0, 1)]
    return pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4), index=idx, columns=cols
    )


def _make_layered_fiber_df() -> pd.DataFrame:
    """2 elements × 3 steps; 2 fibers × 2 layers × 2 IPs (layered+fiber)."""
    idx = pd.MultiIndex.from_product(
        [[1, 2], [0, 1, 2]], names=["element_id", "step"]
    )
    cols = [
        f"sigma_f{f}_l{l}_ip{ip}"
        for f in (0, 1) for l in (0, 1) for ip in (0, 1)
    ]
    return pd.DataFrame(
        np.arange(48, dtype=float).reshape(6, 8), index=idx, columns=cols
    )


def _make_no_fiber_layered_df() -> pd.DataFrame:
    """Layered shell, no fibers — 2 elements × 3 steps; 2 layers × 2 IPs."""
    idx = pd.MultiIndex.from_product(
        [[1, 2], [0, 1, 2]], names=["element_id", "step"]
    )
    cols = [f"d+_l{l}_ip{ip}" for l in (0, 1) for ip in (0, 1)]
    return pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4), index=idx, columns=cols
    )


# ---------------------------------------------------------------------- #
# _filter_canonical_columns
# ---------------------------------------------------------------------- #

class TestFilterCanonicalColumns:
    def test_no_anchors_keeps_everything(self):
        df = _make_fiber_df()
        assert _filter_canonical_columns(df.columns) == list(df.columns)

    def test_ip_anchor(self):
        df = _make_fiber_df()
        kept = _filter_canonical_columns(df.columns, ip_idx=0)
        assert kept == ["sigma_f0_ip0", "sigma_f1_ip0"]

    def test_fiber_anchor(self):
        df = _make_fiber_df()
        kept = _filter_canonical_columns(df.columns, fiber_idx=1)
        assert kept == ["sigma_f1_ip0", "sigma_f1_ip1"]

    def test_layered_with_layer_anchor(self):
        df = _make_layered_fiber_df()
        kept = _filter_canonical_columns(df.columns, layer_idx=1, ip_idx=0)
        # fibers 0 and 1, layer 1, ip 0
        assert kept == ["sigma_f0_l1_ip0", "sigma_f1_l1_ip0"]

    def test_anchor_that_does_not_exist_wipes(self):
        """A layer_idx on a compressed-fiber bucket (no layers) drops all."""
        df = _make_fiber_df()
        assert _filter_canonical_columns(df.columns, layer_idx=0) == []

    def test_fiber_anchor_disambiguates_substrings(self):
        """``_f1`` must not match ``_f11``. Regex anchors the next char."""
        df = pd.DataFrame(
            np.zeros((1, 3)),
            columns=["c_f1_ip0", "c_f10_ip0", "c_f11_ip0"],
        )
        assert _filter_canonical_columns(df.columns, fiber_idx=1) == ["c_f1_ip0"]


# ---------------------------------------------------------------------- #
# _reduce_fiber_ip — correctness
# ---------------------------------------------------------------------- #

class TestReduceCorrectness:
    def test_over_all_mean(self):
        df = _make_fiber_df()
        s, used = _reduce_fiber_ip(df, over="all", reduce="mean")
        # Row 0 = [0, 1, 2, 3] -> mean 1.5; rows step by 4.
        assert list(s) == [1.5, 5.5, 9.5, 13.5, 17.5, 21.5]
        assert len(used) == 4

    def test_over_fibers_at_ip0_max(self):
        df = _make_fiber_df()
        s, used = _reduce_fiber_ip(
            df, over="fibers", ip_idx=0, reduce="max"
        )
        # cols sigma_f0_ip0 and sigma_f1_ip0; row 0 = [0, 2] -> max 2
        assert list(s) == [2.0, 6.0, 10.0, 14.0, 18.0, 22.0]
        assert used == ["sigma_f0_ip0", "sigma_f1_ip0"]

    def test_over_ips_for_fiber1_mean(self):
        df = _make_fiber_df()
        s, used = _reduce_fiber_ip(
            df, over="ips", fiber_idx=1, reduce="mean"
        )
        # cols sigma_f1_ip0 and sigma_f1_ip1; row 0 = [2, 3] -> mean 2.5
        assert list(s) == [2.5, 6.5, 10.5, 14.5, 18.5, 22.5]
        assert used == ["sigma_f1_ip0", "sigma_f1_ip1"]

    def test_layered_over_layers_at_fiber0_ip0(self):
        df = _make_layered_fiber_df()
        s, used = _reduce_fiber_ip(
            df, over="layers", fiber_idx=0, ip_idx=0, reduce="mean"
        )
        assert used == ["sigma_f0_l0_ip0", "sigma_f0_l1_ip0"]
        # row 0 columns: sigma_f0_l0_ip0=0, sigma_f0_l1_ip0=2 -> mean 1
        assert list(s) == [1.0, 9.0, 17.0, 25.0, 33.0, 41.0]

    def test_extra_pre_filter_on_layered(self):
        """``over="fibers"`` at IP 0 with ``layer_idx=1`` selects only
        fibers within layer 1 at IP 0 — orthogonal pre-filter."""
        df = _make_layered_fiber_df()
        s, used = _reduce_fiber_ip(
            df, over="fibers", ip_idx=0, layer_idx=1, reduce="mean"
        )
        assert used == ["sigma_f0_l1_ip0", "sigma_f1_l1_ip0"]

    def test_layered_no_fibers_over_all(self):
        df = _make_no_fiber_layered_df()
        s, used = _reduce_fiber_ip(df, over="all", reduce="sum")
        assert len(used) == 4
        # row 0 = [0, 1, 2, 3] -> sum 6
        assert list(s) == [6.0, 22.0, 38.0, 54.0, 70.0, 86.0]

    @pytest.mark.parametrize(
        "op,expected_row0",
        [
            ("mean",    1.5),
            ("median",  1.5),
            ("max",     3.0),
            ("min",     0.0),
            ("sum",     6.0),
        ],
    )
    def test_all_reduce_ops_on_row0(self, op, expected_row0):
        df = _make_fiber_df()
        s, _ = _reduce_fiber_ip(df, over="all", reduce=op)
        assert s.iloc[0] == expected_row0

    def test_abs_max_returns_magnitude(self):
        """abs_max must return the *magnitude* (positive), not the signed
        peak — distinguishes from plain max when negatives dominate."""
        df = _make_fiber_df()
        df.iloc[0] = [-5.0, 2.0, -3.0, 1.0]   # |max| = 5, signed max = 2
        s, _ = _reduce_fiber_ip(df, over="all", reduce="abs_max")
        assert s.iloc[0] == 5.0


# ---------------------------------------------------------------------- #
# _reduce_fiber_ip — validation
# ---------------------------------------------------------------------- #

class TestReduceValidation:
    def test_over_fibers_requires_ip_idx(self):
        df = _make_fiber_df()
        with pytest.raises(ValueError, match="requires.*ip_idx"):
            _reduce_fiber_ip(df, over="fibers", reduce="mean")

    def test_over_ips_requires_fiber_idx(self):
        df = _make_fiber_df()
        with pytest.raises(ValueError, match="requires.*fiber_idx"):
            _reduce_fiber_ip(df, over="ips", reduce="mean")

    def test_over_layers_requires_both(self):
        df = _make_layered_fiber_df()
        with pytest.raises(ValueError, match="requires.*ip_idx"):
            _reduce_fiber_ip(
                df, over="layers", fiber_idx=0, reduce="mean"
            )

    def test_unknown_over_lists_valid(self):
        df = _make_fiber_df()
        with pytest.raises(ValueError, match="over=.*not in"):
            _reduce_fiber_ip(df, over="bogus", reduce="mean")

    def test_unknown_reduce_lists_valid(self):
        df = _make_fiber_df()
        with pytest.raises(ValueError, match="reduce=.*not in"):
            _reduce_fiber_ip(df, over="all", reduce="bogus")

    def test_filter_wipes_everything_lists_present_tags(self):
        df = _make_fiber_df()
        with pytest.raises(ValueError) as exc:
            _reduce_fiber_ip(df, over="all", reduce="mean", layer_idx=99)
        msg = str(exc.value)
        assert "layer_idx" in msg
        # Tail-tags listing should include all axes that actually exist.
        assert "_f0" in msg and "_f1" in msg
        assert "_ip0" in msg and "_ip1" in msg
