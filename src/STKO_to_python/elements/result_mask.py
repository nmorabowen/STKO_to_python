"""ResultMask — threshold / time-window filters on :class:`ElementResults`.

The selector layer (``ElementSelector``) decides *which elements* before
any HDF5 read; this layer decides *which of the fetched elements
satisfy a value condition*. Output is a per-element boolean mask that
can be combined with ``&`` / ``|`` / ``~`` and applied to the parent
``ElementResults`` via ``er[mask]`` to get a fresh, trimmed result.

The chain shape is::

    er.where(time=...)                # default time window (optional)
      .component("Mz_ip0")            # or .canonical("axial_force")
      .abs_peak(time=...)             # reduction over time → scalar/elem
      .gt(50)                         # comparator → ResultMask

Each step returns a new immutable object — chains are safe to share.

Reductions
----------
``at_step(s)``         scalar at one step (s as ``int`` step index)
``at_time(t)``         scalar at the step nearest to time ``t`` (``float``)
``peak(time=...)``     signed max over the window
``trough(time=...)``   signed min over the window
``abs_peak(time=...)`` max of \\|.\\| over the window
``mean(time=...)``     arithmetic mean over the window
``residual(time=...)`` last step in the window
``over_threshold(v, frac=..., time=...)``
                       fraction of steps where the value is above ``v``;
                       ``frac`` can be omitted (returns the fraction
                       itself, ready for ``.gt(...)``).

Comparators
-----------
``gt(v)``, ``lt(v)``, ``ge(v)``, ``le(v)``, ``between(lo, hi)``,
``outside(lo, hi)``, ``eq(v, atol=0.0)``, ``near(v, atol)``.

Time grammar (``time=`` argument)
---------------------------------
``None``                all steps in the parent ``time`` array
``int``                 single step index
``float``               step nearest to that time value
``slice(t0, t1)``       half-open *time* range (``t0`` ≤ time < ``t1``)
``(t0, t1)`` tuple       same as ``slice(t0, t1)``
``list[int]``            explicit step indices
``list[float]``          nearest step for each time value
``np.ndarray[int]``      explicit step indices (must be 1-D)

The default window from ``er.where(time=...)`` is overridden by any
explicit ``time=`` on a reduction.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .element_results import ElementResults


TimeSpec = Union[None, int, float, slice, Tuple[float, float], Sequence[int], Sequence[float], np.ndarray]


# Sentinel for "no override; use the chain's default time window".
_DEFAULT: Any = object()


# ---------------------------------------------------------------------- #
# Time resolver                                                           #
# ---------------------------------------------------------------------- #

def resolve_step_indices(spec: TimeSpec, time_arr: np.ndarray) -> np.ndarray:
    """Map a time-spec to an int64 array of step indices.

    The spec semantics are documented in the module docstring.
    Returns a sorted, deduplicated ``np.ndarray[int64]``.
    """
    if spec is None:
        return np.arange(time_arr.size, dtype=np.int64)

    if isinstance(spec, (int, np.integer)):
        s = int(spec)
        if s < 0:
            s += time_arr.size
        if s < 0 or s >= time_arr.size:
            raise IndexError(
                f"time step index {spec} out of range [0, {time_arr.size})."
            )
        return np.asarray([s], dtype=np.int64)

    if isinstance(spec, float):
        if time_arr.size == 0:
            raise ValueError("Cannot resolve time= float against empty time array.")
        return np.asarray(
            [int(np.argmin(np.abs(time_arr - spec)))], dtype=np.int64
        )

    if isinstance(spec, slice):
        t0, t1 = spec.start, spec.stop
        if spec.step is not None:
            raise ValueError("time= slice with step is not supported.")
        return _time_range(time_arr, t0, t1)

    if isinstance(spec, tuple) and len(spec) == 2:
        return _time_range(time_arr, spec[0], spec[1])

    if isinstance(spec, np.ndarray):
        if spec.dtype.kind in ("i", "u"):
            return np.unique(spec.astype(np.int64))
        if spec.dtype.kind == "f":
            return _nearest_steps(time_arr, spec)
        raise TypeError(
            f"time=ndarray must be int or float dtype, got {spec.dtype!r}."
        )

    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return np.empty(0, dtype=np.int64)
        first = spec[0]
        if isinstance(first, (int, np.integer)):
            return np.unique(np.asarray(spec, dtype=np.int64))
        if isinstance(first, float):
            return _nearest_steps(time_arr, np.asarray(spec, dtype=np.float64))
        raise TypeError(
            f"time= list/tuple element type {type(first).__name__} not supported."
        )

    raise TypeError(f"Unsupported time= spec type: {type(spec).__name__}.")


def _time_range(
    time_arr: np.ndarray,
    t0: Optional[float],
    t1: Optional[float],
) -> np.ndarray:
    if time_arr.size == 0:
        return np.empty(0, dtype=np.int64)
    lo = -np.inf if t0 is None else float(t0)
    hi = np.inf if t1 is None else float(t1)
    mask = (time_arr >= lo) & (time_arr < hi)
    return np.flatnonzero(mask).astype(np.int64)


def _nearest_steps(time_arr: np.ndarray, targets: np.ndarray) -> np.ndarray:
    if time_arr.size == 0:
        raise ValueError("Cannot resolve nearest time against empty time array.")
    out = np.empty(targets.size, dtype=np.int64)
    for i, t in enumerate(targets):
        out[i] = int(np.argmin(np.abs(time_arr - t)))
    return np.unique(out)


# ---------------------------------------------------------------------- #
# ResultMask — boolean mask with composition                              #
# ---------------------------------------------------------------------- #

class ResultMask:
    """Per-element boolean mask, composable via ``& / | / ~``.

    Created by the comparator step of a ``er.where().<...>`` chain or
    by the ``predicate(fn)`` escape hatch. Apply via ``er[mask]`` (a
    fresh :class:`ElementResults`) or extract the matching ids via
    ``mask.ids()``.
    """

    __slots__ = ("_er", "_series")

    def __init__(self, er: "ElementResults", series: pd.Series) -> None:
        if not isinstance(series, pd.Series):
            raise TypeError("ResultMask: series must be a pandas Series.")
        # Coerce to bool, fill NaNs with False (NaN reductions can't
        # match thresholds — defensive default).
        s = series.astype("boolean").fillna(False).astype(bool)
        # Reindex to the canonical element_id order from the parent
        # results so that & / | over masks built from different
        # reductions align correctly.
        canonical_idx = pd.Index(np.asarray(er.element_ids, dtype=np.int64))
        s = s.reindex(canonical_idx, fill_value=False)
        s.name = "mask"
        s.index.name = "element_id"
        self._er = er
        self._series = s

    # ------------------------------------------------------------------ #
    # Outputs                                                            #
    # ------------------------------------------------------------------ #
    def mask(self) -> pd.Series:
        """Boolean Series indexed by ``element_id``."""
        return self._series.copy()

    def ids(self) -> np.ndarray:
        """``int64`` array of element IDs where the mask is ``True``."""
        return self._series.index[self._series].to_numpy(dtype=np.int64)

    def count(self) -> int:
        """Number of elements where the mask is ``True``."""
        return int(self._series.sum())

    def apply(self) -> "ElementResults":
        """Return a fresh :class:`ElementResults` trimmed to matched ids."""
        return _subset_er(self._er, self.ids())

    # ------------------------------------------------------------------ #
    # Boolean composition                                                 #
    # ------------------------------------------------------------------ #
    def __and__(self, other: "ResultMask") -> "ResultMask":
        if not isinstance(other, ResultMask):
            return NotImplemented  # type: ignore[return-value]
        if self._er is not other._er:
            raise ValueError(
                "Cannot AND masks from different ElementResults instances."
            )
        return ResultMask(self._er, self._series & other._series)

    def __or__(self, other: "ResultMask") -> "ResultMask":
        if not isinstance(other, ResultMask):
            return NotImplemented  # type: ignore[return-value]
        if self._er is not other._er:
            raise ValueError(
                "Cannot OR masks from different ElementResults instances."
            )
        return ResultMask(self._er, self._series | other._series)

    def __invert__(self) -> "ResultMask":
        return ResultMask(self._er, ~self._series)

    def __repr__(self) -> str:
        return (
            f"ResultMask(n_true={self.count()}, "
            f"n_total={len(self._series)})"
        )

    def __len__(self) -> int:
        return int(self.count())


# ---------------------------------------------------------------------- #
# _ScalarPerElement — reduced values, exposes comparators                 #
# ---------------------------------------------------------------------- #

class _ScalarPerElement:
    """One float per element, ready for a comparator step.

    Construction is internal — produced by :class:`_ComponentQuery`
    reductions. The series is indexed by ``element_id`` and aligned
    with the parent ``ElementResults.element_ids``.
    """

    __slots__ = ("_er", "_values")

    def __init__(self, er: "ElementResults", values: pd.Series) -> None:
        self._er = er
        # Align on element_ids; missing rows fill NaN so comparators
        # produce False (defensive).
        canonical_idx = pd.Index(np.asarray(er.element_ids, dtype=np.int64))
        self._values = values.reindex(canonical_idx)
        self._values.index.name = "element_id"

    # -- introspection --------------------------------------------------
    def values(self) -> pd.Series:
        """The underlying scalar Series (indexed by ``element_id``)."""
        return self._values.copy()

    # -- comparators ----------------------------------------------------
    def gt(self, value: float) -> ResultMask:
        return ResultMask(self._er, self._values > value)

    def lt(self, value: float) -> ResultMask:
        return ResultMask(self._er, self._values < value)

    def ge(self, value: float) -> ResultMask:
        return ResultMask(self._er, self._values >= value)

    def le(self, value: float) -> ResultMask:
        return ResultMask(self._er, self._values <= value)

    def between(self, lo: float, hi: float, *, inclusive: bool = True) -> ResultMask:
        if inclusive:
            s = (self._values >= lo) & (self._values <= hi)
        else:
            s = (self._values > lo) & (self._values < hi)
        return ResultMask(self._er, s)

    def outside(self, lo: float, hi: float, *, inclusive: bool = False) -> ResultMask:
        if inclusive:
            s = (self._values <= lo) | (self._values >= hi)
        else:
            s = (self._values < lo) | (self._values > hi)
        return ResultMask(self._er, s)

    def eq(self, value: float, *, atol: float = 0.0) -> ResultMask:
        if atol == 0.0:
            return ResultMask(self._er, self._values == value)
        return self.near(value, atol=atol)

    def near(self, value: float, *, atol: float) -> ResultMask:
        return ResultMask(self._er, (self._values - value).abs() <= atol)

    def __repr__(self) -> str:
        return f"_ScalarPerElement(n={len(self._values)})"


# ---------------------------------------------------------------------- #
# _ComponentQuery — selects column(s), exposes reductions                 #
# ---------------------------------------------------------------------- #

class _ComponentQuery:
    """One column of an :class:`ElementResults`, with reductions over
    the time axis. Produced by ``er.where(...).component(name)``.
    """

    __slots__ = ("_er", "_column", "_default_time")

    def __init__(
        self,
        er: "ElementResults",
        column: str,
        default_time: TimeSpec,
    ) -> None:
        self._er = er
        if column not in er.df.columns:
            raise ValueError(
                f"component {column!r} not in this ElementResults. "
                f"Available: {list(er.df.columns)}"
            )
        self._column = column
        self._default_time = default_time

    # ------------------------------------------------------------------ #
    # Reductions                                                          #
    # ------------------------------------------------------------------ #
    def at_step(self, step: int) -> _ScalarPerElement:
        s = self._slice_steps([int(step)])
        return _ScalarPerElement(self._er, s.droplevel("step"))

    def at_time(self, t: float) -> _ScalarPerElement:
        idx = resolve_step_indices(float(t), self._time_arr())
        if idx.size == 0:
            raise ValueError(f"at_time({t}): no step found.")
        s = self._slice_steps([int(idx[0])])
        return _ScalarPerElement(self._er, s.droplevel("step"))

    def peak(self, *, time: Any = _DEFAULT) -> _ScalarPerElement:
        return self._reduce_over_time(time, "max")

    def trough(self, *, time: Any = _DEFAULT) -> _ScalarPerElement:
        return self._reduce_over_time(time, "min")

    def abs_peak(self, *, time: Any = _DEFAULT) -> _ScalarPerElement:
        return self._reduce_over_time(time, "abs_max")

    def mean(self, *, time: Any = _DEFAULT) -> _ScalarPerElement:
        return self._reduce_over_time(time, "mean")

    def residual(self, *, time: Any = _DEFAULT) -> _ScalarPerElement:
        return self._reduce_over_time(time, "last")

    def over_threshold(
        self,
        value: float,
        *,
        time: Any = _DEFAULT,
    ) -> _ScalarPerElement:
        """Fraction of steps in the window where value > ``threshold``.

        Chain a comparator (e.g. ``.gt(0.1)``) to mask elements that
        spend at least 10% of the window above the threshold.
        """
        eff = self._effective_time(time)
        steps = resolve_step_indices(eff, self._time_arr())
        if steps.size == 0:
            raise ValueError("over_threshold: empty step window.")
        df = self._sliced_frame(steps)
        ser = (df > value).groupby(level="element_id").mean()[self._column]
        return _ScalarPerElement(self._er, ser)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _time_arr(self) -> np.ndarray:
        t = self._er.time
        if not isinstance(t, np.ndarray):
            t = np.asarray(t, dtype=np.float64)
        return t

    def _effective_time(self, time: Any) -> TimeSpec:
        return self._default_time if time is _DEFAULT else time

    def _slice_steps(self, steps: Sequence[int]) -> pd.Series:
        df = self._er.df
        lvl = df.index.get_level_values("step")
        sub = df.loc[lvl.isin(np.asarray(steps, dtype=np.int64))][self._column]
        return sub

    def _sliced_frame(self, steps: np.ndarray) -> pd.DataFrame:
        df = self._er.df
        lvl = df.index.get_level_values("step")
        return df.loc[lvl.isin(steps)][[self._column]]

    def _reduce_over_time(self, time: TimeSpec, op: str) -> _ScalarPerElement:
        eff = self._effective_time(time)
        steps = resolve_step_indices(eff, self._time_arr())
        if steps.size == 0:
            raise ValueError(f"{op}: empty step window.")
        df = self._sliced_frame(steps)[self._column]
        if op == "max":
            ser = df.groupby(level="element_id").max()
        elif op == "min":
            ser = df.groupby(level="element_id").min()
        elif op == "abs_max":
            ser = df.abs().groupby(level="element_id").max()
        elif op == "mean":
            ser = df.groupby(level="element_id").mean()
        elif op == "last":
            # Pick the largest selected step per element.
            sorted_df = df.reset_index().sort_values("step").set_index(
                ["element_id", "step"]
            )[self._column]
            ser = sorted_df.groupby(level="element_id").last()
        else:
            raise ValueError(f"unknown reduction {op!r}")
        return _ScalarPerElement(self._er, ser)

    def __repr__(self) -> str:
        return f"_ComponentQuery({self._column!r})"


# ---------------------------------------------------------------------- #
# _ResultQuery — er.where() entry point                                   #
# ---------------------------------------------------------------------- #

class _ResultQuery:
    """Entry point produced by :meth:`ElementResults.where`.

    Carries the default time window for the chain; delegates to
    :class:`_ComponentQuery` for column-level work and to
    :func:`_predicate_mask` for the escape hatch.
    """

    __slots__ = ("_er", "_default_time")

    def __init__(self, er: "ElementResults", default_time: TimeSpec) -> None:
        self._er = er
        self._default_time = default_time

    def component(self, name: str) -> _ComponentQuery:
        """Pick one column by exact name (e.g. ``"Mz_ip2"``, ``"Px_1"``)."""
        return _ComponentQuery(self._er, name, self._default_time)

    def canonical(self, name: str) -> _ComponentQuery:
        """Pick a column via canonical engineering name.

        Restricted to canonicals that resolve to exactly one column. For
        multi-IP buckets the user must pick a specific column with
        :meth:`component` (Phase 4 will add a multi-column reduction).
        """
        cols = self._er.canonical_columns(name)
        if not cols:
            raise ValueError(
                f"canonical {name!r}: no matching column. "
                f"Available canonicals: {self._er.list_canonicals()}"
            )
        if len(cols) > 1:
            raise ValueError(
                f"canonical {name!r} resolves to {len(cols)} columns "
                f"({list(cols)}); pick one via .component(name)."
            )
        return _ComponentQuery(self._er, cols[0], self._default_time)

    def predicate(
        self,
        fn: Callable[[pd.DataFrame], Union[pd.Series, np.ndarray]],
    ) -> ResultMask:
        """Escape hatch — ``fn`` receives the parent ``df`` and must
        return a bool mask aligned with ``element_id`` (length =
        ``n_elements``) or aligned with the full ``(element_id, step)``
        index (in which case it is reduced via ``any``).
        """
        df = self._er.df
        out = fn(df)
        arr = np.asarray(out)
        n_elems = len(self._er.element_ids)
        if arr.shape == (n_elems,):
            ser = pd.Series(
                arr.astype(bool),
                index=pd.Index(
                    np.asarray(self._er.element_ids, dtype=np.int64),
                    name="element_id",
                ),
            )
        elif arr.shape == (len(df),):
            tmp = pd.Series(arr.astype(bool), index=df.index)
            ser = tmp.groupby(level="element_id").any()
        else:
            raise ValueError(
                f"predicate(fn): returned shape {arr.shape}; expected "
                f"({n_elems},) or ({len(df)},)."
            )
        return ResultMask(self._er, ser)

    def __repr__(self) -> str:
        return f"_ResultQuery(default_time={self._default_time!r})"


# ---------------------------------------------------------------------- #
# Subsetting helper — fresh ElementResults from a mask                    #
# ---------------------------------------------------------------------- #

def _subset_er(er: "ElementResults", ids: np.ndarray) -> "ElementResults":
    """Build a fresh :class:`ElementResults` keeping only ``ids``.

    Trims ``df``, ``element_ids``, ``element_node_coords``, and
    ``element_node_ids``. Preserves ``time``, ``gp_*``, ``model_*``,
    ``results_name``, ``element_type``, and ``name`` unchanged.
    """
    from .element_results import ElementResults

    keep = np.asarray(sorted(int(x) for x in ids), dtype=np.int64)
    if keep.size == 0:
        new_df = er.df.iloc[0:0]
        new_coords = (
            er.element_node_coords[0:0]
            if er.element_node_coords is not None
            else None
        )
        new_ids = (
            er.element_node_ids[0:0]
            if er.element_node_ids is not None
            else None
        )
    else:
        new_df = er.df.loc[
            er.df.index.get_level_values("element_id").isin(keep)
        ]
        # Build coords/ids subset aligned to the kept-id sorted order.
        if (
            er.element_node_coords is not None
            and er.element_node_ids is not None
        ):
            id_to_pos = {
                int(eid): i for i, eid in enumerate(er.element_ids)
            }
            row_idx = np.array(
                [id_to_pos[int(eid)] for eid in keep], dtype=np.int64
            )
            new_coords = er.element_node_coords[row_idx]
            new_ids = er.element_node_ids[row_idx]
        else:
            new_coords = None
            new_ids = None

    return ElementResults(
        df=new_df,
        time=er.time,
        name=er.name,
        element_ids=tuple(int(x) for x in keep),
        element_type=er.element_type,
        results_name=er.results_name,
        model_stage=er.model_stage,
        model_stages=er.model_stages,
        stage_step_ranges=dict(er.stage_step_ranges),
        gp_xi=er.gp_xi,
        gp_natural=er.gp_natural,
        gp_weights=er.gp_weights,
        element_node_coords=new_coords,
        element_node_ids=new_ids,
    )


__all__ = ["ResultMask", "resolve_step_indices"]
