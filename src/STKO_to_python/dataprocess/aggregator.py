# ── STKO_to_python/postproc/aggregator.py ───────────────────────────────
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Union, Literal, overload

import pandas as pd

# ↓ canonical list of built-in operations for type checkers & IDEs
StrOp = Literal[
    "Sum", "Mean", "Max", "Min", "Std",
    "Percentile", "Envelope",
    "Cumulative", "SignedCumulative", "RunningEnvelope",
]

class Aggregator:
    """
    Fast 1-D aggregation helper for MPCO results DataFrames.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``"step"`` level/index and the given *direction* column.
    direction : str | int
        Column name or integer index to aggregate (``'x'``, ``'y'``, …).

    Notes
    -----
    The class caches heavy statistics so repeated calls on the *same* dataframe
    are almost free:

    >>> agg = Aggregator(df, "x")
    >>> env = agg.envelope()         # computes min/max once
    >>> std = agg.std()              # reuses cached std
    """

    __slots__ = ("group", "_cache")
    _CORE = ("sum", "mean", "std", "min", "max")   # allowed cached stats

    # ------------------------------------------------------------------ #
    def __init__(self, df: pd.DataFrame, direction: str | int) -> None:
        if direction not in df.columns:
            raise KeyError(
                f"'{direction}' not found. Available columns: {list(df.columns)}"
            )
        # one shared group-by object across all operations
        self.group = df.groupby("step")[direction]
        self._cache: dict[str, pd.Series] = {}

    # =======================  private helpers  ======================== #
    def _stat(self, name: str) -> pd.Series:
        """Compute *name* lazily and memoise."""
        if name not in self._cache:
            match name:
                case "sum":  s = self.group.sum()
                case "mean": s = self.group.mean()
                case "std":  s = self.group.std(ddof=0)
                case "min":  s = self.group.min()
                case "max":  s = self.group.max()
                case _:
                    raise AttributeError(f"Unknown stat '{name}'")
            self._cache[name] = s
        return self._cache[name]

    # =======================  public operations  ====================== #
    # simple scalars --------------------------------------------------- #
    def sum(self)  -> pd.Series: return self._stat("sum").rename("Sum")
    def mean(self) -> pd.Series: return self._stat("mean").rename("Mean")
    def max(self)  -> pd.Series: return self._stat("max").rename("Max")
    def min(self)  -> pd.Series: return self._stat("min").rename("Min")

    # derived ---------------------------------------------------------- #
    def std(self) -> pd.DataFrame:
        s = self._stat("std")
        return pd.concat(
            {"Std+1":  s, "Std-1": -s, "Std+2": 2*s, "Std-2": -2*s},
            axis=1,
        )

    def percentile(self, q: float) -> pd.Series:
        if not (0.0 <= q <= 100.0):
            raise ValueError("percentile must be 0–100")
        return self.group.quantile(q / 100).rename(f"P{q:g}")

    def envelope(self) -> pd.DataFrame:
        return pd.concat({"Min": self.min(), "Max": self.max()}, axis=1)

    def cumulative(self) -> pd.Series:
        return self._stat("sum").cumsum().rename("Cumulative")

    def signed_cumulative(self) -> pd.DataFrame:
        s = self._stat("sum")
        return pd.concat(
            {"CumPos": s.where(s > 0, 0).cumsum(),
             "CumNeg": s.where(s < 0, 0).cumsum()},
            axis=1,
        )

    def running_envelope(self) -> pd.DataFrame:
        return pd.concat(
            {"RunMin": self.min().cummin(),
             "RunMax": self.max().cummax()},
            axis=1,
        )

    # =======================  dispatcher API  ======================== #
    _DISPATCH: Mapping[str, str] = {
        "sum":   "sum",
        "mean":  "mean",
        "max":   "max",
        "min":   "min",
        "std":   "std",
        "percentile":       "percentile",
        "envelope":         "envelope",
        "cumulative":       "cumulative",
        "signedcumulative": "signed_cumulative",
        "runningenvelope":  "running_envelope",
    }

    @overload
    def compute(
        self,
        *,
        operation: StrOp,
        percentile: float | None = None,
    ) -> pd.Series | pd.DataFrame: ...
    @overload
    def compute(
        self,
        *,
        operation: Sequence[StrOp],
        percentile: float | None = None,
    ) -> pd.DataFrame: ...

    def compute(
        self,
        *,
        operation: Union[StrOp, Sequence[StrOp]] = "Sum",
        percentile: float | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Back-compat “string” interface.

        Examples
        --------
        >>> agg.compute(operation="Envelope")
        >>> agg.compute(operation=("Max", "Min", "Std"))
        """
        if callable(operation):                      # user-supplied function
            return self.group.apply(operation)

        ops = (operation,) if isinstance(operation, str) else operation
        out: list[pd.Series | pd.DataFrame] = []

        for op in ops:
            key = str(op).lower()
            if key not in self._DISPATCH:
                raise ValueError(f"Unknown operation '{op}'")

            method_name = self._DISPATCH[key]
            if method_name == "percentile":
                if percentile is None:
                    raise ValueError("`percentile=` required for Percentile op")
                out.append(self.percentile(percentile))
            else:
                out.append(getattr(self, method_name)())

        return out[0] if len(out) == 1 else pd.concat(out, axis=1)

    # =======================  dunder niceties  ======================= #
    def __call__(self, *a, **kw):
        """Alias to :meth:`compute` for one-liners."""
        return self.compute(*a, **kw)

    def __repr__(self) -> str:                      # pragma: no cover
        return f"<Aggregator columns={list(self.group.obj.name)!r}>"

__all__ = ["Aggregator"]
