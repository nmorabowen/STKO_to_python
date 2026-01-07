# ── STKO_to_python/postproc/aggregator.py ───────────────────────────────
from __future__ import annotations

from typing import Mapping, Sequence, Union, Literal, overload

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
        Must have some notion of "step", either:
        - a column named 'step', or
        - an index level named 'step', or
        - a MultiIndex whose last level is 'step', or
        - a simple index that *represents* step.
    direction : str | int | None
        Column name or integer index to aggregate (``'x'``, ``'y'``, …).
        If None and df has exactly one column, that column is used.
    """

    __slots__ = ("group", "_cache")
    _CORE = ("sum", "mean", "std", "min", "max")   # allowed cached stats

    # ------------------------------------------------------------------ #
    def __init__(self, df: pd.DataFrame, direction: str | int | None) -> None:
        cols = list(df.columns)

        # ---- resolve direction ---------------------------------------- #
        if direction is None:
            if len(cols) == 1:
                direction = cols[0]
            else:
                raise ValueError(
                    "Aggregator: 'direction' is None but DataFrame has multiple "
                    f"columns {cols}. Specify a column name or index."
                )
        elif isinstance(direction, int) and direction not in df.columns:
            # treat as positional index
            try:
                direction = cols[direction]
            except IndexError as e:
                raise KeyError(
                    f"Integer direction index {direction} is out of range "
                    f"for columns {cols}"
                ) from e

        if direction not in df.columns:
            raise KeyError(
                f"'{direction}' not found. Available columns: {cols}"
            )

        # ---- build a groupby on "step" -------------------------------- #
        # Prefer a 'step' column if present
        if "step" in df.columns:
            self.group = df.groupby("step")[direction]
        else:
            idx = df.index
            nlevels = getattr(idx, "nlevels", 1)

            if nlevels > 1:
                # MultiIndex: try level named 'step', else last level
                names = list(idx.names) if idx.names is not None else []
                if "step" in names:
                    self.group = df.groupby(level="step")[direction]
                else:
                    # assume last level is step
                    self.group = df.groupby(level=-1)[direction]
            else:
                # Simple Index: treat index values as steps
                self.group = df.groupby(idx)[direction]

        self._cache: dict[str, pd.Series] = {}

    # =======================  private helpers  ======================== #
    def _stat(self, name: str) -> pd.Series:
        """Compute *name* lazily and memoise."""
        if name not in self._cache:
            match name:
                case "sum":
                    s = self.group.sum()
                case "mean":
                    s = self.group.mean()
                case "std":
                    s = self.group.std(ddof=0)
                case "min":
                    s = self.group.min()
                case "max":
                    s = self.group.max()
                case _:
                    raise AttributeError(f"Unknown stat '{name}'")
            self._cache[name] = s
        return self._cache[name]

    # =======================  public operations  ====================== #
    def sum(self) -> pd.Series:
        return self._stat("sum").rename("Sum")

    def mean(self) -> pd.Series:
        return self._stat("mean").rename("Mean")

    def max(self) -> pd.Series:
        return self._stat("max").rename("Max")

    def min(self) -> pd.Series:
        return self._stat("min").rename("Min")

    def std(self) -> pd.DataFrame:
        s = self._stat("std")
        return pd.concat(
            {"Std+1": s, "Std-1": -s, "Std+2": 2 * s, "Std-2": -2 * s},
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
            {
                "CumPos": s.where(s > 0, 0).cumsum(),
                "CumNeg": s.where(s < 0, 0).cumsum(),
            },
            axis=1,
        )

    def running_envelope(self) -> pd.DataFrame:
        return pd.concat(
            {"RunMin": self.min().cummin(), "RunMax": self.max().cummax()},
            axis=1,
        )

    # =======================  dispatcher API  ======================== #
    _DISPATCH: Mapping[str, str] = {
        "sum": "sum",
        "mean": "mean",
        "max": "max",
        "min": "min",
        "std": "std",
        "percentile": "percentile",
        "envelope": "envelope",
        "cumulative": "cumulative",
        "signedcumulative": "signed_cumulative",
        "runningenvelope": "running_envelope",
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
        if callable(operation):  # user-supplied function
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

    def __repr__(self) -> str:  # pragma: no cover
        col = getattr(self.group.obj, "name", None)
        return f"<Aggregator column={col!r}>"
