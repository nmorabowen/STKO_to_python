"""
ElementResults — rich container for element-level results.

Mirrors the NodalResults pattern:
  - _ResultView proxy for attribute-style component access
  - fetch() / list_components() introspection
  - pickle serialization support
  - time array attachment
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# ResultView proxy
# ------------------------------------------------------------------ #

class _ElementResultView:
    """
    Lightweight proxy for a single value column in element results.

    Allows:
        results.val_1                   -> all elements, all steps
        results.val_1[element_ids]      -> filtered by element
    """

    def __init__(self, parent: "ElementResults", col_name: str) -> None:
        self._parent = parent
        self._col_name = col_name

    def __getitem__(
        self, key: Union[int, Sequence[int], slice]
    ) -> pd.Series:
        """
        Subscript by element_ids.

        Examples
        --------
        >>> view[42]           # single element
        >>> view[[42, 99]]     # multiple elements
        >>> view[:]            # all elements (same as view)
        """
        if isinstance(key, slice) and key == slice(None):
            return self._parent.fetch(self._col_name)

        if isinstance(key, (int, np.integer)):
            element_ids = [int(key)]
        else:
            element_ids = list(key)

        return self._parent.fetch(self._col_name, element_ids=element_ids)

    @property
    def series(self) -> pd.Series:
        """Return the full column as a Series."""
        return self._parent.fetch(self._col_name)

    def __repr__(self) -> str:
        return f"<ElementResultView {self._col_name!r}>"


# ------------------------------------------------------------------ #
# ElementResults container
# ------------------------------------------------------------------ #

class ElementResults:
    """
    Container for element-level results from MPCO HDF5 files.

    Expected df shape:
      - index: (element_id, step)
      - columns: val_1, val_2, ..., val_N  (component values)

    Attributes
    ----------
    df : pd.DataFrame
        Results data indexed by (element_id, step).
    time : np.ndarray
        Time array corresponding to steps.
    name : str or None
        Model name.
    element_ids : tuple[int, ...]
        Element IDs in this result set.
    element_type : str
        Base element type used for the query.
    results_name : str
        HDF5 result group name (e.g. 'globalForces').
    model_stage : str
        Model stage this data comes from.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time: Any,
        *,
        name: Optional[str] = None,
        element_ids: Optional[Tuple[int, ...]] = None,
        element_type: Optional[str] = None,
        results_name: Optional[str] = None,
        model_stage: Optional[str] = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name
        self.element_ids = element_ids or ()
        self.element_type = element_type or ""
        self.results_name = results_name or ""
        self.model_stage = model_stage or ""

        self._views: Dict[str, _ElementResultView] = {}
        self._build_views()

    # ------------------------------------------------------------------ #
    # View construction
    # ------------------------------------------------------------------ #

    def _build_views(self) -> None:
        """Build dynamic ResultView proxies for each value column."""
        self._views.clear()
        if self.df.empty:
            return

        for col in self.df.columns:
            col_str = str(col)
            self._views[col_str] = _ElementResultView(self, col_str)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to result columns (e.g., results.val_1)."""
        if name.startswith("_"):
            raise AttributeError(name)
        views = self.__dict__.get("_views", {})
        if name in views:
            return views[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available columns: {list(views.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def list_components(self) -> Tuple[str, ...]:
        """Return the names of all value columns."""
        if self.df.empty:
            return ()
        return tuple(str(c) for c in self.df.columns)

    @property
    def n_components(self) -> int:
        """Number of result components per element per step."""
        return len(self.df.columns) if not self.df.empty else 0

    @property
    def n_elements(self) -> int:
        """Number of unique elements in this result set."""
        return len(self.element_ids)

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        if self.df.empty:
            return 0
        return self.df.index.get_level_values("step").nunique()

    @property
    def empty(self) -> bool:
        return self.df.empty

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        component: Optional[str] = None,
        *,
        element_ids: Union[int, Sequence[int], None] = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Fetch results with optional element and component filtering.

        Parameters
        ----------
        component : str or None
            Column name (e.g. 'val_1'). If None, returns all columns.
        element_ids : int, list[int], or None
            Filter to specific elements. If None, returns all.

        Returns
        -------
        pd.Series or pd.DataFrame
        """
        df = self.df

        # Filter by element_ids
        if element_ids is not None:
            if isinstance(element_ids, (int, np.integer)):
                element_ids = [int(element_ids)]
            else:
                element_ids = [int(e) for e in element_ids]

            lvl = df.index.get_level_values("element_id")
            df = df.loc[lvl.isin(element_ids)]

            if df.empty:
                raise ValueError(
                    f"None of the requested element_ids are present. "
                    f"Available (sample): {sorted(self.element_ids)[:10]}"
                )

        # Filter by component
        if component is not None:
            if component not in df.columns:
                raise ValueError(
                    f"Component '{component}' not found. "
                    f"Available: {self.list_components()}"
                )
            return df[component]

        return df

    def to_dataframe(self, include_time: bool = True) -> pd.DataFrame:
        """
        Return a flat DataFrame with time column attached.

        Parameters
        ----------
        include_time : bool
            If True and time array is available, add a 'time' column.
        """
        df = self.df.copy()

        if include_time and isinstance(self.time, np.ndarray) and self.time.size > 0:
            steps = df.index.get_level_values("step")
            # Map step index to time value (safe for out-of-range)
            time_map = np.full(steps.max() + 1, np.nan)
            time_map[: len(self.time)] = self.time
            df["time"] = time_map[steps]

        return df

    # ------------------------------------------------------------------ #
    # Aggregation helpers
    # ------------------------------------------------------------------ #

    def envelope(
        self,
        component: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute min/max envelope over steps for each element.

        Parameters
        ----------
        component : str or None
            If provided, envelope for that component only.

        Returns
        -------
        pd.DataFrame
            Indexed by element_id, columns: [component_min, component_max, ...]
        """
        df = self.df if component is None else self.df[[component]]
        grouped = df.groupby("element_id")
        mins = grouped.min().add_suffix("_min")
        maxs = grouped.max().add_suffix("_max")
        return pd.concat([mins, maxs], axis=1).sort_index()

    def at_step(self, step: int) -> pd.DataFrame:
        """Extract results at a specific step, indexed by element_id."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.xs(step, level="step")

    def at_time(self, t: float, tol: float = 1e-10) -> pd.DataFrame:
        """
        Extract results at the time step closest to *t*.

        Parameters
        ----------
        t : float
            Target time.
        tol : float
            Tolerance for matching.
        """
        if not isinstance(self.time, np.ndarray) or self.time.size == 0:
            raise ValueError("Time array not available.")

        idx = int(np.argmin(np.abs(self.time - t)))
        if abs(self.time[idx] - t) > tol:
            # Just use closest
            pass
        return self.at_step(idx)

    # ------------------------------------------------------------------ #
    # Pickle support
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_views"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._views = {}
        self._build_views()

    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Save to pickle (optionally gzipped)."""
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        opener = gzip.open if compress else open
        with opener(p, "wb") as f:  # type: ignore
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(
        cls,
        path: str | Path,
        *,
        compress: bool | None = None,
    ) -> "ElementResults":
        """Load from pickle."""
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        opener = gzip.open if compress else open
        with opener(p, "rb") as f:  # type: ignore
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {p} is {type(obj)!r}, expected {cls.__name__}."
            )
        return obj

    # ------------------------------------------------------------------ #
    # Dunder niceties
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"ElementResults("
            f"results_name={self.results_name!r}, "
            f"element_type={self.element_type!r}, "
            f"n_elements={self.n_elements}, "
            f"n_steps={self.n_steps}, "
            f"n_components={self.n_components})"
        )

    def __str__(self) -> str:
        return (
            f"ElementResults: {self.results_name} on {self.element_type}\n"
            f"  Elements: {self.n_elements}, Steps: {self.n_steps}, "
            f"Components: {self.n_components}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __bool__(self) -> bool:
        return not self.df.empty
