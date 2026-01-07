from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any
from pathlib import Path
import gzip
import pickle

import pandas as pd
import numpy as np

from .nodal_results_plotting import NodalResultsPlotter
from .nodal_results_info import NodalResultsInfo

if TYPE_CHECKING:
    from ..plotting.plot_dataclasses import ModelPlotSettings


class _ResultView:
    """
    Lightweight proxy for a single result type.

    Allows:
        results.ACCELERATION[1]  -> component 1
        results.ACCELERATION[:]  -> all components
    """

    def __init__(self, parent: "NodalResults", result_name: str):
        self._parent = parent
        self._result_name = result_name

    def __getitem__(self, component) -> pd.Series | pd.DataFrame:
        if component is None or component == slice(None) or component == ":":
            return self._parent.fetch(self._result_name, component=None)
        return self._parent.fetch(self._result_name, component=component)

    def __repr__(self) -> str:
        try:
            comps = self._parent.list_components(self._result_name)
        except Exception:
            comps = ()
        return f"<ResultView {self._result_name!r}, components={comps}>"


class NodalResults:
    """
    Container for generic nodal results.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time: Any,
        name: str,
        *,
        nodes_ids: Optional[Tuple[int, ...]] = None,
        nodes_info: Optional[pd.DataFrame] = None,
        results_components: Optional[Tuple[str, ...]] = None,
        model_stages: Optional[Tuple[str, ...]] = None,
        plot_settings: Optional["ModelPlotSettings"] = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name

        self.info = NodalResultsInfo(
            nodes_info=nodes_info,
            nodes_ids=nodes_ids,
            model_stages=model_stages,
            results_components=results_components,
        )

        self.plot_settings = plot_settings

        self._views: Dict[str, _ResultView] = {}
        self._build_views()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_views(self) -> None:
        """Build dynamic ResultView proxies based on DataFrame columns."""
        self._views.clear()

        cols = self.df.columns
        if isinstance(cols, pd.MultiIndex):
            names = sorted({str(c0) for (c0, _) in cols})
            for rname in names:
                self._views[rname] = _ResultView(self, rname)

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
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        if compress:
            with gzip.open(p, "wb") as f:
                pickle.dump(self, f, protocol=protocol)
        else:
            with open(p, "wb") as f:
                pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(
        cls,
        path: str | Path,
        *,
        compress: bool | None = None,
    ) -> "NodalResults":
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        if compress:
            with gzip.open(p, "rb") as f:
                obj = pickle.load(f)
        else:
            with open(p, "rb") as f:
                obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(f"Pickle at {p} is {type(obj)!r}, expected {cls.__name__}.")
        return obj

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #

    def list_results(self) -> Tuple[str, ...]:
        cols = self.df.columns
        if isinstance(cols, pd.MultiIndex):
            names = sorted({str(level0) for (level0, _) in cols})
        else:
            names = sorted({str(c) for c in cols})
        return tuple(names)

    def list_components(self, result_name: Optional[str] = None) -> Tuple[str, ...]:
        cols = self.df.columns

        if isinstance(cols, pd.MultiIndex):
            if result_name is None:
                return tuple(sorted({str(c1) for (_, c1) in cols}))

            comps = {str(c1) for (c0, c1) in cols if str(c0) == str(result_name)}
            if not comps:
                raise ValueError(
                    f"Result '{result_name}' not found.\n"
                    f"Available result types: {self.list_results()}"
                )
            return tuple(sorted(comps))

        if result_name is not None:
            raise ValueError(
                "Single-level columns: do not pass result_name.\n"
                f"Available components: {tuple(map(str, cols))}"
            )

        return tuple(map(str, cols))

    # ------------------------------------------------------------------ #
    # Data access (single public accessor)
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        result_name: Optional[str] = None,
        component: Optional[object] = None,
    ) -> pd.Series | pd.DataFrame:
        cols = self.df.columns

        # MultiIndex columns: (result_name, component)
        if isinstance(cols, pd.MultiIndex):
            if result_name is None:
                raise ValueError(
                    "result_name must be provided.\n"
                    f"Available results: {self.list_results()}"
                )

            if component is None:
                sub_cols = [c for c in cols if str(c[0]) == str(result_name)]
                if not sub_cols:
                    raise ValueError(f"No components found for result '{result_name}'.")
                return self.df.loc[:, sub_cols]

            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    return self.df.loc[:, (c0, c1)]

            raise ValueError(
                f"Component '{component}' not found for result '{result_name}'.\n"
                f"Available components: {self.list_components(result_name)}"
            )

        # Single-level columns
        if result_name is not None:
            raise ValueError("Single-level columns: use fetch(component=...) only.")

        if component is None:
            return self.df

        if component in cols:
            return self.df[component]

        comp_str = str(component)
        if comp_str in cols:
            return self.df[comp_str]

        raise ValueError(
            f"Component '{component}' not found.\n"
            f"Available components: {tuple(map(str, cols))}"
        )

    # ------------------------------------------------------------------ #
    # Dynamic attribute access
    # ------------------------------------------------------------------ #

    def __getattr__(self, item: str) -> Any:
        if "_views" in self.__dict__ and item in self._views:
            return self._views[item]
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    def __dir__(self):
        base = set(super().__dir__())
        base.update(self._views.keys())
        return sorted(base)

    # ------------------------------------------------------------------ #
    # Plotting
    # ------------------------------------------------------------------ #

    @property
    def plot(self) -> NodalResultsPlotter:
        return NodalResultsPlotter(self)

    # ------------------------------------------------------------------ #
    # Representation
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        results = self.list_results()
        first = results[0] if results else None
        comps = self.list_components(first) if first is not None else ()
        stages = self.info.model_stages or ()
        return (
            f"NodalResults(name={self.name!r}, "
            f"results={results}, "
            f"components={comps}, "
            f"stages={stages})"
        )
