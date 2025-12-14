from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from pathlib import Path
import gzip
import pickle

from .nodal_results_plotting import NodalResultsPlotter

import pandas as pd

if TYPE_CHECKING:
    from .nodal_results_plotting import NodalResultsPlotter
    from ..plotting.plot_dataclasses import ModelPlotSettings


# ───────────────────────────────────────────────────────────────────── #
# Helper proxy for attribute-style access: results.ACCELERATION[1]
# ───────────────────────────────────────────────────────────────────── #

class _ResultView:
    """
    Lightweight proxy for a single result type.

    Allows:
        results.ACCELERATION[1]  -> component 1
        results.ACCELERATION[:]  -> all components (same as get('ACCELERATION'))
    """
    def __init__(self, parent: "NodalResults", result_name: str):
        self._parent = parent
        self._result_name = result_name

    def __getitem__(self, component) -> pd.Series | pd.DataFrame:
        """
        Component indexing:

        - view[None] or view[:] → all components of this result
        - view[1]               → component 1
        - view['1']             → same as above
        """
        # Treat ":" or slice(None) as "all components"
        if component is None or component == slice(None) or component == ":":
            return self._parent.get(self._result_name, component=None)
        return self._parent.get(self._result_name, component=component)

    def __repr__(self) -> str:
        try:
            comps = self._parent.list_components(self._result_name)
        except Exception:
            comps = ()
        return f"<ResultView {self._result_name!r}, components={comps}>"


# ───────────────────────────────────────────────────────────────────── #
# Main dataclass
# ───────────────────────────────────────────────────────────────────── #

@dataclass(frozen=True, slots=True)
class NodalResults:
    """
    Container for generic nodal results.

    Attributes
    ----------
    df
        Results dataframe. For a single stage, index is typically (node_id, step).
        For multi-stage output, index is (stage, node_id, step).
    time
        Reference time table or per-stage dict (whatever MPCODataSet provides).
    name
        Dataset name.

    node_ids
        Optional tuple of unique node IDs involved in this result set.
    coords_map
        Optional mapping node_id -> {'x': float, 'y': float, 'z': float}.
    component_names
        Optional tuple of flattened component/column names.
    stages
        Optional tuple of stages used in this result (for multi-stage cases).
    """
    df: pd.DataFrame
    time: Any
    name: str

    node_ids: Optional[Tuple[int, ...]] = None
    coords_map: Optional[Dict[int, Dict[str, float]]] = None
    component_names: Optional[Tuple[str, ...]] = None
    stages: Optional[Tuple[str, ...]] = None
    plot_settings: ModelPlotSettings | None = None

    # internal cache of _ResultView objects (result_name -> _ResultView)
    _views: Dict[str, _ResultView] = None  # set in __post_init__

    # ------------------------------------------------------------------ #
    # Post-init to build dynamic views
    # ------------------------------------------------------------------ #

    def __post_init__(self):
        # Build the dict of views once, based on whatever is in df.columns
        views: Dict[str, _ResultView] = {}

        cols = self.df.columns
        if isinstance(cols, pd.MultiIndex):
            # names are at level 0
            names = sorted({str(c0) for (c0, _) in cols})
        else:
            # single-level: we treat columns as "components only", so no views
            names = []

        for rname in names:
            views[rname] = _ResultView(self, rname)

        # frozen dataclass, so we must bypass normal setattr
        object.__setattr__(self, "_views", views)

    # ------------------------------------------------------------------ #
    # Pickle support
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict[str, Any]:
        """
        Control what gets pickled.

        We purposely drop `_views` because it contains objects referencing `self`.
        We'll rebuild it in `__setstate__`.
        """
        state = {slot: getattr(self, slot) for slot in self.__slots__}  # type: ignore[attr-defined]
        state["_views"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore state and rebuild `_views`.
        """
        for k, v in state.items():
            object.__setattr__(self, k, v)
        # rebuild views after restoring df
        self.__post_init__()

    def save_pickle(self, path: str | Path, *, compress: bool | None = None, protocol: int = pickle.HIGHEST_PROTOCOL) -> Path:
        """
        Save this object to a pickle file.

        - If `compress` is None: infer from suffix (.gz).
        - If `compress` is True: gzip.
        """
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
    def load_pickle(cls, path: str | Path, *, compress: bool | None = None) -> "NodalResults":
        """
        Load a pickled NodalResults.
        """
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
        """
        List distinct result types present in the DataFrame.

        For MultiIndex columns (result_name, component) this returns the
        unique first-level names (e.g. 'DISPLACEMENT', 'ACCELERATION').

        For single-level columns it returns the stringified column names.
        """
        cols = self.df.columns

        if isinstance(cols, pd.MultiIndex):
            names = sorted({str(level0) for (level0, _) in cols})
        else:
            names = sorted({str(c) for c in cols})

        return tuple(names)

    def list_components(self, result_name: Optional[str] = None) -> Tuple[str, ...]:
        """
        List available component labels.

        - For MultiIndex columns:
            • If result_name is given, list components for that result.
            • If result_name is None, list all distinct components.
        - For single-level columns:
            • Only result_name=None is supported, returns column names.
        """
        cols = self.df.columns

        # MultiIndex: (result_name, component)
        if isinstance(cols, pd.MultiIndex):
            if result_name is None:
                comps = sorted({str(c1) for (_, c1) in cols})
                return tuple(comps)

            # Filter by given result_name
            comps = {str(c1) for (c0, c1) in cols if str(c0) == str(result_name)}
            if not comps:
                raise ValueError(
                    f"Result '{result_name}' not found.\n"
                    f"Available result types: {self.list_results()}"
                )
            return tuple(sorted(comps))

        # Single-level: no explicit result_name layer
        if result_name is not None:
            raise ValueError(
                "This NodalResults instance has no separate result_name level.\n"
                "Columns are directly components. Do not pass result_name.\n"
                f"Available components: {tuple(map(str, cols))}"
            )

        return tuple(map(str, cols))

    def get(
        self,
        result_name: Optional[str] = None,
        component: Optional[object] = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Safe accessor for the underlying DataFrame.

        Cases
        -----
        MultiIndex columns (result_name, component):

            • get(result_name='ACCELERATION', component=1)
                → Series with that single component.

            • get(result_name='ACCELERATION', component=None)
                → DataFrame with all components of that result.

            • get(result_name=None, ...)
                → Error: result_name is required when multiple results exist.

        Single-level columns:

            • get(result_name=None, component='<colname>' or index)
                → Series of that column.

            • get(result_name=None, component=None)
                → Full DataFrame.

            • Passing result_name != None
                → Error explaining usage.

        Raises
        ------
        ValueError
            With clear messages listing valid result types / components.
        """
        cols = self.df.columns

        # -------------------------------------------------------------- #
        # MultiIndex: (result_name, component)
        # -------------------------------------------------------------- #
        if isinstance(cols, pd.MultiIndex):
            # Require result_name in this case
            if result_name is None:
                raise ValueError(
                    "result_name must be provided for this NodalResults.\n"
                    f"Available result types: {self.list_results()}"
                )

            # Validate result_name
            available_results = self.list_results()
            if str(result_name) not in available_results:
                raise ValueError(
                    f"Unknown result_name '{result_name}'.\n"
                    f"Available result types: {available_results}"
                )

            # If component is None -> return all components of that result
            if component is None:
                sub_cols = [c for c in cols if str(c[0]) == str(result_name)]
                if not sub_cols:
                    raise ValueError(
                        f"No components found for result '{result_name}'."
                    )
                return self.df.loc[:, sub_cols]

            # Component specified -> try to match by string equality
            target = None
            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    target = (c0, c1)
                    break

            if target is None:
                comps = self.list_components(result_name)
                raise ValueError(
                    f"Component '{component}' not found for result '{result_name}'.\n"
                    f"Available components for '{result_name}': {comps}"
                )

            return self.df.loc[:, target]

        # -------------------------------------------------------------- #
        # Single-level columns: only components
        # -------------------------------------------------------------- #
        if result_name is not None:
            raise ValueError(
                "This NodalResults has single-level columns and does not "
                "distinguish result_name vs component.\n"
                "Use get(component=...) only.\n"
                f"Available components: {tuple(map(str, cols))}"
            )

        # No component → return full DataFrame
        if component is None:
            return self.df

        # Try to resolve component by exact match on column name or integer
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
    # Dynamic attribute access: results.ACCELERATION[1]
    # ------------------------------------------------------------------ #

    def __getattr__(self, item: str) -> Any:
        """
        If `item` matches a result_name, return its _ResultView proxy.

        This enables:
            results.DISPLACEMENT[1]
            results.ACCELERATION[:]
        """
        views = object.__getattribute__(self, "_views")
        if item in views:
            return views[item]
        # Fall back to default behaviour
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {item!r}")

    @property
    def plot(self) -> "NodalResultsPlotter":
        """
        Access a plotting helper bound to this NodalResults object.

        Examples
        --------
        results.plot.xy(
            y="ACCELERATION",
            y_direction=1,
            y_operation="Sum",
            x="TIME",
        )
        """
        return NodalResultsPlotter(self)
    
    def __dir__(self):
        """
        Extend dir() so that interactive tools (like VS Code's REPL)
        can see dynamic attributes (result names) as well.
        """
        base = set(super().__dir__())
        base.update(self._views.keys())
        return sorted(base)

    # ------------------------------------------------------------------ #
    # Pretty repr
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        results = self.list_results()
        comps = self.list_components(results[0]) if results else ()
        stages = self.stages or ()
        return (
            f"NodalResults(name={self.name!r}, "
            f"results={results}, "
            f"components={comps}, "
            f"stages={stages})"
        )
