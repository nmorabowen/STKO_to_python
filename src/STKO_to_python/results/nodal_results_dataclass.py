from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any, Sequence
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
        results.ACCELERATION[1]             -> component 1 (all nodes)
        results.ACCELERATION[:]             -> all components (all nodes)
        results.ACCELERATION[1, [14, 25]]   -> component 1, nodes 14 & 25
        results.ACCELERATION[:, [14, 25]]   -> all components, nodes 14 & 25
    """

    def __init__(self, parent: "NodalResults", result_name: str):
        self._parent = parent
        self._result_name = result_name

    def __getitem__(self, key) -> pd.Series | pd.DataFrame:
        # Support tuple indexing: view[component, node_ids]
        if isinstance(key, tuple):
            if len(key) == 0:
                component = None
                node_ids = None
            elif len(key) == 1:
                component = key[0]
                node_ids = None
            elif len(key) == 2:
                component, node_ids = key
            else:
                raise TypeError(
                    f"Too many indices for ResultView: got {len(key)}. "
                    "Use view[component] or view[component, node_ids]."
                )
        else:
            component = key
            node_ids = None

        if component is None or component == slice(None) or component == ":":
            return self._parent.fetch(self._result_name, component=None, node_ids=node_ids)

        return self._parent.fetch(self._result_name, component=component, node_ids=node_ids)

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
        selection_set: Optional[dict] = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name

        self.info = NodalResultsInfo(
            nodes_info=nodes_info,
            nodes_ids=nodes_ids,
            model_stages=model_stages,
            results_components=results_components,
            selection_set=selection_set,
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
        *,
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Fetch results with optional node filtering.

        You can filter by:
        - node_ids=...
        - selection_set_id=...   (resolved via self.info.selection_set_node_ids)

        Notes
        -----
        Assumes the DataFrame index includes node_id either:
        - named level 'node_id', or
        - (node_id, step), or
        - (stage, node_id, step).
        """
        # ------------------------------------------------------------------ #
        # Resolve node filter
        # ------------------------------------------------------------------ #
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Use either node_ids or selection_set_id, not both.")

        if selection_set_id is not None:
            node_ids = self.info.selection_set_node_ids(selection_set_id)

        df = self.df

        # ------------------------------------------------------------------ #
        # Optional node filtering (node_ids)
        # ------------------------------------------------------------------ #
        if node_ids is not None:
            if isinstance(node_ids, (int, np.integer)):
                node_ids_arr = np.asarray([int(node_ids)], dtype=np.int64)
            else:
                node_ids_arr = np.asarray(list(node_ids), dtype=np.int64)
                if node_ids_arr.size == 0:
                    raise ValueError("node_ids is empty.")

            idx = df.index
            if not isinstance(idx, pd.MultiIndex):
                raise ValueError(
                    "[fetch] Expected a MultiIndex containing node_id. "
                    f"Got index type={type(idx).__name__}."
                )

            nlevels = idx.nlevels
            names = list(idx.names) if idx.names is not None else [None] * nlevels

            if "node_id" in names:
                node_level = names.index("node_id")
            else:
                if nlevels == 2:
                    node_level = 0  # (node_id, step)
                elif nlevels == 3:
                    node_level = 1  # (stage, node_id, step)
                else:
                    raise ValueError(
                        "[fetch] Cannot infer node_id level. "
                        f"Index nlevels={nlevels}, names={names}."
                    )

            lvl = idx.get_level_values(node_level)
            df = df.loc[lvl.isin(node_ids_arr)]
            if df.empty:
                raise ValueError(
                    f"[fetch] None of the requested node_ids are present. "
                    f"Requested (sample): {node_ids_arr[:10].tolist()}"
                )

        cols = df.columns

        # ------------------------------------------------------------------ #
        # MultiIndex columns: (result_name, component)
        # ------------------------------------------------------------------ #
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
                return df.loc[:, sub_cols]

            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    return df.loc[:, (c0, c1)]

            raise ValueError(
                f"Component '{component}' not found for result '{result_name}'.\n"
                f"Available components: {self.list_components(result_name)}"
            )

        # ------------------------------------------------------------------ #
        # Single-level columns
        # ------------------------------------------------------------------ #
        if result_name is not None:
            raise ValueError("Single-level columns: use fetch(component=...) only.")

        if component is None:
            return df

        if component in cols:
            return df[component]

        comp_str = str(component)
        if comp_str in cols:
            return df[comp_str]

        raise ValueError(
            f"Component '{component}' not found.\n"
            f"Available components: {tuple(map(str, cols))}"
        )


from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any, Sequence
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
        results.ACCELERATION[1]             -> component 1 (all nodes)
        results.ACCELERATION[:]             -> all components (all nodes)
        results.ACCELERATION[1, [14, 25]]   -> component 1, nodes 14 & 25
        results.ACCELERATION[:, [14, 25]]   -> all components, nodes 14 & 25
    """

    def __init__(self, parent: "NodalResults", result_name: str):
        self._parent = parent
        self._result_name = result_name

    def __getitem__(self, key) -> pd.Series | pd.DataFrame:
        # Support tuple indexing: view[component, node_ids]
        if isinstance(key, tuple):
            if len(key) == 0:
                component = None
                node_ids = None
            elif len(key) == 1:
                component = key[0]
                node_ids = None
            elif len(key) == 2:
                component, node_ids = key
            else:
                raise TypeError(
                    f"Too many indices for ResultView: got {len(key)}. "
                    "Use view[component] or view[component, node_ids]."
                )
        else:
            component = key
            node_ids = None

        if component is None or component == slice(None) or component == ":":
            return self._parent.fetch(self._result_name, component=None, node_ids=node_ids)

        return self._parent.fetch(self._result_name, component=component, node_ids=node_ids)

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
        selection_set: Optional[dict] = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name

        self.info = NodalResultsInfo(
            nodes_info=nodes_info,
            nodes_ids=nodes_ids,
            model_stages=model_stages,
            results_components=results_components,
            selection_set=selection_set,
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
        *,
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Fetch results with optional node filtering.

        You can filter by:
        - node_ids=...
        - selection_set_id=...   (resolved via self.info.selection_set_node_ids)

        Notes
        -----
        Assumes the DataFrame index includes node_id either:
        - named level 'node_id', or
        - (node_id, step), or
        - (stage, node_id, step).
        """
        # ------------------------------------------------------------------ #
        # Resolve node filter
        # ------------------------------------------------------------------ #
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Use either node_ids or selection_set_id, not both.")

        if selection_set_id is not None:
            node_ids = self.info.selection_set_node_ids(selection_set_id)

        df = self.df

        # ------------------------------------------------------------------ #
        # Optional node filtering (node_ids)
        # ------------------------------------------------------------------ #
        if node_ids is not None:
            if isinstance(node_ids, (int, np.integer)):
                node_ids_arr = np.asarray([int(node_ids)], dtype=np.int64)
            else:
                node_ids_arr = np.asarray(list(node_ids), dtype=np.int64)
                if node_ids_arr.size == 0:
                    raise ValueError("node_ids is empty.")

            idx = df.index
            if not isinstance(idx, pd.MultiIndex):
                raise ValueError(
                    "[fetch] Expected a MultiIndex containing node_id. "
                    f"Got index type={type(idx).__name__}."
                )

            nlevels = idx.nlevels
            names = list(idx.names) if idx.names is not None else [None] * nlevels

            if "node_id" in names:
                node_level = names.index("node_id")
            else:
                if nlevels == 2:
                    node_level = 0  # (node_id, step)
                elif nlevels == 3:
                    node_level = 1  # (stage, node_id, step)
                else:
                    raise ValueError(
                        "[fetch] Cannot infer node_id level. "
                        f"Index nlevels={nlevels}, names={names}."
                    )

            lvl = idx.get_level_values(node_level)
            df = df.loc[lvl.isin(node_ids_arr)]
            if df.empty:
                raise ValueError(
                    f"[fetch] None of the requested node_ids are present. "
                    f"Requested (sample): {node_ids_arr[:10].tolist()}"
                )

        cols = df.columns

        # ------------------------------------------------------------------ #
        # MultiIndex columns: (result_name, component)
        # ------------------------------------------------------------------ #
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
                return df.loc[:, sub_cols]

            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    return df.loc[:, (c0, c1)]

            raise ValueError(
                f"Component '{component}' not found for result '{result_name}'.\n"
                f"Available components: {self.list_components(result_name)}"
            )

        # ------------------------------------------------------------------ #
        # Single-level columns
        # ------------------------------------------------------------------ #
        if result_name is not None:
            raise ValueError("Single-level columns: use fetch(component=...) only.")

        if component is None:
            return df

        if component in cols:
            return df[component]

        comp_str = str(component)
        if comp_str in cols:
            return df[comp_str]

        raise ValueError(
            f"Component '{component}' not found.\n"
            f"Available components: {tuple(map(str, cols))}"
        )

    def fetch_nearest(
        self,
        *,
        points: Sequence[Sequence[float]],
        result_name: Optional[str] = None,
        component: Optional[object] = None,
        file_id: Optional[int] = None,
        return_nodes: bool = False,
    ) -> pd.Series | pd.DataFrame | tuple[pd.Series | pd.DataFrame, list[int]]:
        """
        Convenience: resolve coordinates -> nearest node_ids, then fetch().
        """
        node_ids = self.info.nearest_node_id(points, file_id=file_id, return_distance=False)
        out = self.fetch(result_name=result_name, component=component, node_ids=node_ids)
        return (out, node_ids) if return_nodes else out

    def drift(
        self,
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",   # "series" | "max"
    ) -> pd.Series | float:
        """
        Drift between two nodes:

            drift(t) = (u_top(t) - u_bottom(t)) / (z_top - z_bottom)

        Parameters
        ----------
        top, bottom
            Node id (int) or coordinates (x,y) or (x,y,z).
            Coordinates are resolved to nearest node.
        component
            REQUIRED displacement component (e.g. 1, 'X', 'Ux').
        result_name
            Typically 'DISPLACEMENT'.
        stage
            Required if results are multi-stage (stage, node_id, step).
        signed
            If False, uses |u_top - u_bottom|.
        reduce
            - "series" → return full drift time-history (default)
            - "max"    → return max(|drift(t)|)

        Returns
        -------
        pd.Series or float
            Drift time-history or absolute maximum drift.
        """

        # ---------------------------
        # Resolve node ids
        # ---------------------------
        def _as_node_id(v: int | Sequence[float], *, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)

            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"{name} must be a node id or coordinates (x,y) or (x,y,z)."
                )

            coords = tuple(float(x) for x in v)
            if len(coords) not in (2, 3):
                raise TypeError(
                    f"{name} coordinates must have length 2 or 3. Got {len(coords)}."
                )

            return int(self.info.nearest_node_id([coords])[0])

        top_id = _as_node_id(top, name="top")
        bot_id = _as_node_id(bottom, name="bottom")

        # ---------------------------
        # dz from nodes_info
        # ---------------------------
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. z-coordinates are required for drift().")
        if not isinstance(self.info.nodes_info, pd.DataFrame):
            raise TypeError("drift() expects nodes_info as a pandas DataFrame.")

        ni = self.info.nodes_info
        zcol = self.info._resolve_column(ni, "z", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        def _z_of(nid: int) -> float:
            if nid_col is not None:
                row = ni.loc[ni[nid_col].to_numpy() == nid]
                if row.empty:
                    raise ValueError(f"node_id={nid} not found in nodes_info.")
                return float(row.iloc[0][zcol])
            if nid not in ni.index:
                raise ValueError(f"node_id={nid} not found in nodes_info index.")
            return float(ni.loc[nid, zcol])

        z_top = _z_of(top_id)
        z_bot = _z_of(bot_id)
        dz = float(z_top - z_bot)
        if dz == 0.0:
            raise ValueError("z_top == z_bottom → dz = 0. Cannot compute drift.")

        # ---------------------------
        # Fetch displacement
        # ---------------------------
        s = self.fetch(
            result_name=result_name,
            component=component,
            node_ids=[top_id, bot_id],
        )

        # Multi-stage handling
        idx = s.index
        if isinstance(idx, pd.MultiIndex) and idx.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
                raise ValueError(
                    "Multi-stage results detected. Provide stage=...\n"
                    f"Available stages: {stages}"
                )
            s = s.xs(str(stage), level=0)

        # Expect (node_id, step)
        idx = s.index
        if not (isinstance(idx, pd.MultiIndex) and idx.nlevels == 2):
            raise ValueError(
                "drift() expects index (node_id, step) after stage selection."
            )

        u_top = s.xs(top_id, level=0).sort_index()
        u_bot = s.xs(bot_id, level=0).sort_index()
        u_top, u_bot = u_top.align(u_bot, join="inner")

        du = u_top - u_bot
        if not signed:
            du = du.abs()

        drift_series = du / dz
        drift_series.name = f"drift({result_name}:{component})"

        # ---------------------------
        # Reduction
        # ---------------------------
        if reduce == "series":
            return drift_series

        if reduce == "max":
            return float(np.nanmax(np.abs(drift_series.to_numpy())))

        raise ValueError(
            f"Unknown reduce='{reduce}'. Use 'series' or 'max'."
        )

    def interstory_drift_envelope(
        self,
        *,
        component: object,
        selection_set_id: int | Sequence[int] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Interstory drift envelope (MAX and MIN) from a vertical stack of nodes.
        Nodes are sorted by z, then consecutive pairs are used as stories.
        """

        provided = sum(x is not None for x in (selection_set_id, node_ids, coordinates))
        if provided != 1:
            raise ValueError("Provide exactly ONE of: selection_set_id, node_ids, coordinates.")

        # resolve ids
        if selection_set_id is not None:
            ids = self.info.selection_set_node_ids(selection_set_id)
        elif node_ids is not None:
            if len(node_ids) == 0:
                raise ValueError("node_ids is empty.")
            ids = [int(i) for i in node_ids]
        else:
            if coordinates is None or len(coordinates) == 0:
                raise ValueError("coordinates is empty.")
            ids = self.info.nearest_node_id(coordinates, return_distance=False)

        ids = sorted(set(ids))
        if len(ids) < 2:
            raise ValueError("Need at least 2 nodes to compute interstory drift.")

        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. Need nodes_info with z-coordinates.")
        ni = self.info.nodes_info
        zcol = self.info._resolve_column(ni, "z", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        # build z lookup once
        if nid_col is not None:
            sub = ni.loc[ni[nid_col].isin(ids), [nid_col, zcol]]
            if sub.empty:
                raise ValueError("None of the node ids were found in nodes_info.")
            z_map = dict(zip(sub[nid_col].to_numpy(dtype=int), sub[zcol].to_numpy(dtype=float)))
        else:
            missing = [i for i in ids if i not in ni.index]
            if missing:
                raise ValueError(f"node_id(s) not found in nodes_info index: {missing[:10]}")
            z_map = {int(i): float(ni.loc[int(i), zcol]) for i in ids}

        z_vals = np.asarray([z_map[i] for i in ids], dtype=float)
        order = np.argsort(z_vals, kind="mergesort")
        ids_sorted = [ids[i] for i in order]
        z_sorted = z_vals[order]

        # drop duplicate z levels
        keep = [0]
        for i in range(1, len(z_sorted)):
            if z_sorted[i] != z_sorted[keep[-1]]:
                keep.append(i)
        ids_sorted = [ids_sorted[i] for i in keep]
        z_sorted = z_sorted[keep]

        if len(ids_sorted) < 2:
            raise ValueError("After sorting/dedup by z, fewer than 2 unique z-levels remain.")

        rows: list[dict[str, float]] = []
        idx_pairs: list[tuple[int, int]] = []

        for lower_id, upper_id, z_lo, z_up in zip(ids_sorted[:-1], ids_sorted[1:], z_sorted[:-1], z_sorted[1:]):
            dz = float(z_up - z_lo)
            if dz == 0.0:
                continue

            dr = self.drift(
                top=upper_id,
                bottom=lower_id,
                component=component,
                result_name=result_name,
                stage=stage,
                signed=True,
                reduce="series",
            )

            arr = dr.to_numpy(dtype=float)
            rows.append(
                {
                    "dz": dz,
                    "z_lower": float(z_lo),
                    "z_upper": float(z_up),
                    "max_drift": float(np.nanmax(arr)),
                    "min_drift": float(np.nanmin(arr)),
                }
            )
            idx_pairs.append((int(lower_id), int(upper_id)))

        if not rows:
            raise ValueError("No valid story pairs were produced (check z-coordinates).")

        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx_pairs, names=("lower_node", "upper_node")),
        )
        return out.sort_values("z_lower", kind="mergesort")



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


    def drift(
        self,
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",   # "series" | "max"
    ) -> pd.Series | float:
        """
        Drift between two nodes:

            drift(t) = (u_top(t) - u_bottom(t)) / (z_top - z_bottom)

        Parameters
        ----------
        top, bottom
            Node id (int) or coordinates (x,y) or (x,y,z).
            Coordinates are resolved to nearest node.
        component
            REQUIRED displacement component (e.g. 1, 'X', 'Ux').
        result_name
            Typically 'DISPLACEMENT'.
        stage
            Required if results are multi-stage (stage, node_id, step).
        signed
            If False, uses |u_top - u_bottom|.
        reduce
            - "series" → return full drift time-history (default)
            - "max"    → return max(|drift(t)|)

        Returns
        -------
        pd.Series or float
            Drift time-history or absolute maximum drift.
        """

        # ---------------------------
        # Resolve node ids
        # ---------------------------
        def _as_node_id(v: int | Sequence[float], *, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)

            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(
                    f"{name} must be a node id or coordinates (x,y) or (x,y,z)."
                )

            coords = tuple(float(x) for x in v)
            if len(coords) not in (2, 3):
                raise TypeError(
                    f"{name} coordinates must have length 2 or 3. Got {len(coords)}."
                )

            return int(self.info.nearest_node_id([coords])[0])

        top_id = _as_node_id(top, name="top")
        bot_id = _as_node_id(bottom, name="bottom")

        # ---------------------------
        # dz from nodes_info
        # ---------------------------
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. z-coordinates are required for drift().")
        if not isinstance(self.info.nodes_info, pd.DataFrame):
            raise TypeError("drift() expects nodes_info as a pandas DataFrame.")

        ni = self.info.nodes_info
        zcol = self.info._resolve_column(ni, "z", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        def _z_of(nid: int) -> float:
            if nid_col is not None:
                row = ni.loc[ni[nid_col].to_numpy() == nid]
                if row.empty:
                    raise ValueError(f"node_id={nid} not found in nodes_info.")
                return float(row.iloc[0][zcol])
            if nid not in ni.index:
                raise ValueError(f"node_id={nid} not found in nodes_info index.")
            return float(ni.loc[nid, zcol])

        z_top = _z_of(top_id)
        z_bot = _z_of(bot_id)
        dz = float(z_top - z_bot)
        if dz == 0.0:
            raise ValueError("z_top == z_bottom → dz = 0. Cannot compute drift.")

        # ---------------------------
        # Fetch displacement
        # ---------------------------
        s = self.fetch(
            result_name=result_name,
            component=component,
            node_ids=[top_id, bot_id],
        )

        # Multi-stage handling
        idx = s.index
        if isinstance(idx, pd.MultiIndex) and idx.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in idx.get_level_values(0)}))
                raise ValueError(
                    "Multi-stage results detected. Provide stage=...\n"
                    f"Available stages: {stages}"
                )
            s = s.xs(str(stage), level=0)

        # Expect (node_id, step)
        idx = s.index
        if not (isinstance(idx, pd.MultiIndex) and idx.nlevels == 2):
            raise ValueError(
                "drift() expects index (node_id, step) after stage selection."
            )

        u_top = s.xs(top_id, level=0).sort_index()
        u_bot = s.xs(bot_id, level=0).sort_index()
        u_top, u_bot = u_top.align(u_bot, join="inner")

        du = u_top - u_bot
        if not signed:
            du = du.abs()

        drift_series = du / dz
        drift_series.name = f"drift({result_name}:{component})"

        # ---------------------------
        # Reduction
        # ---------------------------
        if reduce == "series":
            return drift_series

        if reduce == "max":
            return float(np.nanmax(np.abs(drift_series.to_numpy())))

        raise ValueError(
            f"Unknown reduce='{reduce}'. Use 'series' or 'max'."
        )

    def interstory_drift_envelope(
        self,
        *,
        component: object,
        selection_set_id: int | Sequence[int] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Interstory drift envelope (MAX and MIN) from a vertical stack of nodes.

        Provide exactly ONE of:
        - selection_set_id
        - node_ids
        - coordinates  -> nearest nodes

        Steps:
        1) Resolve -> node ids
        2) Read z from nodes_info
        3) Sort by z
        4) For each consecutive pair (lower, upper):
                drift(t) = (u_upper(t)-u_lower(t)) / (z_upper-z_lower)
            store:
                max_drift = max(drift(t))
                min_drift = min(drift(t))

        Returns
        -------
        pd.DataFrame
            MultiIndex (lower_node, upper_node) with columns:
                dz, z_lower, z_upper, max_drift, min_drift

            You can derive absolute max later as:
                abs_max = max(|max_drift|, |min_drift|)
        """
        # ---------------------------
        # Validate exclusive inputs
        # ---------------------------
        provided = sum(x is not None for x in (selection_set_id, node_ids, coordinates))
        if provided != 1:
            raise ValueError("Provide exactly ONE of: selection_set_id, node_ids, coordinates.")

        # ---------------------------
        # Resolve node ids
        # ---------------------------
        if selection_set_id is not None:
            ids = self.info.selection_set_node_ids(selection_set_id)
        elif node_ids is not None:
            if len(node_ids) == 0:
                raise ValueError("node_ids is empty.")
            ids = [int(i) for i in node_ids]
        else:
            if coordinates is None or len(coordinates) == 0:
                raise ValueError("coordinates is empty.")
            ids = self.info.nearest_node_id(coordinates, return_distance=False)

        ids = sorted(set(ids))
        if len(ids) < 2:
            raise ValueError("Need at least 2 nodes to compute interstory drift.")

        # ---------------------------
        # Get z for each node and sort by z
        # ---------------------------
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. Need nodes_info with z-coordinates.")
        if not isinstance(self.info.nodes_info, pd.DataFrame):
            raise TypeError("interstory_drift_envelope expects info.nodes_info as a DataFrame.")

        ni = self.info.nodes_info
        zcol = self.info._resolve_column(ni, "z", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        def _z_of(nid: int) -> float:
            if nid_col is not None:
                row = ni.loc[ni[nid_col].to_numpy() == nid]
                if row.empty:
                    raise ValueError(f"node_id={nid} not found in nodes_info.")
                return float(row.iloc[0][zcol])
            if nid not in ni.index:
                raise ValueError(f"node_id={nid} not found in nodes_info index.")
            return float(ni.loc[nid, zcol])

        z_vals = np.asarray([_z_of(i) for i in ids], dtype=float)

        order = np.argsort(z_vals, kind="mergesort")
        ids_sorted = [ids[i] for i in order]
        z_sorted = z_vals[order]

        # Drop duplicate z-levels (keep first occurrence)
        keep = [0]
        for i in range(1, len(z_sorted)):
            if z_sorted[i] != z_sorted[keep[-1]]:
                keep.append(i)
        ids_sorted = [ids_sorted[i] for i in keep]
        z_sorted = z_sorted[keep]

        if len(ids_sorted) < 2:
            raise ValueError("After sorting/dedup by z, fewer than 2 unique z-levels remain.")

        # ---------------------------
        # Compute per-story envelopes
        # ---------------------------
        rows: list[dict[str, float]] = []
        idx_pairs: list[tuple[int, int]] = []

        for lower_id, upper_id, z_lo, z_up in zip(
            ids_sorted[:-1], ids_sorted[1:], z_sorted[:-1], z_sorted[1:]
        ):
            dz = float(z_up - z_lo)
            if dz == 0.0:
                continue

            dr = self.drift(
                top=upper_id,
                bottom=lower_id,
                component=component,
                result_name=result_name,
                stage=stage,
                signed=True,
                reduce="series",
            )

            arr = dr.to_numpy(dtype=float)
            rows.append(
                {
                    "dz": dz,
                    "z_lower": float(z_lo),
                    "z_upper": float(z_up),
                    "max_drift": float(np.nanmax(arr)),
                    "min_drift": float(np.nanmin(arr)),
                }
            )
            idx_pairs.append((int(lower_id), int(upper_id)))

        if not rows:
            raise ValueError("No valid story pairs were produced (check z-coordinates).")

        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx_pairs, names=("lower_node", "upper_node")),
        )

        return out.sort_values("z_lower", kind="mergesort")


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
