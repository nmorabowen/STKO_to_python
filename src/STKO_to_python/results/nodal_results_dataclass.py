# results/nodal_results_dataclass.py

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union
from pathlib import Path
import gzip
import pickle

import numpy as np
import pandas as pd

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

    Expected df shape:
      - index: (node_id, step) OR (stage, node_id, step)
      - columns: MultiIndex (result, component) OR single-level
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time: Any,
        *,
        name: Optional[str],
        nodes_ids: Optional[Tuple[int, ...]] = None,
        nodes_info: Optional[pd.DataFrame] = None,
        results_components: Optional[Tuple[str, ...]] = None,
        model_stages: Optional[Tuple[str, ...]] = None,
        plot_settings: Optional["ModelPlotSettings"] = None,
        selection_set: Optional[dict] = None,
        analysis_time: Optional[float] = None,
        size: Optional[int] = None,
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
            analysis_time=analysis_time,
            size=size,
            name=name,
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
    # Data access
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        result_name: Optional[str] = None,
        component: Optional[object] = None,
        *,
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        only_available: bool = True,
    ) -> pd.Series | pd.DataFrame:
        """
        Fetch results with optional node filtering.

        You can filter by any combination of:
          - node_ids
          - selection_set_id
          - selection_set_name

        Semantics: UNION of all node sources.

        only_available:
          Passed to selection_set resolver(s) to optionally intersect with self.info.nodes_ids.
        """
        df = self.df
        gathered: list[np.ndarray] = []

        # ---- selection by id ----
        if selection_set_id is not None:
            ids = self.info.selection_set_node_ids(selection_set_id, only_available=only_available)
            gathered.append(np.asarray(ids, dtype=np.int64))

        # ---- selection by name ----
        if selection_set_name is not None:
            ids = self.info.selection_set_node_ids_by_name(selection_set_name, only_available=only_available)
            gathered.append(np.asarray(ids, dtype=np.int64))

        # ---- explicit node_ids ----
        if node_ids is not None:
            if isinstance(node_ids, (int, np.integer)):
                gathered.append(np.asarray([int(node_ids)], dtype=np.int64))
            else:
                arr = np.asarray(list(node_ids), dtype=np.int64)
                if arr.size == 0:
                    raise ValueError("node_ids is empty.")
                gathered.append(arr)

        # ---- apply node filter ----
        if gathered:
            node_ids_arr = np.unique(np.concatenate(gathered))
            if node_ids_arr.size == 0:
                raise ValueError("Resolved node set is empty.")

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

        # ---- MultiIndex columns: (result_name, component) ----
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

        # ---- Single-level columns ----
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
        return_nodes: bool = False,
    ) -> pd.Series | pd.DataFrame | tuple[pd.Series | pd.DataFrame, list[int]]:
        """
        Convenience: resolve coordinates -> nearest node_ids, then fetch().
        """
        node_ids = self.info.nearest_node_id(points, return_distance=False)
        out = self.fetch(result_name=result_name, component=component, node_ids=node_ids)
        return (out, node_ids) if return_nodes else out

    # ------------------------------------------------------------------ #
    # Drift utilities
    # ------------------------------------------------------------------ #

    def drift(
        self,
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",  # "series" | "abs_max"
    ) -> pd.Series | float:
        """
        Drift between two nodes:

            drift(t) = (u_top(t) - u_bottom(t)) / (z_top - z_bottom)

        top, bottom:
            - node id (int), or
            - coordinates (x,y) or (x,y,z) resolved to nearest node.
        """

        def _as_node_id(v: int | Sequence[float], *, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)

            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(f"{name} must be a node id or coordinates (x,y) or (x,y,z).")

            coords = tuple(float(x) for x in v)
            if len(coords) not in (2, 3):
                raise TypeError(f"{name} coordinates must have length 2 or 3. Got {len(coords)}.")

            return int(self.info.nearest_node_id([coords])[0])

        top_id = _as_node_id(top, name="top")
        bot_id = _as_node_id(bottom, name="bottom")

        # ---- z coords ----
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. z-coordinates are required for drift().")
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

        # ---- fetch displacement for both nodes ----
        s = self.fetch(result_name=result_name, component=component, node_ids=[top_id, bot_id])

        # multi-stage -> require stage
        if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
            s = s.xs(str(stage), level=0)

        if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
            raise ValueError("drift() expects index (node_id, step) after stage selection.")

        u_top = s.xs(top_id, level=0).sort_index()
        u_bot = s.xs(bot_id, level=0).sort_index()
        u_top, u_bot = u_top.align(u_bot, join="inner")

        du = u_top - u_bot
        if not signed:
            du = du.abs()

        drift_series = du / dz
        drift_series.name = f"drift({result_name}:{component})"

        if reduce == "series":
            return drift_series
        if reduce == "abs_max":
            return float(np.nanmax(np.abs(drift_series.to_numpy(dtype=float))))
        raise ValueError(f"Unknown reduce='{reduce}'. Use 'series' or 'abs_max'.")

    def _resolve_story_nodes_by_z_tol(
        self,
        *,
        selection_set_id: int | Sequence[int] | None,
        selection_set_name: str | Sequence[str] | None,
        node_ids: Sequence[int] | None,
        coordinates: Sequence[Sequence[float]] | None,
        dz_tol: float,
    ) -> list[tuple[float, list[int]]]:
        """
        Returns: [(z_ref, [node_ids_at_story]), ...] sorted by z_ref.
        z_ref is the first z encountered in the cluster (deterministic after sorting).
        """
        provided = sum(x is not None for x in (selection_set_id, selection_set_name, node_ids, coordinates))
        if provided != 1:
            raise ValueError(
                "Provide exactly ONE of: selection_set_id, selection_set_name, node_ids, coordinates."
            )

        # ---- resolve ids ----
        if selection_set_id is not None:
            ids = self.info.selection_set_node_ids(selection_set_id)
        elif selection_set_name is not None:
            ids = self.info.selection_set_node_ids_by_name(selection_set_name)
        elif node_ids is not None:
            if len(node_ids) == 0:
                raise ValueError("node_ids is empty.")
            ids = [int(i) for i in node_ids]
        else:
            assert coordinates is not None
            if len(coordinates) == 0:
                raise ValueError("coordinates is empty.")
            ids = self.info.nearest_node_id(coordinates, return_distance=False)

        ids = sorted(set(int(i) for i in ids))
        if len(ids) == 0:
            raise ValueError("Resolved node list is empty.")

        if self.info.nodes_info is None:
            raise ValueError("nodes_info is None. Need nodes_info with z-coordinates.")
        ni = self.info.nodes_info
        zcol = self.info._resolve_column(ni, "z", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        # ---- build (node_id, z) pairs ----
        pairs: list[tuple[int, float]] = []
        if nid_col is not None:
            sub = ni.loc[ni[nid_col].isin(ids), [nid_col, zcol]]
            if sub.empty:
                raise ValueError("None of the node ids were found in nodes_info.")
            for nid, z in zip(sub[nid_col], sub[zcol]):
                pairs.append((int(nid), float(z)))
        else:
            missing = [i for i in ids if i not in ni.index]
            if missing:
                raise ValueError(f"node_id(s) not found in nodes_info index: {missing[:10]}")
            for nid in ids:
                pairs.append((int(nid), float(ni.loc[int(nid), zcol])))

        # ---- sort and cluster by tolerance ----
        pairs.sort(key=lambda x: x[1])

        stories: list[tuple[float, list[int]]] = []
        for nid, z in pairs:
            if not stories:
                stories.append((z, [nid]))
                continue
            z_ref, members = stories[-1]
            if abs(z - z_ref) <= float(dz_tol):
                members.append(nid)
            else:
                stories.append((z, [nid]))

        return stories


    def interstory_drift_envelope(
        self,
        *,
        component: object,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        dz_tol: float = 1e-3,
        representative: str = "min_id",  # "min_id" | "max_abs_peak"
    ) -> pd.DataFrame:
        """
        Interstory drift envelope (MAX and MIN signed) using z-tolerance clustering.

        representative:
        - "min_id": uses min node_id in each story (fast, deterministic)
        - "max_abs_peak": chooses node in each story with largest abs peak response
                            (more robust if multiple nodes per floor)

        Returns
        -------
        DataFrame indexed by (z_lower, z_upper) with:
            lower_node, upper_node, dz, max_drift, min_drift, max_abs_drift
        """
        stories = self._resolve_story_nodes_by_z_tol(
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )
        if len(stories) < 2:
            raise ValueError("Need at least 2 story levels after z clustering.")

        # Helper: pick a representative node per story
        def _pick_node(nodes: list[int]) -> int:
            if representative == "min_id":
                return int(min(nodes))
            if representative == "max_abs_peak":
                # choose node with max abs peak displacement component in that story
                s = self.fetch(result_name=result_name, component=component, node_ids=nodes)
                if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                    if stage is None:
                        stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                        raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
                    s = s.xs(str(stage), level=0)
                wide = s.unstack(level=-1)  # rows=node, cols=step
                A = wide.to_numpy(dtype=float)
                peaks = np.nanmax(np.abs(A), axis=1)
                return int(wide.index.to_numpy(dtype=int)[int(np.nanargmax(peaks))])
            raise ValueError(f"Unknown representative='{representative}'")

        rows: list[dict[str, float]] = []
        idx: list[tuple[float, float]] = []

        for (z_lo, nodes_lo), (z_up, nodes_up) in zip(stories[:-1], stories[1:]):
            dz = float(z_up - z_lo)
            if dz == 0.0:
                continue

            n_lo = _pick_node(nodes_lo)
            n_up = _pick_node(nodes_up)

            dr = self.drift(
                top=n_up,
                bottom=n_lo,
                component=component,
                result_name=result_name,
                stage=stage,
                signed=True,
                reduce="series",
            )
            arr = dr.to_numpy(dtype=float)

            rows.append(
                {
                    "lower_node": float(n_lo),
                    "upper_node": float(n_up),
                    "dz": dz,
                    "max_drift": float(np.nanmax(arr)),
                    "min_drift": float(np.nanmin(arr)),
                    "max_abs_drift": float(np.nanmax(np.abs(arr))),
                }
            )
            idx.append((float(z_lo), float(z_up)))

        if not rows:
            raise ValueError("No valid story pairs were produced (check z-coordinates).")

        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=("z_lower", "z_upper")),
        )
        # store node ids as ints (came through float in dict)
        out["lower_node"] = out["lower_node"].astype(int)
        out["upper_node"] = out["upper_node"].astype(int)
        return out


    def story_pga_envelope(
        self,
        *,
        component: object,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "ACCELERATION",
        stage: Optional[str] = None,
        dz_tol: float = 1e-3,
        to_g: bool = False,
        g_value: float = 9.80665,
        reduce_nodes: str = "max_abs",  # "max_abs" | "max" | "min"
    ) -> pd.DataFrame:
        """
        Story acceleration envelope (max, min, pga) using z-tolerance clustering.

        reduce_nodes:
        - "max_abs": pga is max abs over nodes at story (typical)
        - "max":     story peak = max over nodes then over time (positive)
        - "min":     story peak = min over nodes then over time (negative)

        Returns
        -------
        DataFrame indexed by story_z with:
            n_nodes, max_acc, min_acc, pga, ctrl_node_max, ctrl_node_min, ctrl_node_pga
        """
        stories = self._resolve_story_nodes_by_z_tol(
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )

        # all ids (union)
        all_ids: list[int] = sorted({nid for _, nodes in stories for nid in nodes})
        if not all_ids:
            raise ValueError("No nodes resolved.")

        s = self.fetch(result_name=result_name, component=component, node_ids=all_ids)

        # multi-stage -> require stage
        if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
            s = s.xs(str(stage), level=0)

        if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
            raise ValueError("story_pga_envelope() expects index (node_id, step) after stage selection.")

        wide = s.unstack(level=-1)  # rows=node, cols=step
        A = wide.to_numpy(dtype=float)
        node_index = wide.index.to_numpy(dtype=int)

        if to_g:
            A = A / float(g_value)

        # per-node peaks
        max_node = np.nanmax(A, axis=1)
        min_node = np.nanmin(A, axis=1)
        pga_node = np.nanmax(np.abs(A), axis=1)

        # map node -> row
        row_of = {int(n): i for i, n in enumerate(node_index)}

        rows = []
        for z, nodes in stories:
            ridx = np.asarray([row_of[int(n)] for n in nodes if int(n) in row_of], dtype=int)
            if ridx.size == 0:
                continue

            arr_max = max_node[ridx]
            arr_min = min_node[ridx]
            arr_pga = pga_node[ridx]

            i_max = int(np.nanargmax(arr_max))
            i_min = int(np.nanargmin(arr_min))
            i_pga = int(np.nanargmax(arr_pga))

            rows.append(
                {
                    "story_z": float(z),
                    "n_nodes": int(len(nodes)),
                    "max_acc": float(arr_max[i_max]),
                    "min_acc": float(arr_min[i_min]),
                    "pga": float(arr_pga[i_pga]),
                    "ctrl_node_max": int(nodes[i_max]),
                    "ctrl_node_min": int(nodes[i_min]),
                    "ctrl_node_pga": int(nodes[i_pga]),
                }
            )

        if not rows:
            raise ValueError("No story rows produced. Check dz_tol and node selection.")

        return pd.DataFrame(rows).set_index("story_z").sort_index()



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
