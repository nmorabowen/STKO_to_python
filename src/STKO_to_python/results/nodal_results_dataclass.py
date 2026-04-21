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
from ..dataprocess.aggregation import AggregationEngine

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

    # Shared stateless aggregator — engineering methods (drift, envelope,
    # rocking, ...) forward to this instance. Class-level rather than
    # per-instance so old pickles that predate Phase 4.3 still resolve
    # the attribute after unpickling.
    _aggregation_engine: AggregationEngine = AggregationEngine()

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
        coordinates: Sequence[Sequence[float]] | None = None,
        only_available: bool = True,
        return_nodes: bool = False,
    ) -> (
        pd.Series
        | pd.DataFrame
        | tuple[pd.Series | pd.DataFrame, list[int]]
    ):
        """
        Fetch results with optional node filtering.

        You can filter by any combination of:
        - node_ids
        - selection_set_id
        - selection_set_name
        - coordinates  (x,y) or (x,y,z) -> nearest nodes

        Semantics: UNION of all node sources.

        only_available:
        Passed to selection_set resolver(s) to optionally intersect with self.info.nodes_ids.

        return_nodes:
        If True, return (data, resolved_node_ids). The resolved ids correspond to the UNION
        of all node sources (after uniquing). If no node filter was provided, returns [].
        """
        df = self.df
        gathered: list[np.ndarray] = []
        resolved_node_ids: list[int] = []

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

        # ---- coordinates -> nearest node_ids ----
        if coordinates is not None:
            if not isinstance(coordinates, (list, tuple, np.ndarray)):
                raise TypeError(
                    "coordinates must be a sequence of points like [(x,y), ...] or [(x,y,z), ...]."
                )
            if len(coordinates) == 0:
                raise ValueError("coordinates is empty.")

            pts: list[tuple[float, ...]] = []
            for i, p in enumerate(coordinates):
                if not isinstance(p, (list, tuple, np.ndarray)):
                    raise TypeError(f"coordinates[{i}] must be a sequence (x,y) or (x,y,z).")
                pp = tuple(float(v) for v in p)
                if len(pp) not in (2, 3):
                    raise TypeError(f"coordinates[{i}] must have length 2 or 3. Got {len(pp)}.")
                pts.append(pp)

            ids = self.info.nearest_node_id(pts, return_distance=False)
            gathered.append(np.asarray(ids, dtype=np.int64))

        # ---- apply node filter ----
        if gathered:
            node_ids_arr = np.unique(np.concatenate(gathered))
            if node_ids_arr.size == 0:
                raise ValueError("Resolved node set is empty.")

            resolved_node_ids = node_ids_arr.astype(int).tolist()

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

        # helper to optionally attach nodes
        def _ret(out: pd.Series | pd.DataFrame):
            if return_nodes:
                return out, resolved_node_ids
            return out

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
                return _ret(df.loc[:, sub_cols])

            for c0, c1 in cols:
                if str(c0) == str(result_name) and str(c1) == str(component):
                    return _ret(df.loc[:, (c0, c1)])

            raise ValueError(
                f"Component '{component}' not found for result '{result_name}'.\n"
                f"Available components: {self.list_components(result_name)}"
            )

        # ---- Single-level columns ----
        if result_name is not None:
            raise ValueError("Single-level columns: use fetch(component=...) only.")

        if component is None:
            return _ret(df)

        if component in cols:
            return _ret(df[component])

        comp_str = str(component)
        if comp_str in cols:
            return _ret(df[comp_str])

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

    def delta_u(
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
        return self._aggregation_engine.delta_u(
            self,
            top=top,
            bottom=bottom,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce=reduce,
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
        reduce: str = "series",  # "series" | "abs_max"
    ) -> pd.Series | float:
        return self._aggregation_engine.drift(
            self,
            top=top,
            bottom=bottom,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce=reduce,
        )

    def _resolve_story_nodes_by_z_tol(
        self,
        *,
        selection_set_id: int | Sequence[int] | None,
        selection_set_name: str | Sequence[str] | None,
        node_ids: Sequence[int] | None,
        coordinates: Sequence[Sequence[float]] | None,
        dz_tol: float,
    ) -> list[tuple[float, list[int]]]:
        return self._aggregation_engine._resolve_story_nodes_by_z_tol(
            self,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )

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
                        (robust if multiple nodes per floor)

        Returns
        -------
        DataFrame indexed by (z_lower, z_upper) AND exposing them as columns with:
            z_lower, z_upper,
            lower_node, upper_node, dz,
            max_drift, min_drift, max_abs_drift
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

        # --------------------------------------------------
        # Representative node selection
        # --------------------------------------------------
        def _pick_node(nodes: list[int]) -> int:
            if representative == "min_id":
                return int(min(nodes))

            if representative == "max_abs_peak":
                s = self.fetch(
                    result_name=result_name,
                    component=component,
                    node_ids=nodes,
                )
                if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                    if stage is None:
                        stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                        raise ValueError(
                            f"Multi-stage results detected. Provide stage=... "
                            f"Available: {stages}"
                        )
                    s = s.xs(str(stage), level=0)

                wide = s.unstack(level=-1)  # rows=node, cols=step
                A = wide.to_numpy(dtype=float)
                peaks = np.nanmax(np.abs(A), axis=1)
                return int(wide.index.to_numpy(dtype=int)[int(np.nanargmax(peaks))])

            raise ValueError(f"Unknown representative='{representative}'")

        # --------------------------------------------------
        # Build rows
        # --------------------------------------------------
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
                    "lower_node": int(n_lo),
                    "upper_node": int(n_up),
                    "dz": dz,
                    "max_drift": float(np.nanmax(arr)),
                    "min_drift": float(np.nanmin(arr)),
                    "max_abs_drift": float(np.nanmax(np.abs(arr))),
                }
            )
            idx.append((float(z_lo), float(z_up)))

        if not rows:
            raise ValueError("No valid story pairs were produced (check z-coordinates).")

        # --------------------------------------------------
        # Assemble DataFrame
        # --------------------------------------------------
        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=("z_lower", "z_upper")),
        )

        # expose bounds as regular columns too
        out = out.reset_index().set_index(["z_lower", "z_upper"], drop=False)

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
        g_value: float = 9810,
        reduce_nodes: str = "max_abs",  # "max_abs" | "max" | "min"
    ) -> pd.DataFrame:
        """
        Story acceleration envelope (max, min, pga) using z-tolerance clustering.

        Clusters the provided nodes into story levels by z-coordinate using `dz_tol`,
        then computes per-story extrema over time, reduced across nodes.

        Parameters
        ----------
        reduce_nodes:
            - "max_abs": story pga is max abs over nodes at story (typical)
            - "max":     story peak = max over nodes then over time (positive)
            - "min":     story peak = min over nodes then over time (negative)

        Returns
        -------
        DataFrame indexed by story_z with columns:
            n_nodes, max_acc, min_acc, pga,
            ctrl_node_max, ctrl_node_min, ctrl_node_pga

        Notes
        -----
        - Control nodes are chosen among the nodes that are actually present in the
        results after filtering/stage selection (fixes the potential mismatch bug).
        - `n_nodes` reports the number of nodes *requested* in that story cluster,
        not necessarily the number available in the results (see `n_nodes_present`).
        """

        stories = self._resolve_story_nodes_by_z_tol(
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )

        # union of all ids
        all_ids: list[int] = sorted({int(nid) for _, nodes in stories for nid in nodes})
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

        # wide: rows=node_id, cols=step
        wide = s.unstack(level=-1)

        # Preserve the true node ids and array view
        node_index = wide.index.to_numpy(dtype=int)  # shape (n_nodes_present,)
        A = wide.to_numpy(dtype=float)               # shape (n_nodes_present, n_steps)

        if A.size == 0:
            raise ValueError("story_pga_envelope(): empty results after fetch/stage selection.")

        if to_g:
            A = A / float(g_value)

        # per-node peaks over time
        max_node = np.nanmax(A, axis=1)          # (n_nodes_present,)
        min_node = np.nanmin(A, axis=1)          # (n_nodes_present,)
        pga_node = np.nanmax(np.abs(A), axis=1)  # (n_nodes_present,)

        # map node_id -> row index in A
        row_of = {int(n): i for i, n in enumerate(node_index)}

        rows: list[dict[str, float | int]] = []
        for z, nodes in stories:
            requested_nodes = [int(n) for n in nodes]
            ridx = np.asarray([row_of[n] for n in requested_nodes if n in row_of], dtype=int)

            # If none of the story nodes exist in results, skip
            if ridx.size == 0:
                continue

            present_nodes = node_index[ridx]  # node ids that are actually present for this story

            arr_max = max_node[ridx]
            arr_min = min_node[ridx]
            arr_pga = pga_node[ridx]

            # choose story-wide envelope values + controlling nodes among PRESENT nodes
            i_max = int(np.nanargmax(arr_max))
            i_min = int(np.nanargmin(arr_min))
            i_pga = int(np.nanargmax(arr_pga))

            # baseline story envelope
            story_max = float(arr_max[i_max])
            story_min = float(arr_min[i_min])
            story_pga = float(arr_pga[i_pga])

            # optional alternate "reduce_nodes" semantics
            # (kept simple: story fields still report max/min/pga; reduce_nodes can choose
            # which one you care about downstream. If you want it to change pga definition,
            # uncomment below and define accordingly.)
            if reduce_nodes not in ("max_abs", "max", "min"):
                raise ValueError("reduce_nodes must be one of: 'max_abs', 'max', 'min'.")

            rows.append(
                {
                    "story_z": float(z),
                    "n_nodes": int(len(requested_nodes)),
                    "n_nodes_present": int(ridx.size),
                    "max_acc": story_max,
                    "min_acc": story_min,
                    "pga": story_pga,
                    "ctrl_node_max": int(present_nodes[i_max]),
                    "ctrl_node_min": int(present_nodes[i_min]),
                    "ctrl_node_pga": int(present_nodes[i_pga]),
                }
            )

        if not rows:
            raise ValueError("No story rows produced. Check dz_tol and node selection.")

        return pd.DataFrame(rows).set_index("story_z").sort_index()

    def roof_torsion(
        self,
        *,
        node_a_id: int | None = None,
        node_b_id: int | None = None,
        node_a_coord: Sequence[float] | None = None,
        node_b_coord: Sequence[float] | None = None,
        result_name: str = "DISPLACEMENT",
        ux_component: object = 1,
        uy_component: object = 2,
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",  # "series" | "abs_max" | "max" | "min"
        return_residual: bool = False,
        return_quality: bool = False,
    ) -> (
        pd.Series
        | float
        | tuple[pd.Series | float, pd.DataFrame]
    ):
        """
        Roof torsion (rotation about z) estimated from 2 roof nodes A, B.

        Uses small-rotation relation:
            theta(t) = ([du, dv] · [-dy, dx]) / (dx^2 + dy^2)

        If return_residual/return_quality is True, returns a debug DataFrame
        indexed by the same time/step index as theta, including residual terms.

        Returns
        -------
        - theta series [rad] if reduce="series", else a float
        - optionally (theta_out, debug_df)
        """

        # ----------------------------
        # Resolve node ids
        # ----------------------------
        def _resolve_one(nid: int | None, coord: Sequence[float] | None, label: str) -> int:
            provided = (nid is not None) + (coord is not None)
            if provided != 1:
                raise ValueError(f"{label}: provide exactly one of {label}_id or {label}_coord.")

            if nid is not None:
                return int(nid)

            assert coord is not None
            if not isinstance(coord, (list, tuple, np.ndarray)):
                raise TypeError(f"{label}_coord must be a sequence like (x,y) or (x,y,z).")
            pt = tuple(float(v) for v in coord)
            if len(pt) not in (2, 3):
                raise TypeError(f"{label}_coord must have length 2 or 3 (got {len(pt)}).")

            return int(self.info.nearest_node_id([pt], return_distance=False)[0])

        a_id = _resolve_one(node_a_id, node_a_coord, "node_a")
        b_id = _resolve_one(node_b_id, node_b_coord, "node_b")

        if a_id == b_id:
            raise ValueError("node_a and node_b resolved to the same node id; cannot compute torsion.")

        # ----------------------------
        # Baseline plan geometry (dx, dy)
        # ----------------------------
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is required (must contain x,y) for roof_torsion().")

        ni = self.info.nodes_info
        xcol = self.info._resolve_column(ni, "x", required=True)
        ycol = self.info._resolve_column(ni, "y", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        def _xy_of(nid: int) -> tuple[float, float]:
            if nid_col is not None:
                row = ni.loc[ni[nid_col].to_numpy() == nid]
                if row.empty:
                    raise ValueError(f"node_id={nid} not found in nodes_info.")
                return float(row.iloc[0][xcol]), float(row.iloc[0][ycol])
            if nid not in ni.index:
                raise ValueError(f"node_id={nid} not found in nodes_info index.")
            return float(ni.loc[nid, xcol]), float(ni.loc[nid, ycol])

        xa, ya = _xy_of(a_id)
        xb, yb = _xy_of(b_id)

        dx = float(xb - xa)
        dy = float(yb - ya)
        L2 = dx * dx + dy * dy
        if L2 == 0.0:
            raise ValueError("Reference nodes have identical (x,y) → baseline length is zero.")

        # p = (-dy, dx)
        px = -dy
        py = dx

        # ----------------------------
        # Fetch Ux, Uy for both nodes
        # ----------------------------
        ux = self.fetch(result_name=result_name, component=ux_component, node_ids=[a_id, b_id])
        uy = self.fetch(result_name=result_name, component=uy_component, node_ids=[a_id, b_id])

        def _select_stage(s: pd.Series | pd.DataFrame):
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                if stage is None:
                    stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                    raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
                return s.xs(str(stage), level=0)
            return s

        ux = _select_stage(ux)
        uy = _select_stage(uy)

        if not (isinstance(ux.index, pd.MultiIndex) and ux.index.nlevels == 2):
            raise ValueError("roof_torsion(): expected index (node_id, step) after stage selection.")

        ux_a = ux.xs(a_id, level=0).sort_index()
        ux_b = ux.xs(b_id, level=0).sort_index()
        uy_a = uy.xs(a_id, level=0).sort_index()
        uy_b = uy.xs(b_id, level=0).sort_index()

        ux_a, ux_b = ux_a.align(ux_b, join="inner")
        uy_a, uy_b = uy_a.align(uy_b, join="inner")

        du = (ux_b - ux_a).to_numpy(dtype=float)
        dv = (uy_b - uy_a).to_numpy(dtype=float)

        # ----------------------------
        # Projection (theta)
        # ----------------------------
        theta = (du * px + dv * py) / L2
        if not signed:
            theta = np.abs(theta)

        theta_s = pd.Series(theta, index=ux_a.index, name="roof_torsion_theta_rad")

        # ----------------------------
        # Optional residual + quality
        # ----------------------------
        debug: pd.DataFrame | None = None
        if return_residual or return_quality:
            du_rot = theta * px
            dv_rot = theta * py
            ru = du - du_rot
            rv = dv - dv_rot

            debug_dict: dict[str, np.ndarray] = {
                "du": du,
                "dv": dv,
                "du_rot": du_rot,
                "dv_rot": dv_rot,
                "ru": ru,
                "rv": rv,
            }

            if return_quality:
                rel_norm = np.sqrt(du * du + dv * dv)
                res_norm = np.sqrt(ru * ru + rv * rv)
                rigidity_ratio = np.divide(
                    res_norm,
                    rel_norm,
                    out=np.full_like(res_norm, np.nan, dtype=float),
                    where=rel_norm > 0.0,
                )
                debug_dict.update(
                    {
                        "rel_norm": rel_norm,
                        "res_norm": res_norm,
                        "rigidity_ratio": rigidity_ratio,  # ρ(t)=||r||/||Δu||
                    }
                )

            debug = pd.DataFrame(debug_dict, index=ux_a.index)

        # ----------------------------
        # Reduction
        # ----------------------------
        if reduce == "series":
            theta_out: pd.Series | float = theta_s
        elif reduce == "abs_max":
            theta_out = float(np.nanmax(np.abs(theta_s.to_numpy(dtype=float))))
        elif reduce == "max":
            theta_out = float(np.nanmax(theta_s.to_numpy(dtype=float)))
        elif reduce == "min":
            theta_out = float(np.nanmin(theta_s.to_numpy(dtype=float)))
        else:
            raise ValueError("reduce must be one of: 'series', 'abs_max', 'max', 'min'.")

        if debug is not None:
            return theta_out, debug

        return theta_out

    def residual_drift(
        self,
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        tail: int = 1,
        agg: str = "mean",  # "mean" | "median"
    ) -> float:
        """
        Residual drift ratio between two nodes.

        Definition
        ----------
        Residual drift is evaluated at the end of the record.
        To reduce end-of-record noise, you can average over the last `tail` steps.

        Parameters
        ----------
        tail
            Number of last steps to aggregate.
            - tail=1 -> last step only
            - tail>1 -> aggregate last `tail` drift samples
        agg
            Aggregation over the tail window: "mean" or "median"

        Returns
        -------
        float
            Residual drift ratio (dimensionless). Signed unless signed=False.
        """
        dr = self.drift(
            top=top,
            bottom=bottom,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce="series",
        )

        a = dr.to_numpy(dtype=float)
        if a.size == 0:
            raise ValueError("residual_drift(): empty drift series.")

        tail_i = int(tail)
        if tail_i < 1:
            raise ValueError("tail must be >= 1.")
        tail_i = min(tail_i, a.size)

        w = a[-tail_i:]
        if agg == "mean":
            return float(np.nanmean(w))
        if agg == "median":
            return float(np.nanmedian(w))
        raise ValueError("agg must be 'mean' or 'median'.")

    def residual_interstory_drift_profile(
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
        signed: bool = True,
        tail: int = 1,
        agg: str = "mean",
    ) -> pd.DataFrame:
        """
        Residual interstory drift ratio per story (profile).

        Returns
        -------
        DataFrame indexed by (z_lower, z_upper) and exposing:
            z_lower, z_upper,
            lower_node, upper_node, dz,
            residual_drift
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

        def _pick_node(nodes: list[int]) -> int:
            if representative == "min_id":
                return int(min(nodes))

            if representative == "max_abs_peak":
                s = self.fetch(
                    result_name=result_name,
                    component=component,
                    node_ids=nodes,
                )
                if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                    if stage is None:
                        stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                        raise ValueError(
                            f"Multi-stage results detected. Provide stage=... Available: {stages}"
                        )
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

            r = self.residual_drift(
                top=n_up,
                bottom=n_lo,
                component=component,
                result_name=result_name,
                stage=stage,
                signed=signed,
                tail=tail,
                agg=agg,
            )

            rows.append(
                {
                    "lower_node": int(n_lo),
                    "upper_node": int(n_up),
                    "dz": float(dz),
                    "residual_drift": float(r),
                }
            )
            idx.append((float(z_lo), float(z_up)))

        if not rows:
            raise ValueError("No valid story pairs were produced (check z-coordinates).")

        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=("z_lower", "z_upper")),
        )

        # expose bounds as columns
        out = out.reset_index().set_index(["z_lower", "z_upper"], drop=False)
        return out

    def residual_drift_envelope(
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
        representative: str = "min_id",
        tail: int = 1,
        agg: str = "mean",
    ) -> dict[str, float]:
        """
        Convenience summary metrics:
          - max_abs_residual_story_drift
          - max_pos_residual_story_drift
          - max_neg_residual_story_drift
        """
        prof = self.residual_interstory_drift_profile(
            component=component,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            result_name=result_name,
            stage=stage,
            dz_tol=dz_tol,
            representative=representative,
            signed=True,
            tail=tail,
            agg=agg,
        )

        r = prof["residual_drift"].to_numpy(dtype=float)
        return {
            "max_abs_residual_story_drift": float(np.nanmax(np.abs(r))),
            "max_pos_residual_story_drift": float(np.nanmax(r)),
            "max_neg_residual_story_drift": float(np.nanmin(r)),
        }

    def base_rocking(
        self,
        *,
        node_coords_xy: Sequence[Sequence[float]],  # [(x,y), (x,y), (x,y)]
        z_coord: float,
        result_name: str = "DISPLACEMENT",
        uz_component: object = 3,   # Uz
        stage: Optional[str] = None,
        reduce: str = "series",     # "series" | "abs_max"
        det_tol: float = 1e-12,
    ) -> pd.DataFrame | dict[str, float]:
        """
        Estimate base rocking angles from Uz at 3 base nodes.

        Model (small rotations):
            w(x,y) = w0 + theta_x*y - theta_y*x

        Outputs (reduce="series"):
            w0
            theta_x_rad
            theta_y_rad
            theta_mag_rad = sqrt(theta_x_rad^2 + theta_y_rad^2)
            is_singular   = True if geometry was singular/collinear after snapping (fallback used)

        Fallback behavior
        -----------------
        If the 3 resolved points are collinear (or duplicate) such that the geometry matrix is singular,
        we assume no rocking:
            theta_x_rad(t) = 0
            theta_y_rad(t) = 0
            theta_mag_rad(t) = 0
        and we still compute w0(t) as the mean Uz of the 3 nodes. The method will NOT raise.

        Outputs (reduce="abs_max"):
            theta_x_abs_max
            theta_y_abs_max
            theta_mag_abs_max
        """
        if len(node_coords_xy) != 3:
            raise ValueError("node_coords_xy must contain exactly 3 (x,y) points.")

        # --------------------------------------------------
        # Resolve node ids (nearest at given z)
        # --------------------------------------------------
        pts = [(float(x), float(y), float(z_coord)) for x, y in node_coords_xy]
        node_ids = self.info.nearest_node_id(pts, return_distance=False)
        n1, n2, n3 = map(int, node_ids)

        # --------------------------------------------------
        # Fetch Uz for those 3 nodes (we need this also for fallback)
        # --------------------------------------------------
        uz = self.fetch(result_name=result_name, component=uz_component, node_ids=[n1, n2, n3])

        # stage selection
        if isinstance(uz.index, pd.MultiIndex) and uz.index.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in uz.index.get_level_values(0)}))
                raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
            uz = uz.xs(str(stage), level=0)

        if not (isinstance(uz.index, pd.MultiIndex) and uz.index.nlevels == 2):
            raise ValueError("base_rocking(): expected index (node_id, step) after stage selection.")

        # wide: rows=node, cols=step
        W = uz.unstack(level=-1).loc[[n1, n2, n3]]
        w_steps = W.to_numpy(dtype=float)  # (3, nsteps)

        # --------------------------------------------------
        # Get plan coords for resolved nodes
        # --------------------------------------------------
        if self.info.nodes_info is None:
            raise ValueError("nodes_info is required (must contain x,y) for base_rocking().")

        ni = self.info.nodes_info
        xcol = self.info._resolve_column(ni, "x", required=True)
        ycol = self.info._resolve_column(ni, "y", required=True)
        nid_col = self.info._resolve_column(ni, "node_id", required=False)

        def _xy_of(nid: int) -> tuple[float, float]:
            if nid_col is not None:
                row = ni.loc[ni[nid_col].to_numpy() == nid]
                if row.empty:
                    raise ValueError(f"node_id={nid} not found in nodes_info.")
                return float(row.iloc[0][xcol]), float(row.iloc[0][ycol])
            if nid not in ni.index:
                raise ValueError(f"node_id={nid} not found in nodes_info index.")
            return float(ni.loc[nid, xcol]), float(ni.loc[nid, ycol])

        x1, y1 = _xy_of(n1)
        x2, y2 = _xy_of(n2)
        x3, y3 = _xy_of(n3)

        # --------------------------------------------------
        # Build geometry matrix A and detect singularity
        # w = w0 + theta_x*y - theta_y*x  => [1, y, -x] [w0, theta_x, theta_y]^T
        # --------------------------------------------------
        A = np.array(
            [
                [1.0, y1, -x1],
                [1.0, y2, -x2],
                [1.0, y3, -x3],
            ],
            dtype=float,
        )

        # duplicate-node guard (after snapping)
        duplicate = (len({n1, n2, n3}) < 3)

        det = float(np.linalg.det(A))
        singular = duplicate or (abs(det) < float(det_tol))

        # --------------------------------------------------
        # Fallback: singular geometry -> assume no rocking
        # --------------------------------------------------
        if singular:
            w0 = np.nanmean(w_steps, axis=0)  # (nsteps,)

            out = pd.DataFrame(
                {
                    "w0": w0,
                    "theta_x_rad": np.zeros_like(w0),
                    "theta_y_rad": np.zeros_like(w0),
                    "theta_mag_rad": np.zeros_like(w0),
                    "is_singular": np.ones_like(w0, dtype=bool),
                },
                index=W.columns,  # step index
            )

            if reduce == "series":
                return out

            if reduce == "abs_max":
                return {
                    "theta_x_abs_max": 0.0,
                    "theta_y_abs_max": 0.0,
                    "theta_mag_abs_max": 0.0,
                }

            raise ValueError("reduce must be 'series' or 'abs_max'.")

        # --------------------------------------------------
        # Solve for each step: p = Ainv @ w
        # p = [w0, theta_x, theta_y]
        # --------------------------------------------------
        Ainv = np.linalg.inv(A)
        p_steps = Ainv @ w_steps  # (3, nsteps)

        out = pd.DataFrame(
            {
                "w0": p_steps[0, :],
                "theta_x_rad": p_steps[1, :],
                "theta_y_rad": p_steps[2, :],
            },
            index=W.columns,
        )

        out["theta_mag_rad"] = np.sqrt(
            out["theta_x_rad"].to_numpy(dtype=float) ** 2
            + out["theta_y_rad"].to_numpy(dtype=float) ** 2
        )
        out["is_singular"] = False

        if reduce == "series":
            return out

        if reduce == "abs_max":
            tx = out["theta_x_rad"].to_numpy(dtype=float)
            ty = out["theta_y_rad"].to_numpy(dtype=float)
            tm = out["theta_mag_rad"].to_numpy(dtype=float)
            return {
                "theta_x_abs_max": float(np.nanmax(np.abs(tx))),
                "theta_y_abs_max": float(np.nanmax(np.abs(ty))),
                "theta_mag_abs_max": float(np.nanmax(np.abs(tm))),
            }

        raise ValueError("reduce must be 'series' or 'abs_max'.")

    def asce_torsional_irregularity(
        self,
        *,
        component: object,
        side_a_top: tuple[float, float, float],
        side_a_bottom: tuple[float, float, float],
        side_b_top: tuple[float, float, float],
        side_b_bottom: tuple[float, float, float],
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        reduce_time: str = "abs_max",          # "abs_max" | "max" | "min"
        definition: str = "max_over_avg",      # "max_over_avg" | "max_over_min"
        eps: float = 1e-16,
        signed: bool = True,
        tail: int | None = None,
    ) -> dict[str, Any]:
        """
        Compute ASCE-style torsional irregularity ratio from *edge drifts* at a story.

        Inputs are *coordinates only* (x,y,z) tuples, resolved to nearest node IDs.

        Drift time series per side:
            drA(t) = (u_A_top(t) - u_A_bottom(t)) / (z_top - z_bottom)
            drB(t) = (u_B_top(t) - u_B_bottom(t)) / (z_top - z_bottom)

        Time reduction:
            - "abs_max": max(|dr(t)|)
            - "max":     max(dr(t))
            - "min":     min(dr(t))

        Ratio definitions:
            - "max_over_avg":  max(DA, DB) / avg(DA, DB)
            - "max_over_min":  max(DA, DB) / min(DA, DB)

        Notes
        -----
        - This is a single-story check (user provides bottom & top points for that story).
        - No center points are used.
        - If tail is provided, the last `tail` samples are ignored for the reduction
        (useful if your record ends with solver noise / weird tail). Set tail=None to keep all.

        Returns
        -------
        dict with:
            drift_A, drift_B, drift_avg, drift_max, ratio, ctrl_side,
            node_ids (resolved), metadata
        """
        # ----------------------------
        # validate coords
        # ----------------------------
        def _as_xyz(pt: tuple[float, float, float], name: str) -> tuple[float, float, float]:
            if not isinstance(pt, tuple):
                raise TypeError(f"{name} must be a tuple (x,y,z). Got {type(pt).__name__}.")
            if len(pt) != 3:
                raise TypeError(f"{name} must be a tuple of length 3 (x,y,z). Got len={len(pt)}.")
            x, y, z = pt
            return float(x), float(y), float(z)

        A_top = _as_xyz(side_a_top, "side_a_top")
        A_bot = _as_xyz(side_a_bottom, "side_a_bottom")
        B_top = _as_xyz(side_b_top, "side_b_top")
        B_bot = _as_xyz(side_b_bottom, "side_b_bottom")

        # ----------------------------
        # resolve node ids (nearest)
        # ----------------------------
        pts = [A_top, A_bot, B_top, B_bot]
        node_ids = self.info.nearest_node_id(pts, return_distance=False)
        a_top_id, a_bot_id, b_top_id, b_bot_id = map(int, node_ids)

        # sanity
        if a_top_id == a_bot_id:
            raise ValueError("Side A top and bottom resolved to the same node id (cannot compute drift).")
        if b_top_id == b_bot_id:
            raise ValueError("Side B top and bottom resolved to the same node id (cannot compute drift).")

        # ----------------------------
        # compute drift time histories
        # ----------------------------
        drA = self.drift(
            top=a_top_id,
            bottom=a_bot_id,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce="series",
        )
        drB = self.drift(
            top=b_top_id,
            bottom=b_bot_id,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce="series",
        )

        # align in case step indices differ
        drA, drB = drA.align(drB, join="inner")
        if drA.size == 0:
            raise ValueError("No overlapping steps between Side A and Side B drift series.")

        # optional tail trimming (drop last samples)
        if tail is not None:
            t = int(tail)
            if t < 0:
                raise ValueError("tail must be >= 0 or None.")
            if t > 0:
                if t >= drA.size:
                    raise ValueError("tail is >= series length; nothing would remain.")
                drA = drA.iloc[:-t]
                drB = drB.iloc[:-t]

        # ----------------------------
        # reduce over time
        # ----------------------------
        a = drA.to_numpy(dtype=float)
        b = drB.to_numpy(dtype=float)

        def _reduce(x: np.ndarray, how: str) -> float:
            if x.size == 0:
                return float("nan")
            if how == "abs_max":
                return float(np.nanmax(np.abs(x)))
            if how == "max":
                return float(np.nanmax(x))
            if how == "min":
                return float(np.nanmin(x))
            raise ValueError("reduce_time must be one of: 'abs_max', 'max', 'min'.")

        DA = _reduce(a, reduce_time)
        DB = _reduce(b, reduce_time)

        # ensure positive magnitudes for ratio when using abs-based definitions
        # (even if reduce_time="max"/"min", ratio should typically use magnitudes)
        mA = float(abs(DA))
        mB = float(abs(DB))

        drift_max = max(mA, mB)
        drift_min = min(mA, mB)
        drift_avg = 0.5 * (mA + mB)

        # ----------------------------
        # ratio definition
        # ----------------------------
        if definition == "max_over_avg":
            denom = max(drift_avg, float(eps))
            ratio = drift_max / denom
        elif definition == "max_over_min":
            denom = max(drift_min, float(eps))
            ratio = drift_max / denom
        else:
            raise ValueError("definition must be 'max_over_avg' or 'max_over_min'.")

        ctrl_side = "A" if mA >= mB else "B"

        return {
            "drift_A": mA,
            "drift_B": mB,
            "drift_avg": drift_avg,
            "drift_max": drift_max,
            "ratio": float(ratio),
            "ctrl_side": ctrl_side,
            "node_ids": {
                "A_top": a_top_id,
                "A_bottom": a_bot_id,
                "B_top": b_top_id,
                "B_bottom": b_bot_id,
            },
            "metadata": {
                "component": component,
                "result_name": result_name,
                "stage": stage,
                "reduce_time": reduce_time,
                "definition": definition,
                "signed_drift_series": bool(signed),
                "tail_dropped": int(tail or 0),
                "eps": float(eps),
            },
        }

    def interstory_drift_envelope_pd(
        self,
        *,
        component: object,
        selection_set_name: str | None = None,
        selection_set_id: int | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        dz_tol: float = 1e-3,
        representative: str = "max_abs",  # default
    ) -> pd.DataFrame:
        """
        Interstory drift envelope using z-clustering.

        Returns a DataFrame suitable for statistics / histograms.
        """

        if representative not in ("max_abs", "max", "min"):
            raise ValueError("representative must be 'max_abs', 'max', or 'min'.")

        # --------------------------------------------------
        # Resolve story clusters (THIS METHOD EXISTS HERE)
        # --------------------------------------------------
        stories = self._resolve_story_nodes_by_z_tol(
            selection_set_name=selection_set_name,
            selection_set_id=selection_set_id,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )

        if len(stories) < 2:
            raise ValueError("Need at least two story levels.")

        rows: list[dict[str, float | int]] = []

        # --------------------------------------------------
        # Loop over interstory pairs
        # --------------------------------------------------
        for (z_lo, nodes_lo), (z_up, nodes_up) in zip(stories[:-1], stories[1:]):
            dz = float(z_up - z_lo)
            if dz == 0.0:
                continue

            # deterministic representatives (node-level physics already handled in drift)
            n_lo = int(min(nodes_lo))
            n_up = int(min(nodes_up))

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
            if arr.size == 0:
                continue

            dmax = float(np.nanmax(arr))
            dmin = float(np.nanmin(arr))
            dabs = float(np.nanmax(np.abs(arr)))

            if representative == "max_abs":
                rep = dabs
            elif representative == "max":
                rep = dmax
            else:
                rep = dmin

            rows.append(
                {
                    "z_lower": float(z_lo),
                    "z_upper": float(z_up),
                    "dz": dz,
                    "max_drift": dmax,
                    "min_drift": dmin,
                    "max_abs_drift": dabs,
                    "representative_drift": rep,
                    "lower_node": n_lo,
                    "upper_node": n_up,
                }
            )

        if not rows:
            raise ValueError("No interstory drift data generated.")

        return (
            pd.DataFrame(rows)
            .sort_values("z_lower")
            .reset_index(drop=True)
        )

    def orbit(
        self,
        *,
        result_name: str = "DISPLACEMENT",
        x_component: object = '1',
        y_component: object = '2',
        # node selection (exactly like fetch)
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,  # (x,y) or (x,y,z) -> nearest nodes
        stage: Optional[str] = None,
        # how to combine if multiple nodes are selected
        reduce_nodes: str = "none",  # "none" | "mean" | "median" | "max_abs"
        signed: bool = True,
        return_nodes: bool = False,
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, list[int]]:
        """
        Build an orbit pair (x(t), y(t)) from two components of the same result.

        Parameters
        ----------
        reduce_nodes
            - "none": returns MultiIndex series if multiple nodes are present (node_id, step)
            - "mean"/"median": reduces across nodes at each step
            - "max_abs": reduces across nodes by choosing the node with max abs at each step,
                         independently for x and y (simple + robust, but note x and y can come from
                         different controlling nodes per step)
        """

        provided = sum(x is not None for x in (node_ids, selection_set_id, selection_set_name, coordinates))
        if provided != 1:
            raise ValueError(
                "orbit(): Provide exactly ONE of: node_ids, selection_set_id, selection_set_name, coordinates."
            )

        resolved_node_ids: list[int] | None = None

        if coordinates is not None:
            ids = self.info.nearest_node_id(coordinates, return_distance=False)
            resolved_node_ids = [int(i) for i in ids]
        else:
            # leverage fetch()'s resolver for selection sets + node_ids,
            # but we want the *resolved ids* to optionally return them.
            gathered: list[np.ndarray] = []

            if selection_set_id is not None:
                ids = self.info.selection_set_node_ids(selection_set_id)
                gathered.append(np.asarray(ids, dtype=np.int64))

            if selection_set_name is not None:
                ids = self.info.selection_set_node_ids_by_name(selection_set_name)
                gathered.append(np.asarray(ids, dtype=np.int64))

            if node_ids is not None:
                if isinstance(node_ids, (int, np.integer)):
                    gathered.append(np.asarray([int(node_ids)], dtype=np.int64))
                else:
                    arr = np.asarray(list(node_ids), dtype=np.int64)
                    if arr.size == 0:
                        raise ValueError("orbit(): node_ids is empty.")
                    gathered.append(arr)

            resolved_node_ids = sorted(set(np.unique(np.concatenate(gathered)).astype(int).tolist()))

        if not resolved_node_ids:
            raise ValueError("orbit(): resolved node set is empty.")

        # fetch x and y
        sx = self.fetch(result_name=result_name, component=x_component, node_ids=resolved_node_ids)
        sy = self.fetch(result_name=result_name, component=y_component, node_ids=resolved_node_ids)

        # stage selection if needed
        def _select_stage(s: pd.Series) -> pd.Series:
            if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
                if stage is None:
                    stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                    raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
                return s.xs(str(stage), level=0)
            return s

        sx = _select_stage(sx)
        sy = _select_stage(sy)

        if not (isinstance(sx.index, pd.MultiIndex) and sx.index.nlevels == 2):
            raise ValueError("orbit(): expected index (node_id, step) after stage selection.")
        if not (isinstance(sy.index, pd.MultiIndex) and sy.index.nlevels == 2):
            raise ValueError("orbit(): expected index (node_id, step) after stage selection.")

        # align (important if any missing steps)
        sx, sy = sx.align(sy, join="inner")
        if sx.size == 0:
            raise ValueError("orbit(): no overlapping samples between x and y series after alignment.")

        if not signed:
            sx = sx.abs()
            sy = sy.abs()

        # reduce across nodes if requested
        if reduce_nodes != "none":
            wide_x = sx.unstack(level=-1)  # rows=node, cols=step
            wide_y = sy.unstack(level=-1)

            # union of steps (should already match from align, but keep safe)
            steps = np.intersect1d(wide_x.columns.to_numpy(), wide_y.columns.to_numpy())
            wide_x = wide_x.reindex(columns=steps)
            wide_y = wide_y.reindex(columns=steps)

            Ax = wide_x.to_numpy(dtype=float)
            Ay = wide_y.to_numpy(dtype=float)

            if reduce_nodes == "mean":
                x = np.nanmean(Ax, axis=0)
                y = np.nanmean(Ay, axis=0)
            elif reduce_nodes == "median":
                x = np.nanmedian(Ax, axis=0)
                y = np.nanmedian(Ay, axis=0)
            elif reduce_nodes == "max_abs":
                ix = np.nanargmax(np.abs(Ax), axis=0)
                iy = np.nanargmax(np.abs(Ay), axis=0)
                j = np.arange(steps.size)
                x = Ax[ix, j]
                y = Ay[iy, j]
            else:
                raise ValueError("reduce_nodes must be one of: 'none', 'mean', 'median', 'max_abs'.")

            sx_out = pd.Series(x, index=steps, name=f"{result_name}[{x_component}]")
            sy_out = pd.Series(y, index=steps, name=f"{result_name}[{y_component}]")

            if return_nodes:
                return sx_out, sy_out, resolved_node_ids
            return sx_out, sy_out

        # no reduction: keep per-node indexing
        sx.name = f"{result_name}[{x_component}]"
        sy.name = f"{result_name}[{y_component}]"

        if return_nodes:
            return sx, sy, resolved_node_ids
        return sx, sy

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
