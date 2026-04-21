"""AggregationEngine — engineering aggregations over NodalResults.

This module owns the seismic-engineering aggregations (drift, envelope,
residual drift, rocking, torsional irregularity, orbit) that NodalResults
historically carried as methods. See docs/architecture-refactor-proposal.md
§4.3. The engine is introduced as a skeleton in Phase 4.3.1; real bodies
move in one method per commit in Phase 4.3.2. NodalResults keeps every
public method as a thin forwarder.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence

import pandas as pd

if TYPE_CHECKING:
    from ..results.nodal_results_dataclass import NodalResults


class AggregationEngine:
    """Engineering aggregations over a NodalResults view.

    Methods take the NodalResults instance as the first positional argument
    and otherwise mirror the original NodalResults method signatures so that
    ``NodalResults.drift(...)`` can forward to
    ``self._aggregation_engine.drift(self, ...)`` without argument renaming.

    The engine holds no dataset state; it is safe to share a single instance
    across many NodalResults objects.
    """

    __slots__ = ()

    def __init__(self) -> None:
        # No state today. Reserved for future caches (e.g. resolved story
        # clusters keyed by (selection, dz_tol)).
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    # ------------------------------------------------------------------ #
    # Internal helper (engineering-only, moves here from NodalResults)
    # ------------------------------------------------------------------ #

    def _resolve_story_nodes_by_z_tol(
        self,
        results: "NodalResults",
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
            ids = results.info.selection_set_node_ids(selection_set_id)
        elif selection_set_name is not None:
            ids = results.info.selection_set_node_ids_by_name(selection_set_name)
        elif node_ids is not None:
            if len(node_ids) == 0:
                raise ValueError("node_ids is empty.")
            ids = [int(i) for i in node_ids]
        else:
            assert coordinates is not None
            if len(coordinates) == 0:
                raise ValueError("coordinates is empty.")
            ids = results.info.nearest_node_id(coordinates, return_distance=False)

        ids = sorted(set(int(i) for i in ids))
        if len(ids) == 0:
            raise ValueError("Resolved node list is empty.")

        if results.info.nodes_info is None:
            raise ValueError("nodes_info is None. Need nodes_info with z-coordinates.")
        ni = results.info.nodes_info
        zcol = results.info._resolve_column(ni, "z", required=True)
        nid_col = results.info._resolve_column(ni, "node_id", required=False)

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

    # ------------------------------------------------------------------ #
    # Pair-wise drift utilities
    # ------------------------------------------------------------------ #

    def delta_u(
        self,
        results: "NodalResults",
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",
    ) -> pd.Series | float:
        """
        Relative displacement between two nodes:

            du(t) = u_top(t) - u_bottom(t)

        top, bottom:
            - node id (int), or
            - coordinates (x,y) or (x,y,z) resolved to nearest node.
        """
        import numpy as np  # local import keeps module import cheap

        def _as_node_id(v: int | Sequence[float], *, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)
            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(f"{name} must be a node id or coordinates (x,y) or (x,y,z).")
            coords = tuple(float(x) for x in v)
            if len(coords) not in (2, 3):
                raise TypeError(f"{name} coordinates must have length 2 or 3. Got {len(coords)}.")
            return int(results.info.nearest_node_id([coords])[0])

        top_id = _as_node_id(top, name="top")
        bot_id = _as_node_id(bottom, name="bottom")

        s = results.fetch(result_name=result_name, component=component, node_ids=[top_id, bot_id])

        # multi-stage -> require stage
        if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
            s = s.xs(str(stage), level=0)

        if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
            raise ValueError("delta_u() expects index (node_id, step) after stage selection.")

        u_top = s.xs(top_id, level=0).sort_index()
        u_bot = s.xs(bot_id, level=0).sort_index()
        u_top, u_bot = u_top.align(u_bot, join="inner")

        du = u_top - u_bot
        if not signed:
            du = du.abs()

        du.name = f"delta_u({result_name}:{component})"

        if reduce == "series":
            return du
        if reduce == "abs_max":
            return float(np.nanmax(np.abs(du.to_numpy(dtype=float))))
        raise ValueError(f"Unknown reduce='{reduce}'. Use 'series' or 'abs_max'.")

    def drift(
        self,
        results: "NodalResults",
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        reduce: str = "series",
    ) -> pd.Series | float:
        """
        Drift between two nodes:

            drift(t) = (u_top(t) - u_bottom(t)) / (z_top - z_bottom)

        top, bottom:
            - node id (int), or
            - coordinates (x,y) or (x,y,z) resolved to nearest node.
        """
        import numpy as np

        def _as_node_id(v: int | Sequence[float], *, name: str) -> int:
            if isinstance(v, (int, np.integer)):
                return int(v)

            if not isinstance(v, (list, tuple, np.ndarray)):
                raise TypeError(f"{name} must be a node id or coordinates (x,y) or (x,y,z).")

            coords = tuple(float(x) for x in v)
            if len(coords) not in (2, 3):
                raise TypeError(f"{name} coordinates must have length 2 or 3. Got {len(coords)}.")

            return int(results.info.nearest_node_id([coords])[0])

        top_id = _as_node_id(top, name="top")
        bot_id = _as_node_id(bottom, name="bottom")

        # ---- z coords ----
        if results.info.nodes_info is None:
            raise ValueError("nodes_info is None. z-coordinates are required for drift().")
        ni = results.info.nodes_info
        zcol = results.info._resolve_column(ni, "z", required=True)
        nid_col = results.info._resolve_column(ni, "node_id", required=False)

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
        s = results.fetch(result_name=result_name, component=component, node_ids=[top_id, bot_id])

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

    def residual_drift(
        self,
        results: "NodalResults",
        *,
        top: int | Sequence[float],
        bottom: int | Sequence[float],
        component: object,
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        signed: bool = True,
        tail: int = 1,
        agg: str = "mean",
    ) -> float:
        """
        Residual drift ratio between two nodes, evaluated at the end of the
        record. ``tail`` aggregates the last N drift samples to reduce
        end-of-record noise; ``agg`` chooses mean/median across that window.
        """
        import numpy as np

        dr = self.drift(
            results,
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

    # ------------------------------------------------------------------ #
    # Story profiles / envelopes
    # ------------------------------------------------------------------ #

    def interstory_drift_envelope(
        self,
        results: "NodalResults",
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
    ) -> pd.DataFrame:
        """
        Interstory drift envelope (MAX and MIN signed) using z-tolerance
        clustering. ``representative`` picks which node at each story
        carries the drift — "min_id" for deterministic picks,
        "max_abs_peak" to choose the node with the largest |response|.
        """
        import numpy as np

        stories = self._resolve_story_nodes_by_z_tol(
            results,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )
        if len(stories) < 2:
            raise ValueError("Need at least 2 story levels after z clustering.")

        # Representative node selection
        def _pick_node(nodes: list[int]) -> int:
            if representative == "min_id":
                return int(min(nodes))

            if representative == "max_abs_peak":
                s = results.fetch(
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

                wide = s.unstack(level=-1)
                A = wide.to_numpy(dtype=float)
                peaks = np.nanmax(np.abs(A), axis=1)
                return int(wide.index.to_numpy(dtype=int)[int(np.nanargmax(peaks))])

            raise ValueError(f"Unknown representative='{representative}'")

        # Build rows
        rows: list[dict[str, float]] = []
        idx: list[tuple[float, float]] = []

        for (z_lo, nodes_lo), (z_up, nodes_up) in zip(stories[:-1], stories[1:]):
            dz = float(z_up - z_lo)
            if dz == 0.0:
                continue

            n_lo = _pick_node(nodes_lo)
            n_up = _pick_node(nodes_up)

            dr = self.drift(
                results,
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

        out = pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_tuples(idx, names=("z_lower", "z_upper")),
        )

        # expose bounds as regular columns too
        out = out.reset_index().set_index(["z_lower", "z_upper"], drop=False)

        return out

    def interstory_drift_envelope_pd(
        self,
        results: "NodalResults",
        *,
        component: object,
        selection_set_name: str | None = None,
        selection_set_id: int | None = None,
        node_ids: Sequence[int] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        result_name: str = "DISPLACEMENT",
        stage: str | None = None,
        dz_tol: float = 1e-3,
        representative: str = "max_abs",
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AggregationEngine.interstory_drift_envelope_pd not implemented yet; "
            "filled in Phase 4.3.2."
        )

    def story_pga_envelope(
        self,
        results: "NodalResults",
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
        reduce_nodes: str = "max_abs",
    ) -> pd.DataFrame:
        """
        Story acceleration envelope (max, min, pga) using z-tolerance
        clustering. Controlling nodes are chosen among the nodes that
        are actually present in the results after filtering + stage
        selection (so ``n_nodes_present`` can be smaller than
        ``n_nodes``).
        """
        import numpy as np

        stories = self._resolve_story_nodes_by_z_tol(
            results,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            node_ids=node_ids,
            coordinates=coordinates,
            dz_tol=dz_tol,
        )

        all_ids: list[int] = sorted({int(nid) for _, nodes in stories for nid in nodes})
        if not all_ids:
            raise ValueError("No nodes resolved.")

        s = results.fetch(result_name=result_name, component=component, node_ids=all_ids)

        if isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 3:
            if stage is None:
                stages = tuple(sorted({str(x) for x in s.index.get_level_values(0)}))
                raise ValueError(f"Multi-stage results detected. Provide stage=... Available: {stages}")
            s = s.xs(str(stage), level=0)

        if not (isinstance(s.index, pd.MultiIndex) and s.index.nlevels == 2):
            raise ValueError("story_pga_envelope() expects index (node_id, step) after stage selection.")

        wide = s.unstack(level=-1)

        node_index = wide.index.to_numpy(dtype=int)
        A = wide.to_numpy(dtype=float)

        if A.size == 0:
            raise ValueError("story_pga_envelope(): empty results after fetch/stage selection.")

        if to_g:
            A = A / float(g_value)

        max_node = np.nanmax(A, axis=1)
        min_node = np.nanmin(A, axis=1)
        pga_node = np.nanmax(np.abs(A), axis=1)

        row_of = {int(n): i for i, n in enumerate(node_index)}

        rows: list[dict[str, float | int]] = []
        for z, nodes in stories:
            requested_nodes = [int(n) for n in nodes]
            ridx = np.asarray([row_of[n] for n in requested_nodes if n in row_of], dtype=int)

            if ridx.size == 0:
                continue

            present_nodes = node_index[ridx]

            arr_max = max_node[ridx]
            arr_min = min_node[ridx]
            arr_pga = pga_node[ridx]

            i_max = int(np.nanargmax(arr_max))
            i_min = int(np.nanargmin(arr_min))
            i_pga = int(np.nanargmax(arr_pga))

            story_max = float(arr_max[i_max])
            story_min = float(arr_min[i_min])
            story_pga = float(arr_pga[i_pga])

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

    def residual_interstory_drift_profile(
        self,
        results: "NodalResults",
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
        signed: bool = True,
        tail: int = 1,
        agg: str = "mean",
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "AggregationEngine.residual_interstory_drift_profile not implemented yet; "
            "filled in Phase 4.3.2."
        )

    def residual_drift_envelope(
        self,
        results: "NodalResults",
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
        raise NotImplementedError(
            "AggregationEngine.residual_drift_envelope not implemented yet; "
            "filled in Phase 4.3.2."
        )

    # ------------------------------------------------------------------ #
    # Rotation / torsion / rocking
    # ------------------------------------------------------------------ #

    def roof_torsion(
        self,
        results: "NodalResults",
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
        reduce: str = "series",
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
        """
        import numpy as np

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

            return int(results.info.nearest_node_id([pt], return_distance=False)[0])

        a_id = _resolve_one(node_a_id, node_a_coord, "node_a")
        b_id = _resolve_one(node_b_id, node_b_coord, "node_b")

        if a_id == b_id:
            raise ValueError("node_a and node_b resolved to the same node id; cannot compute torsion.")

        # ----------------------------
        # Baseline plan geometry (dx, dy)
        # ----------------------------
        if results.info.nodes_info is None:
            raise ValueError("nodes_info is required (must contain x,y) for roof_torsion().")

        ni = results.info.nodes_info
        xcol = results.info._resolve_column(ni, "x", required=True)
        ycol = results.info._resolve_column(ni, "y", required=True)
        nid_col = results.info._resolve_column(ni, "node_id", required=False)

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
        ux = results.fetch(result_name=result_name, component=ux_component, node_ids=[a_id, b_id])
        uy = results.fetch(result_name=result_name, component=uy_component, node_ids=[a_id, b_id])

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
                        "rigidity_ratio": rigidity_ratio,
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

    def base_rocking(
        self,
        results: "NodalResults",
        *,
        node_coords_xy: Sequence[Sequence[float]],
        z_coord: float,
        result_name: str = "DISPLACEMENT",
        uz_component: object = 3,
        stage: Optional[str] = None,
        reduce: str = "series",
        det_tol: float = 1e-12,
    ) -> pd.DataFrame | dict[str, float]:
        """
        Estimate base rocking angles from Uz at 3 base nodes.

        Model (small rotations):
            w(x,y) = w0 + theta_x*y - theta_y*x

        When the 3 resolved points are collinear or duplicated (geometry
        matrix singular), returns a no-rocking fallback where theta_x,
        theta_y, theta_mag are all zero and ``is_singular=True``. w0 is
        still computed as the mean Uz across the 3 nodes.
        """
        import numpy as np

        if len(node_coords_xy) != 3:
            raise ValueError("node_coords_xy must contain exactly 3 (x,y) points.")

        # --------------------------------------------------
        # Resolve node ids (nearest at given z)
        # --------------------------------------------------
        pts = [(float(x), float(y), float(z_coord)) for x, y in node_coords_xy]
        node_ids = results.info.nearest_node_id(pts, return_distance=False)
        n1, n2, n3 = map(int, node_ids)

        # --------------------------------------------------
        # Fetch Uz for those 3 nodes (also used for fallback)
        # --------------------------------------------------
        uz = results.fetch(result_name=result_name, component=uz_component, node_ids=[n1, n2, n3])

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
        if results.info.nodes_info is None:
            raise ValueError("nodes_info is required (must contain x,y) for base_rocking().")

        ni = results.info.nodes_info
        xcol = results.info._resolve_column(ni, "x", required=True)
        ycol = results.info._resolve_column(ni, "y", required=True)
        nid_col = results.info._resolve_column(ni, "node_id", required=False)

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
        # Geometry matrix A and singularity check
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

        duplicate = (len({n1, n2, n3}) < 3)
        det = float(np.linalg.det(A))
        singular = duplicate or (abs(det) < float(det_tol))

        # --------------------------------------------------
        # Fallback: singular geometry -> assume no rocking
        # --------------------------------------------------
        if singular:
            w0 = np.nanmean(w_steps, axis=0)

            out = pd.DataFrame(
                {
                    "w0": w0,
                    "theta_x_rad": np.zeros_like(w0),
                    "theta_y_rad": np.zeros_like(w0),
                    "theta_mag_rad": np.zeros_like(w0),
                    "is_singular": np.ones_like(w0, dtype=bool),
                },
                index=W.columns,
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
        p_steps = Ainv @ w_steps

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
        results: "NodalResults",
        *,
        component: object,
        side_a_top: tuple[float, float, float],
        side_a_bottom: tuple[float, float, float],
        side_b_top: tuple[float, float, float],
        side_b_bottom: tuple[float, float, float],
        result_name: str = "DISPLACEMENT",
        stage: Optional[str] = None,
        reduce_time: str = "abs_max",
        definition: str = "max_over_avg",
        eps: float = 1e-16,
        signed: bool = True,
        tail: int | None = None,
    ) -> dict[str, Any]:
        """
        ASCE-style torsional irregularity ratio from edge drifts at a story.

        Inputs are (x, y, z) tuples, resolved to nearest node IDs. The
        two sides A and B each contribute a drift time history; the
        ratio compares their time-reduced magnitudes.
        """
        import numpy as np

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
        node_ids = results.info.nearest_node_id(pts, return_distance=False)
        a_top_id, a_bot_id, b_top_id, b_bot_id = map(int, node_ids)

        if a_top_id == a_bot_id:
            raise ValueError("Side A top and bottom resolved to the same node id (cannot compute drift).")
        if b_top_id == b_bot_id:
            raise ValueError("Side B top and bottom resolved to the same node id (cannot compute drift).")

        # ----------------------------
        # compute drift time histories
        # ----------------------------
        drA = self.drift(
            results,
            top=a_top_id,
            bottom=a_bot_id,
            component=component,
            result_name=result_name,
            stage=stage,
            signed=signed,
            reduce="series",
        )
        drB = self.drift(
            results,
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

    # ------------------------------------------------------------------ #
    # Orbit
    # ------------------------------------------------------------------ #

    def orbit(
        self,
        results: "NodalResults",
        *,
        result_name: str = "DISPLACEMENT",
        x_component: object = '1',
        y_component: object = '2',
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        coordinates: Sequence[Sequence[float]] | None = None,
        stage: Optional[str] = None,
        reduce_nodes: str = "none",
        signed: bool = True,
        return_nodes: bool = False,
    ) -> tuple[pd.Series, pd.Series] | tuple[pd.Series, pd.Series, list[int]]:
        raise NotImplementedError(
            "AggregationEngine.orbit not implemented yet; "
            "filled in Phase 4.3.2."
        )


__all__ = ["AggregationEngine"]
