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
        raise NotImplementedError(
            "AggregationEngine.interstory_drift_envelope not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
        raise NotImplementedError(
            "AggregationEngine.story_pga_envelope not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
        raise NotImplementedError(
            "AggregationEngine.base_rocking not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
        raise NotImplementedError(
            "AggregationEngine.asce_torsional_irregularity not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
