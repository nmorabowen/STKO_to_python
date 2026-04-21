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
        raise NotImplementedError(
            "AggregationEngine._resolve_story_nodes_by_z_tol not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
        raise NotImplementedError(
            "AggregationEngine.residual_drift not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
        raise NotImplementedError(
            "AggregationEngine.roof_torsion not implemented yet; "
            "filled in Phase 4.3.2."
        )

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
