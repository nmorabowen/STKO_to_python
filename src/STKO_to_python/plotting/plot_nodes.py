# ── STKO_to_python/plotting/plot_nodes.py ───────────────────────────────
# -*- coding: utf-8 -*-
from __future__ import annotations

# internal ----------------------------------------------------------------
from ..dataprocess.aggregator import Aggregator     # ← centralised stats helper

# standard lib ------------------------------------------------------------
from typing import (
    TYPE_CHECKING,
    Sequence,
    Tuple,
    Dict,
    Any,
    Optional,
    Union,
)
import warnings

# third-party -------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if TYPE_CHECKING:          # avoid runtime circular import
    from ..core.dataset import MPCODataSet


class PlotNodes:
    """
    Post-processing helpers for an :class:`MPCODataSet`.

    Supported aggregation operations (``operation=``)
    -------------------------------------------------
    Sum, Mean, Max, Min, Std (Std±1/2), Percentile, Envelope, Cumulative,
    SignedCumulative, RunningEnvelope ― or any callable ``pd.Series → scalar``.

    Accepted directions
    -------------------
    Component name ``"x"``, ``"y"``, ``"z"`` **or** its numeric index.
    Use ``None`` when the result is not directional.
    """

    # ------------------------------------------------------------------ #
    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset

    def plot_nodal_results(
        self,
        model_stage: str,
        # ▸ V-axis ------------------------------------------------------------- #
        results_name_verticalAxis: str | None = None,
        node_ids_verticalAxis: Sequence[int] | None = None,
        selection_set_id_verticalAxis: int | None = None,
        direction_verticalAxis: str | int | None | Callable[[pd.Series], float] = None,
        values_operation_verticalAxis: str | Sequence[str] = "Sum",
        scaling_factor_verticalAxis: float = 1.0,
        # ▸ H-axis ------------------------------------------------------------- #
        results_name_horizontalAxis: str | None = None,
        node_ids_horizontalAxis: Sequence[int] | None = None,
        selection_set_id_horizontalAxis: int | None = None,
        direction_horizontalAxis: str | int | None | Callable[[pd.Series], float] = None,
        values_operation_horizontalAxis: str | Sequence[str] = "Sum",
        scaling_factor_horizontalAxis: float = 1.0,
        # ▸ Extra parameters ---------------------------------------------------- #
        operation_kwargs: dict[str, Any] | None = None,
        # ▸ Cosmetics ----------------------------------------------------------- #
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (10, 6),
        linewidth: float = 1.2,
        label: str | None = None,
        marker: str | None = None,
        **line_kwargs,
    ) -> tuple[plt.Axes | None, dict[str, Any]]:
        """
        Plot X-Y curves of nodal results (or TIME/STEP vs. results).

        Parameters
        ----------
        …
        operation_kwargs
            Extra keyword arguments forwarded verbatim to
            :meth:`Aggregator.compute` – e.g. ``{'percentile': 95}``.
        colour_cycle
            Optional custom colour list.  If *None*, uses Tableau colours
            with red-dominant hues pruned (better for protanopia).
        marker
            Matplotlib marker code ('.', 'o', etc.) applied to all lines
            unless overriden via **line_kwargs.

        Notes
        -----
        • Either X **or** Y may be multi-column, but **not both**.
        • If no horizontal result is given, we default to `"TIME"` when
        available, otherwise `"STEP"`.
        • Returns a dict with raw ``x_array`` / ``y_array`` and the
        DataFrame(s) (if multi-column) under key ``"dataframe"``.
        """
        operation_kwargs = operation_kwargs or {}

        # ── helper: fetch & aggregate ------------------------------------- #
        def _axis(
            rname: str | None,
            nids,
            ssid,
            direction,
            op,
            scale: float,
        ):
            # implicit TIME/STEP fallbacks
            if rname is None:
                rname = "TIME" if "TIME" in self.dataset.time.loc[model_stage].columns else "STEP"

            if rname == "STEP":
                return self.dataset.time.loc[model_stage].index.to_numpy() * scale
            if rname == "TIME":
                return self.dataset.time.loc[model_stage]["TIME"].to_numpy() * scale

            df = self.dataset.nodes.get_nodal_results(
                model_stage=model_stage,
                results_name=rname,
                node_ids=nids,
                selection_set_id=ssid,
            )
            if df is None or df.empty:
                raise ValueError(f"No data for result '{rname}'")

            agg = Aggregator(df, direction)
            return agg.compute(operation=op, **operation_kwargs) * scale

        try:
            y = _axis(results_name_verticalAxis,   node_ids_verticalAxis,
                    selection_set_id_verticalAxis, direction_verticalAxis,
                    values_operation_verticalAxis, scaling_factor_verticalAxis)

            x = _axis(results_name_horizontalAxis, node_ids_horizontalAxis,
                    selection_set_id_horizontalAxis, direction_horizontalAxis,
                    values_operation_horizontalAxis, scaling_factor_horizontalAxis)
        except Exception as exc:
            warnings.warn(f"[plot_nodal_results] {exc}", RuntimeWarning)
            return None, {}

        if len(x) != len(y):
            warnings.warn("[plot_nodal_results] X–Y length mismatch.", RuntimeWarning)
            return None, {}

        multi_x = isinstance(x, pd.DataFrame)
        multi_y = isinstance(y, pd.DataFrame)
        if multi_x and multi_y:
            warnings.warn("Both X and Y are multi-column; plot skipped.", RuntimeWarning)
            return None, {}

        # ── axes & colour handling ---------------------------------------- #
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        plot_kwargs = dict(lw=linewidth, marker=marker or None, **line_kwargs)

        # ── draw curves ---------------------------------------------------- #
        if not multi_x and not multi_y:
            ax.plot(x, y, label=label or getattr(y, "name", None), rasterized=True, **plot_kwargs)

        elif not multi_x and multi_y:
            for col in sorted(y.columns):
                ax.plot(x, y[col], label=f"{label + ': ' if label else ''}{col}", rasterized=True, **plot_kwargs)

        else:  # multi_x and not multi_y
            for col in sorted(x.columns):
                ax.plot(x[col], y, label=f"{label + ': ' if label else ''}{col}", rasterized=True, **plot_kwargs)

        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        ax.set_xlabel(results_name_horizontalAxis or ("TIME" if "TIME" in str(x) else "STEP"))
        ax.set_ylabel(results_name_verticalAxis or "Result")
        ax.grid(True)

        meta = {
            "x_array": np.asarray(x),
            "y_array": np.asarray(y),
        }
        if multi_x or multi_y:
            meta["dataframe"] = x if multi_x else y

        return ax, meta

    def plot_time_history(
        self,
        model_stage: str,
        results_name: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        direction: str | int | None = None,
        split_subplots: bool = False,
        scaling_factor: float = 1.0,
        sort_by: str = "z",
        reverse_sort: bool = False,
        figsize: tuple[int, int] = (8, 3),
        linewidth: float = 1.2,
        marker: str | None = None,
        sharey: bool = True,
        **line_kwargs,
    ) -> tuple[plt.Figure | None, dict[str, Any]]:
        """
        Plot raw nodal time-history curves (no aggregation) with TIME on the X-axis.
        """
        # ── resolve node_ids ------------------------------------------------ #
        if node_ids is None and selection_set_id is not None:
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)
        elif node_ids is None:
            node_ids = self.dataset.nodes_info["dataframe"]["node_id"].to_numpy()
        node_ids = tuple(np.unique(node_ids))
        if not node_ids:
            warnings.warn("[plot_time_history] No node IDs to plot.", RuntimeWarning)
            return None, {}

        # ── TIME ------------------------------------------------------------ #
        time_df = self.dataset.time.loc[model_stage]
        time_arr = (time_df["TIME"] if "TIME" in time_df else time_df.index).to_numpy()

        # ── coordinates & sorting ------------------------------------------ #
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        if sort_by not in {"x", "y", "z"}:
            warnings.warn(f"sort_by '{sort_by}' not recognised. Using 'z'.", RuntimeWarning)
            sort_by = "z"

        coords_subset = coords_df.loc[list(node_ids), ["x", "y", "z"]].sort_values(
            by=sort_by, ascending=not reverse_sort
        )
        node_ids = tuple(coords_subset.index)
        coords_map = coords_subset.to_dict("index")

        # ── single batch fetch of results ---------------------------------- #
        df_all = self.dataset.nodes.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=list(node_ids),
        )
        if df_all is None or df_all.empty:
            warnings.warn(f"No data found for result '{results_name}'.", RuntimeWarning)
            return None, {}

        # ── direction index mapping ---------------------------------------- #
        if isinstance(direction, str):
            direction = {"x": 1, "y": 2, "z": 3}.get(direction.lower(), None)

        # ── setup figure --------------------------------------------------- #
        if split_subplots:
            fig_height = figsize[1] * len(node_ids)
            fig, axes = plt.subplots(
                len(node_ids), 1,
                figsize=(figsize[0], fig_height),
                sharex=True,
                sharey=sharey,
            )
            axes = np.atleast_1d(axes)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = np.array([ax])
        axes_iter = iter(axes)

        meta: dict[str, Any] = {"time": time_arr}
        global_ymin, global_ymax = np.inf, -np.inf

        # ── plotting loop over node_ids ------------------------------------ #
        for nid in node_ids:
            ax_i = next(axes_iter) if split_subplots else axes[0]
            try:
                df_node = df_all.xs(nid, level=0)
            except KeyError:
                warnings.warn(f"No data for node {nid}.", RuntimeWarning)
                continue

            if direction is None:
                if df_node.shape[1] > 1:
                    warnings.warn(
                        f"Result '{results_name}' has multiple components; "
                        "defaulting to first column. Set 'direction=' to choose.",
                        RuntimeWarning,
                    )
                y = df_node.iloc[:, 0].to_numpy()
            elif direction - 1 < df_node.shape[1]:
                y = df_node.iloc[:, direction - 1].to_numpy()
            else:
                warnings.warn(f"Direction index {direction} out of range for node {nid}.", RuntimeWarning)
                continue

            y *= scaling_factor
            meta[nid] = y

            global_ymin = min(global_ymin, y.min())
            global_ymax = max(global_ymax, y.max())

            c = coords_map[nid]
            label = f"Node {nid} @({c['x']:.2f}, {c['y']:.2f}, {c['z']:.2f})"
            ax_i.plot(time_arr, y, lw=linewidth, marker=marker, label=label, **line_kwargs)
            ax_i.grid(True)
            if split_subplots:
                ax_i.set_ylabel(results_name)
                ax_i.legend(fontsize="small")

        # ── unify limits and finalize plot --------------------------------- #
        if split_subplots:
            for ax in axes:
                ax.set_ylim(global_ymin, global_ymax)

        axes[-1].set_xlabel("Time [s]")
        if not split_subplots:
            axes[0].set_ylabel(results_name)
            axes[0].legend(fontsize="small")

        fig.tight_layout()
        return fig, meta
    
    def plot_roof_drift(
        self,
        model_stage: str,
        direction: str | int,
        ax: plt.Axes | None = None,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        normalize: bool = True,
        scaling_factor: float = 1.0,
        z_round: int = 3,
        top_z: float | None = None,
        bottom_z: float | None = None,
        aggregate: str | Callable[[pd.Series], float] = "Mean",   # Mean, Median, Max, Min, or callable
        figsize: tuple[int, int] = (10, 4),
        linewidth: float = 1.4,
        marker: str | None = None,
        label: str | None = None,
        rasterized: bool = False,
        **plot_kwargs: Any,
    ) -> tuple[plt.Axes, dict[str, Any]]:
        """
        Plot roof drift Δu(t) = u_top(t) − u_bot(t) (optionally normalized by height)
        for a set of nodes (direct or via selection set). If multiple nodes exist at
        a Z-level, their displacements are aggregated into a single trace.

        Parameters
        ----------
        model_stage : str
            Stage name, e.g. 'MODEL_STAGE[5]'.
        direction : {'x','y','z'} or {1,2,3}
            Component to plot. 1/2/3 map to x/y/z.
        ax : plt.Axes | None
            Existing axes to draw on. If None, a new figure/axes is created.
        node_ids : Sequence[int] | None
            Explicit node IDs. Mutually exclusive with selection_set_id.
        selection_set_id : int | None
            Selection set that defines nodes. Mutually exclusive with node_ids.
        normalize : bool
            If True, plot drift ratio (Δu/h). Otherwise plot Δu in result units.
        scaling_factor : float
            Additional scaling (applied last).
        z_round : int
            Decimal rounding for Z to define floor/roof groups (default 3).
        top_z, bottom_z : float | None
            Force specific Z levels (in model units) *after rounding*. If None, use
            max/min Z levels present in the selected nodes.
        aggregate : str | Callable
            Aggregation across nodes within one Z-level. One of:
            'Mean','Median','Max','Min' or a callable(pd.Series) -> scalar.
        figsize : tuple
            Only used if ax is None.
        linewidth, marker, label, rasterized
            Matplotlib styling.
        **plot_kwargs
            Forwarded to `Axes.plot` (safe—sanitized here).

        Returns
        -------
        (ax, meta) : (plt.Axes, dict)
            meta = {'time','drift','top_z','bottom_z','top_ids','bottom_ids','height'}
        """
        # ── input checks ──────────────────────────────────────────────────────
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Provide either `node_ids` or `selection_set_id` (not both).")

        # direction → column index or name
        if isinstance(direction, str):
            dmap = {"x": 0, "y": 1, "z": 2}
            dkey = direction.lower()
            if dkey not in dmap:
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = dmap[dkey]
            dir_lbl = dkey
        else:
            if direction not in (1, 2, 3):
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = direction - 1
            dir_lbl = ["x", "y", "z"][dir_idx]

        # node set
        if node_ids is None:
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(np.asarray(node_ids, dtype=int))
        if node_ids.size < 2:
            raise ValueError("Need at least two nodes (top & bottom) to compute roof drift.")

        # ── coordinates + Z grouping ──────────────────────────────────────────
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        )
        coords_df = coords_df.drop_duplicates("node_id").set_index("node_id")
        try:
            coords_sub = coords_df.loc[node_ids, ["x", "y", "z"]].copy()
        except KeyError as e:
            missing = set(node_ids) - set(coords_df.index)
            raise ValueError(f"Some nodes missing in nodes_info: {sorted(missing)[:10]} ...") from e

        coords_sub["_z_group"] = coords_sub["z"].round(z_round)
        level_groups = coords_sub.groupby("_z_group")
        if len(level_groups) < 2:
            raise ValueError("Need at least two Z-levels to compute roof drift.")

        # choose bottom/top Z
        z_levels = np.array(sorted(level_groups.groups.keys()), dtype=float)
        z_bot = float(bottom_z) if bottom_z is not None else float(z_levels.min())
        z_top = float(top_z) if top_z is not None else float(z_levels.max())
        if z_bot == z_top:
            raise ValueError("Top and bottom Z-levels coincide (height = 0).")
        if z_bot not in level_groups.groups or z_top not in level_groups.groups:
            raise ValueError(
                f"Requested z-levels not found. Available: {z_levels.tolist()}, "
                f"requested top={z_top}, bottom={z_bot} (after rounding to {z_round})."
            )
        ids_bot = level_groups.get_group(z_bot).index.to_list()
        ids_top = level_groups.get_group(z_top).index.to_list()
        height = abs(z_top - z_bot)
        if normalize and height == 0:
            raise ValueError("Zero height between top and bottom levels.")

        # ── batched results fetch ─────────────────────────────────────────────
        all_ids = np.unique(np.concatenate([ids_bot, ids_top]))
        df_all = self.dataset.nodes.get_nodal_results(
            model_stage=model_stage,
            results_name="DISPLACEMENT",
            node_ids=all_ids.tolist(),
        )
        if df_all is None or df_all.empty:
            raise ValueError("No displacement data found for the requested nodes/stage.")

        # resolve column for direction: support either ['x','y','z'] labels or numeric columns
        if dir_idx < df_all.shape[1]:
            col_series = lambda df: df.iloc[:, dir_idx]
        else:
            # try labeled columns
            try:
                col_series = lambda df: df[dir_lbl]
            except Exception as e:
                raise ValueError("Unexpected displacement columns layout.") from e

        # aggregator
        if isinstance(aggregate, str):
            agg_key = aggregate.lower()
            if agg_key == "mean":
                agg_fn = np.mean
            elif agg_key == "median":
                agg_fn = np.median
            elif agg_key == "max":
                agg_fn = np.max
            elif agg_key == "min":
                agg_fn = np.min
            else:
                raise ValueError("aggregate must be one of 'Mean','Median','Max','Min' or a callable.")
        else:
            agg_fn = aggregate  # callable(pd.Series) -> scalar

        def avg_disp(ids: list[int]) -> np.ndarray:
            # Collect each node series for chosen direction
            series_list: list[np.ndarray] = []
            for nid in ids:
                try:
                    df_node = df_all.xs(nid, level=0)
                except KeyError:
                    continue
                s = col_series(df_node).to_numpy()
                series_list.append(s)
            if not series_list:
                raise ValueError("No valid displacement data for one of the Z groups.")
            # stack → aggregate across nodes per time step
            arr = np.vstack(series_list)  # (n_nodes, n_steps)
            return agg_fn(arr, axis=0)

        u_top = avg_disp(ids_top)
        u_bot = avg_disp(ids_bot)
        drift = u_top - u_bot

        if normalize:
            drift = drift / height
        drift = drift * float(scaling_factor)

        # time vector
        try:
            time_df = self.dataset.time.loc[model_stage]
            time_arr = (time_df["TIME"] if "TIME" in time_df else time_df.index).to_numpy()
        except Exception:
            # Fallback: infer from dataframe index level 1 if structured like (node_id, step)
            if isinstance(df_all.index, pd.MultiIndex) and df_all.index.nlevels >= 2:
                time_arr = df_all.index.get_level_values(1).unique().to_numpy()
            else:
                raise

        # ── plot ──────────────────────────────────────────────────────────────
        # sanitize kwargs (avoid accidental 'ax' etc.)
        plot_kwargs.pop("ax", None)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        line, = ax.plot(time_arr, drift, lw=linewidth, marker=marker, label=label, rasterized=rasterized, **plot_kwargs)
        ax.grid(True)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Drift ratio (Δu/h)" if normalize else "Δu [DISPLACEMENT]")
        ax.set_title(f"Roof Drift – dir {dir_lbl}  (z={z_bot:.3f} → {z_top:.3f})")

        if label is not None:
            ax.legend()

        meta = {
            "time": time_arr,
            "drift": drift,
            "top_z": z_top,
            "bottom_z": z_bot,
            "height": height,
            "top_ids": ids_top,
            "bottom_ids": ids_bot,
            "line": line,
        }
        return ax, meta


    def plot_story_drifts(
        self,
        model_stage: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        direction: str | int = "x",
        normalize: bool = True,
        scaling_factor: float = 1.0,
        split_subplots: bool = False,
        sort_by: str = "z",
        reverse_sort: bool = False,
        figsize: tuple[int, int] = (10, 4),
        linewidth: float = 1.2,
        marker: str | None = None,
        sharey: bool = False,
        **line_kwargs,
    ) -> tuple[plt.Figure | None, dict[str, Any]]:
        """
        Plot inter-storey drifts (Δu or Δu / height) over time with Z-aware node grouping.
        Labels now include Z-coordinate info for each storey.
        """
        results_name = "DISPLACEMENT"
        if results_name not in self.dataset.node_results_names:
            warnings.warn(f"No '{results_name}' result found. Drift plot may fail.", RuntimeWarning)

        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Specify *either* node_ids *or* selection_set_id (not both).")

        if node_ids is None:
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(node_ids)
        if node_ids.size < 2:
            raise ValueError("Need ≥2 nodes to compute storey drifts.")

        # --- get node coordinates --------------------------------------------- #
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        coords_sub = coords_df.loc[list(node_ids), ["x", "y", "z"]].copy()

        if sort_by not in {"x", "y", "z"}:
            warnings.warn(f"sort_by '{sort_by}' invalid. Using 'z'.", RuntimeWarning)
            sort_by = "z"

        coords_sub["_z_group"] = coords_sub[sort_by].round(3)
        level_groups = coords_sub.groupby("_z_group")

        z_levels = sorted(level_groups.groups.keys(), reverse=reverse_sort)
        if len(z_levels) < 2:
            raise ValueError("Need ≥2 Z levels to compute storey drifts.")

        # --- direction index -------------------------------------------------- #
        if isinstance(direction, str):
            direction = {"x": 1, "y": 2, "z": 3}.get(direction.lower(), None)
        if direction not in (1, 2, 3):
            raise ValueError("direction must be 'x','y','z' or 1,2,3")

        # --- batched results fetch -------------------------------------------- #
        df_all = self.dataset.nodes.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=list(node_ids),
        )
        if df_all is None or df_all.empty:
            raise ValueError("No displacement data available for requested nodes.")

        # --- helper: avg time series across level group ----------------------- #
        def avg_disp(ids: list[int]) -> np.ndarray:
            dfs = []
            for nid in ids:
                try:
                    df = df_all.xs(nid, level=0)
                except KeyError:
                    continue
                dfs.append(df.iloc[:, direction - 1].to_numpy())
            if not dfs:
                raise ValueError("No valid displacement data in level group.")
            return np.mean(np.vstack(dfs), axis=0)

        # --- time array ------------------------------------------------------- #
        time_df = self.dataset.time.loc[model_stage]
        time_arr = (time_df["TIME"] if "TIME" in time_df else time_df.index).to_numpy()

        # --- prepare figure --------------------------------------------------- #
        n_stories = len(z_levels) - 1
        if split_subplots:
            fig_height = figsize[1] * n_stories
            fig, axes = plt.subplots(
                n_stories, 1,
                figsize=(figsize[0], fig_height),
                sharex=True,
                sharey=sharey,
            )
            axes = np.atleast_1d(axes)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = np.array([ax])
        axes_iter = iter(axes)

        meta: dict[str, Any] = {"time": time_arr}
        global_ymin, global_ymax = np.inf, -np.inf

        # --- loop over storeys ------------------------------------------------ #
        for i in range(n_stories):
            z1, z2 = z_levels[i], z_levels[i + 1]
            group1_ids = level_groups.get_group(z1).index.tolist()
            group2_ids = level_groups.get_group(z2).index.tolist()

            u1 = avg_disp(group1_ids)
            u2 = avg_disp(group2_ids)
            Δu = u2 - u1
            height = abs(z2 - z1)

            if normalize:
                if height == 0:
                    warnings.warn(f"Zero height between levels {z1} and {z2}; skipping.", RuntimeWarning)
                    continue
                Δu = Δu / height
            Δu *= scaling_factor

            label = f"{z1:.2f}→{z2:.2f}"
            meta[label] = Δu

            ax_i = next(axes_iter) if split_subplots else axes[0]
            ax_i.plot(time_arr, Δu, lw=linewidth, marker=marker, label=f"Story {label}", **line_kwargs)
            ax_i.grid(True)

            if split_subplots:
                ax_i.set_ylabel("Drift" + (" ratio" if normalize else ""))
                ax_i.legend(fontsize="small")

            global_ymin = min(global_ymin, Δu.min())
            global_ymax = max(global_ymax, Δu.max())

        # --- Y-axis alignment ------------------------------------------------- #
        if split_subplots:
            for ax in axes:
                ax.set_ylim(global_ymin, global_ymax)

        # --- final labels ----------------------------------------------------- #
        axes[-1].set_xlabel("Time [s]")
        if not split_subplots:
            axes[0].set_ylabel("Drift" + (" ratio" if normalize else ""))
            axes[0].legend(fontsize="small")

        fig.tight_layout()
        return fig, meta
    
    def plot_drift_profile(
        self,
        model_stage: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        direction: str | int = "x",
        normalize: bool = True,
        scaling_factor: float = 1.0,
        sort_by: str = "z",
        reverse_sort: bool = False,
        limits: list[float] | None = None,
        fill: bool = False,
        ax: Optional[plt.Axes] = None,
        show_legend: bool = True,
        **plot_kwargs,
    ) -> tuple[plt.Axes, dict[str, Any]]:
        """
        Plot max/min drift envelope per storey (Δu or Δu / height) vs. height using Z-grouping.
        Anchors zero drift at the absolute base height, and uses actual Z-levels as ticks.
        """
        results_name = "DISPLACEMENT"

        # ── validate input ───────────────────────────────────────────────────── #
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Specify *either* node_ids *or* selection_set_id (not both).")
        if node_ids is None:
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(node_ids)
        if node_ids.size < 2:
            raise ValueError("Need ≥2 nodes to compute story drifts.")

        # ── coordinates and grouping ─────────────────────────────────────────── #
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        if sort_by not in {"x", "y", "z"}:
            warnings.warn(f"sort_by '{sort_by}' invalid. Defaulting to 'z'.", RuntimeWarning)
            sort_by = "z"

        coords_sub = coords_df.loc[list(node_ids), ["x", "y", "z"]]
        coords_sub["_z_group"] = coords_sub["z"].round(3)
        level_groups = coords_sub.groupby("_z_group")
        z_levels = sorted(level_groups.groups.keys(), reverse=reverse_sort)

        if len(z_levels) < 2:
            raise ValueError("Need ≥2 Z levels to compute inter-storey drifts.")

        # ── direction index ───────────────────────────────────────────────────── #
        if isinstance(direction, str):
            direction = {"x": 1, "y": 2, "z": 3}.get(direction.lower(), None)
        if direction not in (1, 2, 3):
            raise ValueError("direction must be 'x','y','z' or 1,2,3")

        # ── batched results fetch ─────────────────────────────────────────────── #
        df_all = self.dataset.nodes.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=list(node_ids),
        )
        if df_all is None or df_all.empty:
            raise ValueError("No valid displacement data found.")

        # ── helper: average nodal series ─────────────────────────────────────── #
        def avg_disp(nids: list[int]) -> np.ndarray:
            dfs = []
            for nid in nids:
                try:
                    df = df_all.xs(nid, level=0)
                except KeyError:
                    continue
                dfs.append(df.iloc[:, direction - 1].to_numpy())
            if not dfs:
                raise ValueError("No valid displacement data in level group.")
            return np.mean(np.vstack(dfs), axis=0)

        # ── compute drifts per level transition ──────────────────────────────── #
        z_tops, z_bottoms = [], []
        drift_min, drift_max = [], []

        for z1, z2 in zip(z_levels[:-1], z_levels[1:]):
            ids1 = level_groups.get_group(z1).index.tolist()
            ids2 = level_groups.get_group(z2).index.tolist()

            u1 = avg_disp(ids1)
            u2 = avg_disp(ids2)
            Δu = u2 - u1
            height = abs(z2 - z1)

            if normalize:
                if height == 0:
                    warnings.warn(f"Zero height between levels {z1} and {z2}.", RuntimeWarning)
                    continue
                Δu /= height
            Δu *= scaling_factor

            drift_min.append(Δu.min())
            drift_max.append(Δu.max())
            z_tops.append(z2)
            z_bottoms.append(z1)

        if not drift_min:
            raise ValueError("No valid story drifts computed.")

        # ── prepend base level ───────────────────────────────────────────────── #
        z_base = min(z_levels)
        z_tops = [z_base] + z_tops
        drift_min = [0.0] + drift_min
        drift_max = [0.0] + drift_max

        # ── plot ─────────────────────────────────────────────────────────────── #
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 8))

        ax.plot(drift_min, z_tops, "-", label="Min Drift", **plot_kwargs)
        ax.plot(drift_max, z_tops, "-", label="Max Drift", **plot_kwargs)
        if fill:
            ax.fill_betweenx(z_tops, drift_min, drift_max, alpha=0.2)

        if limits is not None:
            for lim in limits:
                ax.axvline(lim, color="gray", linestyle="--", alpha=0.5, label=f"Limit {lim:g}")

        ax.set_xlabel("Drift Ratio" if normalize else f"Δu [{results_name}]")
        if scaling_factor != 1.0:
            ax.set_xlabel(ax.get_xlabel() + f" ×{scaling_factor:g}")
        ax.set_ylabel("Height [Z]")
        ax.set_yticks(z_levels)
        ax.grid(True)
        if show_legend:
            ax.legend()

        return ax, {
            "story_bottom_z": np.insert(z_bottoms, 0, z_base),
            "story_top_z": np.array(z_tops),
            "drift_min": np.array(drift_min),
            "drift_max": np.array(drift_max),
        }
  
    def plot_orbit(
        self,
        model_stage: str,
        results_name: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        direction_x: str | int = 1,
        direction_y: str | int = 2,
        scaling_factor: float = 1.0,
        equal_aspect: bool = True,
        split_subplots: bool = False,
        figsize: tuple[int, int] = (6, 6),
        linewidth: float = 1.4,
        marker: str | None = None,
        show_legend: bool = True,
        **line_kwargs,
    ) -> tuple[plt.Figure | plt.Axes, dict[str, Any]]:
        """
        Plot 2D orbit trajectories for one or more nodes over time, optionally averaged by Z-group.
        """
        # --- validate node input ------------------------------------------------ #
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Provide either `node_ids` or `selection_set_id` (not both).")
        if node_ids is None:
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(node_ids)

        # --- direction parsing -------------------------------------------------- #
        if isinstance(direction_x, str):
            direction_x = {"x": 1, "y": 2, "z": 3}.get(direction_x.lower(), None)
        if isinstance(direction_y, str):
            direction_y = {"x": 1, "y": 2, "z": 3}.get(direction_y.lower(), None)
        if direction_x is None or direction_y is None:
            raise ValueError("Invalid direction_x or direction_y (must be 'x','y','z' or 1,2,3).")

        # --- coordinates & Z grouping ------------------------------------------ #
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")
        coords_sub = coords_df.loc[node_ids, ["x", "y", "z"]].copy()
        coords_sub["_z_group"] = coords_sub["z"].round(3)
        level_groups = coords_sub.groupby("_z_group")
        z_levels = sorted(level_groups.groups.keys())

        # --- batched result fetch ---------------------------------------------- #
        df_all = self.dataset.nodes.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=list(node_ids),
        )
        if df_all is None or df_all.empty:
            raise ValueError("No nodal data available for orbit plot.")

        # --- prepare figure layout --------------------------------------------- #
        nplots = len(z_levels) + 1
        ncols = 2
        nrows = (nplots + 1) // ncols
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(figsize[0] * ncols, figsize[1] * nrows),
            squeeze=False,
        )
        axes = axes.flatten()

        result_dict: dict[str, dict[str, np.ndarray]] = {}
        all_x, all_y = [], []

        # --- loop over Z-groups ------------------------------------------------ #
        for i, z in enumerate(z_levels):
            ids = level_groups.get_group(z).index.tolist()

            x_list, y_list = [], []
            for nid in ids:
                try:
                    df = df_all.xs(nid, level=0)
                    x = df.iloc[:, direction_x - 1].to_numpy() * scaling_factor
                    y = df.iloc[:, direction_y - 1].to_numpy() * scaling_factor
                except (KeyError, IndexError):
                    continue
                x_list.append(x)
                y_list.append(y)

            if not x_list:
                continue

            ax = axes[i]
            xmat = np.vstack(x_list)
            ymat = np.vstack(y_list)

            x_mean = np.mean(xmat, axis=0)
            y_mean = np.mean(ymat, axis=0)

            ax.plot(x_mean, y_mean, lw=linewidth, marker=marker, label=f"Z={z:.2f}", **line_kwargs)
            ax.set_title(f"Z = {z:.2f}")
            ax.set_xlabel(f"{results_name}[{direction_x}]")
            ax.set_ylabel(f"{results_name}[{direction_y}]")
            ax.grid(True)
            if equal_aspect:
                ax.set_aspect("equal", adjustable="box")
            if show_legend:
                ax.legend(fontsize="small")

            result_dict[f"Z={z:.2f}"] = {"x": x_mean, "y": y_mean}
            all_x.append(x_mean)
            all_y.append(y_mean)

        # --- plot total average ------------------------------------------------ #
        ax = axes[len(z_levels)]
        x_global = np.mean(np.vstack(all_x), axis=0)
        y_global = np.mean(np.vstack(all_y), axis=0)

        ax.plot(x_global, y_global, lw=linewidth, marker=marker, label="Global Avg", **line_kwargs)
        ax.set_title("Global Average")
        ax.set_xlabel(f"{results_name}[{direction_x}]")
        ax.set_ylabel(f"{results_name}[{direction_y}]")
        ax.grid(True)
        if equal_aspect:
            ax.set_aspect("equal", adjustable="box")
        if show_legend:
            ax.legend(fontsize="small")
        result_dict["GLOBAL"] = {"x": x_global, "y": y_global}

        all_x.append(x_global)
        all_y.append(y_global)

        # --- fix global axis limits ------------------------------------------- #
        all_x = np.concatenate(all_x)
        all_y = np.concatenate(all_y)
        xlim = (all_x.min(), all_x.max())
        ylim = (all_y.min(), all_y.max())

        for ax in axes[:len(z_levels) + 1]:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        # --- remove unused subplots ------------------------------------------- #
        for j in range(len(z_levels) + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        return fig, result_dict


















