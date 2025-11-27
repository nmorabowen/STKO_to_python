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

            results = self.dataset.nodes.get_nodal_results(
                model_stage=model_stage,
                results_name=rname,
                node_ids=nids,
                selection_set_id=ssid,
            )
            
            df = results.df
            
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
        Uses Nodes.get_time_history() for all data fetching & sorting.
        """

        # ---- fetch prepared data -------------------------------------------- #
        try:
            bundle = self.dataset.nodes.get_time_history(
                model_stage=model_stage,
                results_name=results_name,
                node_ids=node_ids,
                selection_set_id=selection_set_id,
                scaling_factor=scaling_factor,
                sort_by=sort_by,
                reverse_sort=reverse_sort,
            )
        except RuntimeError as e:
            warnings.warn(str(e), RuntimeWarning)
            return None, {}

        time_arr = bundle.time
        df_all   = bundle.df              # MultiIndex rows: (node_id, step)
        node_ids = bundle.node_ids        # already sorted per sort_by
        coords_map = bundle.coords_map
        comp_names = bundle.component_names

        if not node_ids:
            warnings.warn("[plot_time_history] No node IDs to plot.", RuntimeWarning)
            return None, {}

        # ---- map direction (1-based like before; allow 'x','y','z') --------- #
        if isinstance(direction, str):
            direction = {"x": 1, "y": 2, "z": 3}.get(direction.lower(), None)

        # ---- figure ---------------------------------------------------------- #
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

        meta: dict[str, Any] = {"time": time_arr, "steps": getattr(bundle, "steps", None)}
        global_ymin, global_ymax = np.inf, -np.inf
        last_ylabel = results_name  # will be refined with component label below

        # ---- plotting loop --------------------------------------------------- #
        for nid in node_ids:
            ax_i = next(axes_iter) if split_subplots else axes[0]
            try:
                df_node = df_all.xs(nid, level=0)
            except KeyError:
                warnings.warn(f"No data for node {nid}.", RuntimeWarning)
                continue

            # choose component
            if direction is None:
                if df_node.shape[1] > 1:
                    warnings.warn(
                        f"Result '{results_name}' has multiple components; "
                        "defaulting to first column. Set 'direction=' to choose.",
                        RuntimeWarning,
                    )
                y = df_node.iloc[:, 0].to_numpy()
                cname = comp_names[0] if comp_names else ""
            elif (direction - 1) < df_node.shape[1]:
                y = df_node.iloc[:, direction - 1].to_numpy()
                cname = (
                    comp_names[direction - 1]
                    if comp_names and (direction - 1) < len(comp_names)
                    else f"comp{direction}"
                )
            else:
                warnings.warn(f"Direction index {direction} out of range for node {nid}.", RuntimeWarning)
                continue

            last_ylabel = f"{results_name} ({cname})" if cname else results_name
            meta[nid] = y

            global_ymin = min(global_ymin, float(np.min(y)))
            global_ymax = max(global_ymax, float(np.max(y)))

            c = coords_map[nid]
            label = f"Node {nid} @({c['x']:.2f}, {c['y']:.2f}, {c['z']:.2f})"
            ax_i.plot(time_arr, y, lw=linewidth, marker=marker, label=label, **line_kwargs)
            ax_i.grid(True)
            if split_subplots:
                ax_i.set_ylabel(last_ylabel)
                ax_i.legend(fontsize="small")

        # ---- unify limits & finalize ---------------------------------------- #
        if split_subplots:
            for ax in axes:
                ax.set_ylim(global_ymin, global_ymax)

        axes[-1].set_xlabel("Time [s]")
        if not split_subplots:
            axes[0].set_ylabel(last_ylabel)
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
        aggregate: str | Callable[[pd.Series], float] = "Mean",   # accepts str or Series->float for BC
        figsize: tuple[int, int] = (10, 4),
        linewidth: float = 1.4,
        marker: str | None = None,
        label: str | None = None,
        rasterized: bool = False,
        **plot_kwargs: Any,
    ) -> tuple[plt.Axes, dict[str, Any]]:
        """
        Plot roof drift Δu(t) = u_top(t) − u_bot(t) using Nodes.get_roof_drift().
        """

        # enforce exactly one of node_ids / selection_set_id (matches get_roof_drift contract)
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Provide either `node_ids` or `selection_set_id` (not both).")

        # Allow legacy callables that expect a pandas.Series by wrapping into ndarray->float
        agg_param = aggregate
        if callable(aggregate):
            def _agg_wrapper(arr: np.ndarray) -> float:
                try:
                    return float(aggregate(arr))  # user function already ndarray-aware
                except Exception:
                    # fallback for Series-based callables passed by older code
                    return float(aggregate(pd.Series(arr)))
            agg_param = _agg_wrapper  # pass ndarray-compatible callable to nodes layer

        # --- compute drift with the method on Nodes --------------------------------
        res = self.dataset.nodes.get_roof_drift(
            model_stage=model_stage,
            direction=direction,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            normalize=normalize,
            scaling_factor=scaling_factor,
            z_round=z_round,
            top_z=top_z,
            bottom_z=bottom_z,
            aggregate=agg_param,  # str or ndarray->float callable
        )

        # --- plot ------------------------------------------------------------------
        plot_kwargs.pop("ax", None)  # sanitize accidental kw
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        line, = ax.plot(
            res.time, res.drift,
            lw=linewidth, marker=marker, label=label,
            rasterized=rasterized, **plot_kwargs
        )
        ax.grid(True)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Drift ratio (Δu/h)" if normalize else "Δu [DISPLACEMENT]")
        ax.set_title(f"Roof Drift – dir {res.direction}  (z={res.bottom_z:.3f} → {res.top_z:.3f})")

        if label is not None:
            ax.legend()

        meta: dict[str, Any] = {
            "time": res.time,
            "steps": res.steps,
            "drift": res.drift,
            "u_top": res.u_top,
            "u_bot": res.u_bot,
            "top_z": res.top_z,
            "bottom_z": res.bottom_z,
            "height": res.height,
            "top_ids": res.top_ids,
            "bottom_ids": res.bottom_ids,
            "direction": res.direction,
            "component": res.component_name,
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
        z_round: int = 3,
        figsize: tuple[int, int] = (10, 4),
        linewidth: float = 1.2,
        marker: str | None = None,
        sharey: bool = False,
        aggregate: str | Callable[[np.ndarray], float] = "Mean",
        **line_kwargs,
    ) -> tuple["plt.Figure | None", dict[str, Any]]:
        """Thin plotting wrapper over get_story_drifts()."""
 
        res = self.dataset.nodes.get_story_drifts(
            model_stage=model_stage,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            direction=direction,
            normalize=normalize,
            scaling_factor=scaling_factor,
            sort_by=sort_by,
            reverse_sort=reverse_sort,
            z_round=z_round,
            aggregate=aggregate,
        )

        n_stories = len(res.labels)
        if split_subplots:
            fig, axes = plt.subplots(
                n_stories, 1,
                figsize=(figsize[0], figsize[1] * n_stories),
                sharex=True,
                sharey=sharey,
            )
            axes = np.atleast_1d(axes)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = np.array([ax])

        meta: dict[str, Any] = {"time": res.time}
        ymin, ymax = np.inf, -np.inf

        for i, label in enumerate(res.labels):
            y = res.drift[i]
            ax_i = axes[i] if split_subplots else axes[0]
            ax_i.plot(res.time, y, lw=linewidth, marker=marker, label=f"Story {label}", **line_kwargs)
            ax_i.grid(True)
            if split_subplots:
                ax_i.set_ylabel("Drift" + (" ratio" if normalize else ""))
                ax_i.legend(fontsize="small")
            if np.isfinite(y).any():
                ymin = min(ymin, np.nanmin(y))
                ymax = max(ymax, np.nanmax(y))
            meta[label] = y

        if split_subplots and np.isfinite(ymin) and np.isfinite(ymax):
            for ax in axes:
                ax.set_ylim(ymin, ymax)

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
        show_legend: bool = False,
        **plot_kwargs,
    ) -> tuple[plt.Axes, dict[str, Any]]:
        """
        Plot max/min drift envelope per storey (Δu or Δu / height) vs. height.
        Uses Nodes.get_story_drifts(...) to compute the time-histories and envelopes.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # Compute storey drifts + envelopes in one pass (no extra HDF5 reads here)
        res = self.dataset.nodes.get_story_drifts(
            model_stage=model_stage,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            direction=direction,
            normalize=normalize,
            scaling_factor=scaling_factor,
            sort_by=sort_by,
            reverse_sort=reverse_sort,
            z_round=3,           # match previous behaviour
            aggregate="Mean",    # average across nodes at a level
        )

        # Build plotting arrays (anchor at base with 0 drift)
        z_for_plot = np.concatenate([[res.z_base], res.z_tops])
        min_for_plot = np.concatenate([[0.0], res.envelope_min])
        max_for_plot = np.concatenate([[0.0], res.envelope_max])

        # Create axes if needed
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 8))

        # Draw min/max curves and optional fill
        ax.plot(min_for_plot, z_for_plot, "-", label="Min Drift", **plot_kwargs)
        ax.plot(max_for_plot, z_for_plot, "-", label="Max Drift", **plot_kwargs)
        if fill:
            ax.fill_betweenx(z_for_plot, min_for_plot, max_for_plot, alpha=0.2)

        # Optional vertical limit lines
        if limits is not None:
            for lim in limits:
                ax.axvline(lim, color="gray", linestyle="--", alpha=0.5, label=f"Limit {lim:g}")

        # Labels, ticks, grid, legend
        xlabel = "Drift Ratio" if normalize else "Δu [DISPLACEMENT]"
        if scaling_factor != 1.0:
            xlabel += f" ×{scaling_factor:g}"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Height [Z]")
        ax.set_yticks(np.unique(np.concatenate([[res.z_base], res.z_tops])))
        ax.grid(True)
        if show_legend:
            ax.legend()

        # Meta payload (consistent with your previous return)
        meta = {
            "story_bottom_z": np.array([b for b, _ in res.z_pairs]),
            "story_top_z": res.z_tops,
            "drift_min": res.envelope_min,
            "drift_max": res.envelope_max,
            "labels": res.labels,
            "heights": res.heights,
            "direction": res.direction,
            "component_name": res.component_name,
        }
        return ax, meta
  
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


















