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


















