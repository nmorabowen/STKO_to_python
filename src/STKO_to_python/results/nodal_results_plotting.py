# STKO_to_python/results/nodal_results_plotter.py
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..dataprocess.aggregator import StrOp

if TYPE_CHECKING:
    from .nodal_results_dataclass import NodalResults
    from ..plotting.plot_dataclasses import ModelPlotSettings


class NodalResultsPlotter:
    """
    Plotting helper bound to a NodalResults instance.

    Usage
    -----
    Basic X–Y aggregation plot:

        results.plot.xy(
            y_results_name="ACCELERATION",
            y_direction=1,                  # component index / label
            y_operation="Sum",              # one of:
                                           #   'Sum', 'Mean', 'Max', 'Min', 'Std',
                                           #   'Percentile', 'Envelope',
                                           #   'Cumulative', 'SignedCumulative',
                                           #   'RunningEnvelope'
            x_results_name="TIME",          # 'TIME', 'STEP', or another result_name
        )

    Raw time history for specific nodes:

        results.plot.plot_TH(
            result_name="ACCELERATION",
            component=1,
            node_ids=[14, 25],
        )
    """

    def __init__(self, results: "NodalResults"):
        self._results = results

    # ------------------------------------------------------------------ #
    # Small helpers for plot settings
    # ------------------------------------------------------------------ #

    @property
    def _settings(self) -> "ModelPlotSettings | None":
        return getattr(self._results, "plot_settings", None)

    def _build_line_kwargs(
        self,
        *,
        linewidth: float | None = None,
        marker: str | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """
        Merge model-level defaults (ModelPlotSettings) with per-call overrides.

        Precedence:
            ModelPlotSettings  -> base
            explicit args      -> override settings
            **extra            -> override everything
        """
        settings = self._settings

        # Collect only non-None explicit overrides for settings.to_mpl_kwargs
        overrides: dict[str, Any] = {}
        if linewidth is not None:
            overrides["linewidth"] = linewidth
        if marker is not None:
            overrides["marker"] = marker

        if settings is not None:
            base = settings.to_mpl_kwargs(**overrides)
        else:
            base = {}
            base.update(overrides)

        # Extra kwargs from caller always win
        base.update(extra)
        return base

    def _make_label(self, suffix: str | None = None, explicit: str | None = None) -> str | None:
        """
        Decide final label based on:

        1. explicit label (if provided),
        2. ModelPlotSettings.label_base + suffix,
        3. just suffix, if nothing else is set.
        """
        if explicit is not None:
            return explicit

        settings = self._settings
        if settings is None:
            return suffix

        return settings.make_label(suffix=suffix, default=suffix)

    # ------------------------------------------------------------------ #
    # Core X–Y plotting (generic, Aggregator-based)
    # ------------------------------------------------------------------ #
    def xy(
        self,
        *,
        # Y-axis ----------------------------------------------------------- #
        y_results_name: str,
        y_direction: str | int | None = None,
        y_operation: StrOp | Sequence[StrOp] = "Sum",
        y_scale: float = 1.0,
        # X-axis ----------------------------------------------------------- #
        x_results_name: str = "TIME",   # 'TIME', 'STEP', or result_name
        x_direction: str | int | None = None,
        x_operation: StrOp | Sequence[StrOp] = "Sum",
        x_scale: float = 1.0,
        # Aggregator extras ----------------------------------------------- #
        operation_kwargs: dict[str, Any] | None = None,
        # Cosmetics -------------------------------------------------------- #
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (10, 6),
        linewidth: float | None = None,
        marker: str | None = None,
        label: str | None = None,
        **line_kwargs,
    ) -> tuple[plt.Axes | None, dict[str, Any]]:
        """
        Generic X–Y plot for a NodalResults instance.

        For nodal results, this delegates all "what is a component" logic
        to `NodalResults.get()` and uses Aggregator only for the
        aggregation across nodes / time.

        Plot styling:
        -------------
        - Starts from NodalResults.plot_settings (ModelPlotSettings), if present.
        - Per-call `linewidth`, `marker`, and other **line_kwargs override those.
        """
        from ..dataprocess.aggregator import Aggregator  # lazy to avoid cycles

        operation_kwargs = operation_kwargs or {}
        res = self._results
        df = res.df

        # ------------------------------------------------------------------ #
        # helpers: TIME / STEP / nodal result via NodalResults.get()
        # ------------------------------------------------------------------ #
        def _axis_value(
            what: str,
            direction: str | int | None,
            op: StrOp | Sequence[StrOp],
            scale: float,
        ) -> np.ndarray | pd.Series | pd.DataFrame:
            # ---- TIME ------------------------------------------------------ #
            if what.upper() == "TIME":
                t = res.time
                if isinstance(t, dict):
                    # multi-stage case – ambiguous TIME
                    raise ValueError(
                        "[NodalResultsPlotter.xy] TIME axis for multi-stage "
                        "NodalResults is not supported yet.\n"
                        "Either:\n"
                        "  • specify x_results_name='STEP', or\n"
                        "  • pre-select a single stage and build a stage-only "
                        "NodalResults."
                    )
                arr = np.asarray(t, dtype=float).reshape(-1)
                return arr * float(scale)

            # ---- STEP ------------------------------------------------------ #
            if what.upper() == "STEP":
                idx = df.index
                if getattr(idx, "nlevels", 1) >= 1:
                    steps = idx.get_level_values(-1)
                else:
                    steps = np.arange(len(idx))
                arr = steps.to_numpy() if hasattr(steps, "to_numpy") else np.asarray(steps)
                return arr * float(scale)

            # ---- treat as result_name + component (use NodalResults.get) --- #
            if direction is None:
                sub = res.get(result_name=what, component=None)   # all components
            else:
                sub = res.get(result_name=what, component=direction)

            # normalise to DataFrame
            if isinstance(sub, pd.Series):
                sub = sub.to_frame()

            # flatten potential MultiIndex columns
            if isinstance(sub.columns, pd.MultiIndex):
                sub = sub.copy()
                sub.columns = [c1 for (_, c1) in sub.columns]

            # decide what to pass as "direction" to Aggregator
            if sub.shape[1] == 1:
                # <<<<<< key fix here
                eff_dir = sub.columns[0]
            else:
                eff_dir = direction

            from ..dataprocess.aggregator import Aggregator  # keep lazy import
            agg = Aggregator(sub, eff_dir)
            out = agg.compute(operation=op, **operation_kwargs)
            return out * float(scale)


        # build X and Y
        try:
            y_vals = _axis_value(y_results_name, y_direction, y_operation, y_scale)
            x_vals = _axis_value(x_results_name, x_direction, x_operation, x_scale)
        except Exception as exc:
            warnings.warn(f"[NodalResultsPlotter.xy] {exc}", RuntimeWarning)
            return None, {}

        multi_x = isinstance(x_vals, pd.DataFrame)
        multi_y = isinstance(y_vals, pd.DataFrame)

        if len(np.asarray(x_vals)) != len(np.asarray(y_vals)):
            warnings.warn("[NodalResultsPlotter.xy] X–Y length mismatch.", RuntimeWarning)
            return None, {}

        if multi_x and multi_y:
            warnings.warn(
                "[NodalResultsPlotter.xy] Both X and Y are multi-column; plot skipped.",
                RuntimeWarning,
            )
            return None, {}

        # ------------------------------------------------------------------ #
        # axes & plotting
        # ------------------------------------------------------------------ #
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        base_label = label

        # pre-build common line kwargs (style) using global defaults
        common_line_kwargs = self._build_line_kwargs(
            linewidth=linewidth,
            marker=marker,
            **line_kwargs,
        )

        if not multi_x and not multi_y:
            # single curve
            final_label = self._make_label(suffix=None, explicit=base_label)
            ax.plot(
                x_vals,
                y_vals,
                label=final_label,
                rasterized=True,
                **common_line_kwargs,
            )

        elif not multi_x and multi_y:
            # multiple Y columns, one X
            for j, col in enumerate(sorted(y_vals.columns)):
                if base_label is None:
                    # default behaviour: use component name as suffix
                    suffix = str(col)
                    final_label = self._make_label(
                        suffix=suffix,
                        explicit=None,
                    )
                else:
                    # explicit label provided -> only first curve gets it
                    final_label = base_label if j == 0 else None

                ax.plot(
                    x_vals,
                    y_vals[col],
                    label=final_label,
                    rasterized=True,
                    **common_line_kwargs,
                )

        else:  # multi_x and not multi_y
            # multiple X columns, one Y
            for j, col in enumerate(sorted(x_vals.columns)):
                if base_label is None:
                    suffix = str(col)
                    final_label = self._make_label(
                        suffix=suffix,
                        explicit=None,
                    )
                else:
                    # explicit label provided -> only first curve gets it
                    final_label = base_label if j == 0 else None

                ax.plot(
                    x_vals[col],
                    y_vals,
                    label=final_label,
                    rasterized=True,
                    **common_line_kwargs,
                )


        if ax.get_legend_handles_labels()[0]:
            ax.legend()

        ax.set_xlabel(x_results_name)
        ax.set_ylabel(y_results_name)
        ax.grid(True)

        meta = {
            "x_array": np.asarray(x_vals),
            "y_array": np.asarray(y_vals),
        }
        if multi_x or multi_y:
            meta["dataframe"] = x_vals if multi_x else y_vals

        return ax, meta


    # ------------------------------------------------------------------ #
    # Simple time-history plot: plot_TH
    # ------------------------------------------------------------------ #
    def plot_TH(
        self,
        *,
        result_name: str,
        component: object = 1,
        node_ids: Sequence[int] | None = None,
        split_subplots: bool = False,
        figsize: tuple[int, int] = (8, 3),
        linewidth: float | None = None,
        marker: str | None = None,
        sharey: bool = True,
        label_prefix: str | None = "Node",
        **line_kwargs,
    ) -> tuple[plt.Figure | None, dict[str, Any]]:
        """
        Plot raw nodal time-history curves for a single result/component.

        Assumes the NodalResults corresponds to a **single stage**, with
        index (node_id, step). If multiple stages are present in the index,
        this currently raises.

        Parameters
        ----------
        result_name
            e.g. 'DISPLACEMENT', 'ACCELERATION'.
        component
            Component label passed to NodalResults.get(), e.g. 1, '1', 'x'.
        node_ids
            If None → plot all node_ids present.
            If sequence → filter to these node_ids.
        split_subplots
            If True → one subplot per node; otherwise all curves on one Axes.
        """
        res = self._results
        df = res.df

        # ---- TIME ARRAY ----------------------------------------------------- #
        t = res.time
        if isinstance(t, dict):
            raise ValueError(
                "[NodalResultsPlotter.plot_TH] This NodalResults has per-stage "
                "time information. For now, build a stage-specific "
                "NodalResults before calling plot_TH."
            )
        time_arr = np.asarray(t, dtype=float).reshape(-1)

        # ---- extract the single component via NodalResults.get ------------- #
        series_or_df = res.get(result_name=result_name, component=component)

        if isinstance(series_or_df, pd.DataFrame):
            if series_or_df.shape[1] != 1:
                warnings.warn(
                    f"[plot_TH] result '{result_name}' component '{component}' returned "
                    f"{series_or_df.shape[1]} columns; using first.",
                    RuntimeWarning,
                )
            series = series_or_df.iloc[:, 0]
        else:
            series = series_or_df  # already a Series

        idx = series.index
        if getattr(idx, "nlevels", 1) != 2:
            raise ValueError(
                "[NodalResultsPlotter.plot_TH] Expected index (node_id, step). "
                f"Got nlevels={getattr(idx, 'nlevels', 1)}."
            )

        node_level = 0
        step_level = 1

        # ---- which nodes to plot ------------------------------------------- #
        all_nodes = np.unique(idx.get_level_values(node_level).to_numpy())
        if node_ids is None:
            node_ids_use = all_nodes
        else:
            node_ids_use = np.intersect1d(
                all_nodes,
                np.asarray(node_ids, dtype=all_nodes.dtype),
            )
            if node_ids_use.size == 0:
                raise ValueError(
                    "[plot_TH] None of the requested node_ids are present.\n"
                    f"Available node_ids: {all_nodes.tolist()}"
                )

        # ---- figure & axes -------------------------------------------------- #
        if split_subplots:
            fig, axes = plt.subplots(
                len(node_ids_use), 1,
                figsize=(figsize[0], figsize[1] * len(node_ids_use)),
                sharex=True,
                sharey=sharey,
            )
            axes = np.atleast_1d(axes)
        else:
            fig, ax = plt.subplots(figsize=figsize)
            axes = np.array([ax])

        meta: dict[str, Any] = {"time": time_arr}
        global_ymin, global_ymax = np.inf, -np.inf

        # common style for all curves
        common_line_kwargs = self._build_line_kwargs(
            linewidth=linewidth,
            marker=marker,
            **line_kwargs,
        )

        # ---- loop over nodes ----------------------------------------------- #
        for i, nid in enumerate(node_ids_use):
            ax_i = axes[i] if split_subplots else axes[0]

            try:
                s_node = series.xs(nid, level=node_level)  # index = step
            except KeyError:
                warnings.warn(f"[plot_TH] No data for node {nid}.", RuntimeWarning)
                continue

            steps = s_node.index.to_numpy(dtype=int)
            y = s_node.to_numpy(dtype=float)

            # align with time array via step index
            valid = (steps >= 0) & (steps < len(time_arr))
            if not np.all(valid):
                warnings.warn(
                    f"[plot_TH] Node {nid} has {np.count_nonzero(~valid)} "
                    "step(s) outside time range; trimming.",
                    RuntimeWarning,
                )
                steps = steps[valid]
                y = y[valid]

            x = time_arr[steps]

            suffix = f"{label_prefix} {nid}" if label_prefix else f"{nid}"
            final_label = self._make_label(suffix=suffix, explicit=None)

            ax_i.plot(x, y, label=final_label, **common_line_kwargs)
            ax_i.grid(True)

            global_ymin = min(global_ymin, float(np.nanmin(y)))
            global_ymax = max(global_ymax, float(np.nanmax(y)))
            meta[int(nid)] = y

            if split_subplots:
                ax_i.set_ylabel(result_name)
                if ax_i.get_legend_handles_labels()[0]:
                    ax_i.legend(fontsize="small")

        # ---- unify limits & final touches ---------------------------------- #
        if split_subplots and np.isfinite(global_ymin) and np.isfinite(global_ymax):
            for ax_sub in axes:
                ax_sub.set_ylim(global_ymin, global_ymax)

        axes[-1].set_xlabel("Time")
        if not split_subplots:
            axes[0].set_ylabel(result_name)
            if axes[0].get_legend_handles_labels()[0]:
                axes[0].legend(fontsize="small")

        fig.tight_layout()
        return fig, meta
