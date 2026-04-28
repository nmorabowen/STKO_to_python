"""Dataset-level plotting facade.

Per-result plotting lives on ``NodalResults.plot`` (a
``NodalResultsPlotter``). ``Plot`` is a thin dataset-bound facade for
one-shot "fetch + plot" convenience — the "dataset-level convenience
wrapper" from spec §8.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

from .deformed_shape import plot_deformed_shape, plot_undeformed_shape
from .mesh import plot_mesh, plot_mesh_with_contour

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..elements.element_results import ElementResults


class Plot:
    """Dataset-level plotting facade.

    Held on ``MPCODataSet.plot``. Offers one-shot "fetch a result and
    plot it" convenience methods; each method fetches a
    :class:`NodalResults` via ``dataset.nodes.get_nodal_results(...)``
    and delegates the rendering to :class:`NodalResultsPlotter`.

    For repeated plots off the same result, prefer fetching once:

        nr = ds.nodes.get_nodal_results(...)
        nr.plot.xy(...)   # reuses the cached NodalResults
    """

    def __init__(self, dataset: "MPCODataSet") -> None:
        self._dataset = dataset

    def __repr__(self) -> str:
        return f"<Plot facade for {type(self._dataset).__name__}>"

    # ------------------------------------------------------------------ #
    # Dataset-level convenience wrappers
    # ------------------------------------------------------------------ #

    def xy(
        self,
        *,
        # --- fetch args --------------------------------------------------- #
        model_stage: str | Sequence[str],
        results_name: str,
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        # --- plotter args (forwarded verbatim to NodalResultsPlotter.xy) - #
        y_direction: str | int | None = None,
        y_operation: Any = "Sum",
        y_scale: float = 1.0,
        x_results_name: str = "TIME",
        x_direction: str | int | None = None,
        x_operation: Any = "Sum",
        x_scale: float = 1.0,
        operation_kwargs: dict[str, Any] | None = None,
        ax: plt.Axes | None = None,
        figsize: tuple[int, int] = (10, 6),
        linewidth: float | None = None,
        marker: str | None = None,
        label: str | None = None,
        **line_kwargs,
    ) -> tuple[plt.Axes | None, dict[str, Any]]:
        """Fetch a NodalResults then delegate to NodalResultsPlotter.xy.

        ``results_name`` is the Y-axis result; ``x_results_name`` defaults
        to ``"TIME"`` (other valid values: ``"STEP"``, or another result
        name, which will trigger a second fetch inside the plotter).
        """
        nr = self._dataset.nodes.get_nodal_results(
            results_name=results_name,
            model_stage=model_stage,
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )
        return nr.plot.xy(
            y_results_name=results_name,
            y_direction=y_direction,
            y_operation=y_operation,
            y_scale=y_scale,
            x_results_name=x_results_name,
            x_direction=x_direction,
            x_operation=x_operation,
            x_scale=x_scale,
            operation_kwargs=operation_kwargs,
            ax=ax,
            figsize=figsize,
            linewidth=linewidth,
            marker=marker,
            label=label,
            **line_kwargs,
        )


    # ------------------------------------------------------------------ #
    # Deformed-mesh visualization
    # ------------------------------------------------------------------ #

    def deformed_shape(
        self,
        *,
        model_stage: str,
        step: int,
        scale: float = 1.0,
        ax: Any = None,
        show_undeformed: bool = True,
        color: Any = "C0",
        undeformed_color: Any = "0.7",
        linewidth: float = 1.2,
        undeformed_linewidth: float = 0.8,
        alpha: float = 1.0,
        undeformed_alpha: float = 0.6,
        title: Optional[str] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Render the deformed mesh at a given step.

        Computes ``deformed = node_coords + scale * displacement`` for
        every node, then draws element edges class-by-class. Returns
        ``(ax, meta)`` like the other plot helpers; see
        :func:`STKO_to_python.plotting.deformed_shape.plot_deformed_shape`
        for the full parameter list.
        """
        return plot_deformed_shape(
            self._dataset,
            model_stage=model_stage,
            step=step,
            scale=scale,
            ax=ax,
            show_undeformed=show_undeformed,
            color=color,
            undeformed_color=undeformed_color,
            linewidth=linewidth,
            undeformed_linewidth=undeformed_linewidth,
            alpha=alpha,
            undeformed_alpha=undeformed_alpha,
            title=title,
        )

    def undeformed_shape(
        self,
        *,
        ax: Any = None,
        color: Any = "0.3",
        linewidth: float = 1.0,
        alpha: float = 1.0,
        title: Optional[str] = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Render the original (undeformed) mesh — sanity-check helper."""
        return plot_undeformed_shape(
            self._dataset,
            ax=ax,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            title=title,
        )


    # ------------------------------------------------------------------ #
    # Mesh visualization ("show me the model")
    # ------------------------------------------------------------------ #

    def mesh(
        self,
        *,
        model_stage: Optional[str] = None,
        element_type: Optional[str] = None,
        element_ids: Union[int, Sequence[int], np.ndarray, None] = None,
        ax: Any = None,
        edge_color: Any = "lightgray",
        linewidth: float = 0.5,
        alpha: float = 1.0,
        title: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Render element edges from the cached connectivity.

        Useful as a "show me the model" backdrop before or under a
        contour plot. Composes with
        :meth:`ElementResultsPlotter.scatter` — call ``mesh()`` first,
        then pass the returned ``ax`` to ``er.plot.scatter(..., ax=ax)``
        so a single axes carries both the wireframe and the IP scatter.

        See :func:`STKO_to_python.plotting.mesh.plot_mesh` for the full
        parameter list.

        Examples
        --------
        Compose mesh + contour into a single axes::

            ax, _ = ds.plot.mesh(element_type="ASDShellQ4")
            er = ds.elements.get_element_results(
                results_name="section.force",
                element_type="ASDShellQ4",
            )
            er.plot.scatter("bending_moment_xx", step=10, ax=ax)
        """
        return plot_mesh(
            self._dataset,
            model_stage=model_stage,
            element_type=element_type,
            element_ids=element_ids,
            ax=ax,
            edge_color=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            title=title,
            **kwargs,
        )

    def mesh_with_contour(
        self,
        element_results: "ElementResults",
        component_canonical: str,
        *,
        step: int,
        model_stage: Optional[str] = None,
        element_type: Optional[str] = None,
        element_ids: Union[int, Sequence[int], np.ndarray, None] = None,
        ax: Any = None,
        edge_color: Any = "lightgray",
        linewidth: float = 0.5,
        alpha: float = 1.0,
        axes: Tuple[str, str] = ("x", "y"),
        title: Optional[str] = None,
        **scatter_kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Convenience wrapper: ``mesh()`` then ``er.plot.scatter()`` on the same axes.

        See :func:`STKO_to_python.plotting.mesh.plot_mesh_with_contour`.
        """
        return plot_mesh_with_contour(
            self._dataset,
            element_results,
            component_canonical,
            step=step,
            model_stage=model_stage,
            element_type=element_type,
            element_ids=element_ids,
            ax=ax,
            edge_color=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            axes=axes,
            title=title,
            **scatter_kwargs,
        )


__all__ = ["Plot"]
