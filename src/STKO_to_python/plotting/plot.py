"""Dataset-level plotting facade.

Per-result plotting lives on ``NodalResults.plot`` (a
``NodalResultsPlotter``). ``Plot`` is a thin dataset-bound facade for
one-shot "fetch + plot" convenience — the "dataset-level convenience
wrapper" from spec §8.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


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


__all__ = ["Plot"]
