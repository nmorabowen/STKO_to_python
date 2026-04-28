"""Mesh ("show me the model") visualization for an :class:`MPCODataSet`.

Renders element edges from the cached connectivity so users can orient
themselves before — or under — a contour plot. The companion
:func:`plot_mesh_with_contour` overlays an
:class:`~STKO_to_python.elements.element_results.ElementResults` scatter
on top of the same axes for the 80% "show the mesh, color the IPs"
workflow.

Edge topology is shared with the deformed-mesh renderer
(:mod:`STKO_to_python.plotting.deformed_shape`) — line elements get one
segment, triangles three, quads four, hex bricks twelve. Anything else
is skipped with a warning.

The public entry points sit on :class:`STKO_to_python.plotting.plot.Plot`
(``ds.plot.mesh`` / ``ds.plot.mesh_with_contour``); this module is the
implementation.
"""
from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .deformed_shape import (
    _autoscale_axes,
    _build_segments,
    _class_label,
    _decide_3d,
    _draw_segments,
    _edge_topology,
)

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..elements.element_results import ElementResults


def _filter_elements(
    df_elements: pd.DataFrame,
    *,
    element_type: Optional[str],
    element_ids: Union[int, Sequence[int], np.ndarray, None],
) -> pd.DataFrame:
    """Apply element_type / element_ids filters to the element index."""
    df = df_elements
    if element_type is not None:
        base = str(element_type).split("[")[0]
        df = df[df["element_type"].str.startswith(base)]
    if element_ids is not None:
        if isinstance(element_ids, (int, np.integer)):
            ids = {int(element_ids)}
        else:
            ids = {int(e) for e in element_ids}
        df = df[df["element_id"].isin(ids)]
    return df


def plot_mesh(
    dataset: "MPCODataSet",
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
) -> Tuple[Any, Dict[str, Any]]:
    """Render element edges from the cached connectivity.

    Parameters
    ----------
    dataset : MPCODataSet
        Source dataset.
    model_stage : str or None
        Accepted for API symmetry with the rest of the plot facade.
        Node coordinates are stage-independent in the MPCO format, so
        this argument is currently advisory; supply it for clarity in
        plots that combine the mesh with a stage-specific contour.
    element_type : str or None
        Restrict to one element class (matched on the type prefix
        before ``[`` — e.g. ``"203-ASDShellQ4"`` or ``"ASDShellQ4"``).
    element_ids : int, sequence of int, or None
        Restrict to specific element IDs (in addition to
        ``element_type`` if both are passed).
    ax : matplotlib Axes or None
        Existing axes. If ``None``, a new figure is created. The
        renderer auto-selects 2D/3D from the model geometry; if you
        pass ``ax`` you are responsible for matching its projection.
    edge_color : matplotlib color, default ``"lightgray"``
    linewidth : float, default 0.5
    alpha : float, default 1.0
    title : str or None
        Optional axes title.
    **kwargs
        Forwarded to the underlying ``LineCollection`` /
        ``Line3DCollection`` only when no specialized handling exists.
        Currently unused — present so callers can pass extras without
        a ``TypeError``.

    Returns
    -------
    (ax, meta) : tuple
        ``meta`` carries:

        * ``n_edges`` — total segments drawn
        * ``n_elements_drawn`` — element rows that contributed
        * ``edges_per_class`` — ``{class_label: edges_per_element}``
        * ``skipped_classes`` — class labels with unsupported topology
        * ``is_3d`` — whether the axes are 3D
        * ``model_stage`` — passed through verbatim
    """
    df_nodes = dataset.nodes_info["dataframe"]
    df_elements = dataset.elements_info["dataframe"]
    if df_nodes.empty or df_elements.empty:
        raise ValueError("Dataset has no nodes or no elements to draw.")

    df_filtered = _filter_elements(
        df_elements, element_type=element_type, element_ids=element_ids
    )
    if df_filtered.empty:
        raise ValueError(
            "No elements remain after filtering (element_type="
            f"{element_type!r}, element_ids={element_ids!r})."
        )

    coord_lookup = {
        int(row.node_id): np.array([row.x, row.y, row.z], dtype=np.float64)
        for row in df_nodes.itertuples(index=False)
    }

    is_3d = _decide_3d(df_filtered, df_nodes)

    if ax is None:
        import matplotlib.pyplot as plt

        if is_3d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig, ax = plt.subplots()

    segs_per_class, edges_per_class, skipped = _build_segments(
        df_elements=df_filtered, coord_lookup=coord_lookup
    )

    n_edges = 0
    n_elements_drawn = 0
    for label, segs in segs_per_class.items():
        _draw_segments(
            ax,
            segs,
            is_3d=is_3d,
            color=edge_color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=1.0,
        )
        n_edges += int(segs.shape[0])
        epe = edges_per_class.get(label, 0)
        if epe:
            n_elements_drawn += int(segs.shape[0] // epe)

    if skipped:
        warnings.warn(
            "[mesh] Skipped element classes with unsupported topology: "
            f"{skipped}",
            RuntimeWarning,
        )

    coord_stack = np.stack(list(coord_lookup.values()))
    _autoscale_axes(ax, coord_stack, is_3d=is_3d)

    if not is_3d:
        ax.set_aspect("equal", adjustable="datalim")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if is_3d:
        ax.set_zlabel("z")
    if title is not None:
        ax.set_title(title)

    meta: Dict[str, Any] = {
        "n_edges": n_edges,
        "n_elements_drawn": n_elements_drawn,
        "edges_per_class": edges_per_class,
        "skipped_classes": skipped,
        "is_3d": is_3d,
        "model_stage": model_stage,
    }
    return ax, meta


def plot_mesh_with_contour(
    dataset: "MPCODataSet",
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
) -> Tuple[Any, Dict[str, Any]]:
    """One-call mesh + IP-colored scatter overlay.

    Calls :func:`plot_mesh` then ``element_results.plot.scatter`` into
    the same axes — the 80% "show the model, color the IPs" workflow.

    Parameters
    ----------
    dataset : MPCODataSet
    element_results : ElementResults
        Result whose IPs will be colored.
    component_canonical : str
        Forwarded to :meth:`ElementResultsPlotter.scatter`.
    step : int
        Forwarded to :meth:`ElementResultsPlotter.scatter`.
    model_stage, element_type, element_ids, ax, edge_color, linewidth,
    alpha, title :
        Forwarded to :func:`plot_mesh` (mesh styling).
    axes : tuple of two of ``"x"``, ``"y"``, ``"z"``
        Projection plane for the scatter — must match the mesh axes.
    **scatter_kwargs
        Forwarded to ``ax.scatter`` (e.g. ``cmap``, ``s``).

    Returns
    -------
    (ax, meta) : tuple
        ``meta`` carries ``{"mesh": <plot_mesh meta>,
        "scatter": <ElementResultsPlotter.scatter meta>}``.
    """
    ax, mesh_meta = plot_mesh(
        dataset,
        model_stage=model_stage,
        element_type=element_type,
        element_ids=element_ids,
        ax=ax,
        edge_color=edge_color,
        linewidth=linewidth,
        alpha=alpha,
        title=title,
    )
    _, scatter_meta = element_results.plot.scatter(
        component_canonical,
        step=int(step),
        ax=ax,
        axes=axes,
        **scatter_kwargs,
    )
    return ax, {"mesh": mesh_meta, "scatter": scatter_meta}


__all__ = ["plot_mesh", "plot_mesh_with_contour"]
