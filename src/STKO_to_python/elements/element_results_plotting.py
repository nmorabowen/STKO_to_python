"""Plotting helper bound to an :class:`ElementResults` instance.

Three plot families covering the most common engineering workflows:

* :meth:`ElementResultsPlotter.history` — time history of a component
  for one or more elements (and optionally a specific IP).
* :meth:`ElementResultsPlotter.diagram` — for line elements (beams),
  plot a component as a function of physical position along the
  element at a single step. The classic moment / shear / axial
  diagram.
* :meth:`ElementResultsPlotter.scatter` — for shells, plane elements,
  or solids, scatter integration-point physical positions colored by
  the component value at a step. A lightweight contour-style view
  that doesn't need triangulation or a mesh-aware renderer.

All methods accept an optional ``ax`` so they compose cleanly with
user-built figures, and return ``(ax, meta)`` like
:class:`NodalResultsPlotter` for symmetry.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .element_results import ElementResults


# Map "x"/"y"/"z" to physical_coords axis index.
_AXIS_NAME_TO_IDX = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


class ElementResultsPlotter:
    """Plotting helper for :class:`ElementResults`.

    Held as ``ElementResults.plot`` and rebuilt on unpickle.
    """

    __slots__ = ("_results",)

    def __init__(self, results: "ElementResults") -> None:
        self._results = results

    # ------------------------------------------------------------------ #
    # Time history                                                        #
    # ------------------------------------------------------------------ #

    def history(
        self,
        component: str,
        *,
        element_ids: Union[int, Sequence[int], None] = None,
        ax: Any = None,
        x_axis: str = "time",
        **plot_kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Plot the time history of one component for one or more elements.

        Parameters
        ----------
        component : str
            Column name (e.g. ``"Mz_1"``, ``"P_ip2"``,
            ``"sigma11_l0_ip0"``).
        element_ids : int, sequence of int, or None
            Restrict to specific elements. If ``None``, plots every
            element in the result (one curve each — be wary on large
            meshes).
        ax : matplotlib Axes or None
            Existing axes to draw on. If ``None``, a new figure is
            created.
        x_axis : ``"time"`` or ``"step"``
            x-axis source. ``"time"`` requires ``self._results.time``.
        **plot_kwargs
            Forwarded to ``ax.plot``.

        Returns
        -------
        (ax, meta) : tuple
            ``meta`` carries ``{"x", "y_per_element"}`` for inspection.
        """
        import matplotlib.pyplot as plt

        er = self._results
        if component not in er.df.columns:
            raise ValueError(
                f"Component {component!r} not in this result. "
                f"Available: {er.list_components()[:20]}"
                + (" ..." if len(er.list_components()) > 20 else "")
            )

        if element_ids is None:
            ids = list(er.element_ids)
        elif isinstance(element_ids, (int, np.integer)):
            ids = [int(element_ids)]
        else:
            ids = [int(e) for e in element_ids]

        if x_axis == "time":
            if not isinstance(er.time, np.ndarray) or er.time.size == 0:
                raise ValueError(
                    "x_axis='time' requested but no time array on this "
                    "result. Use x_axis='step' instead."
                )
            x_full = er.time
            x_label = "Time"
        elif x_axis == "step":
            x_full = np.arange(er.n_steps, dtype=np.int64)
            x_label = "Step"
        else:
            raise ValueError(f"x_axis must be 'time' or 'step', got {x_axis!r}")

        if ax is None:
            fig, ax = plt.subplots()

        ser = er.df[component]
        per_element: dict[int, np.ndarray] = {}
        for eid in ids:
            try:
                y = ser.xs(eid, level="element_id").sort_index().to_numpy()
            except KeyError:
                continue
            x = x_full[: y.size]
            label = plot_kwargs.pop("label", None) if len(ids) == 1 else f"e{eid}"
            ax.plot(x, y, label=label, **plot_kwargs)
            per_element[eid] = y

        ax.set_xlabel(x_label)
        ax.set_ylabel(component)
        if len(ids) > 1 and len(ids) <= 12:
            ax.legend()
        return ax, {"x": x_full, "y_per_element": per_element}

    # ------------------------------------------------------------------ #
    # Beam diagrams (line elements)                                       #
    # ------------------------------------------------------------------ #

    def diagram(
        self,
        component_canonical: str,
        *,
        element_id: int,
        step: int,
        ax: Any = None,
        x_in_natural: bool = False,
        **plot_kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Plot a force / moment / strain diagram along a line element.

        For a single element and step, plots the *canonical* quantity
        (e.g. ``"axial_force"``, ``"bending_moment_z"``) as a function
        of physical position along the element.

        Only valid for line elements (``gp_dim == 1``) — this is the
        classic beam diagram. For shells / solids see :meth:`scatter`.

        Parameters
        ----------
        component_canonical : str
            Canonical name. Must resolve to ``self._results.n_ip``
            columns (one per IP) — same constraint as
            :meth:`integrate_canonical`.
        element_id : int
        step : int
        ax : matplotlib Axes or None
        x_in_natural : bool, default False
            If ``False``, x-axis is physical position along the beam
            (requires node coords). If ``True``, x-axis is the natural
            ξ ∈ [-1, +1].
        **plot_kwargs
            Forwarded to ``ax.plot``.
        """
        import matplotlib.pyplot as plt

        er = self._results
        if er.gp_dim != 1:
            raise ValueError(
                f"diagram() is only valid for line elements (gp_dim=1). "
                f"This result has gp_dim={er.gp_dim}. Use scatter() for "
                f"shells / solids."
            )

        cols = er.canonical_columns(component_canonical)
        if not cols:
            raise ValueError(
                f"Canonical {component_canonical!r} doesn't match any "
                f"columns in this result. Present canonicals: "
                f"{er.list_canonicals()}"
            )
        if len(cols) != er.n_ip:
            raise ValueError(
                f"Canonical {component_canonical!r} resolves to "
                f"{len(cols)} columns but the bucket has {er.n_ip} IPs. "
                f"diagram() needs one column per IP."
            )

        try:
            row = er.df.xs((int(element_id), int(step)))
        except KeyError as err:
            raise ValueError(
                f"({element_id}, {step}) not in this result"
            ) from err
        y = row[list(cols)].to_numpy(dtype=np.float64)

        if x_in_natural:
            if er.gp_xi is None:
                raise ValueError(
                    "x_in_natural=True but gp_xi is None on this result"
                )
            x = er.gp_xi
            x_label = "ξ (natural)"
        else:
            if er.element_node_coords is None:
                raise ValueError(
                    "Physical x requested but element_node_coords is None. "
                    "Pass x_in_natural=True to plot in parent coordinates."
                )
            # Find the row index for this element_id.
            try:
                eid_row = list(er.element_ids).index(int(element_id))
            except ValueError:
                raise ValueError(
                    f"element_id {element_id} not in self.element_ids"
                )
            nc = er.element_node_coords[eid_row]
            length = float(np.linalg.norm(nc[-1] - nc[0]))
            x = er.physical_x(length)
            x_label = "Position along element"

        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(x, y, marker="o", **plot_kwargs)
        ax.set_xlabel(x_label)
        ax.set_ylabel(component_canonical)
        ax.set_title(f"Element {element_id}, step {step}")
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color="black", linewidth=0.5)
        return ax, {"x": x, "y": y, "columns": list(cols)}

    # ------------------------------------------------------------------ #
    # Spatial scatter (shells / planes / solids)                          #
    # ------------------------------------------------------------------ #

    def scatter(
        self,
        component_canonical: str,
        *,
        step: int,
        ax: Any = None,
        axes: Tuple[str, str] = ("x", "y"),
        **scatter_kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Scatter integration-point physical positions colored by value.

        For shells, plane elements, and solids: at a given step, plot
        a 2-D projection of every IP's physical position colored by
        the canonical value at that IP. Lightweight stand-in for a
        proper contour plot — no mesh-aware rendering, no
        triangulation, just a colored scatter.

        Parameters
        ----------
        component_canonical : str
            Canonical name. Must resolve to ``n_ip`` columns.
        step : int
        ax : matplotlib Axes or None
        axes : tuple of two of ``"x"``, ``"y"``, ``"z"``
            Which physical axes to use for the 2-D plot. Default
            ``("x", "y")`` (top-down view); use ``("x", "z")`` for an
            elevation view.
        **scatter_kwargs
            Forwarded to ``ax.scatter`` (e.g. ``cmap``, ``s``).
        """
        import matplotlib.pyplot as plt

        er = self._results
        phys = er.physical_coords()
        if phys is None:
            raise ValueError(
                "physical_coords() is None on this result — physical "
                "scatter unavailable. Either the bucket is closed-form, "
                "the element class isn't in the shape-function catalog, "
                "or node coords weren't populated at fetch time."
            )

        cols = er.canonical_columns(component_canonical)
        if not cols:
            raise ValueError(
                f"Canonical {component_canonical!r} doesn't match any "
                f"columns. Present canonicals: {er.list_canonicals()}"
            )
        if len(cols) != er.n_ip:
            raise ValueError(
                f"Canonical {component_canonical!r} resolves to "
                f"{len(cols)} columns but the bucket has {er.n_ip} IPs. "
                f"scatter() needs one column per IP."
            )

        try:
            ax0_idx = _AXIS_NAME_TO_IDX[axes[0]]
            ax1_idx = _AXIS_NAME_TO_IDX[axes[1]]
        except KeyError as err:
            raise ValueError(f"axes must be from x/y/z, got {axes!r}") from err

        snap = er.at_step(int(step))[list(cols)]
        # Align IP values to elements (snap is indexed by element_id;
        # element_ids ordering matches phys's first axis).
        try:
            snap_aligned = snap.loc[list(er.element_ids)]
        except KeyError:
            raise ValueError(
                f"step {step}: not all element_ids present"
            )
        values = snap_aligned.to_numpy()  # (n_e, n_ip)
        xs = phys[:, :, ax0_idx].ravel()
        ys = phys[:, :, ax1_idx].ravel()
        cs = values.ravel()

        if ax is None:
            fig, ax = plt.subplots()
        # Sensible defaults
        scatter_kwargs.setdefault("cmap", "viridis")
        scatter_kwargs.setdefault("s", 20)
        sc = ax.scatter(xs, ys, c=cs, **scatter_kwargs)
        # Caller can use ax.figure.colorbar(sc) if they want one.
        ax.set_xlabel(f"{axes[0]} (physical)")
        ax.set_ylabel(f"{axes[1]} (physical)")
        ax.set_title(f"{component_canonical} at step {step}")
        ax.set_aspect("equal", adjustable="datalim")
        return ax, {
            "x": xs,
            "y": ys,
            "values": cs,
            "scatter": sc,
        }
