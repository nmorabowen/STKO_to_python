"""Plotting helper bound to an :class:`ElementResults` instance.

Four plot families covering the most common engineering workflows:

* :meth:`ElementResultsPlotter.history` — time history of a single
  column for one or more elements.
* :meth:`ElementResultsPlotter.history_canonical` — like ``history``
  but takes a canonical engineering name (``"strain_11"``,
  ``"bending_moment_z"``, …) and collapses the fiber / IP / layer
  columns it expands to with a chosen reduction (``mean``, ``max``,
  ``abs_max``, …). One curve per element.
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

import re
from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .element_results import ElementResults


# Map "x"/"y"/"z" to physical_coords axis index.
_AXIS_NAME_TO_IDX = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


# --------------------------------------------------------------------- #
# Private helper for history_canonical                                  #
# --------------------------------------------------------------------- #

# Allowed reductions for history_canonical. Lambdas operate row-wise
# (axis=1) on a DataFrame of fiber / IP / layer columns and return a
# Series indexed identically to the input.
_CANONICAL_REDUCERS: dict[str, Any] = {
    "mean":    lambda df: df.mean(axis=1),
    "median":  lambda df: df.median(axis=1),
    "max":     lambda df: df.max(axis=1),
    "min":     lambda df: df.min(axis=1),
    "abs_max": lambda df: df.abs().max(axis=1),
    "sum":     lambda df: df.sum(axis=1),
}

# ``over`` modes and which anchor index they make mandatory. ``"all"``
# requires nothing; the others require the partner axes to be pinned so
# the reduction collapses the named axis rather than silently folding in
# every other axis too.
_OVER_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "all":    (),
    "fibers": ("ip_idx",),
    "ips":    ("fiber_idx",),
    "layers": ("fiber_idx", "ip_idx"),
}


def _filter_canonical_columns(
    columns: Sequence[str],
    *,
    fiber_idx: Optional[int] = None,
    ip_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> list[str]:
    """Keep only columns whose ``_f<F>``, ``_l<L>``, ``_ip<K>`` tags
    match every supplied filter.

    Tag grammar (per ``io/meta_parser.py``):

    * ``<comp>_f<F>_ip<K>``            — compressed fibers
    * ``<comp>_l<L>_ip<K>``            — layered shell, no fibers
    * ``<comp>_f<F>_l<L>_ip<K>``       — layered shell with fibers
    * ``<comp>_ip<K>``                 — line-station / gauss-level

    Anchors that aren't present in a given column simply don't match —
    e.g. ``layer_idx=2`` on a compressed-fiber bucket drops every column
    (which is the right answer; layers don't exist there).
    """
    kept: list[str] = []
    fiber_re = re.compile(rf"_f{fiber_idx}(_|$)") if fiber_idx is not None else None
    layer_re = re.compile(rf"_l{layer_idx}(_|$)") if layer_idx is not None else None
    ip_re    = re.compile(rf"_ip{ip_idx}$")       if ip_idx    is not None else None
    for c in columns:
        s = str(c)
        if fiber_re is not None and not fiber_re.search(s):
            continue
        if layer_re is not None and not layer_re.search(s):
            continue
        if ip_re is not None and not ip_re.search(s):
            continue
        kept.append(s)
    return kept


def _reduce_fiber_ip(
    df: pd.DataFrame,
    *,
    over: str,
    reduce: str,
    fiber_idx: Optional[int] = None,
    ip_idx: Optional[int] = None,
    layer_idx: Optional[int] = None,
) -> tuple[pd.Series, list[str]]:
    """Filter a fiber/IP/layer DataFrame and row-reduce across its columns.

    Returns ``(reduced_series, columns_used)`` where ``reduced_series``
    has the same index as ``df`` (typically ``(element_id, step)``).
    Pure: no plotting, no dataset access — easy to unit-test against a
    synthetic DataFrame.
    """
    if over not in _OVER_REQUIREMENTS:
        raise ValueError(
            f"over={over!r} not in {tuple(_OVER_REQUIREMENTS)}."
        )
    if reduce not in _CANONICAL_REDUCERS:
        raise ValueError(
            f"reduce={reduce!r} not in {tuple(_CANONICAL_REDUCERS)}."
        )
    required = _OVER_REQUIREMENTS[over]
    locals_map = {"fiber_idx": fiber_idx, "ip_idx": ip_idx, "layer_idx": layer_idx}
    missing = [name for name in required if locals_map[name] is None]
    if missing:
        raise ValueError(
            f"over={over!r} requires {missing}; pass them as keyword args "
            f"so the reduction collapses {over!r} at fixed partner indices."
        )

    cols = _filter_canonical_columns(
        df.columns,
        fiber_idx=fiber_idx,
        ip_idx=ip_idx,
        layer_idx=layer_idx,
    )
    if not cols:
        # Name the filters in the message so users know what wiped them.
        anchors = {
            k: v for k, v in (
                ("fiber_idx", fiber_idx),
                ("ip_idx", ip_idx),
                ("layer_idx", layer_idx),
            ) if v is not None
        }
        tag_re = re.compile(r"_f\d+|_l\d+|_ip\d+")
        present_tags = sorted(
            {tag for c in df.columns for tag in tag_re.findall(str(c))}
        )
        raise ValueError(
            f"No columns matched anchors {anchors or '{}'}. "
            f"Available column tail-tags: {present_tags}"
        )

    series = _CANONICAL_REDUCERS[reduce](df[cols])
    return series, cols


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
        annotate_stage_boundaries: bool = True,
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

        meta: dict[str, Any] = {"x": x_full, "y_per_element": per_element}

        # Stage-boundary annotation for multi-stage results. Positions
        # come from ``stage_step_ranges``; we map them to the chosen
        # x-axis (time or step) so the line sits on the last sample of
        # each completed stage.
        if annotate_stage_boundaries and er.is_multi_stage:
            ranges = er.stage_step_ranges
            stages = er.model_stages
            meta["stage_boundaries"] = []
            for st in stages[:-1]:
                end_step = ranges.get(st, (0, 0))[1]
                if x_axis == "time":
                    if 0 < end_step <= x_full.size:
                        x_b = float(x_full[end_step - 1])
                    else:
                        continue
                else:  # step
                    x_b = float(end_step - 1)
                meta["stage_boundaries"].append((st, x_b))
                ax.axvline(
                    x_b, color="0.5", linestyle="--",
                    linewidth=0.8, alpha=0.6,
                )

        return ax, meta

    # ------------------------------------------------------------------ #
    # Time history of a canonical name, with fiber/IP/layer reduction    #
    # ------------------------------------------------------------------ #

    def history_canonical(
        self,
        canonical_name: str,
        *,
        over: str = "all",
        reduce: str = "mean",
        fiber_idx: Optional[int] = None,
        ip_idx: Optional[int] = None,
        layer_idx: Optional[int] = None,
        element_ids: Union[int, Sequence[int], None] = None,
        ax: Any = None,
        x_axis: str = "time",
        annotate_stage_boundaries: bool = True,
        **plot_kwargs: Any,
    ) -> Tuple[Any, dict]:
        """Time history of a canonical quantity, reduced across fibers /
        IPs / layers — one curve per element.

        Sibling to :meth:`history`. Where ``history`` takes a single raw
        column (e.g. ``"eps11_f0_ip0"``), this method takes the
        engineering name (``"strain_11"``) which typically expands to
        ``n_fibers × n_ip`` (or ``n_fibers × n_layers × n_ip``) columns,
        then collapses them with the chosen reduction.

        Parameters
        ----------
        canonical_name : str
            Engineering name. Must be in :meth:`ElementResults.list_canonicals`.
        over : ``"all"`` | ``"fibers"`` | ``"ips"`` | ``"layers"``
            Which axis the reduction collapses. ``"all"`` folds every
            fiber / IP / layer column. The other three require their
            partner axes to be anchored so the reduction is well-defined:

            * ``"fibers"`` needs ``ip_idx`` (collapses fibers at that IP).
            * ``"ips"`` needs ``fiber_idx`` (collapses IPs of that fiber).
            * ``"layers"`` needs both ``fiber_idx`` and ``ip_idx``
              (collapses layers at that fiber × IP).
        reduce : ``"mean"`` | ``"median"`` | ``"max"`` | ``"min"`` |
                 ``"abs_max"`` | ``"sum"``
            Row-wise reduction across the matched columns.
        fiber_idx, ip_idx, layer_idx : int, optional
            Anchor indices. Required where ``over`` says so; otherwise
            optional pre-filters that further restrict the matched
            columns (e.g. ``layer_idx=2`` with ``over="fibers"`` and
            ``ip_idx=0`` selects fibers within layer 2 at IP 0).
        element_ids : int or sequence of int, optional
            Restrict to specific elements. ``None`` plots every element
            in the result (one curve each).
        ax : matplotlib Axes or None
            Existing axes to draw on; new figure if ``None``.
        x_axis : ``"time"`` | ``"step"``
            x-axis source. ``"time"`` requires ``self._results.time``.
        annotate_stage_boundaries : bool
            For multi-stage results, draw a dashed vertical line at each
            stage transition and record the positions in
            ``meta["stage_boundaries"]``.
        **plot_kwargs
            Forwarded to ``ax.plot``.

        Returns
        -------
        (ax, meta) : tuple
            ``meta`` carries ``{"x", "y_per_element", "columns_used",
            "series", "stage_boundaries"?}``. ``columns_used`` is the
            list of raw fiber / IP / layer columns that were folded into
            the curve — useful for audit and downstream labelling.

        Examples
        --------
        Mean ε11 across every fiber and IP, one curve per element::

            er.plot.history_canonical("strain_11", reduce="mean")

        Max(|ε11|) across fibers at IP 0::

            er.plot.history_canonical("strain_11", over="fibers",
                                       ip_idx=0, reduce="abs_max")

        Max σ11 across IPs of fiber 0, on selected elements::

            er.plot.history_canonical("stress_11", over="ips",
                                       fiber_idx=0, reduce="max",
                                       element_ids=[1, 2, 3])
        """
        import matplotlib.pyplot as plt

        er = self._results

        # Pull the canonical slice (raises with a clear message if the
        # name isn't present in this bucket).
        try:
            full = er.canonical(canonical_name)
        except ValueError:
            # Re-raise with the live list so users can fix the name
            # without an extra ``list_canonicals()`` call.
            raise ValueError(
                f"Canonical {canonical_name!r} not in this result. "
                f"Present canonicals: {er.list_canonicals()}"
            ) from None

        # Filter + reduce across the chosen axis. Pure helper —
        # tested separately in tests/unit/elements/.
        series, columns_used = _reduce_fiber_ip(
            full,
            over=over,
            reduce=reduce,
            fiber_idx=fiber_idx,
            ip_idx=ip_idx,
            layer_idx=layer_idx,
        )

        # Pick elements to draw.
        if element_ids is None:
            ids = list(er.element_ids)
        elif isinstance(element_ids, (int, np.integer)):
            ids = [int(element_ids)]
        else:
            ids = [int(e) for e in element_ids]

        # x-axis source.
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

        per_element: dict[int, np.ndarray] = {}
        for eid in ids:
            try:
                y = series.xs(eid, level="element_id").sort_index().to_numpy()
            except KeyError:
                continue
            x = x_full[: y.size]
            label = plot_kwargs.pop("label", None) if len(ids) == 1 else f"e{eid}"
            ax.plot(x, y, label=label, **plot_kwargs)
            per_element[eid] = y

        ax.set_xlabel(x_label)
        ax.set_ylabel(f"{reduce}({canonical_name})")
        if len(ids) > 1 and len(ids) <= 12:
            ax.legend()

        meta: dict[str, Any] = {
            "x": x_full,
            "y_per_element": per_element,
            "columns_used": columns_used,
            "series": series,
        }

        # Stage-boundary annotation, mirroring ``history``.
        if annotate_stage_boundaries and er.is_multi_stage:
            ranges = er.stage_step_ranges
            stages = er.model_stages
            meta["stage_boundaries"] = []
            for st in stages[:-1]:
                end_step = ranges.get(st, (0, 0))[1]
                if x_axis == "time":
                    if 0 < end_step <= x_full.size:
                        x_b = float(x_full[end_step - 1])
                    else:
                        continue
                else:  # step
                    x_b = float(end_step - 1)
                meta["stage_boundaries"].append((st, x_b))
                ax.axvline(
                    x_b, color="0.5", linestyle="--",
                    linewidth=0.8, alpha=0.6,
                )

        return ax, meta

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

        Examples
        --------
        Compose with the dataset-level mesh outline so the IP scatter
        sits on top of the model edges:

        >>> ax, _ = ds.plot.mesh(element_type="203-ASDShellQ4")
        >>> er.plot.scatter("membrane_xx", step=10, ax=ax)

        Or use the bundled convenience:

        >>> ds.plot.mesh_with_contour(er, "membrane_xx", step=10)
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
