"""
ElementResults — rich container for element-level results.

Mirrors the NodalResults pattern:
  - _ResultView proxy for attribute-style component access
  - fetch() / list_components() introspection
  - pickle serialization support
  - time array attachment
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ #
# ResultView proxy
# ------------------------------------------------------------------ #

class _ElementResultView:
    """
    Lightweight proxy for a single value column in element results.

    Column names are derived from the bucket's META/COMPONENTS — see
    ``docs/mpco_format_conventions.md`` and ``io/meta_parser.py``.
    Examples: ``Px_1`` for closed-form globalForce, ``N_2`` for
    localForce, ``Mz_ip3`` for line-station section.force,
    ``sigma11_f0_ip0`` for compressed fibers.

    Allows:
        results.<name>                  -> all elements, all steps
        results.<name>[element_ids]     -> filtered by element
    """

    def __init__(self, parent: "ElementResults", col_name: str) -> None:
        self._parent = parent
        self._col_name = col_name

    def __getitem__(
        self, key: Union[int, Sequence[int], slice]
    ) -> pd.Series:
        """
        Subscript by element_ids.

        Examples
        --------
        >>> view[42]           # single element
        >>> view[[42, 99]]     # multiple elements
        >>> view[:]            # all elements (same as view)
        """
        if isinstance(key, slice) and key == slice(None):
            return self._parent.fetch(self._col_name)

        if isinstance(key, (int, np.integer)):
            element_ids = [int(key)]
        else:
            element_ids = list(key)

        return self._parent.fetch(self._col_name, element_ids=element_ids)

    @property
    def series(self) -> pd.Series:
        """Return the full column as a Series."""
        return self._parent.fetch(self._col_name)

    def __repr__(self) -> str:
        return f"<ElementResultView {self._col_name!r}>"


# ------------------------------------------------------------------ #
# ElementResults container
# ------------------------------------------------------------------ #

class ElementResults:
    """
    Container for element-level results from MPCO HDF5 files.

    Expected df shape:
      - index: (element_id, step)
      - columns: real component names parsed from META/COMPONENTS
        (e.g. ``Px_1, Py_1, ..., Mz_2`` for closed-form globalForce;
        ``P_ip0, Mz_ip0, ..., T_ip4`` for line-station section.force;
        ``sigma11_f0_ip0, ...`` for compressed fiber buckets).
        Falls back to ``val_1, val_2, ..., val_N`` only when META is
        absent.

    Attributes
    ----------
    df : pd.DataFrame
        Results data indexed by (element_id, step).
    time : np.ndarray
        Time array corresponding to steps.
    name : str or None
        Model name.
    element_ids : tuple[int, ...]
        Element IDs in this result set.
    element_type : str
        Base element type used for the query.
    results_name : str
        HDF5 result group name (e.g. 'globalForces').
    model_stage : str
        Model stage this data comes from.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        time: Any,
        *,
        name: Optional[str] = None,
        element_ids: Optional[Tuple[int, ...]] = None,
        element_type: Optional[str] = None,
        results_name: Optional[str] = None,
        model_stage: Optional[str] = None,
        gp_xi: Optional[np.ndarray] = None,
        gp_natural: Optional[np.ndarray] = None,
        gp_weights: Optional[np.ndarray] = None,
        element_node_coords: Optional[np.ndarray] = None,
        element_node_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.df = df
        self.time = time
        self.name = name
        self.element_ids = element_ids or ()
        self.element_type = element_type or ""
        self.results_name = results_name or ""
        self.model_stage = model_stage or ""

        # Multi-dimensional natural-coord IP positions, shape
        # ``(n_ip, dim)``: dim=1 for lines, 2 for shells / plane
        # elements, 3 for solids. Populated from connectivity ``GP_X``
        # (line elements) or from the static catalog at
        # :mod:`STKO_to_python.utilities.gauss_points` (shells / solids).
        # Always in natural / parent-element coordinates ∈ [-1, +1]^dim.
        # ``None`` when no source is available.
        self.gp_natural: Optional[np.ndarray] = (
            np.asarray(gp_natural, dtype=np.float64)
            if gp_natural is not None
            else None
        )

        # Quadrature weights aligned to ``gp_natural`` (same first
        # axis). Catalog-driven only — line-element custom rules don't
        # write weights to the file. Use for numerical integration
        # over the parent domain (``sum(value * weight * |J|)``) once a
        # Jacobian is supplied for the physical element.
        self.gp_weights: Optional[np.ndarray] = (
            np.asarray(gp_weights, dtype=np.float64)
            if gp_weights is not None
            else None
        )

        # Natural-coordinate integration-point positions (ξ ∈ [-1, +1]).
        # Length equals the number of integration points; ``None`` for
        # closed-form buckets (no IPs) and for buckets whose connectivity
        # dataset lacks a ``GP_X`` attribute.
        # See docs/mpco_format_conventions.md §1, §7.
        self.gp_xi: Optional[np.ndarray] = (
            np.asarray(gp_xi, dtype=np.float64) if gp_xi is not None else None
        )

        # Per-element nodal coordinates and node IDs, ordered to match
        # ``self.element_ids`` (sorted ascending). Shape
        # ``(n_elements, n_nodes_per, 3)`` for coords; ``(n_elements,
        # n_nodes_per)`` for IDs. Required by :meth:`physical_coords`
        # and :meth:`jacobian_dets`. ``None`` for empty results or when
        # the read path couldn't resolve node coordinates.
        self.element_node_coords: Optional[np.ndarray] = (
            np.asarray(element_node_coords, dtype=np.float64)
            if element_node_coords is not None
            else None
        )
        self.element_node_ids: Optional[np.ndarray] = (
            np.asarray(element_node_ids, dtype=np.int64)
            if element_node_ids is not None
            else None
        )

        self._views: Dict[str, _ElementResultView] = {}
        self._build_views()

        # Plotting helper bound to this result. Imported lazily to
        # keep the matplotlib dependency optional at import time.
        from .element_results_plotting import ElementResultsPlotter

        self.plot = ElementResultsPlotter(self)

    # ------------------------------------------------------------------ #
    # View construction
    # ------------------------------------------------------------------ #

    def _build_views(self) -> None:
        """Build dynamic ResultView proxies for each value column."""
        self._views.clear()
        if self.df.empty:
            return

        for col in self.df.columns:
            col_str = str(col)
            self._views[col_str] = _ElementResultView(self, col_str)

    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to result columns (e.g., results.Mz_ip2)."""
        if name.startswith("_"):
            raise AttributeError(name)
        views = self.__dict__.get("_views", {})
        if name in views:
            return views[name]
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'. "
            f"Available columns: {list(views.keys())}"
        )

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def list_components(self) -> Tuple[str, ...]:
        """Return the names of all value columns."""
        if self.df.empty:
            return ()
        return tuple(str(c) for c in self.df.columns)

    @property
    def n_components(self) -> int:
        """Number of result components per element per step."""
        return len(self.df.columns) if not self.df.empty else 0

    @property
    def n_elements(self) -> int:
        """Number of unique elements in this result set."""
        return len(self.element_ids)

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        if self.df.empty:
            return 0
        return self.df.index.get_level_values("step").nunique()

    @property
    def empty(self) -> bool:
        return self.df.empty

    @property
    def n_ip(self) -> int:
        """Number of integration points (0 for closed-form buckets).

        Considers both the line-element ``gp_xi`` and the multi-D
        ``gp_natural`` so the count is consistent for shells / solids
        as well as beams.
        """
        if self.gp_natural is not None:
            return int(self.gp_natural.shape[0])
        if self.gp_xi is not None:
            return int(self.gp_xi.size)
        return 0

    @property
    def gp_dim(self) -> int:
        """Parametric dimensionality of the integration scheme.

        ``1`` for line elements, ``2`` for shells / plane elements,
        ``3`` for solids. ``0`` for closed-form buckets and unknown
        element classes.
        """
        if self.gp_natural is not None:
            return int(self.gp_natural.shape[1])
        if self.gp_xi is not None:
            return 1
        return 0

    # ------------------------------------------------------------------ #
    # Canonical-name access                                              #
    # ------------------------------------------------------------------ #

    def list_canonicals(self) -> Tuple[str, ...]:
        """Canonical-name vocabulary present in this result's columns.

        Returns the engineering-friendly names (e.g. ``"axial_force"``,
        ``"bending_moment_z"``) for which at least one matching column
        exists. See :mod:`STKO_to_python.elements.canonical` for the
        full mapping.
        """
        from .canonical import list_canonical_for_columns

        return list_canonical_for_columns(self.df.columns)

    def canonical_columns(self, name: str) -> Tuple[str, ...]:
        """List the column names that match a canonical engineering name.

        Parameters
        ----------
        name : str
            E.g. ``"axial_force"``, ``"bending_moment_z"``,
            ``"stress_11"``. See ``list_canonicals()`` for what's
            available in this result, or ``available_canonicals()`` in
            :mod:`STKO_to_python.elements.canonical` for the full map.

        Returns
        -------
        tuple of str
            Column names in their on-disk order. Empty tuple if no
            columns match (e.g. asking for ``"axial_force"`` on a
            shell ``section.force`` result).

        Raises
        ------
        ValueError
            If ``name`` is not a known canonical name.
        """
        from .canonical import match_canonical_columns

        return tuple(match_canonical_columns(name, self.df.columns))

    def canonical(self, name: str) -> pd.DataFrame:
        """Return the DataFrame subset for a canonical engineering name.

        Convenience wrapper around :meth:`canonical_columns`. Raises if
        no columns match (catches typos and shells-asking-for-axial-
        force-style misuse loudly).
        """
        cols = self.canonical_columns(name)
        if not cols:
            raise ValueError(
                f"No columns matching canonical name {name!r} in this "
                f"result. Present canonicals: {self.list_canonicals()}"
            )
        return self.df[list(cols)]

    # ------------------------------------------------------------------ #
    # Physical-coordinate mapping (B7b — shells / solids / lines)         #
    # ------------------------------------------------------------------ #

    def physical_coords(self) -> Optional[np.ndarray]:
        """Physical (x, y, z) of each integration point in each element.

        Maps ``gp_natural`` through the element-class shape function
        catalog at :mod:`STKO_to_python.utilities.shape_functions`.

        Returns
        -------
        np.ndarray, shape ``(n_elements, n_ip, 3)``, or ``None``
            First axis aligned with ``self.element_ids``. ``None`` if
            any of the following is missing: ``gp_natural``,
            ``element_node_coords``, or a shape function for this
            element class.

        Examples
        --------
        Brick continuum, plot σ_11 contour at midspan ζ=0:

        >>> phys = er.physical_coords()              # (n_e, 8, 3)
        >>> sigma11 = er.canonical("stress_11").to_numpy()
        >>> # ... pick a step, then scatter(phys[:, :, 0], phys[:, :, 1], c=sigma11)
        """
        if self.gp_natural is None or self.element_node_coords is None:
            return None
        from ..utilities.shape_functions import (
            compute_physical_coords,
            get_shape_functions,
        )

        base = self.element_type.split("[", 1)[0]
        fns = get_shape_functions(base)
        if fns is None:
            return None
        N_fn, _, _ = fns
        return compute_physical_coords(
            self.gp_natural, self.element_node_coords, N_fn
        )

    def jacobian_dets(self) -> Optional[np.ndarray]:
        """Jacobian determinants at each IP for each element.

        - Solids (3-D): ``|det(∂x/∂ξ)|`` — the volume measure.
        - Shells (2-D in 3-D): ``||∂x/∂ξ × ∂x/∂η||`` — the surface
          measure.
        - Line elements: ``||∂x/∂ξ||`` — the line measure.

        Multiplying ``value_at_ip * gp_weights * jacobian_dets`` gives a
        contribution to the integral over the *physical* element.

        Returns
        -------
        np.ndarray, shape ``(n_elements, n_ip)``, or ``None``
            Same alignment + ``None`` conditions as
            :meth:`physical_coords`.

        Examples
        --------
        Volume-integrate σ_11 over each brick element at step 100:

        >>> dets = er.jacobian_dets()                   # (n_e, n_ip)
        >>> sigma11 = er.canonical("stress_11").xs(100, level="step").to_numpy()
        >>> # sigma11 shape: (n_e, n_ip)
        >>> per_elem = (sigma11 * er.gp_weights[None, :] * dets).sum(axis=1)
        """
        if self.gp_natural is None or self.element_node_coords is None:
            return None
        from ..utilities.shape_functions import (
            compute_jacobian_dets,
            get_shape_functions,
        )

        base = self.element_type.split("[", 1)[0]
        fns = get_shape_functions(base)
        if fns is None:
            return None
        _, dN_fn, geom_kind = fns
        return compute_jacobian_dets(
            self.gp_natural, self.element_node_coords, dN_fn, geom_kind
        )

    def integrate_canonical(self, name: str) -> pd.Series:
        """Integrate a canonical quantity over the *physical* element
        for every element and step.

        For each ``(element_id, step)``, computes the quadrature sum::

            ∫ value dΩ ≈ Σ_ip value_at_ip * gp_weights * |J|

        where ``gp_weights`` come from the static Gauss-point catalog
        and ``|J|`` is the appropriate Jacobian measure (volume for
        solids, surface for shells, length for line elements).

        Parameters
        ----------
        name : str
            Canonical engineering name, e.g. ``"stress_11"``,
            ``"membrane_xx"``, ``"axial_force"``. Must resolve to
            exactly one column per integration point — i.e. the same
            quantity at every IP. Compressed-fiber buckets (with
            sub-IP fibers) are rejected; integrate them by selecting
            specific fiber columns directly via :meth:`canonical_columns`.

        Returns
        -------
        pd.Series
            Indexed by ``(element_id, step)`` — same MultiIndex as
            ``self.df``. Use ``.unstack("element_id")`` for a
            step × element matrix.

        Raises
        ------
        ValueError
            If the bucket is closed-form, has no ``gp_weights`` (line
            elements with custom rules), no Jacobian (no node coords
            or unknown element class), or the canonical name doesn't
            match exactly ``n_ip`` columns.

        Examples
        --------
        Volume-integrate σ_11 over each brick element, get a
        Series indexed by (element_id, step):

        >>> s = er.integrate_canonical("stress_11")
        >>> s.unstack("element_id").head()    # step × element matrix

        Area-integrate Mxx (bending moment per unit length) over
        each shell element to get total internal moment:

        >>> moments = er_shell.integrate_canonical("bending_moment_xx")
        """
        if self.gp_weights is None:
            raise ValueError(
                f"integrate_canonical({name!r}): no gp_weights on this "
                f"result. Either the bucket is closed-form (no IPs) "
                f"or this is a line element with a custom rule whose "
                f"weights aren't written to MPCO. Catalog-driven "
                f"shell / solid / plane elements have weights."
            )
        dets = self.jacobian_dets()
        if dets is None:
            raise ValueError(
                f"integrate_canonical({name!r}): jacobian_dets() "
                f"returned None. Likely missing element_node_coords "
                f"or no shape function registered for "
                f"element_type={self.element_type!r}."
            )

        cols = self.canonical_columns(name)
        if not cols:
            raise ValueError(
                f"integrate_canonical({name!r}): no columns matching "
                f"this canonical name. Present canonicals: "
                f"{self.list_canonicals()}"
            )
        if len(cols) != self.n_ip:
            # Compressed-fiber bucket (n_fibers * n_ip) or some other
            # multi-block layout. Per-fiber integration needs section /
            # fiber metadata we don't carry yet — refuse loudly.
            raise ValueError(
                f"integrate_canonical({name!r}): canonical resolves to "
                f"{len(cols)} columns but the bucket has {self.n_ip} "
                f"integration points. Likely a fiber bucket — pick the "
                f"specific columns via canonical_columns() / df[...] "
                f"and integrate manually."
            )

        # cols are in IP order (the META parser emits them sorted by
        # gauss_id for line/gauss-level buckets). values shape: rows
        # of ``self.df`` × n_ip.
        values = self.df[list(cols)].to_numpy(dtype=np.float64)

        # Build a per-row (gp_weights * |J|) array. dets is aligned to
        # ``self.element_ids`` (sorted ascending); look up each row's
        # element to get the right Jacobian.
        eids_in_df = (
            self.df.index.get_level_values("element_id").to_numpy(np.int64)
        )
        elem_id_to_row = {int(eid): i for i, eid in enumerate(self.element_ids)}
        try:
            row_idx = np.array(
                [elem_id_to_row[int(e)] for e in eids_in_df], dtype=np.int64
            )
        except KeyError as err:
            raise ValueError(
                f"integrate_canonical({name!r}): df contains element_id "
                f"{err.args[0]} not in self.element_ids. Index out of sync."
            )
        row_weights = self.gp_weights[None, :] * dets[row_idx]
        integrals = (values * row_weights).sum(axis=1)
        return pd.Series(integrals, index=self.df.index, name=f"integral_{name}")

    def physical_x(self, length: float) -> np.ndarray:
        """Convert ``self.gp_xi`` (natural ξ ∈ [-1, +1]) to physical
        positions along an element of the given ``length``.

        Useful for plotting moment diagrams along beams in physical
        coordinates. See ``utilities/coords.py``.

        Parameters
        ----------
        length : float
            Element length L (positive). Per docs §1, all elements that
            share a connectivity bracket also share the same beam-
            integration rule, but their physical lengths can differ —
            so this method takes ``length`` per call.

        Returns
        -------
        np.ndarray
            Physical positions, length ``n_ip``.

        Raises
        ------
        ValueError
            If the bucket is closed-form (no IPs) or ``length`` is
            non-positive.
        """
        if self.gp_xi is None:
            raise ValueError(
                "physical_x() is only valid for line-station / "
                "gauss-level buckets. This result is closed-form "
                "(no integration points)."
            )
        from ..utilities.coords import xi_natural_to_physical

        return xi_natural_to_physical(self.gp_xi, length)

    def at_ip(self, ip_idx: int) -> pd.DataFrame:
        """Return the columns belonging to a single integration point.

        Column names use the ``..._ip<gauss_id>`` suffix introduced by
        the META parser; this method filters by that suffix.

        Parameters
        ----------
        ip_idx : int
            Sequential integration-point index, ``0..n_ip-1``.

        Returns
        -------
        pd.DataFrame
            Subset of ``self.df`` whose column names end with
            ``_ip<ip_idx>``. Index is preserved (``element_id, step``).

        Raises
        ------
        ValueError
            If the bucket is closed-form (no IPs), if ``ip_idx`` is out
            of range, or if no columns match the suffix.
        """
        n = self.n_ip
        if n == 0:
            raise ValueError(
                "at_ip() is only valid for buckets with integration "
                "points. This result is closed-form (no IPs)."
            )
        if not (0 <= ip_idx < n):
            raise ValueError(
                f"ip_idx={ip_idx} out of range [0, {n})."
            )
        suffix = f"_ip{int(ip_idx)}"
        cols = [c for c in self.df.columns if str(c).endswith(suffix)]
        if not cols:
            raise ValueError(
                f"No columns matching '*{suffix}' in this result. "
                f"Available columns: {list(self.df.columns)}"
            )
        return self.df[cols]

    # ------------------------------------------------------------------ #
    # Data access
    # ------------------------------------------------------------------ #

    def fetch(
        self,
        component: Optional[str] = None,
        *,
        element_ids: Union[int, Sequence[int], None] = None,
    ) -> pd.Series | pd.DataFrame:
        """
        Fetch results with optional element and component filtering.

        Parameters
        ----------
        component : str or None
            Column name (e.g. ``'Mz_ip2'``, ``'Px_1'``). If None, returns
            all columns.
        element_ids : int, list[int], or None
            Filter to specific elements. If None, returns all.

        Returns
        -------
        pd.Series or pd.DataFrame
        """
        df = self.df

        # Filter by element_ids
        if element_ids is not None:
            if isinstance(element_ids, (int, np.integer)):
                element_ids = [int(element_ids)]
            else:
                element_ids = [int(e) for e in element_ids]

            lvl = df.index.get_level_values("element_id")
            df = df.loc[lvl.isin(element_ids)]

            if df.empty:
                raise ValueError(
                    f"None of the requested element_ids are present. "
                    f"Available (sample): {sorted(self.element_ids)[:10]}"
                )

        # Filter by component
        if component is not None:
            if component not in df.columns:
                raise ValueError(
                    f"Component '{component}' not found. "
                    f"Available: {self.list_components()}"
                )
            return df[component]

        return df

    def to_dataframe(self, include_time: bool = True) -> pd.DataFrame:
        """
        Return a flat DataFrame with time column attached.

        Parameters
        ----------
        include_time : bool
            If True and time array is available, add a 'time' column.
        """
        df = self.df.copy()

        if include_time and isinstance(self.time, np.ndarray) and self.time.size > 0:
            steps = df.index.get_level_values("step")
            # Map step index to time value (safe for out-of-range)
            time_map = np.full(steps.max() + 1, np.nan)
            time_map[: len(self.time)] = self.time
            df["time"] = time_map[steps]

        return df

    # ------------------------------------------------------------------ #
    # Aggregation helpers
    # ------------------------------------------------------------------ #

    def envelope(
        self,
        component: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Compute min/max envelope over steps for each element.

        Parameters
        ----------
        component : str or None
            If provided, envelope for that component only.

        Returns
        -------
        pd.DataFrame
            Indexed by element_id, columns: [component_min, component_max, ...]
        """
        df = self.df if component is None else self.df[[component]]
        grouped = df.groupby("element_id")
        mins = grouped.min().add_suffix("_min")
        maxs = grouped.max().add_suffix("_max")
        return pd.concat([mins, maxs], axis=1).sort_index()

    def at_step(self, step: int) -> pd.DataFrame:
        """Extract results at a specific step, indexed by element_id."""
        if self.df.empty:
            return pd.DataFrame()
        return self.df.xs(step, level="step")

    # ------------------------------------------------------------------ #
    # Per-element time-series statistics                                  #
    # ------------------------------------------------------------------ #

    def peak_abs(
        self,
        component: Optional[str] = None,
    ) -> pd.DataFrame:
        """Per-element absolute peak value across all time steps.

        For each element, return ``max(|value|)`` over the entire step
        history. Useful for envelope-style post-processing (e.g.
        peak shear demand per beam).

        Parameters
        ----------
        component : str or None
            Restrict to one column. If ``None``, every column gets a
            ``<col>_peak_abs`` entry.

        Returns
        -------
        pd.DataFrame
            Indexed by ``element_id``. Columns: ``<col>_peak_abs``.
        """
        df = self.df if component is None else self.df[[component]]
        if df.empty:
            return pd.DataFrame()
        grouped = df.abs().groupby("element_id")
        return grouped.max().add_suffix("_peak_abs").sort_index()

    def time_of_peak(
        self,
        component: str,
        *,
        abs: bool = True,
    ) -> pd.Series:
        """Per-element step index at which a component peaks.

        Parameters
        ----------
        component : str
            Required — peak time is single-component by definition.
        abs : bool, default True
            If ``True``, find the step where ``|value|`` peaks (most
            useful for cyclic / dynamic loading). If ``False``, find
            the step where the *signed* value is maximal (useful for
            monotonic pushover-style analyses).

        Returns
        -------
        pd.Series
            Indexed by ``element_id``, values are step indices (int).
        """
        if component not in self.df.columns:
            raise ValueError(
                f"Component {component!r} not in this result. "
                f"Available: {self.list_components()}"
            )
        ser = self.df[component].abs() if abs else self.df[component]
        # idxmax returns the (element_id, step) tuple of the max-row;
        # we want just the step level.
        idx_max = ser.groupby("element_id").idxmax()
        # idx_max is a Series indexed by element_id, values are tuples.
        steps = pd.Series(
            [int(t[1]) for t in idx_max.to_numpy()],
            index=idx_max.index,
            name=f"{component}_argmax",
        )
        return steps.sort_index()

    def cumulative_envelope(
        self,
        component: Optional[str] = None,
    ) -> pd.DataFrame:
        """Running min/max envelope per element over the step axis.

        Like :meth:`envelope` but instead of one ``(min, max)`` pair
        per element, returns the running envelope at every step —
        useful for monotonic-load analyses where you want to see when
        each element first reached its peak.

        Returns
        -------
        pd.DataFrame
            MultiIndex ``(element_id, step)`` matching ``self.df``,
            with columns ``<col>_running_min`` and ``<col>_running_max``
            for each input column.
        """
        df = self.df if component is None else self.df[[component]]
        if df.empty:
            return pd.DataFrame()
        # Ensure deterministic step ordering before cumulative ops.
        df = df.sort_index(level=["element_id", "step"])
        grouped = df.groupby(level="element_id")
        running_max = grouped.cummax().add_suffix("_running_max")
        running_min = grouped.cummin().add_suffix("_running_min")
        return pd.concat([running_min, running_max], axis=1)

    def summary(self) -> pd.DataFrame:
        """One-row-per-element summary of useful per-element statistics.

        Combines: signed peak (max), trough (min), absolute peak,
        residual (last step value), and mean — across the full step
        history, for every component column.

        Returns
        -------
        pd.DataFrame
            Indexed by ``element_id``. For each input column ``<col>``,
            five output columns: ``<col>_max``, ``<col>_min``,
            ``<col>_peak_abs``, ``<col>_residual``, ``<col>_mean``.
        """
        if self.df.empty:
            return pd.DataFrame()
        # Sort so .last() picks the largest step.
        df = self.df.sort_index(level=["element_id", "step"])
        grouped = df.groupby(level="element_id")
        peak_abs = df.abs().groupby(level="element_id").max().add_suffix("_peak_abs")
        out = pd.concat(
            [
                grouped.max().add_suffix("_max"),
                grouped.min().add_suffix("_min"),
                peak_abs,
                grouped.last().add_suffix("_residual"),
                grouped.mean().add_suffix("_mean"),
            ],
            axis=1,
        )
        return out.sort_index()

    def at_time(self, t: float, tol: float = 1e-10) -> pd.DataFrame:
        """
        Extract results at the time step closest to *t*.

        Parameters
        ----------
        t : float
            Target time.
        tol : float
            Tolerance for matching.
        """
        if not isinstance(self.time, np.ndarray) or self.time.size == 0:
            raise ValueError("Time array not available.")

        idx = int(np.argmin(np.abs(self.time - t)))
        if abs(self.time[idx] - t) > tol:
            # Just use closest
            pass
        return self.at_step(idx)

    # ------------------------------------------------------------------ #
    # Pickle support
    # ------------------------------------------------------------------ #

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        # ResultView proxies and the plotter hold a back-reference to
        # ``self``; rebuild them on unpickle rather than serializing
        # the cycle.
        state["_views"] = None
        state.pop("plot", None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._views = {}
        self._build_views()
        from .element_results_plotting import ElementResultsPlotter

        self.plot = ElementResultsPlotter(self)

    def save_pickle(
        self,
        path: str | Path,
        *,
        compress: bool | None = None,
        protocol: int = pickle.HIGHEST_PROTOCOL,
    ) -> Path:
        """Save to pickle (optionally gzipped)."""
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        opener = gzip.open if compress else open
        with opener(p, "wb") as f:  # type: ignore
            pickle.dump(self, f, protocol=protocol)
        return p

    @classmethod
    def load_pickle(
        cls,
        path: str | Path,
        *,
        compress: bool | None = None,
    ) -> "ElementResults":
        """Load from pickle."""
        p = Path(path)
        if compress is None:
            compress = p.suffix.lower() == ".gz"

        opener = gzip.open if compress else open
        with opener(p, "rb") as f:  # type: ignore
            obj = pickle.load(f)

        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle at {p} is {type(obj)!r}, expected {cls.__name__}."
            )
        return obj

    # ------------------------------------------------------------------ #
    # Dunder niceties
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        ip_part = f", n_ip={self.n_ip}" if self.n_ip else ""
        return (
            f"ElementResults("
            f"results_name={self.results_name!r}, "
            f"element_type={self.element_type!r}, "
            f"n_elements={self.n_elements}, "
            f"n_steps={self.n_steps}, "
            f"n_components={self.n_components}"
            f"{ip_part})"
        )

    def __str__(self) -> str:
        return (
            f"ElementResults: {self.results_name} on {self.element_type}\n"
            f"  Elements: {self.n_elements}, Steps: {self.n_steps}, "
            f"Components: {self.n_components}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __bool__(self) -> bool:
        return not self.df.empty
