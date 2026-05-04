"""ElementSelector — lazy, chainable, composable element-id queries.

The selector layer answers *which elements* before any HDF5 result is
read. Every public method returns a new selector (selectors are
immutable); resolution to an ID array happens on demand via
:meth:`ElementSelector.ids` / :meth:`mask` / :meth:`df` / :meth:`count`.

The composition primitives are:

* **Chain** — ``sel.of_type(...).within_box(...).nearest_to(...)`` —
  each call AND-narrows the candidate set in source order.
* **Boolean algebra** — ``a & b``, ``a | b``, ``~a`` — operate on the
  resolved id sets via ``np.intersect1d`` / ``np.union1d`` /
  ``np.setdiff1d`` against a *type-anchored universe*.

Universe rule for negation
--------------------------
``~sel`` is always taken relative to the elements that match ``sel``'s
own ``of_type`` / ``from_selection`` / ``with_ids`` anchor. If no
anchor was set the call raises rather than silently negating against
every element in the model.

Combinators inherit a derived universe:

* ``(a & b)._universe`` = ``a._universe ∩ b._universe``
* ``(a | b)._universe`` = ``a._universe ∪ b._universe``
* ``(~a)._universe``  = ``a._universe``  (idempotent)
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover
    from .element_manager import ElementManager


ArrayLike = Union[Sequence[float], np.ndarray]


# ---------------------------------------------------------------------- #
# Filter ops — one tiny dataclass per primitive, each with .apply(df)    #
# ---------------------------------------------------------------------- #

class _FilterOp:
    """Marker base. Each op narrows a DataFrame; ops compose by stacking."""

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        raise NotImplementedError


@dataclass(frozen=True)
class _WithinBoxOp(_FilterOp):
    bbox_min: Tuple[float, float, float]
    bbox_max: Tuple[float, float, float]
    mode: str  # "centroid" | "any_node" | "all_nodes"

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        bmin = np.asarray(self.bbox_min, dtype=np.float64)
        bmax = np.asarray(self.bbox_max, dtype=np.float64)
        if self.mode == "centroid":
            cx = df["centroid_x"].to_numpy()
            cy = df["centroid_y"].to_numpy()
            cz = df["centroid_z"].to_numpy()
            mask = (
                (cx >= bmin[0]) & (cx <= bmax[0]) &
                (cy >= bmin[1]) & (cy <= bmax[1]) &
                (cz >= bmin[2]) & (cz <= bmax[2])
            )
            return df.loc[mask]
        node_xyz = _node_coord_lookup(manager)
        if node_xyz is None:
            raise ValueError(
                "within_box(mode!='centroid') needs node coordinates "
                "but the dataset has no nodes_info table."
            )
        in_box_mask: list[bool] = []
        for nl in df["node_list"]:
            arr = node_xyz.get_many(nl)
            if arr is None:
                in_box_mask.append(False)
                continue
            inside = (
                (arr[:, 0] >= bmin[0]) & (arr[:, 0] <= bmax[0]) &
                (arr[:, 1] >= bmin[1]) & (arr[:, 1] <= bmax[1]) &
                (arr[:, 2] >= bmin[2]) & (arr[:, 2] <= bmax[2])
            )
            if self.mode == "any_node":
                in_box_mask.append(bool(inside.any()))
            elif self.mode == "all_nodes":
                in_box_mask.append(bool(inside.all()))
            else:
                raise ValueError(
                    f"within_box: unknown mode {self.mode!r}; "
                    f"expected 'centroid', 'any_node', or 'all_nodes'."
                )
        return df.loc[np.asarray(in_box_mask, dtype=bool)]


@dataclass(frozen=True)
class _WithinDistanceOp(_FilterOp):
    point: Tuple[float, float, float]
    radius: float

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        p = np.asarray(self.point, dtype=np.float64)
        cx = df["centroid_x"].to_numpy()
        cy = df["centroid_y"].to_numpy()
        cz = df["centroid_z"].to_numpy()
        d = np.sqrt((cx - p[0]) ** 2 + (cy - p[1]) ** 2 + (cz - p[2]) ** 2)
        return df.loc[d <= self.radius]


@dataclass(frozen=True)
class _NearestToOp(_FilterOp):
    point: Tuple[float, float, float]
    k: int

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty or self.k <= 0:
            return df.iloc[0:0]
        p = np.asarray(self.point, dtype=np.float64)
        cx = df["centroid_x"].to_numpy()
        cy = df["centroid_y"].to_numpy()
        cz = df["centroid_z"].to_numpy()
        d = np.sqrt((cx - p[0]) ** 2 + (cy - p[1]) ** 2 + (cz - p[2]) ** 2)
        if self.k >= len(df):
            return df.iloc[np.argsort(d, kind="stable")]
        idx = np.argpartition(d, self.k - 1)[: self.k]
        # Stable sort within the top-k for deterministic ordering.
        idx = idx[np.argsort(d[idx], kind="stable")]
        return df.iloc[idx]


@dataclass(frozen=True)
class _OnPlaneOp(_FilterOp):
    point: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    tol: float

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        node_xyz = _node_coord_lookup(manager)
        if node_xyz is None:
            raise ValueError(
                "on_plane needs node coordinates but the dataset has "
                "no nodes_info table."
            )
        p = np.asarray(self.point, dtype=np.float64)
        n = np.asarray(self.normal, dtype=np.float64)
        norm = np.linalg.norm(n)
        if norm < 1e-15:
            raise ValueError("on_plane: normal vector has zero length.")
        n = n / norm
        keep: list[bool] = []
        for nl in df["node_list"]:
            arr = node_xyz.get_many(nl)
            if arr is None:
                keep.append(False)
                continue
            signed = (arr - p) @ n
            has_pos = bool(np.any(signed > self.tol))
            has_neg = bool(np.any(signed < -self.tol))
            on_plane = bool(np.any(np.abs(signed) <= self.tol))
            keep.append(on_plane or (has_pos and has_neg))
        return df.loc[np.asarray(keep, dtype=bool)]


@dataclass(frozen=True)
class _NearLineOp(_FilterOp):
    p0: Tuple[float, float, float]
    p1: Tuple[float, float, float]
    radius: float

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        a = np.asarray(self.p0, dtype=np.float64)
        b = np.asarray(self.p1, dtype=np.float64)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-30:
            raise ValueError("near_line: p0 and p1 coincide.")
        c = np.column_stack(
            (
                df["centroid_x"].to_numpy(),
                df["centroid_y"].to_numpy(),
                df["centroid_z"].to_numpy(),
            )
        )
        t = np.clip(((c - a) @ ab) / ab2, 0.0, 1.0)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(c - proj, axis=1)
        return df.loc[d <= self.radius]


@dataclass(frozen=True)
class _CentroidInOp(_FilterOp):
    axis: str
    lo: Optional[float]
    hi: Optional[float]

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        col = f"centroid_{self.axis.lower()}"
        if col not in df.columns:
            raise ValueError(f"centroid_in: unknown axis {self.axis!r}.")
        v = df[col].to_numpy()
        mask = np.ones(len(v), dtype=bool)
        if self.lo is not None:
            mask &= v >= self.lo
        if self.hi is not None:
            mask &= v <= self.hi
        return df.loc[mask]


@dataclass(frozen=True)
class _PredicateOp(_FilterOp):
    fn: Callable[[pd.DataFrame], np.ndarray]

    def apply(self, df: pd.DataFrame, manager: "ElementManager") -> pd.DataFrame:
        if df.empty:
            return df
        mask = self.fn(df)
        mask_arr = np.asarray(mask, dtype=bool)
        if mask_arr.shape != (len(df),):
            raise ValueError(
                f"where(fn): predicate returned shape {mask_arr.shape}, "
                f"expected ({len(df)},)."
            )
        return df.loc[mask_arr]


# ---------------------------------------------------------------------- #
# Lightweight node-coord lookup (cached on the manager via attribute)    #
# ---------------------------------------------------------------------- #

class _NodeCoordLookup:
    """Vectorized node_id -> xyz lookup, built once per manager."""

    __slots__ = ("_pos", "_coords")

    def __init__(self, df_nodes: pd.DataFrame) -> None:
        nids = df_nodes["node_id"].to_numpy(dtype=np.int64)
        coords = df_nodes[["x", "y", "z"]].to_numpy(dtype=np.float64)
        self._pos: dict[int, int] = {int(n): i for i, n in enumerate(nids)}
        self._coords = coords

    def get_many(self, node_ids: Iterable[int]) -> Optional[np.ndarray]:
        out: list[np.ndarray] = []
        for nid in node_ids:
            i = self._pos.get(int(nid))
            if i is None:
                return None
            out.append(self._coords[i])
        return np.asarray(out, dtype=np.float64) if out else None


def _match_type(elem_type_col: pd.Series, query: str) -> pd.Series:
    """Boolean mask matching element-type strings against a query.

    The element index stores the full ``<classTag>-<ClassName>`` form
    (e.g. ``"64-DispBeamColumn3d"``). Users typically pass just the
    class name (``"DispBeamColumn3d"``), but the tagged form is also
    valid. This helper accepts both: the query matches a row when the
    full string equals the query, or when the row's class-tag prefix
    is stripped and the remainder equals the query.
    """
    q = str(query).strip()
    s = elem_type_col.astype(str)
    direct = s == q
    no_tag = s.str.replace(r"^\d+-", "", regex=True) == q
    return direct | no_tag


def _node_coord_lookup(manager: "ElementManager") -> Optional[_NodeCoordLookup]:
    cache = getattr(manager, "_selector_node_lookup", None)
    if cache is not None:
        return cache
    nodes_info = getattr(manager.dataset, "nodes_info", None)
    if not isinstance(nodes_info, dict):
        return None
    df_nodes = nodes_info.get("dataframe")
    if df_nodes is None or df_nodes.empty:
        return None
    cache = _NodeCoordLookup(df_nodes)
    manager._selector_node_lookup = cache  # type: ignore[attr-defined]
    return cache


# ---------------------------------------------------------------------- #
# Anchor — defines the type-bound universe for a selector                #
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Anchor:
    of_type: Optional[str] = None
    selection: Optional[Tuple[Any, ...]] = None  # names or set ids
    explicit_ids: Optional[Tuple[int, ...]] = None

    def is_set(self) -> bool:
        return (
            self.of_type is not None
            or self.selection is not None
            or self.explicit_ids is not None
        )

    def with_(
        self,
        *,
        of_type: Optional[str] = None,
        selection: Optional[Tuple[Any, ...]] = None,
        explicit_ids: Optional[Tuple[int, ...]] = None,
    ) -> "_Anchor":
        return _Anchor(
            of_type=of_type if of_type is not None else self.of_type,
            selection=selection if selection is not None else self.selection,
            explicit_ids=(
                explicit_ids
                if explicit_ids is not None
                else self.explicit_ids
            ),
        )


# ---------------------------------------------------------------------- #
# ElementSelector                                                        #
# ---------------------------------------------------------------------- #

class ElementSelector:
    """Lazy, immutable element-id query.

    Construct via :meth:`ElementManager.select` and chain primitives.
    Resolution to ids is deferred until :meth:`ids`, :meth:`mask`,
    :meth:`df`, or :meth:`count` is called.

    Examples
    --------
    >>> sel = (dataset.elements.select()
    ...        .of_type("DispBeamColumn3d")
    ...        .within_box(min=(0, 0, 0), max=(10, 10, 30))
    ...        .nearest_to((5, 5, 15), k=20))
    >>> ids = sel.ids()                      # np.ndarray[int64]
    >>> df  = sel.df()                       # element-index rows
    >>> mask = sel.mask()                    # bool Series indexed by id
    >>> n   = sel.count()                    # int

    Composition
    -----------
    >>> a = dataset.elements.select().of_type("Beam").within_box(...)
    >>> b = dataset.elements.select().of_type("Beam").from_selection("Core")
    >>> (a & b).ids()        # intersection
    >>> (a | b).ids()        # union
    >>> (~a).ids()           # complement within a's of_type universe
    """

    def __init__(
        self,
        manager: "ElementManager",
        *,
        anchor: Optional[_Anchor] = None,
        ops: Tuple[_FilterOp, ...] = (),
    ) -> None:
        self._manager = manager
        self._anchor = anchor or _Anchor()
        self._ops = ops

    # ------------------------------------------------------------------ #
    # Anchor primitives — narrow the universe                            #
    # ------------------------------------------------------------------ #

    def of_type(self, name: str) -> "ElementSelector":
        """Anchor universe to a base element type (e.g. ``"DispBeamColumn3d"``).

        Strips any decorated ``[bracket]`` suffix — the anchor is on the
        OpenSees class name, not the per-rule decorated key.
        """
        base = str(name).split("[", 1)[0]
        return ElementSelector(
            self._manager, anchor=self._anchor.with_(of_type=base), ops=self._ops
        )

    def from_selection(
        self,
        name_or_id: Union[str, int, Sequence[Union[str, int]]],
    ) -> "ElementSelector":
        """Anchor universe to one or more selection sets (by name or id)."""
        if isinstance(name_or_id, (str, int, np.integer)):
            sel_tup: Tuple[Any, ...] = (name_or_id,)
        else:
            sel_tup = tuple(name_or_id)
        existing = self._anchor.selection or ()
        return ElementSelector(
            self._manager,
            anchor=self._anchor.with_(selection=existing + sel_tup),
            ops=self._ops,
        )

    def with_ids(
        self, ids: Union[int, Sequence[int], np.ndarray]
    ) -> "ElementSelector":
        """Anchor universe to an explicit set of element ids."""
        if isinstance(ids, (int, np.integer)):
            arr = (int(ids),)
        else:
            arr = tuple(int(x) for x in np.asarray(ids).ravel())
        existing = self._anchor.explicit_ids or ()
        return ElementSelector(
            self._manager,
            anchor=self._anchor.with_(explicit_ids=existing + arr),
            ops=self._ops,
        )

    # ------------------------------------------------------------------ #
    # Spatial primitives — append a filter op                            #
    # ------------------------------------------------------------------ #

    def within_box(
        self,
        *,
        min: ArrayLike,
        max: ArrayLike,
        mode: str = "centroid",
    ) -> "ElementSelector":
        """Keep elements inside an axis-aligned bounding box.

        Parameters
        ----------
        min, max : (x, y, z)
            Box corners.
        mode : "centroid" | "any_node" | "all_nodes"
            * ``"centroid"``: element centroid lies in the box (default,
              fastest — uses pre-computed centroid columns).
            * ``"any_node"``: at least one node of the element is in the
              box.
            * ``"all_nodes"``: every node of the element is in the box.
        """
        op = _WithinBoxOp(
            bbox_min=tuple(np.asarray(min, dtype=np.float64).tolist()),
            bbox_max=tuple(np.asarray(max, dtype=np.float64).tolist()),
            mode=mode,
        )
        return self._with_op(op)

    def within_distance(
        self, point: ArrayLike, radius: float
    ) -> "ElementSelector":
        """Keep elements whose centroid is within ``radius`` of ``point``."""
        if radius < 0:
            raise ValueError("within_distance: radius must be non-negative.")
        op = _WithinDistanceOp(
            point=tuple(np.asarray(point, dtype=np.float64).tolist()),
            radius=float(radius),
        )
        return self._with_op(op)

    def nearest_to(self, point: ArrayLike, k: int = 1) -> "ElementSelector":
        """Keep the ``k`` elements whose centroids are nearest to ``point``.

        Result rows are sorted by ascending distance (deterministic, with
        a stable tie-break on the original index order).
        """
        if k < 0:
            raise ValueError("nearest_to: k must be non-negative.")
        op = _NearestToOp(
            point=tuple(np.asarray(point, dtype=np.float64).tolist()),
            k=int(k),
        )
        return self._with_op(op)

    def on_plane(
        self,
        *,
        point: Optional[ArrayLike] = None,
        normal: Optional[ArrayLike] = None,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        tol: float = 1e-6,
    ) -> "ElementSelector":
        """Keep elements that *cross* a plane (any node on each side, or
        any node on the plane within ``tol``).

        Either pass ``point`` + ``normal`` for a general plane, or one
        of ``x=``, ``y=``, ``z=`` for an axis-aligned plane.
        """
        axis_args = sum(arg is not None for arg in (x, y, z))
        if axis_args > 1:
            raise ValueError("on_plane: pass at most one of x=, y=, z=.")
        if axis_args == 1 and (point is not None or normal is not None):
            raise ValueError(
                "on_plane: cannot mix x/y/z= with point/normal."
            )
        if axis_args == 1:
            if x is not None:
                p, n = (x, 0.0, 0.0), (1.0, 0.0, 0.0)
            elif y is not None:
                p, n = (0.0, y, 0.0), (0.0, 1.0, 0.0)
            else:
                p, n = (0.0, 0.0, z), (0.0, 0.0, 1.0)  # type: ignore[arg-type]
        else:
            if point is None or normal is None:
                raise ValueError(
                    "on_plane: pass point=+normal= or one of x=/y=/z=."
                )
            p = tuple(np.asarray(point, dtype=np.float64).tolist())  # type: ignore[assignment]
            n = tuple(np.asarray(normal, dtype=np.float64).tolist())  # type: ignore[assignment]
        op = _OnPlaneOp(point=p, normal=n, tol=float(tol))  # type: ignore[arg-type]
        return self._with_op(op)

    def near_line(
        self,
        p0: ArrayLike,
        p1: ArrayLike,
        radius: float,
    ) -> "ElementSelector":
        """Keep elements whose centroid is within ``radius`` of the line
        segment from ``p0`` to ``p1``."""
        if radius < 0:
            raise ValueError("near_line: radius must be non-negative.")
        op = _NearLineOp(
            p0=tuple(np.asarray(p0, dtype=np.float64).tolist()),
            p1=tuple(np.asarray(p1, dtype=np.float64).tolist()),
            radius=float(radius),
        )
        return self._with_op(op)

    def centroid_in(
        self,
        axis: str,
        *,
        lo: Optional[float] = None,
        hi: Optional[float] = None,
    ) -> "ElementSelector":
        """Keep elements with ``centroid_<axis>`` in ``[lo, hi]``.

        Either bound may be ``None`` for a one-sided constraint. Axis
        is one of ``"x"``, ``"y"``, ``"z"``.
        """
        if lo is None and hi is None:
            raise ValueError("centroid_in: pass at least one of lo=, hi=.")
        return self._with_op(_CentroidInOp(axis=str(axis), lo=lo, hi=hi))

    def where(
        self, fn: Callable[[pd.DataFrame], np.ndarray]
    ) -> "ElementSelector":
        """Predicate escape hatch.

        ``fn`` receives the candidate element-index DataFrame and must
        return a boolean array of equal length.

        Example
        -------
        >>> sel.where(lambda df: df["num_nodes"] >= 4)
        """
        return self._with_op(_PredicateOp(fn=fn))

    # ------------------------------------------------------------------ #
    # Resolution                                                         #
    # ------------------------------------------------------------------ #

    def df(self) -> pd.DataFrame:
        """Return the element-index DataFrame matching this selector."""
        df = self._resolve_universe_df()
        for op in self._ops:
            df = op.apply(df, self._manager)
            if df.empty:
                break
        return df

    def ids(self) -> np.ndarray:
        """Return matched element ids as ``int64`` array.

        Order matches the underlying op chain — for ``nearest_to`` this
        is by ascending distance; otherwise by ascending element_id.
        """
        df = self.df()
        if df.empty:
            return np.empty(0, dtype=np.int64)
        return df["element_id"].to_numpy(dtype=np.int64)

    def mask(self) -> pd.Series:
        """Boolean Series indexed by element_id over the universe."""
        uni = self._universe_ids()
        matched = set(int(x) for x in self.ids().tolist())
        values = np.fromiter(
            (int(eid) in matched for eid in uni), dtype=bool, count=uni.size
        )
        return pd.Series(values, index=uni, name="mask")

    def count(self) -> int:
        """Number of matched elements."""
        return int(self.ids().size)

    # ------------------------------------------------------------------ #
    # Boolean composition                                                #
    # ------------------------------------------------------------------ #

    def __and__(self, other: "ElementSelector") -> "ElementSelector":
        if not isinstance(other, ElementSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="and", parts=(self, other))

    def __or__(self, other: "ElementSelector") -> "ElementSelector":
        if not isinstance(other, ElementSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="or", parts=(self, other))

    def __invert__(self) -> "ElementSelector":
        if not self._anchor.is_set():
            raise ValueError(
                "Cannot negate a selector without an of_type/"
                "from_selection/with_ids anchor — call .of_type(...) "
                "first to define the universe."
            )
        return _CombinedSelector(self._manager, kind="not", parts=(self,))

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #

    def _with_op(self, op: _FilterOp) -> "ElementSelector":
        return ElementSelector(
            self._manager, anchor=self._anchor, ops=self._ops + (op,)
        )

    def _resolve_universe_df(self) -> pd.DataFrame:
        df = self._manager._ensure_elem_index_df()
        a = self._anchor
        if a.of_type is not None:
            df = df[_match_type(df["element_type"], a.of_type)]
        if a.selection is not None:
            names: list[str] = []
            ids: list[int] = []
            for s in a.selection:
                if isinstance(s, (int, np.integer)):
                    ids.append(int(s))
                else:
                    names.append(str(s))
            sel_ids = self._manager.dataset._selection_resolver.resolve_elements(
                names=names or None, ids=ids or None
            )
            df = df[df["element_id"].isin(sel_ids)]
        if a.explicit_ids is not None:
            df = df[df["element_id"].isin(a.explicit_ids)]
        return df

    def _universe_ids(self) -> np.ndarray:
        if not self._anchor.is_set():
            raise ValueError(
                "Selector has no anchor; cannot compute a universe."
            )
        return (
            self._resolve_universe_df()["element_id"]
            .to_numpy(dtype=np.int64)
        )

    # ------------------------------------------------------------------ #
    # Repr                                                               #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        bits: list[str] = []
        if self._anchor.of_type:
            bits.append(f"of_type={self._anchor.of_type!r}")
        if self._anchor.selection:
            bits.append(f"from_selection={list(self._anchor.selection)!r}")
        if self._anchor.explicit_ids:
            bits.append(f"with_ids({len(self._anchor.explicit_ids)})")
        for op in self._ops:
            bits.append(type(op).__name__.strip("_"))
        return f"ElementSelector({', '.join(bits)})"


# ---------------------------------------------------------------------- #
# Combined (AND/OR/NOT) selector                                          #
# ---------------------------------------------------------------------- #

class _CombinedSelector(ElementSelector):
    """Boolean combinator over child selectors. Honors the same protocol
    as :class:`ElementSelector` so combinators are themselves chainable
    via further ``&`` / ``|`` / ``~``.
    """

    def __init__(
        self,
        manager: "ElementManager",
        *,
        kind: str,
        parts: Tuple[ElementSelector, ...],
    ) -> None:
        # Bypass the regular __init__: combinators carry no anchor/ops
        # of their own; resolution is delegated to the parts.
        self._manager = manager
        self._anchor = _Anchor()
        self._ops = ()
        if kind not in {"and", "or", "not"}:
            raise ValueError(f"_CombinedSelector: unknown kind {kind!r}.")
        if kind == "not" and len(parts) != 1:
            raise ValueError("_CombinedSelector('not'): expects one part.")
        if kind in {"and", "or"} and len(parts) < 2:
            raise ValueError(f"_CombinedSelector({kind!r}): expects ≥2 parts.")
        self._kind = kind
        self._parts = parts

    # -- resolution -----------------------------------------------------
    def ids(self) -> np.ndarray:
        if self._kind == "and":
            arrs = [p.ids() for p in self._parts]
            return reduce(np.intersect1d, arrs) if arrs else np.empty(0, np.int64)
        if self._kind == "or":
            arrs = [p.ids() for p in self._parts]
            return (
                reduce(np.union1d, arrs) if arrs else np.empty(0, np.int64)
            )
        # not
        inner = self._parts[0]
        uni = inner._universe_ids()
        return np.setdiff1d(uni, inner.ids(), assume_unique=False)

    def df(self) -> pd.DataFrame:
        ids = self.ids()
        if ids.size == 0:
            return self._manager._ensure_elem_index_df().iloc[0:0]
        df_all = self._manager._ensure_elem_index_df()
        return df_all[df_all["element_id"].isin(ids)]

    def mask(self) -> pd.Series:
        uni = self._universe_ids()
        matched = set(int(x) for x in self.ids().tolist())
        values = np.fromiter(
            (int(eid) in matched for eid in uni), dtype=bool, count=uni.size
        )
        return pd.Series(values, index=uni, name="mask")

    def count(self) -> int:
        return int(self.ids().size)

    # -- universe -------------------------------------------------------
    def _universe_ids(self) -> np.ndarray:
        unis = [p._universe_ids() for p in self._parts]
        if self._kind == "and":
            return reduce(np.intersect1d, unis) if unis else np.empty(0, np.int64)
        if self._kind == "or":
            return reduce(np.union1d, unis) if unis else np.empty(0, np.int64)
        return unis[0]  # not

    # -- composition ----------------------------------------------------
    def __and__(self, other: "ElementSelector") -> "ElementSelector":
        if not isinstance(other, ElementSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="and", parts=(self, other))

    def __or__(self, other: "ElementSelector") -> "ElementSelector":
        if not isinstance(other, ElementSelector):
            return NotImplemented  # type: ignore[return-value]
        return _CombinedSelector(self._manager, kind="or", parts=(self, other))

    def __invert__(self) -> "ElementSelector":
        # All parts must have anchors so the universe is well-defined.
        for p in self._parts:
            try:
                p._universe_ids()
            except ValueError as e:
                raise ValueError(
                    "Cannot negate a combinator whose parts lack anchors. "
                    f"Inner error: {e}"
                ) from e
        return _CombinedSelector(self._manager, kind="not", parts=(self,))

    # Anchor / chain primitives are intentionally not supported on a
    # combinator — they would be ambiguous (which leaf gets the new
    # anchor?). Users should anchor each leaf and combine.
    def of_type(self, name: str) -> "ElementSelector":  # noqa: D401
        raise TypeError(
            "Cannot call .of_type() on a combined selector; anchor each "
            "leaf selector before combining."
        )

    def from_selection(self, name_or_id):  # type: ignore[override]
        raise TypeError(
            "Cannot call .from_selection() on a combined selector; "
            "anchor each leaf selector before combining."
        )

    def with_ids(self, ids):  # type: ignore[override]
        raise TypeError(
            "Cannot call .with_ids() on a combined selector; anchor "
            "each leaf selector before combining."
        )

    def _with_op(self, op: _FilterOp) -> "ElementSelector":
        raise TypeError(
            "Cannot append filter ops directly to a combined selector. "
            "Apply primitives to leaf selectors before combining, or "
            "pass the combined result through .ids() and rebuild."
        )

    # -- repr -----------------------------------------------------------
    def __repr__(self) -> str:
        op = {"and": " & ", "or": " | "}.get(self._kind)
        if self._kind == "not":
            return f"~({self._parts[0]!r})"
        return "(" + op.join(repr(p) for p in self._parts) + ")"  # type: ignore[arg-type]


__all__ = ["ElementSelector"]
