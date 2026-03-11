from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..core.selection import Selection, SelectionBox

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - NumPy fallback is tested explicitly
    cKDTree = None


class NodalResultsInfo:
    """
    Metadata / helpers for nodal results.

    Notes
    -----
    - Assumes `nodes_info` is a pandas DataFrame.
    - `nearest_node_id` accepts query points as a list of points:
        [(x,y), ...] or [(x,y,z), ...]
      and returns a list of nearest node ids (and optionally distances).
    - Selection sets are expected as:
        { id:int : { 'SET_NAME': str, 'NODES': [...], 'ELEMENTS': [...] }, ... }
      but name keys are handled robustly (SET_NAME / NAME / name / Name).
    """

    __slots__ = (
        "nodes_ids",
        "nodes_info",
        "model_stages",
        "results_components",
        "selection_set",
        "analysis_time",
        "size",
        "name",
        "_coord_cols",
        "_node_ids_array",
        "_file_id_array",
        "_coords_2d",
        "_coords_3d",
        "_kdtree_2d",
        "_kdtree_3d",
    )

    def __init__(
        self,
        *,
        nodes_ids: Optional[tuple[int, ...]] = None,
        nodes_info: Optional[pd.DataFrame] = None,
        model_stages: Optional[tuple[str, ...]] = None,
        results_components: Optional[tuple[str, ...]] = None,
        selection_set: Optional[dict] = None,
        analysis_time: Optional[float] = None,
        size: Optional[int] = None,
        name: Optional[str] = None,
    ) -> None:
        if nodes_ids is not None:
            nodes_ids = tuple(int(i) for i in nodes_ids)

        if model_stages is not None:
            model_stages = tuple(str(s) for s in model_stages)

        if results_components is not None:
            results_components = tuple(str(c) for c in results_components)

        if nodes_info is not None and not isinstance(nodes_info, pd.DataFrame):
            raise TypeError(
                "nodes_info must be a pandas DataFrame "
                f"(got {type(nodes_info)!r})."
            )

        if isinstance(nodes_info, pd.DataFrame) and nodes_info.index.name is None:
            nodes_info = nodes_info.rename_axis("node_id")

        self.nodes_ids = nodes_ids
        self.nodes_info = nodes_info
        self.model_stages = model_stages
        self.results_components = results_components
        self.selection_set = selection_set
        self.analysis_time = analysis_time
        self.size = size
        self.name = name

        self._coord_cols: dict[str, str] | None = None
        self._node_ids_array: np.ndarray | None = None
        self._file_id_array: np.ndarray | None = None
        self._coords_2d: np.ndarray | None = None
        self._coords_3d: np.ndarray | None = None
        self._kdtree_2d = None
        self._kdtree_3d = None

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def nearest_node_id(
        self,
        points: Sequence[Sequence[float]],
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ) -> list[int] | Tuple[list[int], list[float]]:
        """
        Find nearest node(s) to a list of query points.
        """
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise TypeError(
                "points must be a sequence of (x,y) or (x,y,z) coordinates. "
                "Example: [(0,0), (1,2)] or [(0,0,0), (1,2,3)]."
            )

        if pts.shape[1] == 2:
            coords = self._coords_array(ndim=2)
        else:
            coords = self._coords_array(ndim=3)

        mask = self._mask_for_file_id(file_id)
        node_ids = self._node_ids()

        if mask is not None:
            coords = coords[mask]
            node_ids = node_ids[mask]

        if coords.size == 0:
            raise ValueError("No candidate nodes are available for nearest-node search.")

        if cKDTree is not None:
            tree = self._get_kdtree(ndim=pts.shape[1], file_id=file_id)
            dist, idx = tree.query(pts)
            idx = np.asarray(idx, dtype=np.int64)
            out_ids = node_ids[idx].astype(np.int64, copy=False).tolist()
            if return_distance:
                return [int(v) for v in out_ids], np.asarray(dist, dtype=float).tolist()
            return [int(v) for v in out_ids]

        diff = coords[None, :, :] - pts[:, None, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(dist2, axis=1)
        out_ids = node_ids[idx].astype(np.int64, copy=False).tolist()
        if return_distance:
            return [int(v) for v in out_ids], np.sqrt(dist2[np.arange(len(idx)), idx]).tolist()
        return [int(v) for v in out_ids]

    def node_ids_in_box(
        self,
        box: SelectionBox,
        *,
        file_id: Optional[int] = None,
    ) -> list[int]:
        coords = self._coords_array(ndim=box.ndim)
        node_ids = self._node_ids()
        mask = self._mask_for_file_id(file_id)

        if mask is not None:
            coords = coords[mask]
            node_ids = node_ids[mask]

        lo = np.asarray(box.min_corner, dtype=float)
        hi = np.asarray(box.max_corner, dtype=float)

        if box.inclusive:
            in_box = np.all((coords >= lo) & (coords <= hi), axis=1)
        else:
            in_box = np.all((coords > lo) & (coords < hi), axis=1)

        out = node_ids[in_box].astype(np.int64, copy=False)
        if out.size == 0:
            raise ValueError(
                f"No nodes found inside selection box {box.min_corner} -> {box.max_corner}."
            )
        return [int(v) for v in np.unique(out).tolist()]

    def resolve_selection(
        self,
        selection: Selection | None = None,
        *,
        only_available: bool = True,
    ) -> list[int]:
        selection = selection or Selection()

        if not isinstance(selection, Selection):
            raise TypeError(
                f"selection must be a Selection instance or None. Got {type(selection)!r}."
            )

        sources: list[np.ndarray] = []

        if selection.ids is not None:
            sources.append(np.asarray(selection.ids, dtype=np.int64))

        if selection.selection_set_id is not None:
            ids = self.selection_set_node_ids(
                selection.selection_set_id,
                only_available=only_available,
            )
            sources.append(np.asarray(ids, dtype=np.int64))

        if selection.selection_set_name is not None:
            ids = self.selection_set_node_ids_by_name(
                selection.selection_set_name,
                only_available=only_available,
            )
            sources.append(np.asarray(ids, dtype=np.int64))

        if selection.coordinates is not None:
            ids = self.nearest_node_id(
                selection.coordinates,
                file_id=selection.file_id,
                return_distance=False,
            )
            sources.append(np.asarray(ids, dtype=np.int64))

        if selection.box is not None:
            ids = self.node_ids_in_box(selection.box, file_id=selection.file_id)
            sources.append(np.asarray(ids, dtype=np.int64))

        if not sources:
            out = self.available_node_ids()
        elif selection.combine == "union":
            out = np.unique(np.concatenate([np.unique(src) for src in sources]))
        else:
            out = np.unique(sources[0])
            for src in sources[1:]:
                out = np.intersect1d(out, np.unique(src), assume_unique=False)

        if only_available and self.nodes_ids is not None:
            out = np.intersect1d(
                out,
                np.asarray(self.nodes_ids, dtype=np.int64),
                assume_unique=False,
            )

        if out.size == 0:
            raise ValueError("Resolved node set is empty.")
        return [int(v) for v in out.tolist()]

    def available_node_ids(self) -> np.ndarray:
        if self.nodes_ids is not None:
            out = np.asarray(self.nodes_ids, dtype=np.int64)
            if out.size:
                return np.unique(out)

        if self.nodes_info is None:
            return np.empty((0,), dtype=np.int64)

        return np.unique(self._node_ids())

    # ------------------------------------------------------------------ #
    # Column helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _norm_col(name: object) -> str:
        s = str(name).strip()
        if s.startswith("#"):
            s = s[1:].strip()
        return s.lower()

    def _normalized_columns(self, df: pd.DataFrame) -> dict[str, str]:
        return {self._norm_col(c): str(c) for c in df.columns}

    def _resolve_column(
        self,
        df: pd.DataFrame,
        key: str,
        *,
        required: bool = True,
    ) -> Optional[str]:
        cols = self._normalized_columns(df)
        k = key.lower()
        if k in cols:
            return cols[k]
        if required:
            raise ValueError(
                f"nodes_info is missing required column '{key}'. "
                f"Available columns (normalized): {sorted(cols.keys())}"
            )
        return None

    # ------------------------------------------------------------------ #
    # Selection set helpers (by id and by name)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalize_selection_names(
        selection_set_name: str | Sequence[str] | None,
    ) -> Tuple[str, ...]:
        if selection_set_name is None:
            return ()
        if isinstance(selection_set_name, str):
            s = selection_set_name.strip()
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                return tuple(parts)
            return (s,)
        out: list[str] = []
        for x in selection_set_name:
            if x is None:
                continue
            sx = str(x).strip()
            if sx:
                out.append(sx)
        return tuple(out)

    def _selection_set_name_for(self, sid: int) -> str:
        if self.selection_set is None:
            return ""
        d = self.selection_set.get(int(sid), {})
        if not isinstance(d, dict):
            return ""

        for k in ("SET_NAME", "set_name", "NAME", "name", "Name"):
            v = d.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        return ""

    def selection_set_ids_from_names(
        self,
        selection_set_name: str | Sequence[str],
    ) -> Tuple[int, ...]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        names = self._normalize_selection_names(selection_set_name)
        if not names:
            raise ValueError("selection_set_name is empty.")

        resolved: list[int] = []
        for raw in names:
            sid_i, _ = self._resolve_selection_set_name(raw)
            resolved.append(sid_i)

        return tuple(resolved)

    def _resolve_selection_set_name(self, raw_name: str) -> tuple[int, str]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        query = str(raw_name).strip()
        if not query:
            raise ValueError("selection_set_name is empty.")

        exact_hits: list[tuple[int, str]] = []
        folded_hits: list[tuple[int, str]] = []
        available: set[str] = set()

        for sid in self.selection_set.keys():
            try:
                sid_i = int(sid)
            except Exception:
                continue

            name = self._selection_set_name_for(sid_i)
            if not name:
                continue

            available.add(name)
            if name == query:
                exact_hits.append((sid_i, name))
            if name.lower() == query.lower():
                folded_hits.append((sid_i, name))

        if len(exact_hits) == 1:
            return exact_hits[0]

        if len(exact_hits) > 1:
            raise ValueError(
                f"Ambiguous selection set name {raw_name!r}: exact matches IDs "
                f"{sorted(sid for sid, _ in exact_hits)}. Use selection_set_id instead."
            )

        if len(folded_hits) == 1:
            return folded_hits[0]

        if len(folded_hits) > 1:
            raise ValueError(
                f"Ambiguous selection set name {raw_name!r}: case-insensitive matches IDs "
                f"{sorted(sid for sid, _ in folded_hits)}. Use selection_set_id instead."
            )

        preview = ", ".join(sorted(available)[:50]) + (" ..." if len(available) > 50 else "")
        raise ValueError(
            f"Selection set name not found: {raw_name!r}. "
            f"Available names include: {preview}"
        )

    def selection_set_node_ids_by_name(
        self,
        selection_set_name: str | Sequence[str],
        *,
        only_available: bool = True,
    ) -> list[int]:
        sids = self.selection_set_ids_from_names(selection_set_name)
        return self.selection_set_node_ids(sids, only_available=only_available)

    def selection_set_node_ids(
        self,
        selection_set_id: int | Sequence[int],
        *,
        only_available: bool = True,
    ) -> list[int]:
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        ids = [selection_set_id] if isinstance(selection_set_id, int) else list(selection_set_id)
        if len(ids) == 0:
            raise ValueError("selection_set_id is empty.")

        gathered: list[np.ndarray] = []
        missing: list[int] = []

        for sid in ids:
            sid_i = int(sid)
            if sid_i not in self.selection_set:
                missing.append(sid_i)
                continue

            entry = self.selection_set.get(sid_i) or {}
            nodes = entry.get("NODES")
            set_name = self._selection_set_name_for(sid_i) or str(sid_i)

            if nodes is None or len(nodes) == 0:
                raise ValueError(
                    f"Selection set '{set_name}' (id={sid_i}) contains 0 nodes "
                    f"in the source .cdata."
                )

            gathered.append(np.asarray(nodes, dtype=np.int64))

        if missing:
            raise ValueError(
                f"Selection set id(s) not found: {missing}. "
                f"Available ids: {sorted(map(int, self.selection_set.keys()))[:50]}"
            )

        out = np.unique(np.concatenate(gathered)).astype(np.int64, copy=False)

        if only_available and self.nodes_ids is not None:
            avail = np.asarray(self.nodes_ids, dtype=np.int64)
            out = out[np.isin(out, avail)]

        if out.size == 0:
            raise ValueError(
                "Resolved selection set node ids are empty "
                "(possibly due to only_available=True filtering)."
            )

        return [int(v) for v in out.tolist()]

    # ------------------------------------------------------------------ #
    # Cached geometry helpers
    # ------------------------------------------------------------------ #
    def _ensure_geometry_cache(self) -> None:
        if self._coords_2d is not None:
            return

        if self.nodes_info is None:
            raise ValueError("nodes_info is None. Cannot resolve geometric node selection.")
        if not isinstance(self.nodes_info, pd.DataFrame):
            raise TypeError("nodes_info must be a pandas DataFrame.")

        df = self.nodes_info
        xcol = self._resolve_column(df, "x", required=True)
        ycol = self._resolve_column(df, "y", required=True)
        zcol = self._resolve_column(df, "z", required=False)
        fid_col = self._resolve_column(df, "file_id", required=False)
        nid_col = self._resolve_column(df, "node_id", required=False)

        x = df[xcol].to_numpy(dtype=float, copy=False)
        y = df[ycol].to_numpy(dtype=float, copy=False)
        if zcol is not None:
            z = df[zcol].to_numpy(dtype=float, copy=False)
        else:
            z = np.zeros_like(x, dtype=float)

        if nid_col is not None:
            node_ids = df[nid_col].to_numpy(dtype=np.int64, copy=False)
        else:
            try:
                node_ids = df.index.to_numpy(dtype=np.int64, copy=False)
            except TypeError as exc:
                raise ValueError(
                    "nodes_info must provide node ids either as a 'node_id' column "
                    "or as an integer index."
                ) from exc

        file_ids = None
        if fid_col is not None:
            file_ids = df[fid_col].to_numpy(dtype=np.int64, copy=False)

        coord_cols = {"x": xcol, "y": ycol}
        if zcol is not None:
            coord_cols["z"] = zcol
        if fid_col is not None:
            coord_cols["file_id"] = fid_col
        if nid_col is not None:
            coord_cols["node_id"] = nid_col

        object.__setattr__(self, "_coord_cols", coord_cols)
        object.__setattr__(
            self,
            "_coords_2d",
            np.column_stack([x, y]).astype(float, copy=False),
        )
        object.__setattr__(
            self,
            "_coords_3d",
            np.column_stack([x, y, z]).astype(float, copy=False),
        )
        object.__setattr__(self, "_node_ids_array", np.asarray(node_ids, dtype=np.int64))
        object.__setattr__(
            self,
            "_file_id_array",
            None if file_ids is None else np.asarray(file_ids, dtype=np.int64),
        )

    def _coords_array(self, *, ndim: int) -> np.ndarray:
        self._ensure_geometry_cache()
        if ndim == 2:
            assert self._coords_2d is not None
            return self._coords_2d
        if ndim == 3:
            assert self._coords_3d is not None
            return self._coords_3d
        raise ValueError(f"Unsupported ndim={ndim}.")

    def _node_ids(self) -> np.ndarray:
        self._ensure_geometry_cache()
        assert self._node_ids_array is not None
        return self._node_ids_array

    def _mask_for_file_id(self, file_id: int | None) -> np.ndarray | None:
        if file_id is None:
            return None
        self._ensure_geometry_cache()
        if self._file_id_array is None:
            raise ValueError("nodes_info does not contain 'file_id'; file_id filtering is unavailable.")

        mask = self._file_id_array == int(file_id)
        if not np.any(mask):
            raise ValueError(f"No nodes found for file_id={int(file_id)}.")
        return mask

    def _get_kdtree(self, *, ndim: int, file_id: int | None):
        coords = self._coords_array(ndim=ndim)
        mask = self._mask_for_file_id(file_id)

        if mask is None:
            attr = "_kdtree_2d" if ndim == 2 else "_kdtree_3d"
            tree = getattr(self, attr)
            if tree is None:
                tree = cKDTree(coords)
                object.__setattr__(self, attr, tree)
            return tree

        return cKDTree(coords[mask])

    # ------------------------------------------------------------------ #
    # Small utilities
    # ------------------------------------------------------------------ #
    def has_nodes_info(self) -> bool:
        return self.nodes_info is not None

    def __repr__(self) -> str:
        n_nodes = len(self.nodes_ids) if self.nodes_ids is not None else None
        stages = self.model_stages or ()
        comps = self.results_components or ()
        info_type = type(self.nodes_info).__name__ if self.nodes_info is not None else None
        return (
            "NodalResultsInfo("
            f"n_nodes={n_nodes}, "
            f"nodes_info={info_type}, "
            f"model_stages={stages}, "
            f"results_components={comps}"
            ")"
        )

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError(f"{type(self).__name__} is immutable; cannot modify '{name}'.")
        super().__setattr__(name, value)
