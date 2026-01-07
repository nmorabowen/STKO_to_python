from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class NodalResultsInfo:
    """
    Metadata / helpers for nodal results.

    Notes
    -----
    - This simplified version assumes `nodes_info` is a pandas DataFrame.
    - `nearest_node_id` ONLY accepts query points as a list of points:
        [(x,y), ...] or [(x,y,z), ...]
      and returns a list of nearest node ids (and optionally distances).
    """

    __slots__ = ("nodes_ids", "nodes_info", "model_stages", "results_components", "selection_set")

    def __init__(
        self,
        *,
        nodes_ids: Optional[tuple[int, ...]] = None,
        nodes_info: Optional[pd.DataFrame] = None,
        model_stages: Optional[tuple[str, ...]] = None,
        results_components: Optional[tuple[str, ...]] = None,
        selection_set: Optional[dict] = None,
    ) -> None:
        # --------------------
        # Normalize
        # --------------------
        if nodes_ids is not None:
            nodes_ids = tuple(int(i) for i in nodes_ids)

        if model_stages is not None:
            model_stages = tuple(str(s) for s in model_stages)

        if results_components is not None:
            results_components = tuple(str(c) for c in results_components)

        # --------------------
        # Basic validation
        # --------------------
        if nodes_info is not None and not isinstance(nodes_info, pd.DataFrame):
            raise TypeError(
                "nodes_info must be a pandas DataFrame "
                f"(got {type(nodes_info)!r})."
            )

        # Optional nicety: name the index if it's node_id-like
        if isinstance(nodes_info, pd.DataFrame) and nodes_info.index.name is None:
            # Don't force it, but it helps downstream debugging/printing
            nodes_info = nodes_info.rename_axis("node_id")

        self.nodes_ids = nodes_ids
        self.nodes_info = nodes_info
        self.model_stages = model_stages
        self.results_components = results_components
        self.selection_set = selection_set

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

        Parameters
        ----------
        points
            Sequence of points with consistent dimensionality:
                [(x, y), ...]    or
                [(x, y, z), ...]
        file_id
            If provided, restrict search to nodes with that file_id
            (requires `nodes_info` to contain a 'file_id' column).
        return_distance
            If True, also return Euclidean distance(s) to the nearest node.

        Returns
        -------
        list[int]
            Nearest node_id for each query point.

        (list[int], list[float])
            If return_distance=True.
        """
        if self.nodes_info is None:
            raise ValueError("nodes_info is None. Cannot search nearest node.")
        if not isinstance(self.nodes_info, pd.DataFrame):
            # should be impossible with current __init__ validation,
            # but keep it defensive if someone bypasses __init__.
            raise TypeError("nearest_node_id expects nodes_info as a pandas DataFrame.")

        df = self.nodes_info

        # ---- optional file filter ------------------------------------- #
        if file_id is not None:
            fid_col = self._resolve_column(df, "file_id")
            mask = df[fid_col].to_numpy() == int(file_id)
            df = df.loc[mask]
            if df.empty:
                raise ValueError(f"No nodes found for file_id={file_id}.")

        # ---- parse points --------------------------------------------- #
        pts = np.asarray(points, dtype=float)

        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise TypeError(
                "points must be a sequence of (x,y) or (x,y,z) coordinates. "
                "Example: [(0,0), (1,2)] or [(0,0,0), (1,2,3)]."
            )

        is_3d = pts.shape[1] == 3

        # ---- resolve coordinate columns ------------------------------- #
        xcol = self._resolve_column(df, "x")
        ycol = self._resolve_column(df, "y")
        zcol = self._resolve_column(df, "z", required=is_3d)

        X = df[xcol].to_numpy(dtype=float, copy=False)
        Y = df[ycol].to_numpy(dtype=float, copy=False)

        if is_3d:
            assert zcol is not None
            Z = df[zcol].to_numpy(dtype=float, copy=False)
        else:
            Z = None

        # ---- node id resolution --------------------------------------- #
        has_node_id_col = "node_id" in self._normalized_columns(df)
        nid_col = self._resolve_column(df, "node_id") if has_node_id_col else None
        idx_values = df.index.to_numpy()

        out_ids: list[int] = []
        out_dist: list[float] = []

        # ---- compute nearest ------------------------------------------ #
        # Loop over queries (usually small) to avoid building huge (n_nodes x n_queries) matrices.
        for p in pts:
            dx = X - p[0]
            dy = Y - p[1]

            if is_3d:
                assert Z is not None
                dz = Z - p[2]
                d2 = dx * dx + dy * dy + dz * dz
            else:
                d2 = dx * dx + dy * dy

            i = int(np.argmin(d2))

            if nid_col is not None:
                out_ids.append(int(df[nid_col].iloc[i]))
            else:
                out_ids.append(int(idx_values[i]))

            if return_distance:
                out_dist.append(float(np.sqrt(d2[i])))

        if return_distance:
            return out_ids, out_dist
        return out_ids

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

    def selection_set_node_ids(
        self,
        selection_set_id: int | Sequence[int],
        *,
        only_available: bool = True,
    ) -> list[int]:
        """
        Return node ids for one or more selection set ids.

        Parameters
        ----------
        selection_set_id
            An int or a sequence of ints (multiple sets will be unioned).
        only_available
            If True, intersect with `self.nodes_ids` when available, so the returned
            ids are guaranteed to exist in this NodalResults object.

        Returns
        -------
        list[int]
            Sorted unique node ids.
        """
        if self.selection_set is None:
            raise ValueError("selection_set is None. No selection sets available.")

        ids = [selection_set_id] if isinstance(selection_set_id, int) else list(selection_set_id)
        if len(ids) == 0:
            raise ValueError("selection_set_id is empty.")

        gathered: list[np.ndarray] = []
        missing: list[int] = []

        for sid in ids:
            if sid not in self.selection_set:
                missing.append(int(sid))
                continue

            entry = self.selection_set.get(sid) or {}
            nodes = entry.get("NODES")

            if nodes is None or len(nodes) == 0:
                # empty selection set is an error (keeps debugging honest)
                raise ValueError(f"Selection set {sid} has no nodes.")

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
