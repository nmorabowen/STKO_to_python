from __future__ import annotations

from typing import Optional, Tuple, Sequence, overload

import numpy as np
import pandas as pd


class NodalResultsInfo:
    __slots__ = ("nodes_ids", "nodes_info", "model_stages", "results_components")

    def __init__(
        self,
        *,
        nodes_ids: Optional[tuple[int, ...]] = None,
        nodes_info: Optional[pd.DataFrame | np.ndarray] = None,
        model_stages: Optional[tuple[str, ...]] = None,
        results_components: Optional[tuple[str, ...]] = None,
    ) -> None:

        if nodes_ids is not None:
            nodes_ids = tuple(int(i) for i in nodes_ids)

        if model_stages is not None:
            model_stages = tuple(str(s) for s in model_stages)

        if results_components is not None:
            results_components = tuple(str(c) for c in results_components)

        # if nodes_info is not None:
        #     if isinstance(nodes_info, pd.DataFrame):
        #         if nodes_info.index.name is None:
        #             nodes_info = nodes_info.rename_axis("node_id")

        #         if nodes_ids is not None:
        #             idx = nodes_info.index
        #             missing = [i for i in nodes_ids if i not in idx]
        #             if missing:
        #                 raise ValueError(
        #                     f"nodes_info DataFrame is missing {len(missing)} node ids "
        #                     f"(e.g. {missing[:5]})."
        #                 )
        #     else:
        #         raise TypeError(
        #             "nodes_info must be a pandas DataFrame"
        #             f"(got {type(nodes_info)!r})."
        #         )

        self.nodes_ids = nodes_ids
        self.nodes_info = nodes_info
        self.model_stages = model_stages
        self.results_components = results_components

    # ------------------------------------------------------------------ #
    # Work methods
    # ------------------------------------------------------------------ #

    @overload
    def nearest_node_id(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ) -> int | Tuple[int, float]: ...

    @overload
    def nearest_node_id(
        self,
        x: Sequence[float],
        y: Sequence[float],
        z: Optional[Sequence[float]] = None,
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ) -> list[int] | Tuple[list[int], list[float]]: ...

    @overload
    def nearest_node_id(
        self,
        x: Sequence[Sequence[float]],
        y: None = None,
        z: None = None,
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ) -> list[int] | Tuple[list[int], list[float]]: ...

    def nearest_node_id(
        self,
        x,
        y=None,
        z=None,
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ):
        """
        Find nearest node(s) to one or more query point(s).

        Accepted inputs
        ---------------
        1) Scalar:
            nearest_node_id(x, y, z=None)

        2) Separate arrays:
            nearest_node_id(x=[...], y=[...], z=[...] or None)

        3) Points list:
            nearest_node_id([(x1,y1), (x2,y2)])
            nearest_node_id([(x1,y1,z1), (x2,y2,z2)])

        Returns
        -------
        - single point -> int (or (int, float) if return_distance)
        - multiple points -> list[int] (or (list[int], list[float]) if return_distance)
        """
        if self.nodes_info is None:
            raise ValueError("nodes_info is None. Cannot search nearest node.")
        if not isinstance(self.nodes_info, pd.DataFrame):
            raise TypeError("nearest_node_id currently expects nodes_info as a pandas DataFrame.")

        df = self.nodes_info

        # optional file filter
        if file_id is not None:
            fid_col = self._resolve_column(df, "file_id")
            df = df.loc[df[fid_col].to_numpy() == int(file_id)]
            if df.empty:
                raise ValueError(f"No nodes found for file_id={file_id}.")

        # resolve columns
        xcol = self._resolve_column(df, "x")
        ycol = self._resolve_column(df, "y")
        zcol = self._resolve_column(df, "z", required=False)

        X = df[xcol].to_numpy(dtype=float, copy=False)
        Y = df[ycol].to_numpy(dtype=float, copy=False)
        Z = df[zcol].to_numpy(dtype=float, copy=False) if zcol is not None else None

        # ---- parse query points into (Qx, Qy, Qz_or_None) arrays ----
        Qx, Qy, Qz, is_single = self._coerce_query_points(x, y, z)

        # if user did not provide z, or df has no z -> do 2D
        use_3d = (Qz is not None) and (Z is not None)

        nQ = Qx.shape[0]
        out_ids = np.empty(nQ, dtype=np.int64)
        out_d = np.empty(nQ, dtype=float)

        # node_id resolution
        has_node_id_col = "node_id" in self._normalized_columns(df)
        nid_col = self._resolve_column(df, "node_id") if has_node_id_col else None
        idx_values = df.index.to_numpy()

        # ---- compute nearest for each query ----
        # (loop over queries; node count can be huge; looping over queries is usually cheaper)
        for k in range(nQ):
            dx = X - Qx[k]
            dy = Y - Qy[k]
            if use_3d:
                dz = Z - Qz[k]  # type: ignore[operator]
                d2 = dx * dx + dy * dy + dz * dz
            else:
                d2 = dx * dx + dy * dy

            i = int(np.argmin(d2))
            out_d[k] = float(np.sqrt(d2[i]))

            if nid_col is not None:
                out_ids[k] = int(df[nid_col].iloc[i])
            else:
                out_ids[k] = int(idx_values[i])

        if is_single:
            nid = int(out_ids[0])
            if return_distance:
                return nid, float(out_d[0])
            return nid

        ids_list = [int(v) for v in out_ids.tolist()]
        if return_distance:
            return ids_list, [float(v) for v in out_d.tolist()]
        return ids_list

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _coerce_query_points(self, x, y, z):
        """
        Normalize query input into (Qx, Qy, Qz_or_None, is_single).
        Supports:
            - scalar x,y,(z)
            - x,y arrays
            - points list: x=[(x1,y1),(x2,y2)] or [(x1,y1,z1),...]
        """
        # Case A: points list passed in x, with y=None
        if y is None and isinstance(x, (list, tuple, np.ndarray)):
            arr = np.asarray(x, dtype=float)
            if arr.ndim != 2 or arr.shape[1] not in (2, 3):
                raise TypeError(
                    "When passing a list of points, use [(x,y), ...] or [(x,y,z), ...]."
                )
            Qx = arr[:, 0]
            Qy = arr[:, 1]
            Qz = arr[:, 2] if arr.shape[1] == 3 else None
            return Qx, Qy, Qz, False

        # Case B: scalar x,y,(z)
        if np.isscalar(x) and np.isscalar(y):
            Qx = np.asarray([float(x)], dtype=float)
            Qy = np.asarray([float(y)], dtype=float)
            Qz = None if z is None else np.asarray([float(z)], dtype=float)
            return Qx, Qy, Qz, True

        # Case C: array-like x,y,(z)
        if y is None:
            raise TypeError("Provide y when x is array-like, or pass points as [(x,y),...].")

        Qx = np.asarray(x, dtype=float).reshape(-1)
        Qy = np.asarray(y, dtype=float).reshape(-1)
        if Qx.shape[0] != Qy.shape[0]:
            raise ValueError(f"x and y must have same length. Got {Qx.shape[0]} and {Qy.shape[0]}.")

        if z is None:
            Qz = None
        else:
            Qz = np.asarray(z, dtype=float).reshape(-1)
            if Qz.shape[0] != Qx.shape[0]:
                raise ValueError(
                    f"z must have same length as x and y. Got {Qz.shape[0]} and {Qx.shape[0]}."
                )

        return Qx, Qy, Qz, False

    @staticmethod
    def _norm_col(name: object) -> str:
        s = str(name).strip()
        if s.startswith("#"):
            s = s[1:].strip()
        return s.lower()

    def _normalized_columns(self, df: pd.DataFrame) -> dict[str, str]:
        return {self._norm_col(c): str(c) for c in df.columns}

    def _resolve_column(self, df: pd.DataFrame, key: str, *, required: bool = True) -> Optional[str]:
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
