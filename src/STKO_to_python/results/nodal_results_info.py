from __future__ import annotations

from typing import Optional, Tuple

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

    def nearest_node_id(
        self,
        x: float,
        y: float,
        z: Optional[float] = None,
        *,
        file_id: Optional[int] = None,
        return_distance: bool = False,
    ) -> int | Tuple[int, float]:

        if self.nodes_info is None:
            raise ValueError("nodes_info is None. Cannot search nearest node.")
        if not isinstance(self.nodes_info, pd.DataFrame):
            raise TypeError("nearest_node_id currently expects nodes_info as a pandas DataFrame.")

        df = self.nodes_info

        if file_id is not None:
            fid_col = self._resolve_column(df, "file_id")
            df = df.loc[df[fid_col].to_numpy() == int(file_id)]
            if df.empty:
                raise ValueError(f"No nodes found for file_id={file_id}.")

        xcol = self._resolve_column(df, "x")
        ycol = self._resolve_column(df, "y")
        zcol = self._resolve_column(df, "z", required=False)

        X = df[xcol].to_numpy(dtype=float, copy=False)
        Y = df[ycol].to_numpy(dtype=float, copy=False)

        if z is None or zcol is None:
            dx = X - float(x)
            dy = Y - float(y)
            d2 = dx * dx + dy * dy
        else:
            Z = df[zcol].to_numpy(dtype=float, copy=False)
            dx = X - float(x)
            dy = Y - float(y)
            dz = Z - float(z)
            d2 = dx * dx + dy * dy + dz * dz

        i = int(np.argmin(d2))

        if "node_id" in self._normalized_columns(df):
            nid_col = self._resolve_column(df, "node_id")
            node_id = int(df[nid_col].iloc[i])
        else:
            node_id = int(df.index[i])

        if return_distance:
            return node_id, float(np.sqrt(d2[i]))
        return node_id

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
