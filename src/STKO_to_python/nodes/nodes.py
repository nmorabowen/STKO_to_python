from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import h5py

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..results.nodal_results_dataclass import NodalResults


ResultName = str
StageName = str


class Nodes:
    """
    Core nodal-results reader.

    Responsibilities
    ---------------
    1) Build/hold a canonical node index: node_id -> (file_id, local_index, coords)
    2) Fetch nodal results from HDF5 (single/multi stage, single/multi result)
    3) Provide time arrays for stages (local or continuous)
    4) Resolve node ids from selection sets
    """

    def __init__(self, dataset: "MPCODataSet") -> None:
        self.dataset = dataset

        # cached node index (built once)
        self._nodes_index_df: Optional[pd.DataFrame] = None
        self._nodes_index_arr: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # 1) Node index (used by MPCODataSet during init)
    # ---------------------------------------------------------------------

    def _get_node_dtype(self):
        return np.dtype(
            [
                ("node_id", "i8"),
                ("file_id", "i8"),
                ("index", "i8"),
                ("x", "f8"),
                ("y", "f8"),
                ("z", "f8"),
            ]
        )

    def _get_all_nodes_ids(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Build node index across partitions: node_id, file_id, local index, coords.
        Returns dict with keys: {'array', 'dataframe'} to preserve your dataset contract.
        """
        dtype = self._get_node_dtype()
        stage0 = self.dataset.model_stages[0]

        parts: list[np.ndarray] = []

        for file_id, path in self.dataset.results_partitions.items():
            with h5py.File(path, "r") as h5:
                nodes_group = h5.get(self.dataset.MODEL_NODES_PATH.format(model_stage=stage0))
                if nodes_group is None:
                    continue

                # STKO convention: datasets named ID* and COORDINATES*
                id_keys = [k for k in nodes_group.keys() if k.startswith("ID")]
                for id_key in id_keys:
                    node_ids = nodes_group[id_key][...].astype(np.int64, copy=False)

                    coord_key = id_key.replace("ID", "COORDINATES")
                    if coord_key not in nodes_group:
                        continue

                    coords = nodes_group[coord_key][...]
                    if coords.shape[1] == 2:
                        x = coords[:, 0]
                        y = coords[:, 1]
                        z = np.zeros_like(x, dtype=float)
                    elif coords.shape[1] == 3:
                        x = coords[:, 0]
                        y = coords[:, 1]
                        z = coords[:, 2]
                    else:
                        raise ValueError(f"Unexpected coords dim: {coords.shape}")

                    out = np.empty(node_ids.shape[0], dtype=dtype)
                    out["node_id"] = node_ids
                    out["file_id"] = int(file_id)
                    out["index"] = np.arange(node_ids.shape[0], dtype=np.int64)
                    out["x"] = x
                    out["y"] = y
                    out["z"] = z
                    parts.append(out)

        if not parts:
            arr = np.empty((0,), dtype=dtype)
            df = pd.DataFrame(columns=["node_id", "file_id", "index", "x", "y", "z"])
        else:
            arr = np.concatenate(parts, axis=0)
            df = pd.DataFrame.from_records(arr)

            # Canonicalize: deterministic mapping node_id -> smallest file_id
            df = (
                df.sort_values(["node_id", "file_id", "index"], kind="mergesort")
                  .drop_duplicates(subset="node_id", keep="first")
                  .sort_values("node_id", kind="mergesort")
                  .reset_index(drop=True)
            )

        self._nodes_index_arr = arr
        self._nodes_index_df = df

        if verbose:
            print(f"[Nodes] node_index rows: {len(df)}")

        return {"array": arr, "dataframe": df}

    def _ensure_node_index(self) -> pd.DataFrame:
        if self._nodes_index_df is None:
            # if dataset already computed it, use that
            ni = getattr(self.dataset, "nodes_info", None)
            if isinstance(ni, dict) and "dataframe" in ni and isinstance(ni["dataframe"], pd.DataFrame):
                self._nodes_index_df = ni["dataframe"]
            else:
                self._get_all_nodes_ids(verbose=False)
        assert self._nodes_index_df is not None
        return self._nodes_index_df

    def get_node_files_and_indices(self, node_ids: Sequence[int]) -> pd.DataFrame:
        """
        Return a DataFrame with columns: node_id, file_id, index for the requested node_ids.
        """
        df = self._ensure_node_index()
        node_ids_arr = np.asarray(node_ids, dtype=np.int64)
        sub = df.loc[df["node_id"].isin(node_ids_arr), ["node_id", "file_id", "index"]]

        if sub.empty:
            raise ValueError("None of the provided node IDs were found in the dataset.")

        # preserve deterministic order: by node_id
        sub = (
            sub.sort_values(["node_id", "file_id", "index"], kind="mergesort")
               .drop_duplicates("node_id", keep="first")
               .sort_values("node_id", kind="mergesort")
               .reset_index(drop=True)
        )
        return sub

    # ---------------------------------------------------------------------
    # 2) Selection set helper (used by your public API)
    # ---------------------------------------------------------------------

    def get_nodes_in_selection_set(self, selection_set_id: int) -> np.ndarray:
        sel = self.dataset.selection_set
        if selection_set_id not in sel:
            raise ValueError(f"Selection set ID '{selection_set_id}' not found.")
        entry = sel[selection_set_id]
        ids = entry.get("NODES", None)
        if not ids:
            raise ValueError(f"Selection set {selection_set_id} does not contain nodes.")
        return np.unique(np.asarray(ids, dtype=np.int64))

    def resolve_node_ids(self, *, node_ids=None, selection_set_id=None) -> np.ndarray:
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Provide exactly one of node_ids or selection_set_id.")
        if selection_set_id is not None:
            return self.get_nodes_in_selection_set(selection_set_id)
        if isinstance(node_ids, (int, np.integer)):
            return np.asarray([node_ids], dtype=np.int64)
        arr = np.asarray(node_ids, dtype=np.int64)
        if arr.size == 0:
            raise ValueError("node_ids is empty.")
        return np.unique(arr)

    # ---------------------------------------------------------------------
    # 3) Time helper (because you attach time into NodalResults)
    # ---------------------------------------------------------------------

    def get_time_array_for_stage(self, stage: str, *, continuous: bool = False) -> np.ndarray:
        ds = self.dataset
        tdf = ds.time.loc[stage]
        if "TIME" in tdf.columns:
            t = tdf["TIME"].to_numpy(dtype=float)
        else:
            t = tdf.index.to_numpy(dtype=float)
        t = t.reshape(-1)

        if not continuous:
            return t

        # offset by sum of previous stage durations (stage order = ds.model_stages)
        offset = 0.0
        for s in ds.model_stages:
            if s == stage:
                break
            prev = ds.time.loc[s]
            last = float(prev["TIME"].iloc[-1]) if "TIME" in prev.columns else float(prev.index[-1])
            offset += last
        return t + offset

    # ---------------------------------------------------------------------
    # 4) Core: get_nodal_results (single/multi stage, single/multi result)
    # ---------------------------------------------------------------------

    def _normalize_stages(self, model_stage: str | Sequence[str] | None) -> Tuple[str, ...]:
        if model_stage is None:
            return tuple(self.dataset.model_stages)
        if isinstance(model_stage, str):
            return (model_stage,)
        return tuple(model_stage)

    def _normalize_results(self, results_name: str | Sequence[str] | None) -> Tuple[str, ...]:
        if results_name is None:
            return tuple(sorted(self.dataset.node_results_names))
        if isinstance(results_name, str):
            return (results_name,)
        return tuple(results_name)

    def _read_one_file_one_stage_one_result(
        self,
        *,
        file_path: str,
        base_path: str,
        node_id_vals: np.ndarray,
        node_local_idx: np.ndarray,
    ) -> pd.DataFrame:
        """
        Reads all steps for (stage, result) from ONE mpco partition file and returns:
        columns = component indices (1..ncomp)
        index cols = node_id, step
        """
        with h5py.File(file_path, "r") as h5:
            g = h5.get(base_path)
            if g is None:
                raise KeyError(f"Missing path: {base_path}")

            step_names = list(g.keys())
            if not step_names:
                raise ValueError(f"No steps in {base_path}")

            # infer ncomp from first step
            first = g[step_names[0]]
            sample = first[node_local_idx[:1]]
            ncomp = sample.shape[1]
            cols = [i + 1 for i in range(ncomp)]

            frames: list[pd.DataFrame] = []
            for step_i, step_name in enumerate(step_names):
                dset = g[step_name]
                data = dset[node_local_idx]  # (n_nodes, ncomp)
                df = pd.DataFrame(data, columns=cols)
                df["node_id"] = node_id_vals
                df["step"] = step_i
                frames.append(df)

        out = pd.concat(frames, axis=0, ignore_index=True, copy=False)
        return out

    def get_nodal_results(
        self,
        *,
        results_name: str | Sequence[str] | None = None,
        model_stage: str | Sequence[str] | None = None,
        node_ids: Sequence[int] | int | None = None,
        selection_set_id: int | None = None,
    ) -> "NodalResults":
        """
        Minimal, deterministic, multi-stage + multi-result reader.

        - single stage -> index (node_id, step)
        - multiple stages -> index (stage, node_id, step)
        - multiple results -> MultiIndex columns (result_name, component)
        """
        stages = self._normalize_stages(model_stage)
        results = self._normalize_results(results_name)

        ids = self.resolve_node_ids(node_ids=node_ids, selection_set_id=selection_set_id)
        # deterministic node order (by coordinate later if you want; by node_id for now)
        ids_sorted = np.sort(ids)

        # node -> (file_id, local index)
        nmap = self.get_node_files_and_indices(ids_sorted.tolist())
        # group by file for fewer open/close operations
        file_groups = {fid: g for fid, g in nmap.groupby("file_id")}

        per_stage_frames: list[pd.DataFrame] = []

        for st in stages:
            per_result_frames: list[pd.DataFrame] = []

            for rname in results:
                base_path = f"{st}/RESULTS/ON_NODES/{rname}/DATA"

                file_frames: list[pd.DataFrame] = []
                for fid, grp in file_groups.items():
                    file_path = self.dataset.results_partitions[int(fid)]
                    node_id_vals = grp["node_id"].to_numpy(np.int64)
                    node_local_idx = grp["index"].to_numpy(np.int64)

                    df_file = self._read_one_file_one_stage_one_result(
                        file_path=file_path,
                        base_path=base_path,
                        node_id_vals=node_id_vals,
                        node_local_idx=node_local_idx,
                    )
                    file_frames.append(df_file)

                df_r = pd.concat(file_frames, axis=0, ignore_index=True, copy=False)
                df_r = df_r.set_index(["node_id", "step"]).sort_index()

                # tag columns for multi-result
                df_r.columns = pd.MultiIndex.from_product([[rname], df_r.columns.to_list()])
                per_result_frames.append(df_r)

            df_stage = pd.concat(per_result_frames, axis=1, copy=False)

            if len(stages) > 1:
                df_stage = df_stage.reset_index()
                df_stage["stage"] = st
                df_stage = df_stage.set_index(["stage", "node_id", "step"]).sort_index()

            per_stage_frames.append(df_stage)

        df = per_stage_frames[0] if len(per_stage_frames) == 1 else pd.concat(per_stage_frames, axis=0, copy=False).sort_index()

        # time output
        if len(stages) == 1:
            time_out = self.get_time_array_for_stage(stages[0])
        else:
            time_out = {s: self.get_time_array_for_stage(s) for s in stages}

        # coords map (needed by your NodalResults)
        idx_df = self._ensure_node_index().drop_duplicates("node_id").set_index("node_id")
        coords_map = idx_df.loc[ids_sorted, ["x", "y", "z"]].to_dict("index")

        # component names (flatten multiindex to strings; you already do this pattern)
        component_names = tuple("|".join(map(str, c)) for c in df.columns.to_list())

        from ..results.nodal_results_dataclass import NodalResults  # local import to avoid cycles

        return NodalResults(
            df=df,
            time=time_out,
            name=self.dataset.name,
            node_ids=tuple(ids_sorted.tolist()),
            coords_map=coords_map,
            component_names=component_names,
            stages=stages,
            plot_settings=self.dataset.plot_settings,
        )
