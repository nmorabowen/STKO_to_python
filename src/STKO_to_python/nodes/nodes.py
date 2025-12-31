from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union
import re
import numpy as np
import pandas as pd
import h5py

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..results.nodal_results_dataclass import NodalResults


class Nodes:
    """
    High-performance nodal-results reader (MPCO/HDF5).

    Public API kept minimal:
        - _get_all_nodes_ids()   (needed by MPCODataSet init)
        - get_nodal_results()    (only public results method)

    Assumed MPCO nodal layout:
        /{stage}/RESULTS/ON_NODES/{result}/DATA/<step_dataset>
    where each <step_dataset> has shape (n_nodes_in_partition, n_comp).
    """

    def __init__(self, dataset: "MPCODataSet") -> None:
        self.dataset = dataset
        self._node_index_df: Optional[pd.DataFrame] = None
        self._node_index_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Dataset init dependency
    # ------------------------------------------------------------------

    @staticmethod
    def _node_dtype() -> np.dtype:
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
        Build canonical node index across partitions and cache it.

        Returns
        -------
        dict with:
            - 'array' : structured ndarray
            - 'dataframe' : DataFrame (node_id, file_id, index, x,y,z)
        """
        dtype = self._node_dtype()
        stage0 = self.dataset.model_stages[0]

        chunks: list[np.ndarray] = []

        for file_id, path in self.dataset.results_partitions.items():
            with h5py.File(path, "r") as h5:
                gpath = self.dataset.MODEL_NODES_PATH.format(model_stage=stage0)
                g = h5.get(gpath)
                if g is None:
                    continue

                id_keys = [k for k in g.keys() if k.startswith("ID")]
                for id_key in id_keys:
                    node_ids = g[id_key][...].astype(np.int64, copy=False)

                    coord_key = id_key.replace("ID", "COORDINATES")
                    if coord_key not in g:
                        continue
                    coords = g[coord_key][...]

                    out = np.empty(node_ids.shape[0], dtype=dtype)
                    out["node_id"] = node_ids
                    out["file_id"] = int(file_id)
                    out["index"] = np.arange(node_ids.shape[0], dtype=np.int64)

                    if coords.shape[1] == 3:
                        out["x"] = coords[:, 0]
                        out["y"] = coords[:, 1]
                        out["z"] = coords[:, 2]
                    elif coords.shape[1] == 2:
                        out["x"] = coords[:, 0]
                        out["y"] = coords[:, 1]
                        out["z"] = 0.0
                    else:
                        raise ValueError(f"Unexpected COORDINATES shape: {coords.shape}")

                    chunks.append(out)

        if chunks:
            arr = np.concatenate(chunks, axis=0)
            df = pd.DataFrame.from_records(arr)

            # Canonicalize: one row per node_id (smallest file_id), deterministic
            df = (
                df.sort_values(["node_id", "file_id", "index"], kind="mergesort")
                  .drop_duplicates(subset="node_id", keep="first")
                  .sort_values("node_id", kind="mergesort")
                  .reset_index(drop=True)
            )
        else:
            arr = np.empty((0,), dtype=dtype)
            df = pd.DataFrame(columns=["node_id", "file_id", "index", "x", "y", "z"])

        self._node_index_arr = arr
        self._node_index_df = df

        if verbose:
            print(f"[Nodes] cached node index: {len(df)} unique nodes")

        return {"array": arr, "dataframe": df}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_node_index_df(self) -> pd.DataFrame:
        if self._node_index_df is not None:
            return self._node_index_df

        # If dataset already built nodes_info in init, reuse it
        ni = getattr(self.dataset, "nodes_info", None)
        if isinstance(ni, dict) and isinstance(ni.get("dataframe", None), pd.DataFrame):
            self._node_index_df = ni["dataframe"]
            return self._node_index_df

        # Otherwise build it now
        self._get_all_nodes_ids(verbose=False)
        assert self._node_index_df is not None
        return self._node_index_df

    @staticmethod
    def _normalize_stages(stages: Union[str, Sequence[str], None], all_stages: Sequence[str]) -> Tuple[str, ...]:
        if stages is None:
            return tuple(all_stages)
        if isinstance(stages, str):
            return (stages,)
        return tuple(stages)

    def _normalize_results(self, results: Union[str, Sequence[str], None]) -> Tuple[str, ...]:
        if results is None:
            return tuple(sorted(self.dataset.node_results_names))
        if isinstance(results, str):
            return (results,)
        return tuple(results)

    def _resolve_node_ids(
        self,
        *,
        node_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, None],
        selection_set_id: Union[int, Sequence[int], None],
    ) -> np.ndarray:
        """
        Resolve node ids from:
          - node_ids: int | seq[int] | seq[seq[int]] | ndarray
          - selection_set_id: int | seq[int]
        Allows BOTH at once -> union.
        Returns unique int64 array.
        """
        gathered: list[np.ndarray] = []

        # ---- selection sets -------------------------------------------------
        if selection_set_id is not None:
            if isinstance(selection_set_id, (int, np.integer)):
                sel_list = [int(selection_set_id)]
            else:
                sel_list = [int(x) for x in selection_set_id]

            sel = self.dataset.selection_set
            for sid in sel_list:
                if sid not in sel:
                    raise ValueError(f"Selection set '{sid}' not found.")
                nodes = sel[sid].get("NODES", None)
                if not nodes:
                    raise ValueError(f"Selection set {sid} has no nodes.")
                gathered.append(np.asarray(nodes, dtype=np.int64))

        # ---- explicit node ids ---------------------------------------------
        if node_ids is not None:
            if isinstance(node_ids, (int, np.integer)):
                gathered.append(np.asarray([int(node_ids)], dtype=np.int64))
            else:
                arr_obj = np.asarray(node_ids, dtype=object)

                # ragged list-of-lists -> flatten
                if (
                    arr_obj.dtype == object
                    and arr_obj.ndim == 1
                    and any(isinstance(x, (list, tuple, np.ndarray)) for x in arr_obj)
                ):
                    flat: list[int] = []
                    for x in node_ids:  # type: ignore[assignment]
                        if isinstance(x, (int, np.integer)):
                            flat.append(int(x))
                        else:
                            flat.extend([int(k) for k in x])
                    gathered.append(np.asarray(flat, dtype=np.int64))
                else:
                    gathered.append(np.asarray(node_ids, dtype=np.int64))

        if not gathered:
            raise ValueError("Provide at least one of node_ids or selection_set_id.")

        out = np.unique(np.concatenate(gathered))
        if out.size == 0:
            raise ValueError("Resolved node set is empty.")
        return out

    def _node_file_map(self, node_ids: np.ndarray) -> pd.DataFrame:
        """
        Return DataFrame with columns: node_id, file_id, index for requested node_ids.
        """
        df = self._ensure_node_index_df()
        sub = df.loc[df["node_id"].isin(node_ids), ["node_id", "file_id", "index"]]
        if sub.empty:
            raise ValueError("None of the provided node IDs were found in nodes index.")

        # Deterministic: one row per node (smallest file_id), sorted by node_id
        sub = (
            sub.sort_values(["node_id", "file_id", "index"], kind="mergesort")
               .drop_duplicates("node_id", keep="first")
               .sort_values("node_id", kind="mergesort")
               .reset_index(drop=True)
        )
        return sub

    @staticmethod
    def _sort_step_keys(keys: Sequence[str]) -> list[str]:
        """
        Sort step datasets robustly:
        - if all keys are ints -> numeric sort
        - else if keys contain a trailing integer (STEP_12, '12') -> sort by that
        - else keep original order
        """
        if not keys:
            return []

        # case 1: pure integer strings
        try:
            ints = [int(k) for k in keys]
            return [k for _, k in sorted(zip(ints, keys))]
        except Exception:
            pass

        # case 2: extract last integer in string
        rx = re.compile(r"(\d+)(?!.*\d)")
        nums = []
        ok = True
        for k in keys:
            m = rx.search(k)
            if not m:
                ok = False
                break
            nums.append(int(m.group(1)))

        if ok:
            return [k for _, k in sorted(zip(nums, keys))]

        return list(keys)

    @staticmethod
    def _read_all_steps_for_nodes(
        *,
        h5: h5py.File,
        base_path: str,
        node_id_vals: np.ndarray,       # (n_nodes,)
        node_local_idx: np.ndarray,     # (n_nodes,)
    ) -> pd.DataFrame:
        """
        Fast read all steps for one (stage, result) inside an already open h5 file.

        Returns a DataFrame with columns [1..ncomp] + ['node_id','step'].
        """
        g = h5.get(base_path)
        if g is None:
            raise KeyError(f"Missing path: {base_path}")

        step_names = Nodes._sort_step_keys(list(g.keys()))
        if not step_names:
            raise ValueError(f"No steps in {base_path}")

        node_id_vals = np.asarray(node_id_vals, dtype=np.int64)
        node_local_idx = np.asarray(node_local_idx, dtype=np.int64)

        # HDF5 fancy indexing: sort indices for speed/compat
        order = np.argsort(node_local_idx, kind="mergesort")
        idx_sorted = node_local_idx[order]

        inv = np.empty_like(order)
        inv[order] = np.arange(order.size, dtype=order.dtype)

        # infer ncomp
        first = g[step_names[0]]
        sample = first[idx_sorted[:1]]
        if sample.ndim != 2:
            raise ValueError(f"Expected (n_nodes, n_comp); got {sample.shape} at {base_path}/{step_names[0]}")
        ncomp = int(sample.shape[1])
        cols = [i + 1 for i in range(ncomp)]

        n_steps = len(step_names)
        n_nodes = idx_sorted.size
        n_rows = n_steps * n_nodes

        out_vals = np.empty((n_rows, ncomp), dtype=np.float64)

        # build node_id, step vectors once (requested/original order)
        out_node = np.tile(node_id_vals, n_steps)
        out_step = np.repeat(np.arange(n_steps, dtype=np.int32), n_nodes)

        for s, step_name in enumerate(step_names):
            dset = g[step_name]
            block = dset[idx_sorted]          # (n_nodes, ncomp) in sorted idx order
            if block.shape[1] != ncomp:
                raise ValueError(
                    f"Inconsistent ncomp at {base_path}/{step_name}: expected {ncomp}, got {block.shape[1]}"
                )
            block = block[inv, :]             # restore requested node order
            i0 = s * n_nodes
            i1 = i0 + n_nodes
            out_vals[i0:i1, :] = block

        df = pd.DataFrame(out_vals, columns=cols)
        df["node_id"] = out_node
        df["step"] = out_step
        return df

    def _time_array_for_stage(self, stage: str) -> np.ndarray:
        tdf = self.dataset.time.loc[stage]
        if "TIME" in tdf.columns:
            return tdf["TIME"].to_numpy(dtype=float).reshape(-1)
        return tdf.index.to_numpy(dtype=float).reshape(-1)

    # ------------------------------------------------------------------
    # Public API: ONLY get_nodal_results
    # ------------------------------------------------------------------

    def get_nodal_results(
        self,
        *,
        results_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        node_ids: Union[int, Sequence[int], Sequence[Sequence[int]], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
    ) -> "NodalResults":
        """
        Get nodal results as NodalResults (fast).

        Nodes selection:
        - node_ids can be: int | [int] | [[int]] | ndarray
        - selection_set_id can be: int | [int]
        - You can provide BOTH -> union(unique)

        Output:
        - stages:
            * single stage -> index (node_id, step)
            * multiple stages -> index (stage, node_id, step)
        - columns:
            * always MultiIndex (result_name, component_index)
        """
        # --- resolve stages/results/nodes ---------------------------------
        stages = self._normalize_stages(model_stage, self.dataset.model_stages)
        results = self._normalize_results(results_name)

        ids = self._resolve_node_ids(node_ids=node_ids, selection_set_id=selection_set_id)
        ids_sorted = np.sort(ids)

        # node -> file mapping once
        nmap = self._node_file_map(ids_sorted)
        file_groups = {fid: grp for fid, grp in nmap.groupby("file_id")}

        # coords_map once
        idx_df = self._ensure_node_index_df().drop_duplicates("node_id").set_index("node_id")
        coords_map = idx_df.loc[ids_sorted, ["x", "y", "z"]].to_dict("index")

        # --- main loop -----------------------------------------------------
        stage_frames: list[pd.DataFrame] = []

        for st in stages:
            # open each file ONCE per stage, read ALL requested results inside that open
            per_result_collect: dict[str, list[pd.DataFrame]] = {r: [] for r in results}

            for fid, grp in file_groups.items():
                file_path = self.dataset.results_partitions[int(fid)]
                node_id_vals = grp["node_id"].to_numpy(dtype=np.int64, copy=False)
                node_local_idx = grp["index"].to_numpy(dtype=np.int64, copy=False)

                with h5py.File(file_path, "r") as h5:
                    for rname in results:
                        base_path = f"{st}/RESULTS/ON_NODES/{rname}/DATA"
                        df_raw = self._read_all_steps_for_nodes(
                            h5=h5,
                            base_path=base_path,
                            node_id_vals=node_id_vals,
                            node_local_idx=node_local_idx,
                        )
                        per_result_collect[rname].append(df_raw)

            # build per-result stage frame (index + MultiIndex columns)
            per_stage_result_frames: list[pd.DataFrame] = []
            for rname in results:
                df_r = pd.concat(per_result_collect[rname], axis=0, ignore_index=True, copy=False)
                df_r = df_r.set_index(["node_id", "step"]).sort_index()

                comp_cols = [c for c in df_r.columns if c not in ("node_id", "step")]
                df_r = df_r[comp_cols]
                df_r.columns = pd.MultiIndex.from_product([[rname], df_r.columns.to_list()])
                per_stage_result_frames.append(df_r)

            df_stage = pd.concat(per_stage_result_frames, axis=1, copy=False)

            if len(stages) > 1:
                df_stage = df_stage.reset_index()
                df_stage["stage"] = st
                df_stage = df_stage.set_index(["stage", "node_id", "step"]).sort_index()

            stage_frames.append(df_stage)

        df_out = stage_frames[0] if len(stage_frames) == 1 else pd.concat(stage_frames, axis=0, copy=False).sort_index()

        # --- time output ---------------------------------------------------
        if len(stages) == 1:
            time_out = self._time_array_for_stage(stages[0])
        else:
            time_out = {s: self._time_array_for_stage(s) for s in stages}

        # component_names (flatten)
        component_names = tuple("|".join(map(str, c)) for c in df_out.columns.to_list())

        from ..results.nodal_results_dataclass import NodalResults  # avoid circular import

        return NodalResults(
            df=df_out,
            time=time_out,
            name=self.dataset.name,
            node_ids=tuple(ids_sorted.tolist()),
            coords_map=coords_map,
            component_names=component_names,
            stages=stages,
            plot_settings=self.dataset.plot_settings,
        )
