from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Optional, Iterable
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import h5py
import gc

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..results.nodal_results_dataclass import NodalResults


class Nodes:
    """
    Fast nodal-results reader.

    Key performance choices:
    - Build a canonical node index once (node_id -> file_id, local index, coords).
    - For each (stage, result, file_id): build ONE DataFrame per file (not per step).
    - Read HDF5 fancy-index with sorted indices for speed, then restore requested order.
    - Minimize pandas object churn: few concats, no per-step DataFrames.
    """

    MAX_MEMORY_BUDGET_MB = 2048
    DEFAULT_CHUNK_SIZE = 5000

    def __init__(self, dataset: "MPCODataSet") -> None:
        self.dataset = dataset
        self._cache: dict[str, object] = {}

    # -------------------------------------------------------------------------
    # Nodes index cache (node_id -> file_id, index, coords)
    # -------------------------------------------------------------------------
    @staticmethod
    def _node_dtype():
        return [
            ("node_id", "i8"),
            ("file_id", "i8"),
            ("index", "i8"),
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
        ]

    def _estimate_node_count(self) -> int:
        """Estimate the total number of nodes without loading all data."""
        total_nodes = 0
        model_stage = self.dataset.model_stages[0]
        
        for part_number, partition_path in self.dataset.results_partitions.items():
            try:
                with h5py.File(partition_path, 'r') as partition:
                    nodes_group = partition.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                    if nodes_group is None:
                        continue
                    
                    for key in nodes_group.keys():
                        if key.startswith("ID"):
                            node_count = len(nodes_group[key])
                            total_nodes += node_count
                            # No need to load actual data, just get the count
                            break
            except Exception as e:
                logger.warning(f"Error estimating nodes in partition {part_number}: {str(e)}")
                
        return total_nodes

    def _get_all_nodes_ids(self, verbose=False, max_workers=4) -> Dict[str, Any]:
        """
        Retrieve all node IDs, file names, indices, and coordinates from the partition files.
        
        Optimized to use pre-allocation, vectorized operations, and parallel processing.

        Args:
            verbose (bool): If True, prints the memory usage of the structured array and DataFrame.
            max_workers (int): Maximum number of worker threads for parallel processing.

        Returns:
            dict: A dictionary containing:
                - 'array': A structured NumPy array with all node IDs, file names, indices, and coordinates.
                - 'dataframe': A pandas DataFrame with the same data.
        """
        # Estimate total node count first to pre-allocate arrays
        estimated_node_count = self._estimate_node_count()
        
        if verbose:
            print(f"Estimated total nodes across all partitions: {estimated_node_count}")
        
        if estimated_node_count == 0:
            return {'array': np.array([], dtype=self._get_node_dtype()), 'dataframe': pd.DataFrame()}
        
        # Check if we need chunked processing based on memory budget
        estimated_memory_mb = estimated_node_count * 48 / 1024 / 1024  # Rough estimate: 6 fields * 8 bytes each
        chunked_processing = estimated_memory_mb > self.MAX_MEMORY_BUDGET_MB
        
        # Define dtype for structured array
        dtype = self._get_node_dtype()
        
        # Function to process a single partition in parallel
        def process_partition(partition_info):
            part_number, partition_path = partition_info
            model_stage = self.dataset.model_stages[0]
            partition_data = []
            
            try:
                with h5py.File(partition_path, 'r') as partition:
                    nodes_group = partition.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                    if nodes_group is None:
                        return []
                    
                    for key in nodes_group.keys():
                        if key.startswith("ID"):
                            file_id = part_number
                            node_ids = nodes_group[key][...]  # Use [...] for immediate loading
                            coord_key = key.replace("ID", "COORDINATES")
                            
                            if coord_key in nodes_group:
                                coords = nodes_group[coord_key][...]
                                
                                # Vectorized operation to create structured data
                                indices = np.arange(len(node_ids))
                                file_ids = np.full_like(node_ids, file_id)
                                
                                # Create structured array directly
                                part_data = np.zeros(len(node_ids), dtype=dtype)
                                part_data['node_id'] = node_ids
                                part_data['file_id'] = file_ids
                                part_data['index'] = indices
                                if coords.shape[1] == 3:
                                    part_data['x'] = coords[:, 0]
                                    part_data['y'] = coords[:, 1]
                                    part_data['z'] = coords[:, 2]
                                elif coords.shape[1] == 2:
                                    part_data['x'] = coords[:, 0]
                                    part_data['y'] = coords[:, 1]
                                    part_data['z'] = 0.0  # Pad with zeros for 2D models
                                else:
                                    raise ValueError(f"Unexpected number of coordinate components: {coords.shape[1]}")
                                
                                return part_data
            
            return []
        
        # Process partitions in parallel
        all_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_partition, self.dataset.results_partitions.items()))
            all_data = [r for r in results if len(r) > 0]
        
        # Combine arrays
        if not all_data:
            return {'array': np.array([], dtype=dtype), 'dataframe': pd.DataFrame()}
        
        results_array = np.concatenate(all_data)
        
        # Convert to DataFrame efficiently using the structured array
        df = pd.DataFrame({
            'node_id': results_array['node_id'],
            'file_id': results_array['file_id'],
            'index': results_array['index'],
            'x': results_array['x'],
            'y': results_array['y'],
            'z': results_array['z']
        })
        
        results_dict = {
            'array': results_array,
            'dataframe': df
        }
        
        if verbose:
            array_memory = results_array.nbytes
            df_memory = df.memory_usage(deep=True).sum()
        
        # Store in cache for future use
        self._node_info_cache['all_nodes'] = results_dict
        
        return results_dict

    def _ensure_nodes_index(self, max_workers: int = 4) -> None:
        """
        Build/cache nodes_df once with one canonical row per node_id.

        nodes_df columns: node_id, file_id, index, x, y, z
        """
        if "nodes_df" in self._cache:
            return

        items = list(self.dataset.results_partitions.items())
        if not items:
            self._cache["nodes_df"] = pd.DataFrame(columns=["node_id", "file_id", "index", "x", "y", "z"])
            return

        model_stage = self.dataset.model_stages[0]
        dtype = self._node_dtype()

        def process_partition(item):
            part_number, partition_path = item
            try:
                with h5py.File(partition_path, "r") as h5:
                    g = h5.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                    if g is None:
                        return None

                    # Find first "ID*" dataset
                    id_key = next((k for k in g.keys() if k.startswith("ID")), None)
                    if id_key is None:
                        return None

                    node_ids = g[id_key][...]
                    coord_key = id_key.replace("ID", "COORDINATES")
                    if coord_key not in g:
                        return None
                    coords = g[coord_key][...]

                    n = int(node_ids.shape[0])
                    arr = np.empty(n, dtype=dtype)
                    arr["node_id"] = node_ids
                    arr["file_id"] = np.int64(part_number)
                    arr["index"] = np.arange(n, dtype=np.int64)

                    if coords.shape[1] == 3:
                        arr["x"] = coords[:, 0]
                        arr["y"] = coords[:, 1]
                        arr["z"] = coords[:, 2]
                    elif coords.shape[1] == 2:
                        arr["x"] = coords[:, 0]
                        arr["y"] = coords[:, 1]
                        arr["z"] = 0.0
                    else:
                        # Unlikely; skip this partition
                        return None

                    return arr
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
            out = list(ex.map(process_partition, items))

        out = [a for a in out if a is not None and len(a)]
        if not out:
            self._cache["nodes_df"] = pd.DataFrame(columns=["node_id", "file_id", "index", "x", "y", "z"])
            return

        arr = np.concatenate(out, axis=0)
        df = pd.DataFrame(
            {
                "node_id": arr["node_id"],
                "file_id": arr["file_id"],
                "index": arr["index"],
                "x": arr["x"],
                "y": arr["y"],
                "z": arr["z"],
            }
        )

        # Canonical: 1 row per node_id (deterministic)
        df = (
            df.sort_values(["node_id", "file_id", "index"], kind="mergesort")
              .drop_duplicates("node_id", keep="first")
              .sort_values("node_id", kind="mergesort")
              .reset_index(drop=True)
        )

        self._cache["nodes_df"] = df

    def get_node_files_and_indices(self, node_ids: Sequence[int] | int | None = None) -> pd.DataFrame:
        """
        Return (node_id, file_id, index) rows for requested nodes (or all if None).
        """
        self._ensure_nodes_index()
        df: pd.DataFrame = self._cache["nodes_df"]  # type: ignore[assignment]
        base = df[["node_id", "file_id", "index"]]

        if node_ids is None:
            return base.copy()

        node_ids_arr = np.atleast_1d(np.asarray(node_ids, dtype=np.int64))
        out = base[base["node_id"].isin(node_ids_arr)]
        if out.empty:
            raise ValueError("None of the provided node IDs were found.")
        return out.copy()

    # -------------------------------------------------------------------------
    # Resolve nodes + validate
    # -------------------------------------------------------------------------
    @staticmethod
    def _as_int_list(x) -> list[int]:
        if x is None:
            return []
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        if isinstance(x, float):
            raise TypeError("IDs cannot be float.")
        return [int(v) for v in x]

    @staticmethod
    def _unique_preserve_order(seq: Iterable[int]) -> list[int]:
        return list(dict.fromkeys(seq))

    def _resolve_node_ids_union(
        self,
        *,
        node_ids: int | Sequence[int] | None,
        selection_set_id: int | Sequence[int] | None,
        validate_exists: bool = True,
    ) -> np.ndarray:
        nodes = self._as_int_list(node_ids)
        set_ids = self._as_int_list(selection_set_id)

        if set_ids:
            ss = self.dataset.selection_set
            for sid in set_ids:
                if sid not in ss:
                    raise KeyError(f"Unknown selection_set_id: {sid}")
                nodes.extend(ss[sid].get("NODES", []))

        nodes = self._unique_preserve_order(nodes)
        if not nodes:
            raise ValueError("No nodes specified (node_ids and/or selection_set_id).")

        arr = np.asarray(nodes, dtype=np.int64)

        if validate_exists:
            self._ensure_nodes_index()
            df: pd.DataFrame = self._cache["nodes_df"]  # type: ignore[assignment]
            known = set(df["node_id"].to_numpy(np.int64))
            missing = [nid for nid in nodes if nid not in known]
            if missing:
                raise KeyError(f"Unknown node IDs: {missing[:20]}" + (" ..." if len(missing) > 20 else ""))

        return arr

    def _cache_available_results(self) -> None:
        if "node_results_names" in self._cache:
            return
        names: set[str] = set()
        stage0 = self.dataset.model_stages[0]
        for p in self.dataset.results_partitions.values():
            try:
                with h5py.File(p, "r") as h5:
                    path = f"{stage0}/RESULTS/ON_NODES"
                    if path in h5:
                        names.update(h5[path].keys())
            except Exception:
                pass
        self._cache["node_results_names"] = names

    def _normalize_results(self, results_name: str | Sequence[str] | None) -> tuple[str, ...]:
        self._cache_available_results()
        avail: set[str] = self._cache["node_results_names"]  # type: ignore[assignment]

        if results_name is None:
            if not avail:
                raise RuntimeError("No nodal results discoverable in dataset.")
            return tuple(sorted(avail))

        if isinstance(results_name, str):
            r = (results_name,)
        else:
            r = tuple(results_name)

        missing = [x for x in r if x not in avail]
        if missing:
            raise ValueError(f"Result(s) not found: {missing}. Available: {sorted(avail)}")
        return r

    def _validate_stage(self, model_stage) -> tuple[str, ...]:
        if model_stage is None:
            return tuple(self.dataset.model_stages)
        if isinstance(model_stage, str):
            if model_stage not in self.dataset.model_stages:
                raise ValueError(f"Unknown model_stage '{model_stage}'.")
            return (model_stage,)
        stages = tuple(model_stage)
        bad = [s for s in stages if s not in self.dataset.model_stages]
        if bad:
            raise ValueError(f"Unknown stages: {bad}")
        return stages

    # -------------------------------------------------------------------------
    # HDF5 reading (FAST PATH)
    # -------------------------------------------------------------------------
    def _process_file_results(self, file_id: int, group: pd.DataFrame, base_path: str) -> pd.DataFrame:
        """
        Read ALL steps for a given file_id and return ONE DataFrame:

            columns: [1..ncomp] + node_id + step

        This avoids creating per-step DataFrames (major speedup).
        """
        file_path = self.dataset.results_partitions[int(file_id)]

        with h5py.File(file_path, "r") as h5:
            data_group = h5.get(base_path)
            if data_group is None:
                return pd.DataFrame()

            step_names = list(data_group.keys())
            if not step_names:
                return pd.DataFrame()

            # node indices + ids
            node_idx = group["index"].to_numpy(np.int64, copy=False)
            node_ids = group["node_id"].to_numpy(np.int64, copy=False)

            # Sort indices for faster fancy indexing in HDF5, then invert
            order = np.argsort(node_idx)
            node_idx_s = node_idx[order]
            inv = np.empty_like(order)
            inv[order] = np.arange(order.size)

            first = data_group[step_names[0]]
            comp_n = int(first.shape[1])

            n_nodes = node_idx_s.size
            n_steps = len(step_names)

            # Allocate output array and fill by slices
            data2d = np.empty((n_steps * n_nodes, comp_n), dtype=first.dtype)

            row0 = 0
            for step_name in step_names:
                ds = data_group[step_name]
                block = ds[node_idx_s]          # (n_nodes, comp_n), sorted indices
                block = block[inv]              # restore requested order (same as group order)
                data2d[row0:row0 + n_nodes, :] = block
                row0 += n_nodes

        # Build node_id/step columns (vectorized)
        step_col = np.repeat(np.arange(n_steps, dtype=np.int64), n_nodes)
        node_col = np.tile(node_ids, n_steps)

        cols = [i + 1 for i in range(comp_n)]
        df = pd.DataFrame(data2d, columns=cols)
        df["node_id"] = node_col
        df["step"] = step_col
        return df

    def _get_stage_results(
        self,
        stage: str,
        result_name: str,
        node_ids: np.ndarray,
        chunk_size: Optional[int],
    ) -> pd.DataFrame:
        # chunked mode (recursive)
        if chunk_size and len(node_ids) > chunk_size:
            parts = []
            for i in range(0, len(node_ids), chunk_size):
                parts.append(self._get_stage_results(stage, result_name, node_ids[i:i + chunk_size], None))
                gc.collect()
            return pd.concat(parts, axis=0, copy=False).sort_index()

        nodes_info = self.get_node_files_and_indices(node_ids=node_ids)
        base_path = f"{stage}/RESULTS/ON_NODES/{result_name}/DATA"

        file_groups = list(nodes_info.groupby("file_id", sort=False))
        if not file_groups:
            raise ValueError(f"No node mapping found for stage='{stage}', result='{result_name}'.")

        frames: list[pd.DataFrame] = []
        frames_append = frames.append

        with ThreadPoolExecutor(max_workers=min(len(file_groups), 4)) as ex:
            futs = [
                ex.submit(self._process_file_results, int(fid), g, base_path)
                for fid, g in file_groups
            ]
            for f in futs:
                df_file = f.result()
                if df_file is not None and not df_file.empty:
                    frames_append(df_file)

        if not frames:
            raise ValueError(f"No results found for stage='{stage}', result='{result_name}'.")

        df = pd.concat(frames, axis=0, copy=False)
        return df.set_index(["node_id", "step"]).sort_index()

    def _fetch(
        self,
        *,
        stages: tuple[str, ...],
        results: tuple[str, ...],
        node_ids: np.ndarray,
        chunk_size: Optional[int],
    ) -> pd.DataFrame:
        # single stage
        if len(stages) == 1:
            st = stages[0]
            if len(results) == 1:
                return self._get_stage_results(st, results[0], node_ids, chunk_size)

            per = []
            for r in results:
                part = self._get_stage_results(st, r, node_ids, chunk_size)
                part.columns = pd.MultiIndex.from_product([[r], part.columns])
                per.append(part)
            return pd.concat(per, axis=1, copy=False)

        # multi stage
        all_frames = []
        for st in stages:
            per = []
            for r in results:
                part = self._get_stage_results(st, r, node_ids, chunk_size)
                part.columns = pd.MultiIndex.from_product([[r], part.columns])
                per.append(part)

            stage_df = pd.concat(per, axis=1, copy=False).reset_index()
            stage_df["stage"] = st
            all_frames.append(stage_df)
            gc.collect()

        df = pd.concat(all_frames, axis=0, copy=False)
        return df.set_index(["stage", "node_id", "step"]).sort_index()

    # -------------------------------------------------------------------------
    # Public: get_nodal_results (main API)
    # -------------------------------------------------------------------------
    def get_time_array_for_stage(self, model_stage: str) -> np.ndarray:
        tdf = self.dataset.time.loc[model_stage]
        if "TIME" in tdf.columns:
            return tdf["TIME"].to_numpy(float).reshape(-1)
        return tdf.index.to_numpy(float).reshape(-1)

    def get_nodal_results(
        self,
        results_name: str | Sequence[str] | None = None,
        model_stage=None,
        node_ids: int | Sequence[int] | None = None,
        selection_set_id: int | Sequence[int] | None = None,
        chunk_size: Optional[int] = None,
        memory_limit_mb: Optional[int] = None,
    ) -> "NodalResults":
        """
        Return NodalResults(df, time, coords_map, ...).
        Everything post-processing should live in NodalResults methods.
        """
        from ..results.nodal_results_dataclass import NodalResults

        if memory_limit_mb is not None:
            self.MAX_MEMORY_BUDGET_MB = int(memory_limit_mb)

        if chunk_size is None:
            chunk_size = self.DEFAULT_CHUNK_SIZE

        results = self._normalize_results(results_name)
        stages = self._validate_stage(model_stage)

        node_ids_arr = self._resolve_node_ids_union(
            node_ids=node_ids,
            selection_set_id=selection_set_id,
            validate_exists=True,
        )

        # --- memory estimate (cheap heuristic) ---
        n_steps = sum(self.dataset.number_of_steps.get(s, 0) for s in stages) or 10
        est_comp = 6 * len(results)  # heuristic
        est_mb = (len(node_ids_arr) * n_steps * est_comp * 8) / (1024 * 1024)
        use_chunk = est_mb > self.MAX_MEMORY_BUDGET_MB
        eff_chunk = int(chunk_size) if use_chunk else None

        # --- fetch df ---
        df = self._fetch(stages=stages, results=results, node_ids=node_ids_arr, chunk_size=eff_chunk)

        # --- coords_map (fast, no iterrows) ---
        self._ensure_nodes_index()
        nodes_df: pd.DataFrame = self._cache["nodes_df"]  # type: ignore[assignment]

        uniq = np.unique(node_ids_arr)
        coords_subset = (
            nodes_df.drop_duplicates("node_id")
                    .set_index("node_id")[["x", "y", "z"]]
                    .loc[uniq]
                    .sort_values("z", ascending=True)
        )

        node_ids_sorted = tuple(map(int, coords_subset.index.to_numpy()))
        coords_vals = coords_subset.to_numpy()
        coords_keys = coords_subset.index.to_numpy()
        coords_map = {
            int(k): {"x": float(v0), "y": float(v1), "z": float(v2)}
            for k, (v0, v1, v2) in zip(coords_keys, coords_vals)
        }

        # --- time output ---
        if len(stages) == 1:
            time_out = self.get_time_array_for_stage(stages[0])
        else:
            time_out = {s: self.get_time_array_for_stage(s) for s in stages}

        # --- component_names ---
        if isinstance(df.columns, pd.MultiIndex):
            component_names = tuple("|".join(map(str, c)) for c in df.columns.to_list())
        else:
            component_names = tuple(map(str, df.columns.to_list()))

        return NodalResults(
            df=df,
            time=time_out,
            name=self.dataset.name,
            node_ids=node_ids_sorted,
            coords_map=coords_map,
            component_names=component_names,
            stages=stages,
            plot_settings=self.dataset.plot_settings,
        )
