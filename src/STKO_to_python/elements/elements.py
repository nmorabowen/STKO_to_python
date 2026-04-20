from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from .element_results import ElementResults

logger = logging.getLogger(__name__)


class Elements:
    """
    High-performance MPCO element reader.

    Public API:
        - _get_all_element_index()   (called by MPCODataSet during init)
        - get_element_results()
        - get_elements_at_z_levels()
        - get_elements_in_selection_at_z_levels()
        - get_element_results_by_selection_and_z()
        - get_available_element_results()

    Optimizations:
        - Vectorized element index construction (no row-by-row Python loops)
        - Vectorized Z-level filtering using exploded connectivity
        - Cached element index reuse (no redundant HDF5 reads)
        - Sorted fancy indexing for HDF5 reads
    """

    # ------------------------------------------------------------------ #
    # Canonical dtype for element index
    # ------------------------------------------------------------------ #
    _ELEM_DTYPE = np.dtype([
        ("element_id", "i8"),
        ("element_idx", "i8"),
        ("file_id", "i8"),
        ("element_type", object),
        ("node_list", object),
        ("num_nodes", "i8"),
        ("centroid_x", "f8"),
        ("centroid_y", "f8"),
        ("centroid_z", "f8"),
    ])

    def __init__(self, dataset: "MPCODataSet") -> None:
        self.dataset = dataset
        self._elem_index_df: Optional[pd.DataFrame] = None
        self._elem_index_arr: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Element index (required by MPCODataSet)
    # ------------------------------------------------------------------ #

    def _get_all_element_index(
        self,
        element_type: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Build canonical element index: one row per element_id.

        Vectorized implementation: reads entire connectivity datasets at once,
        computes centroids via numpy lookup arrays, and concatenates in bulk.

        Parameters
        ----------
        element_type : str or None
            Base element type to filter (e.g., '203-ASDShellQ4').
            If None, fetches all element types.
        verbose : bool
            Print memory usage and progress info.

        Returns
        -------
        dict
            {'array': structured np.ndarray, 'dataframe': pd.DataFrame}
        """
        stage0 = self.dataset.model_stages[0]

        # -- Build node coordinate lookup arrays for vectorized centroid --
        node_coord_arr = None
        node_id_to_pos = None
        if hasattr(self.dataset, "nodes_info") and isinstance(
            self.dataset.nodes_info, dict
        ):
            df_nodes = self.dataset.nodes_info.get("dataframe")
            if df_nodes is not None and not df_nodes.empty:
                nids = df_nodes["node_id"].to_numpy(dtype=np.int64)
                coords = df_nodes[["x", "y", "z"]].to_numpy(dtype=np.float64)
                # Map node_id -> sequential position
                node_id_to_pos = {int(nid): i for i, nid in enumerate(nids)}
                node_coord_arr = coords  # shape (N, 3)

        # -- Determine which base element types to fetch --
        if element_type is None:
            raw_types = self.dataset.element_types.get("unique_element_types", [])
            if isinstance(raw_types, set):
                raw_types = list(raw_types)
            base_elements = sorted({e.split("[")[0] for e in raw_types})
        else:
            base_elements = [element_type.split("[")[0]]

        if verbose:
            print(f"[Elements] Fetching types: {base_elements}")

        # -- Collect element data from HDF5 partitions --
        chunks: list[pd.DataFrame] = []

        for file_id, partition_path in self.dataset.results_partitions.items():
            with h5py.File(partition_path, "r") as h5:
                elem_group_path = self.dataset.MODEL_ELEMENTS_PATH.format(
                    model_stage=stage0
                )
                element_group = h5.get(elem_group_path)
                if element_group is None:
                    if verbose:
                        print(
                            f"[Elements] No MODEL/ELEMENTS group in partition {file_id}"
                        )
                    continue

                for etype in base_elements:
                    for dset_name in element_group.keys():
                        base_name = dset_name.split("[")[0]
                        if base_name != etype:
                            continue

                        dset = element_group[dset_name]
                        data = dset[:]  # shape (n_elems, 1 + n_nodes_per_elem)

                        if data.size == 0:
                            continue

                        n_elems = data.shape[0]
                        elem_ids = data[:, 0].astype(np.int64)
                        connectivity = data[:, 1:]  # shape (n_elems, n_nodes)
                        n_nodes_per = connectivity.shape[1]

                        # Build node_list as tuple per element (immutable)
                        node_lists = [
                            tuple(int(nid) for nid in connectivity[i])
                            for i in range(n_elems)
                        ]

                        # Vectorized centroid computation
                        if node_coord_arr is not None and node_id_to_pos is not None:
                            cx, cy, cz = self._compute_centroids_vectorized(
                                connectivity, node_coord_arr, node_id_to_pos
                            )
                        else:
                            cx = np.full(n_elems, np.nan)
                            cy = np.full(n_elems, np.nan)
                            cz = np.full(n_elems, np.nan)

                        chunk_df = pd.DataFrame(
                            {
                                "element_id": elem_ids,
                                "element_idx": np.arange(n_elems, dtype=np.int64),
                                "file_id": int(file_id),
                                "element_type": etype,
                                "node_list": node_lists,
                                "num_nodes": n_nodes_per,
                                "centroid_x": cx,
                                "centroid_y": cy,
                                "centroid_z": cz,
                            }
                        )
                        chunks.append(chunk_df)

        # -- Assemble final result --
        if chunks:
            df = pd.concat(chunks, ignore_index=True)
            df = (
                df.sort_values(
                    ["element_id", "file_id", "element_idx"], kind="mergesort"
                )
                .drop_duplicates("element_id", keep="first")
                .sort_values("element_id", kind="mergesort")
                .reset_index(drop=True)
            )

            # Build structured array
            arr = np.empty(len(df), dtype=self._ELEM_DTYPE)
            arr["element_id"] = df["element_id"].to_numpy()
            arr["element_idx"] = df["element_idx"].to_numpy()
            arr["file_id"] = df["file_id"].to_numpy()
            arr["element_type"] = df["element_type"].to_numpy()
            arr["node_list"] = df["node_list"].to_numpy()
            arr["num_nodes"] = df["num_nodes"].to_numpy()
            arr["centroid_x"] = df["centroid_x"].to_numpy()
            arr["centroid_y"] = df["centroid_y"].to_numpy()
            arr["centroid_z"] = df["centroid_z"].to_numpy()
        else:
            arr = np.empty(0, dtype=self._ELEM_DTYPE)
            df = pd.DataFrame(
                columns=[
                    "element_id",
                    "element_idx",
                    "file_id",
                    "element_type",
                    "node_list",
                    "num_nodes",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                ]
            )

        self._elem_index_arr = arr
        self._elem_index_df = df

        if verbose:
            print(f"[Elements] Indexed {len(df)} unique elements")
            if len(df) > 0:
                arr_mem = arr.nbytes
                df_mem = df.memory_usage(deep=True).sum()
                print(
                    f"[Elements] Memory: array={arr_mem / 1024**2:.2f} MB, "
                    f"df={df_mem / 1024**2:.2f} MB"
                )

        return {"array": arr, "dataframe": df}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_centroids_vectorized(
        connectivity: np.ndarray,
        node_coord_arr: np.ndarray,
        node_id_to_pos: dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute element centroids using vectorized numpy operations.

        Parameters
        ----------
        connectivity : ndarray, shape (n_elems, n_nodes_per_elem)
        node_coord_arr : ndarray, shape (total_nodes, 3)
        node_id_to_pos : dict mapping node_id -> position in node_coord_arr

        Returns
        -------
        cx, cy, cz : ndarrays of shape (n_elems,)
        """
        n_elems, n_nodes_per = connectivity.shape

        # Flatten connectivity and map to positions
        flat_ids = connectivity.ravel().astype(np.int64)
        flat_pos = np.array(
            [node_id_to_pos.get(int(nid), -1) for nid in flat_ids],
            dtype=np.int64,
        )

        # Handle missing nodes: use (0,0,0) for unmapped
        valid = flat_pos >= 0
        coords_flat = np.zeros((len(flat_pos), 3), dtype=np.float64)
        coords_flat[valid] = node_coord_arr[flat_pos[valid]]

        # Reshape to (n_elems, n_nodes_per, 3) and average
        coords_3d = coords_flat.reshape(n_elems, n_nodes_per, 3)
        centroids = coords_3d.mean(axis=1)  # (n_elems, 3)

        return centroids[:, 0], centroids[:, 1], centroids[:, 2]

    def _ensure_elem_index_df(self) -> pd.DataFrame:
        """Return the cached element index DataFrame, building it if needed."""
        if self._elem_index_df is not None:
            return self._elem_index_df

        ei = getattr(self.dataset, "elements_info", None)
        if isinstance(ei, dict) and isinstance(ei.get("dataframe"), pd.DataFrame):
            self._elem_index_df = ei["dataframe"]
            return self._elem_index_df

        self._get_all_element_index()
        return self._elem_index_df  # type: ignore

    def _get_cached_elements_df(
        self,
        element_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Return the element index from cache, optionally filtered by base type.
        Never re-reads HDF5 — always uses the cached dataset.elements_info.
        """
        df = self._ensure_elem_index_df()
        if element_type is not None:
            base = element_type.split("[")[0]
            df = df[df["element_type"].str.startswith(base)]
        return df

    # ------------------------------------------------------------------ #
    # Element ID resolution (mirrors Nodes._resolve_node_ids)
    # ------------------------------------------------------------------ #

    def _selection_set_name_for(self, sid: int) -> str:
        """Best-effort extraction of selection set name."""
        d = self.dataset.selection_set.get(int(sid), {})
        if not isinstance(d, dict):
            return ""
        name = d.get("SET_NAME", d.get("name", d.get("Name", "")))
        return "" if name is None else str(name)

    def _selection_set_ids_from_names(
        self, names: Sequence[str]
    ) -> Tuple[int, ...]:
        """Resolve selection set names -> IDs (case-insensitive)."""
        if not names:
            return ()

        buckets: Dict[str, list[int]] = {}
        for sid in self.dataset.selection_set.keys():
            try:
                sid_i = int(sid)
            except Exception:
                continue
            nm = self._selection_set_name_for(sid_i).strip().lower()
            if nm:
                buckets.setdefault(nm, []).append(sid_i)

        resolved: list[int] = []
        for raw in names:
            key = str(raw).strip().lower()
            if not key:
                continue
            hits = buckets.get(key, [])
            if len(hits) == 0:
                available = sorted(buckets.keys())
                preview = ", ".join(available[:30])
                raise ValueError(
                    f"Selection set name not found: {raw!r}. "
                    f"Available: {preview}"
                )
            if len(hits) > 1:
                raise ValueError(
                    f"Ambiguous selection set name {raw!r}: matches IDs {sorted(hits)}."
                )
            resolved.append(hits[0])
        return tuple(resolved)

    def _resolve_element_ids(
        self,
        *,
        element_ids: Union[
            int, Sequence[int], np.ndarray, None
        ] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
    ) -> np.ndarray:
        """
        Resolve element IDs from multiple sources (union semantics).

        Mirrors the Nodes._resolve_node_ids() API.

        Parameters
        ----------
        element_ids : int, list[int], ndarray, or None
        selection_set_id : int, list[int], or None
        selection_set_name : str, list[str], or None

        Returns
        -------
        np.ndarray of int64
            Unique, sorted element IDs.
        """
        gathered: list[np.ndarray] = []

        # -- by name --
        if selection_set_name is not None:
            names = (
                [selection_set_name]
                if isinstance(selection_set_name, str)
                else list(selection_set_name)
            )
            ids_from_names = self._selection_set_ids_from_names(names)
            for sid in ids_from_names:
                elems = self.dataset.selection_set.get(int(sid), {}).get("ELEMENTS")
                if not elems:
                    raise ValueError(
                        f"Selection set {sid} empty or missing ELEMENTS."
                    )
                gathered.append(np.asarray(elems, dtype=np.int64))

        # -- by id --
        if selection_set_id is not None:
            sel_ids = (
                [selection_set_id]
                if isinstance(selection_set_id, int)
                else selection_set_id
            )
            for sid in sel_ids:
                elems = self.dataset.selection_set.get(int(sid), {}).get("ELEMENTS")
                if not elems:
                    raise ValueError(
                        f"Selection set {sid} empty or missing ELEMENTS."
                    )
                gathered.append(np.asarray(elems, dtype=np.int64))

        # -- explicit --
        if element_ids is not None:
            if isinstance(element_ids, (int, np.integer)):
                gathered.append(np.asarray([element_ids], dtype=np.int64))
            else:
                gathered.append(np.asarray(element_ids, dtype=np.int64))

        if not gathered:
            raise ValueError(
                "Provide element_ids and/or selection_set_id and/or selection_set_name."
            )

        out = np.unique(np.concatenate(gathered))
        if out.size == 0:
            raise ValueError("Resolved element set is empty.")
        return out

    # ------------------------------------------------------------------ #
    # Vectorized Z-level filtering
    # ------------------------------------------------------------------ #

    def _compute_z_bounds(
        self,
        df_elements: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add 'z_min' and 'z_max' columns to df_elements by exploding
        connectivity and joining with node Z coordinates.

        Fully vectorized — no iterrows().
        """
        if not hasattr(self.dataset, "nodes_info") or not isinstance(
            self.dataset.nodes_info, dict
        ):
            raise ValueError("Node information is not available in the dataset.")

        df_nodes = self.dataset.nodes_info["dataframe"]
        node_z = df_nodes.set_index("node_id")["z"]

        # Explode node_list to one row per (element_id, node_id)
        exploded = df_elements[["element_id", "node_list"]].copy()
        exploded = exploded.explode("node_list")
        exploded.columns = ["element_id", "node_id"]
        exploded["node_id"] = exploded["node_id"].astype(np.int64)

        # Join Z coordinates
        exploded = exploded.join(node_z, on="node_id", how="left")

        # Group by element to get min/max Z
        z_bounds = exploded.groupby("element_id")["z"].agg(["min", "max"])
        z_bounds.columns = ["z_min", "z_max"]

        return df_elements.join(z_bounds, on="element_id", how="left")

    def get_elements_at_z_levels(
        self,
        list_z: list[float],
        element_type: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return elements that intersect horizontal planes at multiple Z-levels.

        Uses cached element index (no redundant HDF5 reads) and vectorized
        Z-filtering (no iterrows).

        Parameters
        ----------
        list_z : list of float
            Z-coordinates where horizontal slicing planes are defined.
        element_type : str or None
            Base element type to filter. If None, all types.
        verbose : bool
            Print count per Z-level.

        Returns
        -------
        pd.DataFrame
            Intersecting elements with a 'z_level' column.
        """
        df_elements = self._get_cached_elements_df(element_type=element_type)
        if df_elements.empty:
            return pd.DataFrame()

        df_with_z = self._compute_z_bounds(df_elements)

        all_filtered = []
        for z_level in list_z:
            mask = (df_with_z["z_min"] <= z_level) & (df_with_z["z_max"] >= z_level)
            df_hit = df_with_z.loc[mask].copy()
            df_hit["z_level"] = z_level

            if verbose:
                print(f"[Z = {z_level}] Elements found: {len(df_hit)}")

            all_filtered.append(df_hit)

        if all_filtered:
            result = pd.concat(all_filtered, ignore_index=True)
            return result.drop(columns=["z_min", "z_max"], errors="ignore")
        return pd.DataFrame()

    def get_elements_in_selection_at_z_levels(
        self,
        list_z: list[float],
        *,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        element_ids: Union[int, Sequence[int], np.ndarray, None] = None,
        element_type: Optional[str] = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        """
        Return elements from a selection that intersect horizontal Z planes.

        Supports the unified _resolve_element_ids() API: by name, by id,
        or explicit element_ids.

        Parameters
        ----------
        list_z : list of float
            Z-levels for intersection planes.
        selection_set_id : int or list[int], optional
        selection_set_name : str or list[str], optional
        element_ids : int, list[int], ndarray, or None
        element_type : str or None
            Base element type filter.
        verbose : bool

        Returns
        -------
        pd.DataFrame
            Intersecting elements with 'z_level' column.
        """
        resolved_ids = self._resolve_element_ids(
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )

        df_elements = self._get_cached_elements_df(element_type=element_type)
        df_elements = df_elements[df_elements["element_id"].isin(resolved_ids)]

        if df_elements.empty:
            if verbose:
                print("[Elements] No elements match the selection criteria.")
            return pd.DataFrame()

        df_with_z = self._compute_z_bounds(df_elements)

        all_filtered = []
        for z_level in list_z:
            mask = (df_with_z["z_min"] <= z_level) & (df_with_z["z_max"] >= z_level)
            df_hit = df_with_z.loc[mask].copy()
            df_hit["z_level"] = z_level

            if verbose:
                print(f"[Z = {z_level}] Elements in selection: {len(df_hit)}")

            all_filtered.append(df_hit)

        if all_filtered:
            result = pd.concat(all_filtered, ignore_index=True)
            return result.drop(columns=["z_min", "z_max"], errors="ignore")
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    # Available results introspection
    # ------------------------------------------------------------------ #

    def get_available_element_results(
        self,
        element_type: Optional[str] = None,
    ) -> dict[str, dict[str, list[str]]]:
        """
        List available element result types across partitions.

        Parameters
        ----------
        element_type : str, optional
            Base or decorated name. If None, includes all.

        Returns
        -------
        dict
            {partition_id: {result_name: [matching decorated types]}}
        """
        results_by_partition: dict[str, dict[str, list[str]]] = {}

        for part_id, filepath in self.dataset.results_partitions.items():
            with h5py.File(filepath, "r") as f:
                try:
                    partition_results: dict[str, list[str]] = {}

                    for stage in self.dataset.model_stages:
                        group_path = f"{stage}/RESULTS/ON_ELEMENTS"
                        if group_path not in f:
                            continue

                        on_elements = f[group_path]
                        for result_name in on_elements:
                            result_group = on_elements[result_name]
                            matched = []
                            for etype_name in result_group:
                                if element_type is None:
                                    matched.append(etype_name)
                                elif (
                                    etype_name == element_type
                                    or etype_name.startswith(element_type)
                                ):
                                    matched.append(etype_name)

                            if matched:
                                partition_results[result_name] = matched

                    if partition_results:
                        results_by_partition[part_id] = partition_results

                except Exception as e:
                    print(f"[{filepath}] Error reading results: {e}")

        return results_by_partition

    # ------------------------------------------------------------------ #
    # Core results reader
    # ------------------------------------------------------------------ #

    @staticmethod
    def _sort_step_keys(keys: Sequence[str]) -> list[str]:
        """Sort HDF5 step keys numerically."""
        try:
            return [k for _, k in sorted((int(k), k) for k in keys)]
        except Exception:
            rx = re.compile(r"(\d+)(?!.*\d)")
            try:
                return [
                    k
                    for _, k in sorted(
                        (int(rx.search(k).group(1)), k) for k in keys  # type: ignore
                    )
                ]
            except Exception:
                return list(keys)

    def get_element_results(
        self,
        results_name: str,
        element_type: str,
        *,
        element_ids: Union[list[int], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        model_stage: Optional[str] = None,
        verbose: bool = False,
    ) -> "ElementResults":
        """
        Retrieve element results and return an ElementResults container.

        Uses the unified _resolve_element_ids() API for element selection.
        Reads HDF5 with sorted fancy indexing for performance.

        Parameters
        ----------
        results_name : str
            e.g. 'globalForces', 'section_deformation'
        element_type : str
            Base type (e.g. '203-ASDShellQ4') — not the decorated name.
        element_ids : list[int] or ndarray, optional
        selection_set_id : int or list[int], optional
        selection_set_name : str or list[str], optional
        model_stage : str, optional
            Defaults to the first model stage.
        verbose : bool

        Returns
        -------
        ElementResults
            Container with results DataFrame, time, and metadata.
        """
        # Resolve IDs
        ids = self._resolve_element_ids(
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )

        if model_stage is None:
            model_stage = self.dataset.model_stages[0]

        # Filter from cached index
        df_info = self._ensure_elem_index_df()
        base = element_type.split("[")[0]
        df_info = df_info[df_info["element_type"].str.startswith(base)]
        df_info = df_info[df_info["element_id"].isin(ids)]

        if df_info.empty:
            raise ValueError(
                f"No matching elements found for base type '{element_type}'."
            )

        if verbose:
            print(
                f"[Elements] {len(df_info)} matching elements for '{element_type}'"
            )

        collected: list[pd.DataFrame] = []

        # Group by partition for efficient HDF5 access
        for file_id, df_group in df_info.groupby("file_id"):
            file_path = self.dataset.results_partitions[int(file_id)]
            idx_arr = df_group["element_idx"].to_numpy(dtype=np.int64)
            id_arr = df_group["element_id"].to_numpy(dtype=np.int64)

            # Sort indices for efficient HDF5 fancy indexing
            sort_order = np.argsort(idx_arr, kind="mergesort")
            idx_sorted = idx_arr[sort_order]
            id_sorted = id_arr[sort_order]
            # Inverse to restore original order
            inv = np.empty_like(sort_order)
            inv[sort_order] = np.arange(sort_order.size)

            with h5py.File(file_path, "r") as f:
                base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}"
                if base_path not in f:
                    if verbose:
                        print(
                            f"[WARN] '{base_path}' not found in partition {file_id}"
                        )
                    continue

                candidates = list(f[base_path].keys())
                matching = [n for n in candidates if n.startswith(base)]

                if not matching:
                    if verbose:
                        print(
                            f"[WARN] No match for '{base}' under '{base_path}'"
                        )
                    continue

                for decorated_type in matching:
                    h5_data_path = f"{base_path}/{decorated_type}/DATA"
                    if h5_data_path not in f:
                        if verbose:
                            print(f"[WARN] Path not found: {h5_data_path}")
                        continue

                    data_group = f[h5_data_path]
                    step_names = self._sort_step_keys(data_group.keys())
                    n_steps = len(step_names)
                    n_elems = len(idx_sorted)

                    # Read one step to determine shape
                    sample = data_group[step_names[0]][idx_sorted[:1]]
                    n_comp = sample.shape[1]

                    # Pre-allocate and read all steps
                    out = np.empty((n_steps * n_elems, n_comp), dtype=np.float64)
                    for s, sname in enumerate(step_names):
                        raw = data_group[sname][idx_sorted]
                        out[s * n_elems : (s + 1) * n_elems, :] = raw[inv]

                    col_names = [f"val_{i + 1}" for i in range(n_comp)]
                    df_chunk = pd.DataFrame(out, columns=col_names)
                    df_chunk["element_id"] = np.tile(id_arr, n_steps)
                    df_chunk["step"] = np.repeat(np.arange(n_steps), n_elems)
                    collected.append(df_chunk)

        if not collected:
            if verbose:
                print("[Elements] No result data collected.")
            # Return empty ElementResults
            from .element_results import ElementResults

            return ElementResults(
                df=pd.DataFrame(),
                time=np.array([]),
                name=self.dataset.name,
                element_ids=tuple(),
                element_type=element_type,
                results_name=results_name,
                model_stage=model_stage,
            )

        result_df = pd.concat(collected, ignore_index=True)
        result_df = result_df.set_index(["element_id", "step"]).sort_index()

        # Extract time array for the model stage
        try:
            time_arr = self.dataset.time.loc[model_stage]["TIME"].to_numpy()
        except (KeyError, AttributeError):
            time_arr = np.array([])

        from .element_results import ElementResults

        return ElementResults(
            df=result_df,
            time=time_arr,
            name=self.dataset.name,
            element_ids=tuple(sorted(ids.tolist())),
            element_type=element_type,
            results_name=results_name,
            model_stage=model_stage,
        )

    def get_element_results_by_selection_and_z(
        self,
        results_name: str,
        list_z: list[float],
        *,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        element_type: Optional[str] = None,
        model_stage: Optional[str] = None,
        verbose: bool = False,
    ) -> dict[str, "ElementResults"]:
        """
        Filter elements by selection set + Z-levels, then fetch results.

        Returns ElementResults containers grouped by decorated element type.
        """
        df_filtered = self.get_elements_in_selection_at_z_levels(
            list_z=list_z,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            element_type=element_type,
            verbose=verbose,
        )

        if df_filtered.empty:
            if verbose:
                print("[Elements] No elements found at Z-levels in selection.")
            return {}

        # Get decorated type info from cached index
        df_info = self._ensure_elem_index_df()[
            ["element_id", "file_id", "element_type"]
        ]
        df_merged = pd.merge(
            df_filtered,
            df_info,
            on="element_id",
            suffixes=("", "_full"),
        )

        results_by_type: dict[str, "ElementResults"] = {}

        for decorated_type, df_group in df_merged.groupby("element_type_full"):
            eids = df_group["element_id"].unique().tolist()

            if verbose:
                print(f"  -> {decorated_type}: {len(eids)} elements")

            elem_results = self.get_element_results(
                results_name=results_name,
                element_type=str(decorated_type),
                element_ids=eids,
                model_stage=model_stage,
                verbose=verbose,
            )

            if elem_results.df.empty:
                continue

            results_by_type[str(decorated_type)] = elem_results

        return results_by_type
