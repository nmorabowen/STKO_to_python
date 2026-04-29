from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from .element_results import ElementResults

logger = logging.getLogger(__name__)


class ElementManager:
    """
    High-performance MPCO element reader and domain manager.

    Canonical name for the refactored Layer 3 domain manager. The legacy
    name ``Elements`` is preserved as an alias at the bottom of this module
    for back-compat; new code should prefer ``ElementManager``.

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
        # ``decorated_type`` is the 2-field connectivity bracket
        # (e.g. "64-DispBeamColumn3d[1000:1]"). Required to disambiguate
        # results buckets when a model contains multiple integration
        # rules per element class — see docs/mpco_format_conventions.md
        # §1, §4.
        ("decorated_type", object),
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
        policy = self.dataset._format_policy

        for file_id in self.dataset.results_partitions:
            with self.dataset._pool.with_partition(file_id) as h5:
                elem_group_path = policy.model_elements_path(stage0)
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
                                "decorated_type": dset_name,
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
            arr["decorated_type"] = df["decorated_type"].to_numpy()
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
                    "decorated_type",
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

        Mirrors the Nodes._resolve_node_ids() API. Delegates to the
        dataset-owned :class:`SelectionSetResolver`.

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
        return self.dataset._selection_resolver.resolve_elements(
            names=selection_set_name,
            ids=selection_set_id,
            explicit_ids=element_ids,
        )

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

        for part_id in self.dataset.results_partitions:
            with self.dataset._pool.with_partition(part_id) as f:
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

    def _build_element_node_coords(
        self,
        sorted_ids: Tuple[int, ...],
        df_info: pd.DataFrame,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Build (element_node_coords, element_node_ids) aligned to
        ``sorted_ids`` (ascending element_id).

        Returns ``(None, None)`` if the dataset has no node coordinate
        table or the element index is missing — physical-coord
        computations on the resulting :class:`ElementResults` will
        return ``None``, but the named columns and IP layouts still
        come through cleanly.

        Shape:
            element_node_coords : ``(n_elements, n_nodes_per, 3)``
            element_node_ids    : ``(n_elements, n_nodes_per)``

        Assumes ``num_nodes`` is uniform across the matched elements
        (true within a single base class — different decorated brackets
        of the same class still share node count).
        """
        if not sorted_ids:
            return None, None
        nodes_info = getattr(self.dataset, "nodes_info", None)
        if not isinstance(nodes_info, dict):
            return None, None
        df_nodes = nodes_info.get("dataframe")
        if df_nodes is None or df_nodes.empty:
            return None, None

        # Map node_id → row index in the coord array.
        nids = df_nodes["node_id"].to_numpy(dtype=np.int64)
        coords = df_nodes[["x", "y", "z"]].to_numpy(dtype=np.float64)
        node_id_to_pos = {int(n): i for i, n in enumerate(nids)}

        # Slice df_info to one row per element_id, in sorted order.
        df_unique = df_info.drop_duplicates("element_id").set_index(
            "element_id", drop=False
        )
        try:
            df_sorted = df_unique.loc[list(sorted_ids)]
        except KeyError:
            return None, None

        node_lists = df_sorted["node_list"].tolist()
        if not node_lists:
            return None, None
        n_nodes_per = len(node_lists[0])
        n_elems = len(node_lists)

        out_ids = np.empty((n_elems, n_nodes_per), dtype=np.int64)
        out_xyz = np.empty((n_elems, n_nodes_per, 3), dtype=np.float64)
        for i, nl in enumerate(node_lists):
            if len(nl) != n_nodes_per:
                # Unexpected mixed node count — bail out rather than
                # silently producing a ragged array.
                return None, None
            for j, nid in enumerate(nl):
                out_ids[i, j] = int(nid)
                pos = node_id_to_pos.get(int(nid))
                if pos is None:
                    out_xyz[i, j, :] = np.nan
                else:
                    out_xyz[i, j, :] = coords[pos]
        return out_xyz, out_ids

    @staticmethod
    def _validate_homogeneous_layouts_across_stages(
        stage_buckets: dict[str, dict[str, tuple[str, ...]]],
        *,
        results_name: str,
        element_type: str,
    ) -> None:
        """Refuse silently mis-aligning frames across MODEL_STAGE groups
        when the column set for any shared bucket differs across stages.

        Mirrors :meth:`_validate_homogeneous_layouts` (which catches
        cross-bucket mismatches within a single stage); this one catches
        cross-stage mismatches that would otherwise produce NaN-padded
        rows in the contiguous-step concatenation. The bucket-by-bucket
        detail in the message lets the caller see which stage / bucket
        is the odd one out.
        """
        # Union of all bucket paths seen across stages.
        all_buckets: set[str] = set()
        for buckets in stage_buckets.values():
            all_buckets.update(buckets.keys())

        mismatches: dict[str, dict[str, tuple[str, ...] | None]] = {}
        for bucket in sorted(all_buckets):
            per_stage: dict[str, tuple[str, ...] | None] = {}
            for stage, buckets in stage_buckets.items():
                per_stage[stage] = buckets.get(bucket)
            unique = {v for v in per_stage.values() if v is not None}
            if len(unique) > 1:
                mismatches[bucket] = per_stage

        if not mismatches:
            return

        from ..io.meta_parser import MpcoFormatError

        raise MpcoFormatError(
            f"Heterogeneous bucket layouts across MODEL_STAGE groups "
            f"for results_name={results_name!r}, "
            f"element_type={element_type!r}: "
            f"{len(mismatches)} bucket(s) disagree across "
            f"{len(stage_buckets)} stages. "
            f"This usually means the model topology / integration "
            f"rules changed between stages (restart-style); query "
            f"each stage separately. Mismatches: "
            f"{ {b: {s: list(c) if c is not None else None for s, c in by_stage.items()} for b, by_stage in mismatches.items()} }"
        )

    @staticmethod
    def _validate_homogeneous_layouts(
        collected,
        *,
        results_name: str,
        element_type: str,
    ) -> None:
        """Refuse silently mis-aligning the result frame when matched
        buckets disagree on column layout.

        ``collected`` is the list of accumulated chunks from the
        per-partition / per-decorated-bracket read loop. Each entry's
        second slot is the ``col_names`` list. Concatenating chunks
        with different ``col_names`` would silently produce NaN-padded
        rows under ``pd.concat``, so we refuse loudly and ask the
        caller to query each decorated bracket separately. See
        docs/mpco_format_conventions §4 (the ``customRuleIdx`` axis:
        heterogeneous beam-integration rules in the same model spawn
        multiple results buckets per element class).
        """
        unique_layouts = {tuple(entry[1]) for entry in collected}
        if len(unique_layouts) <= 1:
            return
        from ..io.meta_parser import MpcoFormatError

        sample = {entry[0]: list(entry[1]) for entry in collected}
        raise MpcoFormatError(
            f"Heterogeneous bucket layouts for "
            f"results_name={results_name!r}, "
            f"element_type={element_type!r}: "
            f"{len(unique_layouts)} distinct column layouts across "
            f"{len(collected)} buckets. "
            f"This usually means the model has multiple integration "
            f"rules per element class. Query each decorated bucket "
            f"separately. Layouts encountered: {sample}"
        )

    @staticmethod
    def _resolve_bucket_layout(
        bucket_grp,
        bucket_path: str,
        *,
        verbose: bool = False,
    ):
        """Parse META for a results bucket; return ``None`` if META is
        absent or unreadable so the caller can fall back to the legacy
        ``val_*`` placeholders.

        Hard format violations (NUM_COLUMNS mismatch, GAUSS_IDS shape
        wrong) raise ``MpcoFormatError`` per the fail-loud contract —
        only "META not present" is silently downgraded to a fallback.
        """
        from ..io.meta_parser import MpcoFormatError, parse_bucket_meta

        if "META" not in bucket_grp:
            if verbose:
                print(
                    f"[Elements] No META at {bucket_path}; "
                    f"falling back to val_* column names."
                )
            return None
        try:
            return parse_bucket_meta(bucket_grp, bucket_path=bucket_path)
        except MpcoFormatError:
            raise

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

    @staticmethod
    def _normalize_stages(
        model_stage: Union[str, Sequence[str], None],
        all_stages: Sequence[str],
    ) -> Tuple[str, ...]:
        """Resolve the ``model_stage`` argument to a tuple of stage names.

        - ``None`` defaults to the first available stage (single-stage
          back-compat); not all stages, since users typically don't want
          surprise concatenation across every stage in the file.
        - ``str`` -> ``(stage,)`` (single-stage path).
        - ``Sequence[str]`` -> tuple in the order given (multi-stage
          path, contiguous step axis).
        """
        if model_stage is None:
            return (all_stages[0],) if all_stages else ()
        if isinstance(model_stage, str):
            return (model_stage,)
        out = tuple(str(s) for s in model_stage)
        if not out:
            raise ValueError("model_stage list is empty.")
        return out

    def get_element_results(
        self,
        results_name: str,
        element_type: str,
        *,
        element_ids: Union[list[int], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        verbose: bool = False,
    ) -> "ElementResults":
        """Public entry point — routes through the dataset-owned query
        engine so every call benefits from the LRU cache. Thin wrapper.
        """
        engine = getattr(self.dataset, "_element_query_engine", None)
        if engine is not None:
            return engine.fetch(
                results_name,
                element_type,
                element_ids=element_ids,
                selection_set_id=selection_set_id,
                selection_set_name=selection_set_name,
                model_stage=model_stage,
                verbose=verbose,
            )
        return self._fetch_element_results_uncached(
            results_name=results_name,
            element_type=element_type,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            model_stage=model_stage,
            verbose=verbose,
        )

    def _fetch_element_results_uncached(
        self,
        *,
        results_name: str,
        element_type: str,
        element_ids: Union[list[int], np.ndarray, None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        verbose: bool = False,
    ) -> "ElementResults":
        """Uncached read path for element results.

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
        model_stage : str or sequence of str, optional
            Single stage name (default: first stage), or a sequence of
            stages. Multi-stage fetches concatenate along a *contiguous
            global* step axis (stage 2 starts at the last step of stage
            1 + 1) and stitch the time arrays end-to-end. Bucket layouts
            (column names) must agree across stages — heterogeneous
            layouts raise ``MpcoFormatError``.
        verbose : bool

        Returns
        -------
        ElementResults
            Container with results DataFrame, time, and metadata.
            ``model_stages`` and ``stage_step_ranges`` describe the
            stage decomposition.
        """
        # Resolve IDs
        ids = self._resolve_element_ids(
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
        )

        stages = self._normalize_stages(model_stage, self.dataset.model_stages)
        if not stages:
            raise ValueError("Dataset has no model stages.")

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

        # Each entry:
        #   (results_bucket_path, col_names, gp_xi, gp_natural,
        #    gp_weights, df_chunk).
        #
        # ``gp_xi`` is the natural ξ from the connectivity ``@GP_X``
        # attribute (1-D, custom-rule beams only). ``gp_natural`` is
        # the multi-D natural-coord array (n_ip, dim) — for line
        # elements this is ``gp_xi.reshape(-1, 1)``; for shells/solids
        # it comes from the static catalog at
        # ``utilities/gauss_points.py``. ``gp_weights`` is catalog-
        # provided integration weights (None for line elements).
        # ``collected`` accumulates across all stages; ``stage_buckets``
        # tracks per-stage column layouts for the cross-stage validity
        # check.
        collected: list[
            tuple[
                str,
                list[str],
                Optional[np.ndarray],
                Optional[np.ndarray],
                Optional[np.ndarray],
                pd.DataFrame,
            ]
        ] = []
        stage_buckets: dict[str, dict[str, tuple[str, ...]]] = {
            st: {} for st in stages
        }
        stage_n_steps: dict[str, int] = {}
        step_offset = 0

        # Outer loop: stages. For multi-stage fetches we read each stage
        # in turn, tag its chunks with the contiguous global step offset,
        # and accumulate. Time arrays are stitched after the read loop.
        for model_stage in stages:
            stage_step_count = 0

            # Group by partition for efficient HDF5 access
            for file_id, df_group in df_info.groupby("file_id"):
                with self.dataset._pool.with_partition(int(file_id)) as f:
                    base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}"
                    if base_path not in f:
                        if verbose:
                            print(
                                f"[WARN] '{base_path}' not found in partition {file_id}"
                            )
                        continue

                    candidates = list(f[base_path].keys())

                    # Group by *connectivity* decorated bracket (2-field —
                    # the element_idx values are positions within that
                    # specific MODEL/ELEMENTS/<bracket> dataset and only
                    # apply to results buckets sharing the same rule:cust
                    # prefix).
                    if "decorated_type" not in df_group.columns:
                        raise RuntimeError(
                            "element index missing 'decorated_type' column; "
                            "rebuild the dataset to populate it."
                        )

                    for conn_decorated, sub_group in df_group.groupby(
                        "decorated_type"
                    ):
                        if not conn_decorated.endswith("]"):
                            # Defensive: connectivity brackets always end in ']'.
                            if verbose:
                                print(
                                    f"[WARN] Unexpected decorated_type "
                                    f"{conn_decorated!r}; skipping."
                                )
                            continue

                        # 3-field results bucket = 2-field connectivity
                        # bracket with ']' replaced by ':' followed by the
                        # response stream index, then ']'.
                        bracket_prefix = conn_decorated[:-1] + ":"
                        matching_results = [
                            n
                            for n in candidates
                            if n.startswith(bracket_prefix) and n.endswith("]")
                        ]

                        if not matching_results:
                            if verbose:
                                print(
                                    f"[WARN] No results bucket for connectivity "
                                    f"bracket {conn_decorated!r} under "
                                    f"{base_path}"
                                )
                            continue

                        # Read GP_X from the connectivity dataset (only
                        # present on custom-rule beam-columns; None for
                        # closed-form connectivities and continuum classes).
                        # See docs/mpco_format_conventions §1.
                        conn_path = (
                            f"{model_stage}/MODEL/ELEMENTS/{conn_decorated}"
                        )
                        conn_grp = f.get(conn_path)
                        gp_xi: Optional[np.ndarray] = None
                        if conn_grp is not None and "GP_X" in conn_grp.attrs:
                            gp_xi = (
                                np.asarray(conn_grp.attrs["GP_X"])
                                .flatten()
                                .astype(np.float64)
                            )

                        # Pre-compute fancy-indexing arrays for this group
                        # (element_idx is local to this connectivity bucket).
                        idx_arr = sub_group["element_idx"].to_numpy(dtype=np.int64)
                        id_arr = sub_group["element_id"].to_numpy(dtype=np.int64)
                        sort_order = np.argsort(idx_arr, kind="mergesort")
                        idx_sorted = idx_arr[sort_order]
                        inv = np.empty_like(sort_order)
                        inv[sort_order] = np.arange(sort_order.size)

                        # If there are multiple response streams under the
                        # same connectivity bracket (rare; ":1", ":2", ...),
                        # take the first stream and warn — concatenating
                        # them would silently duplicate (element_id, step)
                        # rows. Users wanting other streams should query
                        # the decorated_type explicitly.
                        if len(matching_results) > 1:
                            matching_results = sorted(matching_results)
                            if verbose:
                                print(
                                    f"[WARN] Multiple response streams under "
                                    f"{conn_decorated!r}: {matching_results}; "
                                    f"using {matching_results[0]!r}."
                                )
                            matching_results = matching_results[:1]

                        for results_decorated in matching_results:
                            bucket_path = f"{base_path}/{results_decorated}"
                            h5_data_path = f"{bucket_path}/DATA"
                            if h5_data_path not in f:
                                if verbose:
                                    print(f"[WARN] Path not found: {h5_data_path}")
                                continue

                            bucket_grp = f[bucket_path]
                            layout = self._resolve_bucket_layout(
                                bucket_grp, bucket_path, verbose=verbose
                            )

                            data_group = f[h5_data_path]
                            step_names = self._sort_step_keys(data_group.keys())
                            n_steps = len(step_names)
                            n_elems = len(idx_sorted)

                            if layout is not None:
                                n_comp = layout.num_columns
                                col_names = list(layout.flat_columns)
                            else:
                                sample = data_group[step_names[0]][idx_sorted[:1]]
                                n_comp = sample.shape[1]
                                col_names = [f"val_{i + 1}" for i in range(n_comp)]

                            out = np.empty(
                                (n_steps * n_elems, n_comp), dtype=np.float64
                            )
                            for s, sname in enumerate(step_names):
                                raw = data_group[sname][idx_sorted]
                                if raw.shape[1] != n_comp:
                                    from ..io.meta_parser import MpcoFormatError

                                    raise MpcoFormatError(
                                        f"{bucket_path}/DATA/{sname}: width "
                                        f"{raw.shape[1]} disagrees with expected "
                                        f"{n_comp} "
                                        f"(from {'META' if layout else 'first step'})"
                                    )
                                out[s * n_elems : (s + 1) * n_elems, :] = raw[inv]

                            df_chunk = pd.DataFrame(out, columns=col_names)
                            df_chunk["element_id"] = np.tile(id_arr, n_steps)
                            # Apply the contiguous-global step offset so
                            # multi-stage fetches concatenate without
                            # overlap. Stage-local step numbering is
                            # 0..n_steps-1; we add ``step_offset`` so
                            # stage 2 starts where stage 1 ended.
                            df_chunk["step"] = np.repeat(
                                np.arange(n_steps) + step_offset, n_elems
                            )
                            # GP_X (1-D ξ) is only meaningful for non-closed-
                            # form line buckets. For closed-form, drop it
                            # even if the connectivity dataset happened to
                            # carry one.
                            chunk_gp_xi = (
                                gp_xi
                                if (layout is not None and not layout.closed_form)
                                else None
                            )

                            # Resolve the multi-D natural-coord layout. For
                            # line elements with GP_X we get a 1×n_ip vector
                            # which we promote to (n_ip, 1). For shells /
                            # solids we consult the catalog by base class +
                            # IP count. Closed-form buckets stay None.
                            chunk_gp_natural: Optional[np.ndarray] = None
                            chunk_gp_weights: Optional[np.ndarray] = None
                            if layout is not None and not layout.closed_form:
                                if chunk_gp_xi is not None:
                                    chunk_gp_natural = chunk_gp_xi.reshape(-1, 1)
                                    # Line-element custom rules: weights
                                    # are not in the file. Leave as None.
                                else:
                                    # Look up catalog by the connectivity
                                    # bracket's base class (strip the
                                    # ``[rule:cust]`` suffix).
                                    from ..utilities.gauss_points import (
                                        get_ip_layout,
                                    )

                                    conn_base = conn_decorated.split("[", 1)[0]
                                    cat = get_ip_layout(conn_base, layout.n_ip)
                                    if cat is not None:
                                        chunk_gp_natural = np.asarray(
                                            cat[0], dtype=np.float64
                                        )
                                        chunk_gp_weights = np.asarray(
                                            cat[1], dtype=np.float64
                                        )

                            collected.append(
                                (
                                    bucket_path,
                                    col_names,
                                    chunk_gp_xi,
                                    chunk_gp_natural,
                                    chunk_gp_weights,
                                    df_chunk,
                                )
                            )
                            # Stage-independent bucket key: the
                            # ``<connectivity>/<results_decorated>`` pair
                            # uniquely identifies a logical bucket
                            # across stages (the stage prefix in
                            # ``bucket_path`` would otherwise make every
                            # cross-stage comparison vacuously distinct).
                            bucket_key = (
                                f"{conn_decorated}/{results_decorated}"
                            )
                            stage_buckets[model_stage][bucket_key] = tuple(
                                col_names
                            )
                            stage_step_count = max(stage_step_count, n_steps)

            stage_n_steps[model_stage] = stage_step_count
            step_offset += stage_step_count

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
                model_stage=stages[0],
                model_stages=stages,
                stage_step_ranges={},
            )

        # Cross-bucket layout consistency check (refuses heterogeneous
        # integration rules — see docs/mpco_format_conventions §4).
        self._validate_homogeneous_layouts(
            collected,
            results_name=results_name,
            element_type=element_type,
        )

        # Cross-stage layout consistency: the column set for each bucket
        # must agree across stages. Restart-style models (where the
        # column layout differs between stages) raise the same loud
        # MpcoFormatError as cross-bucket mismatches — silent NaN
        # padding under pd.concat would be a footgun.
        if len(stages) > 1:
            self._validate_homogeneous_layouts_across_stages(
                stage_buckets,
                results_name=results_name,
                element_type=element_type,
            )

        # Reconcile gp_xi / gp_natural / gp_weights across chunks.
        # All non-None values for a given attribute must agree; if not,
        # that's another flavor of heterogeneous integration (already
        # caught by the layout check above unless someone changes the
        # catalog out from under us). Pick the first non-None.
        merged_gp_xi: Optional[np.ndarray] = None
        merged_gp_natural: Optional[np.ndarray] = None
        merged_gp_weights: Optional[np.ndarray] = None
        for _, _, c_xi, c_nat, c_w, _ in collected:
            if c_xi is not None:
                if merged_gp_xi is None:
                    merged_gp_xi = c_xi
                elif not np.allclose(
                    merged_gp_xi, c_xi, rtol=0.0, atol=1e-12
                ):
                    from ..io.meta_parser import MpcoFormatError

                    raise MpcoFormatError(
                        f"GP_X disagrees across buckets for "
                        f"results_name={results_name!r}, "
                        f"element_type={element_type!r}: "
                        f"{merged_gp_xi.tolist()} vs {c_xi.tolist()}. "
                        f"This means heterogeneous beam-integration "
                        f"rules; query each decorated bucket separately."
                    )
            if c_nat is not None and merged_gp_natural is None:
                merged_gp_natural = c_nat
            if c_w is not None and merged_gp_weights is None:
                merged_gp_weights = c_w

        result_df = pd.concat(
            [entry[5] for entry in collected], ignore_index=True
        )
        result_df = result_df.set_index(["element_id", "step"]).sort_index()

        # Stitch per-stage time arrays end-to-end. Stage time arrays
        # come from f[stage].attrs["TIME"] via dataset.time; the total
        # ``time`` is naturally monotonic because each stage records
        # absolute time. Build ``stage_step_ranges`` alongside.
        stage_step_ranges: Dict[str, Tuple[int, int]] = {}
        time_chunks: list[np.ndarray] = []
        running = 0
        for st in stages:
            n = stage_n_steps.get(st, 0)
            stage_step_ranges[st] = (running, running + n)
            running += n
            try:
                t = self.dataset.time.loc[st]["TIME"].to_numpy()
                # Truncate or extend to match this stage's actual step
                # count; if the time DataFrame is shorter (e.g. partial
                # write) we accept the shorter array.
                time_chunks.append(np.asarray(t, dtype=np.float64)[:n])
            except (KeyError, AttributeError):
                time_chunks.append(np.full(n, np.nan, dtype=np.float64))
        time_arr = (
            np.concatenate(time_chunks) if time_chunks else np.array([])
        )

        from .element_results import ElementResults

        # Build per-element node-coordinate arrays for B7b physical-
        # coordinate / Jacobian computations. Aligned with the sorted
        # element_ids that the result df uses.
        sorted_ids_tup = tuple(sorted(ids.tolist()))
        elem_node_coords, elem_node_ids = self._build_element_node_coords(
            sorted_ids_tup, df_info
        )

        return ElementResults(
            df=result_df,
            time=time_arr,
            name=self.dataset.name,
            element_ids=sorted_ids_tup,
            element_type=element_type,
            results_name=results_name,
            model_stage=stages[0],
            model_stages=stages,
            stage_step_ranges=stage_step_ranges,
            gp_xi=merged_gp_xi,
            gp_natural=merged_gp_natural,
            gp_weights=merged_gp_weights,
            element_node_coords=elem_node_coords,
            element_node_ids=elem_node_ids,
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


# Back-compat alias — the legacy name ``Elements`` is guaranteed
# importable from ``STKO_to_python.elements.elements`` and from the
# ``STKO_to_python.elements`` package.
Elements = ElementManager
