from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union
import re

import h5py
import numpy as np
import pandas as pd

from ..core.element_selection import ElementSelection
from ..core.selection import SelectionBox

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from ..results.element_results_dataclass import ElementResults


class Elements:
    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset
        self._selection_info = None

    def _get_all_element_index(
        self,
        element_type: str | Sequence[str] | None = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        stage0 = self.dataset.model_stages[0]

        requested_types: tuple[str, ...] | None = None
        if element_type is not None:
            if isinstance(element_type, str):
                requested_types = (element_type.strip(),)
            else:
                requested_types = tuple(
                    str(x).strip() for x in element_type if str(x).strip()
                )

        node_coord_map = {}
        if hasattr(self.dataset, "nodes_info") and "dataframe" in self.dataset.nodes_info:
            df_nodes = self.dataset.nodes_info["dataframe"]
            for node_id, x, y, z in zip(
                df_nodes["node_id"],
                df_nodes["x"],
                df_nodes["y"],
                df_nodes["z"],
            ):
                node_coord_map[int(node_id)] = (float(x), float(y), float(z))
        else:
            node_coord_map = None

        elements_info: list[dict[str, Any]] = []

        for part_number, partition_path in self.dataset.results_partitions.items():
            with h5py.File(partition_path, "r") as partition:
                elem_path = self.dataset.MODEL_ELEMENTS_PATH.format(model_stage=stage0)
                element_group = partition.get(elem_path)
                if element_group is None:
                    continue

                for element_name in element_group.keys():
                    base_name = str(element_name).split("[")[0]
                    if requested_types is not None and not any(
                        element_name == req
                        or base_name == req
                        or str(element_name).startswith(req + "[")
                        for req in requested_types
                    ):
                        continue

                    dataset = element_group[element_name]
                    data = dataset[:]
                    for idx, element_data in enumerate(data):
                        element_id = int(element_data[0])
                        node_list = [int(nid) for nid in element_data[1:]]

                        if node_coord_map:
                            sx = sy = sz = 0.0
                            for nid in node_list:
                                x, y, z = node_coord_map.get(nid, (0.0, 0.0, 0.0))
                                sx += x
                                sy += y
                                sz += z
                            num_nodes = len(node_list)
                            centroid_x = sx / num_nodes
                            centroid_y = sy / num_nodes
                            centroid_z = sz / num_nodes
                        else:
                            num_nodes = len(node_list)
                            centroid_x = centroid_y = centroid_z = None

                        elements_info.append(
                            {
                                "element_id": element_id,
                                "element_idx": int(idx),
                                "file_name": int(part_number),
                                "element_type": str(element_name),
                                "element_type_base": base_name,
                                "node_list": node_list,
                                "num_nodes": num_nodes,
                                "centroid_x": centroid_x,
                                "centroid_y": centroid_y,
                                "centroid_z": centroid_z,
                            }
                        )

        if elements_info:
            dtype = [
                ("element_id", "i8"),
                ("element_idx", "i8"),
                ("file_name", "i8"),
                ("element_type", object),
                ("element_type_base", object),
                ("node_list", object),
                ("num_nodes", "i8"),
                ("centroid_x", "f8"),
                ("centroid_y", "f8"),
                ("centroid_z", "f8"),
            ]
            structured_data = [
                (
                    elem["element_id"],
                    elem["element_idx"],
                    elem["file_name"],
                    elem["element_type"],
                    elem["element_type_base"],
                    elem["node_list"],
                    elem["num_nodes"],
                    elem["centroid_x"] if elem["centroid_x"] is not None else np.nan,
                    elem["centroid_y"] if elem["centroid_y"] is not None else np.nan,
                    elem["centroid_z"] if elem["centroid_z"] is not None else np.nan,
                )
                for elem in elements_info
            ]
            results_array = np.array(structured_data, dtype=dtype)
            df = pd.DataFrame(elements_info)

            if verbose:
                array_memory = results_array.nbytes
                df_memory = df.memory_usage(deep=True).sum()
                print(
                    f"Memory usage for structured array (ELEMENTS): {array_memory / 1024**2:.2f} MB"
                )
                print(
                    f"Memory usage for DataFrame (ELEMENTS): {df_memory / 1024**2:.2f} MB"
                )

            return {
                "array": results_array,
                "dataframe": df,
            }

        if verbose:
            print("No elements found.")

        return {
            "array": np.array(
                [],
                dtype=[
                    ("element_id", "i8"),
                    ("element_idx", "i8"),
                    ("file_name", "i8"),
                    ("element_type", object),
                    ("element_type_base", object),
                    ("node_list", object),
                    ("num_nodes", "i8"),
                    ("centroid_x", "f8"),
                    ("centroid_y", "f8"),
                    ("centroid_z", "f8"),
                ],
            ),
            "dataframe": pd.DataFrame(),
        }

    def _ensure_element_index_df(self) -> pd.DataFrame:
        ei = getattr(self.dataset, "elements_info", None)
        if isinstance(ei, dict) and isinstance(ei.get("dataframe"), pd.DataFrame):
            return ei["dataframe"]
        out = self._get_all_element_index(verbose=False)
        return out["dataframe"]

    @staticmethod
    def _normalize_stages(stages, all_stages) -> Tuple[str, ...]:
        available = tuple(map(str, all_stages))
        if stages is None:
            return available
        if isinstance(stages, str):
            requested = (stages,)
        else:
            requested = tuple(map(str, stages))

        missing = [stage for stage in requested if stage not in available]
        if missing:
            raise ValueError(
                f"Invalid model_stage value(s): {tuple(missing)}. Available: {available}"
            )
        return requested

    def _normalize_results(self, results) -> Tuple[str, ...]:
        available = tuple(sorted(map(str, self.dataset.element_results_names)))
        if results is None:
            return available
        if isinstance(results, str):
            requested = (results,)
        else:
            requested = tuple(map(str, results))

        missing = tuple(name for name in requested if name not in available)
        if missing:
            raise ValueError(
                f"Invalid results_name value(s): {missing}. "
                f"Available element results: {available}"
            )
        return requested

    def _selection_helper(self):
        if self._selection_info is None:
            from ..results.element_results_info import ElementResultsInfo

            element_df = self._ensure_element_index_df()
            self._selection_info = ElementResultsInfo(
                elements_ids=tuple(
                    element_df["element_id"].to_numpy(dtype=np.int64, copy=False).tolist()
                ),
                elements_info=element_df.set_index("element_id", drop=False),
                selection_set=self.dataset.selection_set,
            )
        return self._selection_info

    def _resolve_element_ids(
        self,
        *,
        selection: ElementSelection | None = None,
        element_ids: Union[int, Sequence[int], None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        element_type: Union[str, Sequence[str], None] = None,
    ) -> np.ndarray:
        selection_info = self._selection_helper()
        resolved = ElementSelection.from_element_filters(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
        )
        return np.asarray(
            selection_info.resolve_selection(resolved, only_available=True),
            dtype=np.int64,
        )

    def _element_file_map(self, element_ids: np.ndarray) -> pd.DataFrame:
        df = self._ensure_element_index_df()
        sub = df[df["element_id"].isin(element_ids)][
            ["element_id", "file_name", "element_idx", "element_type"]
        ]
        if sub.empty:
            raise ValueError("No element IDs found in dataset.")
        return (
            sub.sort_values(
                ["element_id", "file_name", "element_type", "element_idx"],
                kind="mergesort",
            )
            .drop_duplicates("element_id", keep="first")
            .sort_values("element_id", kind="mergesort")
            .reset_index(drop=True)
        )

    @staticmethod
    def _sort_step_keys(keys: Sequence[str]) -> list[str]:
        try:
            return [k for _, k in sorted((int(k), k) for k in keys)]
        except Exception:
            rx = re.compile(r"(\d+)(?!.*\d)")
            try:
                return [k for _, k in sorted((int(rx.search(k).group(1)), k) for k in keys)]
            except Exception:
                return list(keys)

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _read_multi_results_all_steps(
        *,
        h5: h5py.File,
        stage: str,
        results: Sequence[str],
        element_ids: np.ndarray,
        local_idx: np.ndarray,
        decorated_type: str,
    ) -> pd.DataFrame:
        order = np.argsort(local_idx, kind="mergesort")
        idx_sorted = local_idx[order]
        inv = np.empty_like(order)
        inv[order] = np.arange(order.size)

        ref = h5[f"{stage}/RESULTS/ON_ELEMENTS/{results[0]}/{decorated_type}/DATA"]
        step_names = Elements._sort_step_keys(ref.keys())

        n_elements = len(idx_sorted)
        n_steps = len(step_names)

        blocks = []
        lvl0, lvl1 = [], []
        for r in results:
            g = h5[f"{stage}/RESULTS/ON_ELEMENTS/{r}/{decorated_type}/DATA"]
            sample = Elements._ensure_2d(g[step_names[0]][idx_sorted[:1]])
            ncomp = sample.shape[1]

            out = np.empty((n_steps * n_elements, ncomp))
            for s, step in enumerate(step_names):
                arr = Elements._ensure_2d(g[step][idx_sorted])
                out[s * n_elements : (s + 1) * n_elements, :] = arr[inv]

            blocks.append(out)
            lvl0.extend([r] * ncomp)
            lvl1.extend(range(1, ncomp + 1))

        big = np.hstack(blocks)
        cols = pd.MultiIndex.from_arrays([lvl0, lvl1], names=("result", "component"))
        df = pd.DataFrame(big, columns=cols)
        df["element_id"] = np.tile(element_ids, n_steps)
        df["step"] = np.repeat(np.arange(n_steps), n_elements)
        return df

    def _build_element_results(
        self,
        *,
        results_name: Union[str, Sequence[str], None] = None,
        model_stage: Union[str, Sequence[str], None] = None,
        element_ids: Union[int, Sequence[int], None] = None,
        selection_set_id: Union[int, Sequence[int], None] = None,
        selection_set_name: Union[str, Sequence[str], None] = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        selection: ElementSelection | None = None,
        element_type: Union[str, Sequence[str], None] = None,
    ) -> "ElementResults":
        stages = self._normalize_stages(model_stage, self.dataset.model_stages)
        results = self._normalize_results(results_name)

        ids = self._resolve_element_ids(
            selection=selection,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            element_type=element_type,
        )
        ids_sorted = np.sort(ids)

        fmap = self._element_file_map(ids_sorted)
        group_keys = ["file_name", "element_type"]
        file_groups = {key: g for key, g in fmap.groupby(group_keys)}

        info_cols = [
            "file_name",
            "element_idx",
            "element_type",
            "element_type_base",
            "node_list",
            "num_nodes",
            "centroid_x",
            "centroid_y",
            "centroid_z",
        ]
        info_df = (
            self._ensure_element_index_df()
            .drop_duplicates("element_id")
            .set_index("element_id")
            .loc[ids_sorted, info_cols]
            .copy()
        )
        info_df["element_id"] = info_df.index.astype(int)

        stage_frames = []
        for st in stages:
            file_frames = []
            for (file_id, decorated_type), grp in file_groups.items():
                with h5py.File(self.dataset.results_partitions[int(file_id)], "r") as h5:
                    file_frames.append(
                        self._read_multi_results_all_steps(
                            h5=h5,
                            stage=st,
                            results=results,
                            element_ids=grp["element_id"].to_numpy(dtype=np.int64, copy=False),
                            local_idx=grp["element_idx"].to_numpy(dtype=np.int64, copy=False),
                            decorated_type=str(decorated_type),
                        )
                    )

            if not file_frames:
                raise ValueError("No element result data collected for the resolved selection.")

            df_stage = pd.concat(file_frames, ignore_index=True)
            df_stage = df_stage.set_index(["element_id", "step"]).sort_index()

            if len(stages) > 1:
                df_stage = df_stage.reset_index()
                df_stage["stage"] = st
                df_stage = df_stage.set_index(["stage", "element_id", "step"]).sort_index()

            stage_frames.append(df_stage)

        df = stage_frames[0] if len(stage_frames) == 1 else pd.concat(stage_frames).sort_index()
        time = (
            self.dataset.time.loc[stages[0]]["TIME"].to_numpy()
            if len(stages) == 1
            else {s: self.dataset.time.loc[s]["TIME"].to_numpy() for s in stages}
        )

        component_names = tuple("|".join(map(str, c)) for c in df.columns)

        from ..results.element_results_dataclass import ElementResults

        return ElementResults(
            df=df,
            time=time,
            name=self.dataset.name,
            elements_ids=tuple(ids_sorted),
            elements_info=info_df,
            results_components=component_names,
            model_stages=stages,
            selection_set=self.dataset.selection_set,
            analysis_time=self.dataset.info.analysis_time,
            size=self.dataset.info.size,
            plot_settings=self.dataset.plot_settings,
        )

    @staticmethod
    def _legacy_dataframe_from_results(
        results: "ElementResults",
        *,
        result_name: str | Sequence[str] | None,
    ) -> pd.DataFrame:
        if isinstance(result_name, str):
            out = results.fetch(result_name=result_name, component=None)
        else:
            out = results.df.copy()

        if isinstance(out, pd.Series):
            return out.to_frame(name="val_1")
        if isinstance(out.columns, pd.MultiIndex):
            df = out.copy()
            df.columns = [f"val_{i+1}" for i in range(df.shape[1])]
            return df
        return out.copy()

    def get_elements_at_z_levels(
        self,
        list_z: list[float],
        element_type: str | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        result = self._get_all_element_index(element_type=element_type, verbose=False)
        df_elements = result["dataframe"]
        if not hasattr(self.dataset, "nodes_info") or "dataframe" not in self.dataset.nodes_info:
            raise ValueError("Node information is not available in the dataset.")

        df_nodes = self.dataset.nodes_info["dataframe"]
        node_z_map = dict(zip(df_nodes["node_id"], df_nodes["z"]))
        all_filtered = []

        for z_level in list_z:
            filtered_elements = []
            for _, row in df_elements.iterrows():
                node_ids = row["node_list"]
                z_coords = [node_z_map.get(nid, None) for nid in node_ids]
                z_coords = [z for z in z_coords if z is not None]
                if not z_coords:
                    continue
                min_z = min(z_coords)
                max_z = max(z_coords)
                if min_z <= z_level <= max_z:
                    filtered_elements.append(row)

            df_filtered = pd.DataFrame(filtered_elements)
            if not df_filtered.empty:
                df_filtered["z_level"] = z_level
            all_filtered.append(df_filtered)
            if verbose:
                print(f"[Z = {z_level}] Elements found: {len(df_filtered)}")

        if all_filtered:
            return pd.concat(all_filtered, ignore_index=True)
        return pd.DataFrame()

    def get_available_element_results(
        self,
        element_type: str = None,
    ) -> dict[str, dict[str, list[str]]]:
        results_by_partition = {}

        for part_id, filepath in self.dataset.results_partitions.items():
            with h5py.File(filepath, "r") as f:
                try:
                    partition_results = {}
                    for stage in self.dataset.model_stages:
                        group_path = f"{stage}/RESULTS/ON_ELEMENTS"
                        if group_path not in f:
                            continue

                        on_elements = f[group_path]
                        for result_name in on_elements:
                            result_group = on_elements[result_name]
                            matched_element_types = []
                            for etype_name in result_group:
                                if element_type is None:
                                    matched_element_types.append(etype_name)
                                elif etype_name == element_type:
                                    matched_element_types.append(etype_name)
                                elif etype_name.startswith(element_type):
                                    matched_element_types.append(etype_name)

                            if matched_element_types:
                                partition_results[result_name] = matched_element_types

                    if partition_results:
                        results_by_partition[part_id] = partition_results
                except Exception as e:
                    print(f"[{filepath}] → Error reading results: {e}")

        return results_by_partition

    def get_elements_in_selection_at_z_levels(
        self,
        selection_set_id: int,
        list_z: list[float],
        element_type: str | None = None,
        verbose: bool = False,
    ) -> pd.DataFrame:
        result = self._get_all_element_index(element_type=element_type, verbose=False)
        df_elements = result["dataframe"]

        try:
            element_ids = self.dataset.selection_set[selection_set_id]["ELEMENTS"]
        except (AttributeError, KeyError):
            raise ValueError(
                f"Selection set {selection_set_id} not found or has no 'ELEMENTS' key."
            )

        df_elements = df_elements[df_elements["element_id"].isin(element_ids)]
        if not hasattr(self.dataset, "nodes_info") or "dataframe" not in self.dataset.nodes_info:
            raise ValueError("Node information is not available in the dataset.")

        df_nodes = self.dataset.nodes_info["dataframe"]
        node_z_map = dict(zip(df_nodes["node_id"], df_nodes["z"]))
        all_filtered = []

        for z_level in list_z:
            filtered_elements = []
            for _, row in df_elements.iterrows():
                node_ids = row["node_list"]
                z_coords = [node_z_map.get(nid, None) for nid in node_ids]
                z_coords = [z for z in z_coords if z is not None]
                if not z_coords:
                    continue
                if min(z_coords) <= z_level <= max(z_coords):
                    filtered_elements.append(row)

            df_filtered = pd.DataFrame(filtered_elements)
            if not df_filtered.empty:
                df_filtered["z_level"] = z_level
            all_filtered.append(df_filtered)

            if verbose:
                print(f"[Z = {z_level}] Elements in selection set: {len(df_filtered)}")

        if all_filtered:
            return pd.concat(all_filtered, ignore_index=True)
        return pd.DataFrame()

    def get_element_results(
        self,
        results_name: str | Sequence[str] | None = None,
        element_type: str | Sequence[str] | None = None,
        element_ids: int | Sequence[int] | None = None,
        model_stage: str | Sequence[str] | None = None,
        *,
        selection_set_id: int | Sequence[int] | None = None,
        selection_set_name: str | Sequence[str] | None = None,
        selection_box: SelectionBox | Sequence[Sequence[float]] | None = None,
        selection: ElementSelection | None = None,
        return_dataframe: bool | None = None,
        verbose: bool = False,
    ) -> "ElementResults" | pd.DataFrame:
        if verbose:
            print("[Elements] Resolving element results selection")

        legacy_mode = (
            return_dataframe is None
            and selection is None
            and selection_set_id is None
            and selection_set_name is None
            and selection_box is None
            and element_ids is not None
            and element_type is not None
        )
        if return_dataframe is None:
            return_dataframe = legacy_mode

        results = self._build_element_results(
            results_name=results_name,
            model_stage=model_stage,
            element_ids=element_ids,
            selection_set_id=selection_set_id,
            selection_set_name=selection_set_name,
            selection_box=selection_box,
            selection=selection,
            element_type=element_type,
        )

        if return_dataframe:
            return self._legacy_dataframe_from_results(results, result_name=results_name)
        return results

    def get_element_results_by_selection_and_z(
        self,
        results_name: str,
        selection_set_id: int,
        list_z: list[float],
        element_type: str | None = None,
        model_stage: str | None = None,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        df_filtered = self.get_elements_in_selection_at_z_levels(
            selection_set_id=selection_set_id,
            list_z=list_z,
            element_type=element_type,
            verbose=verbose,
        )

        if df_filtered.empty:
            if verbose:
                print("[INFO] No elements found at Z-levels in selection set.")
            return {}

        df_info = self.dataset.elements_info["dataframe"][
            ["element_id", "file_name", "element_type"]
        ]
        df_merged = pd.merge(
            df_filtered,
            df_info,
            on=["element_id", "file_name"],
            suffixes=("", "_decorated"),
        )
        df_merged["element_type"] = df_merged["element_type_decorated"]

        results_by_type: dict[str, pd.DataFrame] = {}
        for decorated_type, df_group in df_merged.groupby("element_type"):
            resolved_ids = df_group["element_id"].unique().tolist()
            if verbose:
                print(f"\n↳ {decorated_type}: {len(resolved_ids)} elements to fetch")

            df_result = self.get_element_results(
                results_name=results_name,
                element_type=decorated_type,
                element_ids=resolved_ids,
                model_stage=model_stage,
                return_dataframe=True,
                verbose=verbose,
            )

            if df_result.empty:
                continue

            df_with_z = (
                pd.merge(
                    df_result.reset_index(),
                    df_group[["element_id", "z_level"]].drop_duplicates(),
                    on="element_id",
                    how="left",
                )
                .set_index(["element_id", "step"])
                .sort_index()
            )
            results_by_type[str(decorated_type)] = df_with_z

        return results_by_type
