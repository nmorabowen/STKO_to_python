import glob
import os
import re
from typing import TYPE_CHECKING
from collections import defaultdict
from typing import Optional, Dict, List, Sequence, Any
import h5py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class ModelInfo:
    """
    This class has a "friend" relationship with MPCODataSet, which is allowed
    to access protected methods.
    """
    def __init__(self, dataset:'MPCODataSet'):
        self.dataset = dataset
        
    def _get_file_list(self, extension: Optional[str] = None, verbose: bool = False) -> Dict[str, Dict[int, str]]:
        """
        Retrieves a mapping of partitioned files from the dataset directory.

        Parameters
        ----------
        extension : str, optional
            The file extension to look for (e.g., 'mpco', 'cdata'). If not provided,
            it uses the extension from `self.MCPODataSet.file_extension`.
        verbose : bool, optional
            If True, prints detailed information about the files found.

        Returns
        -------
        dict
            A nested dictionary structured as:
            {
                'recorder_base_name': {
                    0: '/path/to/recorder_base_name.part-0.<ext>',
                    1: '/path/to/recorder_base_name.part-1.<ext>',
                    ...
                },
                ...
            }

        Raises
        ------
        FileNotFoundError
            If no files with the given extension are found in the specified directory.
        Exception
            For unexpected errors during file parsing.
        """
        if extension is None:
            extension = self.dataset.file_extension.strip("*.")

        results_directory = self.dataset.hdf5_directory

        try:
            files = glob.glob(os.path.join(results_directory, f"*.{extension}"))
            if not files:
                raise FileNotFoundError(f"No .{extension} files found in {results_directory}")

            file_mapping = defaultdict(dict)

            for file in files:
                filename = os.path.basename(file)

                if ".part-" in filename:
                    try:
                        name, part_str = filename.split(".part-", 1)
                        part = int(part_str.split(".")[0])
                        file_mapping[name][part] = file
                    except (ValueError, IndexError):
                        print(f"Skipping file due to unexpected naming format: {file}")
                else:
                    # Handle compound extensions like ".mpco.cdata"
                    if filename.endswith(f".{extension}"):
                        # Remove .<extension> (e.g., .cdata)
                        base_with_possible_extra_ext = filename[: -len(extension) - 1]
                        # Remove any remaining extra extension like .mpco if present
                        base = re.sub(r"\.mpco$", "", base_with_possible_extra_ext)
                        file_mapping[base][0] = file

            if verbose:
                print("\nFound files:")
                for name, parts in file_mapping.items():
                    print(f"\n{name}:")
                    for part, path in sorted(parts.items()):
                        print(f"  Part: {part}, File: {path}")

            return file_mapping

        except Exception as e:
            print(f"Model Info Error during file listing: {e}")
            raise
        
    def _get_file_list_for_results_name(self, extension= None, verbose=False):

        if extension is None:
            extension=self.dataset.file_extension.strip("*.")
            
        file_info=self._get_file_list(extension=extension, verbose=verbose)

        recorder_files = file_info.get(self.dataset.recorder_name)

        if recorder_files is None:
            raise ValueError(f"Model Info Error: Recorder name '{self.dataset.recorder_name}' not found in {extension} files.")

        return recorder_files
    
    def _get_model_stages(self, verbose=False):
        """
        Retrieve model stages from all result partitions.

        Args:
            verbose (bool, optional): If True, prints the model stages.

        Returns:
            list: Sorted list of model stage names from all partitions.
        """
        model_stages = []

        # Use partition paths from the dictionary created by _get_results_partitions
        for _, partition_path in self.dataset.results_partitions.items():
            
            with h5py.File(partition_path, 'r') as results:
                # Get model stages from the current partition file
                partition_stages = [key for key in results.keys() if key.startswith("MODEL_STAGE")]
                model_stages.extend(partition_stages)

        # Remove duplicates by converting to a set, then back to a sorted list
        model_stages = sorted(set(model_stages))

        if not model_stages:
            raise ValueError("Model Info Error: No model stages found in the result partitions.")

        if verbose:
            print(f'The model stages found across partitions are: {model_stages}')

        return model_stages    
    
    def _get_node_results_names(
            self,
            model_stage: Optional[str] = None,
            verbose: bool = False,
            raise_if_empty: bool = False
    ) -> List[str]:
        """
        Retrieve the names of nodal results for a given model stage.

        Args:
            model_stage (str, optional): Model stage name. If None, search all stages.
            verbose (bool, optional): Print the discovered result names.
            raise_if_empty (bool, optional): If True, raise ValueError when nothing is found.
                                            If False (default) return an empty list instead.

        Returns:
            list[str]: Sorted list of nodal result names (may be empty).
        """
        # 1. Determine which stages to inspect
        model_stages = [model_stage] if model_stage else self.dataset.model_stages

        node_results_names: set[str] = set()

        # 2. Scan every partition for every requested stage
        for stage in model_stages:
            for _, partition_path in self.dataset.results_partitions.items():
                with h5py.File(partition_path, "r") as results:
                    nodes_group = results.get(
                        self.dataset.RESULTS_ON_NODES_PATH.format(model_stage=stage)
                    )
                    if nodes_group:
                        node_results_names.update(nodes_group.keys())

            if verbose:
                print(f"Node results in '{stage}': {sorted(node_results_names)}")

        # 3. Handle empty results according to caller’s wishes
        if not node_results_names:
            message = (
                f"No nodal results found for stage(s): {', '.join(model_stages)} "
                f"in {len(self.dataset.results_partitions)} partition(s)."
            )
            if raise_if_empty:
                raise ValueError(message)
            logger.warning(message)

        return sorted(node_results_names)
    
    def _get_elements_results_names(self, model_stage=None, verbose=False):
        """
        Retrieve the names of element results for a given model stage from all result partitions.
        
        Args:
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the element results names.
        
        Returns:
            list: List of element results names across all partitions.
        """
        if model_stage is None:
            model_stages=self.dataset.model_stages
        else:
            # Check for model stage errors
            # self._model_stages_error(model_stage)(ERROR CONTROL PENDING)
            model_stages=[model_stage]
        
        element_results_names = []
        for model_stage in model_stages:
            # Iterate over all result partitions
            for _, partition_path in self.dataset.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    # Get the element results for the given model stage
                    ele_results = results.get(self.dataset.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage))
                    if ele_results is None:
                        continue  # Skip this partition if the element results group is not found
                    
                    # Append the element results names from this partition
                    element_results_names.extend(ele_results.keys())
            
            # Remove duplicates by converting to a set and then back to a list
            element_results_names = list(set(element_results_names))
            
            if not element_results_names:
                raise ValueError(f"Model Info: No element results found for model stage '{model_stage}' in the result partitions.")
            
            if verbose:
                print(f'Model Info: The element results names found across partitions for model stage "{model_stage}" are: {element_results_names}')
        
        return element_results_names
    
    def _get_element_types(self, model_stage=None, results_name=None, verbose=False):
        """
        Retrieve the element types for a given result name and model stage.
        If results_name is None, get types for all results.

        Args:
            model_stage (str, optional): Name of the model stage. If None, retrieve results for all model stages.
            results_name (str, optional): Name of the results group.
            verbose (bool, optional): If True, prints the element types.

        Returns:
            dict: Dictionary mapping results names to their element types.
        """
        if model_stage is None:
            model_stages = self.dataset.model_stages
        else:
            # Check for model stage errors
            # self._model_stages_error(model_stage) (ERROR CONTROL PENDING)
            model_stages = [model_stage]
            
        if results_name is None:
            results_names = self.dataset.element_results_names
        else:
            # self._element_results_name_error(results_name, stage) (ERROR CONTROL PENDING)
            results_names = [results_name]

        element_types_dict = {}
        for stage in model_stages:
            for partition, partition_path in self.dataset.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    for name in results_names:
                        ele_types = results.get(self.dataset.RESULTS_ON_ELEMENTS_PATH.format(model_stage=stage) + f"/{name}")
                        if ele_types is None:
                            raise ValueError(f"Model Info Error: Element types group not found for {name} in partition {partition}")
                        if name not in element_types_dict:
                            element_types_dict[name] = []
                            element_types_dict[name].extend(list(ele_types.keys()))

        unique_element_types = set()
        
        unique_element_types = set()
        for name in element_types_dict:
            unique_element_types.update(element_types_dict[name])

        if verbose:
            print(f'The element types found are: {unique_element_types}')
            
        results = {'element_types_dict': element_types_dict, 'unique_element_types': unique_element_types}
        
        return results
    
    def _get_all_types(self, model_stage=None):
        
        if model_stage is None:
            model_stages = self.dataset.model_stages
        else:
            # Check for model stage errors
            # self._model_stages_error(model_stage) (ERROR CONTROL PENDING)
            model_stages = [model_stage]
        
        element_types = set()
        for model_stage in model_stages:
            for _, partition_path in self.dataset.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    results_names = self._get_elements_results_names(model_stage)
                    for name in results_names:
                        ele_types = results.get(self.dataset.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{name}")
                        if ele_types is None:
                            raise ValueError(f"Model Info Error: Element types group not found for {name}")
                        element_types.update([key.split('[')[0] for key in ele_types.keys()])
            
                
            return sorted(list(element_types))
    
    def _get_time_series_on_nodes_for_stage(self, model_stage, results_name):
        """
        Retrieve and consolidate the unique time series data across all partitions 
        for a given model stage and nodal results name, returning a Pandas DataFrame.

        Args:
            model_stage (str): The model stage to query.
            results_name (str): The nodal results name to query.

        Returns:
            pd.DataFrame: A DataFrame with columns ['STEP', 'TIME'], sorted by STEP.
        """

        time_series_dict = {}  # Dictionary to store STEP -> TIME mapping

        for part_number, partition_path in self.dataset.results_partitions.items():
            try:
                with h5py.File(partition_path, 'r') as partition:
                    base_path = f"{model_stage}/RESULTS/ON_NODES/{results_name}/DATA"
                    data_group = partition.get(base_path)

                    if data_group is None:
                        continue  # Skip if data does not exist

                    # Iterate over all steps and collect STEP & TIME attributes
                    for step_name in data_group.keys():
                        step_group = data_group[step_name]
                        step_value = step_group.attrs.get("STEP")
                        time_value = step_group.attrs.get("TIME")

                        if step_value is not None and time_value is not None:
                            time_series_dict[int(step_value)] = float(time_value)  # Store STEP -> TIME mapping

            except Exception as e:
                print(f"Model Info Error: Get time series error processing partition {part_number} for model stage '{model_stage}', results name '{results_name}': {e}")

        # Convert to DataFrame
        df = pd.DataFrame(list(time_series_dict.items()), columns=['STEP', 'TIME']).sort_values(by='STEP')

        return df
    
    def _get_time_series_on_elements_for_stage(self, model_stage, results_name, element_type):
        """
        Retrieve and consolidate the unique time series data across all partitions 
        for a given model stage, element results name, and specific element type, 
        returning a Pandas DataFrame.

        Args:
            model_stage (str): The model stage to query.
            results_name (str): The element results name to query (e.g., 'force', 'deformation').
            element_type (str): The specific element type to query (e.g., '203-ASDShellQ4').

        Returns:
            pd.DataFrame: A DataFrame with columns ['STEP', 'TIME'], sorted by STEP.
        """

        time_series_dict = {}  # Dictionary to store STEP -> TIME mapping

        for part_number, partition_path in self.dataset.results_partitions.items():
            try:
                with h5py.File(partition_path, 'r') as partition:
                    base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}/{element_type}/DATA"
                    data_group = partition.get(base_path)

                    if data_group is None:
                        continue  # Skip if DATA does not exist

                    # Iterate over all steps and collect STEP & TIME attributes
                    for step_name in data_group.keys():
                        step_group = data_group[step_name]
                        step_value = step_group.attrs.get("STEP")
                        time_value = step_group.attrs.get("TIME")

                        if step_value is not None and time_value is not None:
                            time_series_dict[int(step_value)] = float(time_value)  # Store STEP -> TIME mapping

            except Exception as e:
                print(f"Model Info Error: Get time series error processing partition {part_number} for model stage '{model_stage}', results name '{results_name}', element type '{element_type}': {e}")

        # Convert to DataFrame
        df = pd.DataFrame(list(time_series_dict.items()), columns=['STEP', 'TIME']).sort_values(by='STEP')

        return df
    
    def _get_time_series(self) -> pd.DataFrame:
        """
        Consolidate the unique STEP–TIME pairs for every model stage,
        even if a stage contains only nodal *or* only element results.

        Returns
        -------
        pd.DataFrame
            Multi-index ['MODEL_STAGE', 'STEP'] → ['TIME'].
        """

        # ── convenience ─────────────────────────────────────────────────────────
        node_names: list[str] = (self.dataset.node_results_names or [])
        elem_names: list[str] = (self.dataset.element_results_names or [])
        elem_types_dict: dict[str, list[str]] = self.dataset.element_types.get(
            "element_types_dict", {}
        )

        all_time_series: list[pd.DataFrame] = []

        for stage in self.dataset.model_stages:
            time_df: pd.DataFrame | None = None

            # 1) Try every nodal result (if any) until one yields data
            for n_result in node_names:
                df = self._get_time_series_on_nodes_for_stage(stage, n_result)
                if not df.empty:
                    time_df = df
                    break                                       # ← success!

            # 2) If still empty, try the element results
            if time_df is None or time_df.empty:
                for e_result in elem_names:
                    e_types = elem_types_dict.get(e_result, [])
                    for e_type in e_types:                       # try each element type
                        df = self._get_time_series_on_elements_for_stage(
                            stage, e_result, e_type
                        )
                        if not df.empty:
                            time_df = df
                            break
                    if time_df is not None and not time_df.empty:
                        break                                   # ← success!
            # 3) No data at all → raise a *stage-specific* error
            if time_df is None or time_df.empty:
                raise ValueError(
                    f"Model Info Error: No time-series data found for model stage "
                    f"'{stage}'. Checked {len(node_names)} nodal results and "
                    f"{len(elem_names)} element results."
                )

            # tag with stage and collect
            time_df["MODEL_STAGE"] = stage
            all_time_series.append(time_df)

        # ── union ───────────────────────────────────────────────────────────────
        final_df = (
            pd.concat(all_time_series, copy=False)
            .set_index(["MODEL_STAGE", "STEP"])
            .sort_index()
        )
        return final_df
    
    def _get_number_of_steps(self) -> Dict[str, int]:
        """
        Determine how many analysis steps exist in each model stage,
        regardless of whether the data are stored under ON_NODES or ON_ELEMENTS.

        Returns
        -------
        dict[str, int]
            Mapping {MODEL_STAGE: n_steps}.
        """
        # convenience handles -------------------------------------------------
        node_names: List[str] = self.dataset.node_results_names or []
        elem_names: List[str] = self.dataset.element_results_names or []
        elem_types_dict: Dict[str, List[str]] = self.dataset.element_types.get(
            "element_types_dict", {}
        )
        partitions = list(self.dataset.results_partitions.values())

        steps_info: Dict[str, int] = {}

        for stage in self.dataset.model_stages:
            step_ids: set[int] = set()

            # 1) nodal results ------------------------------------------------
            for n_res in node_names:
                for part_path in partitions:
                    with h5py.File(part_path, "r") as f:
                        grp = f.get(f"{stage}/RESULTS/ON_NODES/{n_res}/DATA")
                        if grp is not None:
                            step_ids.update(self._to_step_int(k) for k in grp.keys())
                if step_ids:
                    break  # found data → stop searching nodal

            # 2) element results ---------------------------------------------
            if not step_ids:
                for e_res in elem_names:
                    for e_type in elem_types_dict.get(e_res, []):
                        for part_path in partitions:
                            with h5py.File(part_path, "r") as f:
                                grp = f.get(
                                    f"{stage}/RESULTS/ON_ELEMENTS/{e_res}/{e_type}/DATA"
                                )
                                if grp is not None:
                                    step_ids.update(
                                        self._to_step_int(k) for k in grp.keys()
                                    )
                        if step_ids:
                            break
                    if step_ids:
                        break

            # 3) error if nothing found --------------------------------------
            if not step_ids:
                raise ValueError(
                    f"Model Info Error: no STEP datasets located for model stage "
                    f"'{stage}'. Checked {len(node_names)} nodal results and "
                    f"{len(elem_names)} element results."
                )

            steps_info[stage] = len(step_ids)

        return steps_info

    def get_node_coordinates(
        self,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        as_dict: bool = False,
    ) -> pd.DataFrame | dict[int, dict[str, Any]]:
        """
        Return full rows (node_id, file_id, index, x, y, z, …) from
        ``self.dataset.nodes_info``.

        Exactly **one** of *node_ids* or *selection_set_id* is required.
        """
        # --- XOR check ---------------------------------------------------- #
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError(
                "Specify **either** 'node_ids' **or** 'selection_set_id' (one, not both)."
            )

        # --- resolve IDs -------------------------------------------------- #
        if node_ids is None:
            if not hasattr(self.dataset.nodes, "get_nodes_in_selection_set"):
                raise AttributeError(
                    "self.dataset.nodes lacks 'get_nodes_in_selection_set'. "
                    "Implement it or pass explicit 'node_ids'."
                )
            node_ids = self.dataset.nodes.get_nodes_in_selection_set(selection_set_id)

        # preserve caller order, drop duplicates
        node_ids = list(dict.fromkeys(node_ids))

        # --- master table ------------------------------------------------- #
        df_all = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict)
            else self.dataset.nodes_info
        )

        # --- verify existence -------------------------------------------- #
        missing = set(node_ids) - set(df_all["node_id"])
        if missing:
            raise KeyError(f"Unknown node IDs: {sorted(missing)}")

        # --- slice while keeping order ----------------------------------- #
        sub = (
            df_all.set_index("node_id")
            .loc[node_ids]              # preserves specified order
            .reset_index()
        )

        # --- return format ----------------------------------------------- #
        if as_dict:
            return {row.node_id: row._asdict() for row in sub.itertuples(index=False)}
        return sub
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _to_step_int(step_key: str | bytes, pattern: str = r"(\d+)$") -> int:
        """
        Convert an HDF5 dataset key to its integer STEP index.

        Parameters
        ----------
        step_key : str | bytes
            Raw dataset key from HDF5 (e.g. 'STEP_0', b'3', 'Step-12').
        pattern : str, optional
            Regex that captures the numeric portion; default grabs trailing digits.

        Returns
        -------
        int
            Numeric STEP value.
        """
        if isinstance(step_key, bytes):           # h5py may yield bytes
            step_key = step_key.decode()

        # Fast path: key is already numeric
        if step_key.isdigit():
            return int(step_key)

        # Fallback: extract digits with regex
        match = re.search(pattern, step_key)
        if match:
            return int(match.group(1))

        raise ValueError(f"Un-recognisable STEP key: {step_key!r}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    