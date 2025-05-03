
from typing import TYPE_CHECKING
import h5py
import numpy as np
import pandas as pd
from typing import Union


if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class Nodes:
    
    def __init__(self, dataset:'MPCODataSet'):
        self.dataset = dataset

    def _get_all_nodes_ids(self, verbose=False):
        """
        Retrieve all node IDs, file names, indices, and coordinates from the partition files.

        This method processes partition files, extracts node IDs and their corresponding coordinates, and returns 
        the results in both a structured NumPy array and a pandas DataFrame. It also provides an option to print 
        memory usage for both data representations.

        Args:
            print_memory (bool): If True, prints the memory usage of the structured array and DataFrame.

        Returns:
            dict: A dictionary containing:
                - 'array': A structured NumPy array with all node IDs, file names, indices, and coordinates (x, y, z).
                - 'dataframe': A pandas DataFrame with the same data.
        """

        node_data = []

        for part_number, partition_path in self.dataset.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                # We are assuming that the nodes remain consistent across all model stages
                model_stage=self.dataset.model_stages[0]
                nodes_group = partition.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                if nodes_group is None:
                    continue  # Skip this partition if the nodes group is not found

                for key in nodes_group.keys():
                    if key.startswith("ID"):
                        file_id = part_number
                        node_ids = nodes_group[key][...]
                        coord_key = key.replace("ID", "COORDINATES")
                        if coord_key in nodes_group:
                            coords = nodes_group[coord_key][...]
                            for index, (node_id, coord) in enumerate(zip(node_ids, coords)):
                                node_data.append((node_id, file_id, index, coord[0], coord[1], coord[2]))

        # Convert the list to a structured NumPy array
        dtype = [
            ('node_id', 'i8'),
            ('file_id', 'i8'),
            ('index', 'i8'),
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8')
        ]

        results_array = np.array(node_data, dtype=dtype)

        # Convert to a Pandas DataFrame
        columns = ['node_id', 'file_id', 'index', 'x', 'y', 'z']
        df = pd.DataFrame(node_data, columns=columns)

        results_dict = {
            'array': results_array, 
            'dataframe': df
        }

        if verbose:
            array_memory = results_array.nbytes
            df_memory = df.memory_usage(deep=True).sum()
            print(f"Memory usage for structured array (NODES): {array_memory / 1024**2:.2f} MB")
            print(f"Memory usage for DataFrame (NODES): {df_memory / 1024**2:.2f} MB")

        return results_dict

    def _validate_and_prepare_inputs(self, model_stage, results_name, node_ids, selection_set_id):
        """
        Validate inputs and return a NumPy array of node IDs.

        Raises:
            ValueError: On invalid combinations or missing/unknown inputs.
        Returns:
            np.ndarray: Array of node IDs.
        """
        # --- Validate mutually exclusive inputs ---
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Provide only one of 'node_ids' or 'selection_set_id', not both.")

        if node_ids is None and selection_set_id is None:
            raise ValueError("You must specify either 'node_ids' or 'selection_set_id'.")

        # --- Validate results_name ---
        if results_name not in self.dataset.node_results_names:
            raise ValueError(
                f"Result name '{results_name}' not found. Available options: {self.dataset.node_results_names}"
            )

        # --- Validate model_stage if provided ---
        if model_stage is not None and model_stage not in self.dataset.model_stages:
            raise ValueError(
                f"Model stage '{model_stage}' not found. Available stages: {self.dataset.model_stages}"
            )

        # --- Resolve selection_set ---
        if selection_set_id is not None:
            if selection_set_id not in self.dataset.selection_set:
                raise ValueError(f"Selection set ID '{selection_set_id}' not found.")
            selection = self.dataset.selection_set[selection_set_id]
            if "NODES" not in selection or not selection["NODES"]:
                raise ValueError(f"Selection set {selection_set_id} does not contain nodes.")
            return np.array(selection["NODES"], dtype=int)

        # --- Resolve node_ids ---
        if isinstance(node_ids, int):
            return np.array([node_ids], dtype=int)

        if isinstance(node_ids, list):
            if not node_ids:
                raise ValueError("'node_ids' list is empty.")
            return np.array(node_ids, dtype=int)

        if isinstance(node_ids, np.ndarray):
            if node_ids.size == 0:
                raise ValueError("'node_ids' array is empty.")
            return node_ids.astype(int)

        raise ValueError("Invalid 'node_ids' format. Must be int, non-empty list, or NumPy array.")

    def _get_stage_results(
        self,
        model_stage: str,
        results_name: str,
        node_ids: Union[np.ndarray, list, int]
    ) -> pd.DataFrame:
        """
        Retrieve nodal results for a given model stage and result type.

        Args:
            model_stage (str): Name of the model stage.
            results_name (str): Type of result to retrieve (e.g., 'Displacement').
            node_ids (np.ndarray | list | int): Node IDs to retrieve.

        Returns:
            pd.DataFrame: DataFrame indexed by (node_id, step) with result components as columns.
        """
        # Resolve node indices and file mapping
        nodes_info: pd.DataFrame = self.get_node_files_and_indices(node_ids=node_ids)
        base_path: str = f"{model_stage}/RESULTS/ON_NODES/{results_name}/DATA"

        # Group all entries by file_id to minimize file access
        file_groups: dict[int, pd.DataFrame] = {
            file_id: group for file_id, group in nodes_info.groupby('file_id')
        }

        all_results: list[pd.DataFrame] = []

        for file_id, group in file_groups.items():
            file_path: str = self.results_partitions[int(file_id)]

            with h5py.File(file_path, 'r') as results_file:
                data_group = results_file.get(base_path)
                if data_group is None:
                    raise ValueError(f"DATA group not found in path '{base_path}'.")

                step_names: list[str] = list(data_group.keys())
                node_indices: np.ndarray = group['index'].to_numpy(dtype=int)
                node_id_vals: np.ndarray = group['node_id'].to_numpy(dtype=int)

                for step_idx, step_name in enumerate(step_names):
                    dataset = data_group[step_name]
                    step_data = dataset[node_indices]

                    step_df = pd.DataFrame(
                        step_data,
                        index=node_id_vals,
                        columns=[f"val_{i + 1}" for i in range(step_data.shape[1])]
                    )
                    step_df['step'] = step_idx
                    step_df['node_id'] = node_id_vals
                    all_results.append(step_df)

        if not all_results:
            raise ValueError(f"No results found for model stage '{model_stage}'.")

        combined_df = pd.concat(all_results, axis=0)
        combined_df.set_index(['node_id', 'step'], inplace=True)
        combined_df.sort_index(inplace=True)

        return combined_df
    
    def get_nodal_results(self, model_stage=None, results_name=None, node_ids=None, selection_set_id=None):
        """
        Get nodal results optimized for numerical operations.
        Returns results as a structured DataFrame for efficient computation.

        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve (e.g., 'Displacement', 'Reaction').
            node_ids (int, list, or np.ndarray, optional): Specific node IDs to filter. Ignored if selection_set_id is used.
            selection_set_id (int, optional): The ID of the selection set to use for filtering node IDs.

        Returns:
            pd.DataFrame: If model_stage is None, returns MultiIndex DataFrame (stage, node_id, step).
                        Otherwise, returns Index (node_id, step). Columns represent result components.
        """
        # --- Validate and determine node_ids ---
        node_ids = self._validate_and_prepare_inputs(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=node_ids,
            selection_set_id=selection_set_id
        )

        if model_stage is None:
            all_results = []

            for stage in self.dataset.model_stages:
                try:
                    stage_df = self._get_stage_results(stage, results_name, node_ids)
                    stage_df['stage'] = stage
                    all_results.append(stage_df)
                except Exception as e:
                    print(f"[Warning] Could not retrieve results for stage '{stage}': {str(e)}")

            if not all_results:
                raise ValueError("No results found for any model stage.")

            # Combine and set a hierarchical index
            combined_df = pd.concat(all_results, axis=0)
            combined_df.set_index(['stage', 'node_id', 'step'], inplace=True)
            return combined_df.sort_index()

        # If a specific model stage is requested
        df = self._get_stage_results(model_stage, results_name, node_ids)
        df.set_index(['node_id', 'step'], inplace=True)
        return df.sort_index()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    