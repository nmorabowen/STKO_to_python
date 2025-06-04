import glob
import os
import re
from typing import TYPE_CHECKING
from collections import defaultdict
from typing import Optional, Dict
import h5py
import pandas as pd

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
            print(f"Error during file listing: {e}")
            raise
        
    def _get_file_list_for_results_name(self, extension= None, verbose=False):

        if extension is None:
            extension=self.dataset.file_extension.strip("*.")
            
        file_info=self._get_file_list(extension=extension, verbose=verbose)

        recorder_files = file_info.get(self.dataset.recorder_name)

        if recorder_files is None:
            raise ValueError(f"Recorder name '{self.dataset.recorder_name}' not found in {extension} files.")

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
            raise ValueError("No model stages found in the result partitions.")

        if verbose:
            print(f'The model stages found across partitions are: {model_stages}')

        return model_stages    
    
    def _get_node_results_names(self, model_stage=None, verbose=False):
        """
        Retrieve the names of node results names for a given model stage.

        Args:
            model_stage (str, optional): Name of the model stage. If None, retrieve results for all model stages.
            verbose (bool, optional): If True, prints the node results names.

        Returns:
            list: List of node results names.
        """
        if model_stage is None:
            model_stages = self.dataset.model_stages
        else:
            # Check for model stage errors
            # self._model_stages_error(model_stage) (ERROR CONTROL PENDING)
            model_stages = [model_stage]

        node_results_names = set()

        for stage in model_stages:
            for _, partition_path in self.dataset.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    nodes_groups = results.get(self.dataset.RESULTS_ON_NODES_PATH.format(model_stage=stage))
                    if nodes_groups is None:
                        continue  # Skip this partition if the nodes group is not found
                    node_results_names.update(nodes_groups.keys())

            if verbose:
                print(f"Node results names for model stage '{stage}': {list(node_results_names)}")

        if not node_results_names:
            raise ValueError(f"No node results found for model stage(s): {', '.join(model_stages)} in the result partitions.")

        return list(node_results_names)
    
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
                raise ValueError(f"No element results found for model stage '{model_stage}' in the result partitions.")
            
            if verbose:
                print(f'The element results names found across partitions for model stage "{model_stage}" are: {element_results_names}')
        
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
                            raise ValueError(f"Element types group not found for {name} in partition {partition}")
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
                            raise ValueError(f"Element types group not found for {name}")
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
                print(f"Error processing partition {part_number}: {e}")

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
                print(f"Error processing partition {part_number}: {e}")

        # Convert to DataFrame
        df = pd.DataFrame(list(time_series_dict.items()), columns=['STEP', 'TIME']).sort_values(by='STEP')

        return df
    
    def _get_time_series(self):
        """
        Retrieve and consolidate the unique time series data across all model stages,
        storing them in a MultiIndex Pandas DataFrame.

        Returns:
            pd.DataFrame: A MultiIndex DataFrame with index ['MODEL_STAGE', 'STEP'] 
                        and column ['TIME'], sorted by MODEL_STAGE and STEP.
        """

        model_stages = self.dataset.model_stages  # Get all model stages
        model_elements_types = self.dataset.element_types['element_types_dict']  # Result name -> Element types

        all_time_series = []  # List to store results for all model stages

        for model_stage in model_stages:
            time_series_df = None  # Initialize per-stage DataFrame

            # Check if nodal results exist
            if self.dataset.node_results_names is not None:
                # Get the time series from nodes
                time_series_df = self._get_time_series_on_nodes_for_stage(model_stage, self.dataset.node_results_names[0])
            
            elif self.dataset.element_results_names is not None:
                # Find the first valid element result
                for result, element_list in model_elements_types.items():
                    if result is not None and element_list:
                        element_type = element_list[0]  # Use the first element type
                        time_series_df = self._get_time_series_on_elements_for_stage(model_stage, result, element_type)
                        break

            if time_series_df is not None:
                # Add MODEL_STAGE as a new column
                time_series_df["MODEL_STAGE"] = model_stage
                all_time_series.append(time_series_df)

            else:
                raise ValueError(f"No nodal or element results found for model stage: {model_stage}")

        # Concatenate all time series DataFrames and set MultiIndex
        final_df = pd.concat(all_time_series).set_index(["MODEL_STAGE", "STEP"]).sort_index()

        return final_df
    
    def _get_number_of_steps(self):
        """
        Retrieves and stores the number of steps (datasets) for each model stage 
        based on available nodal or element results.

        Returns:
            dict: A dictionary mapping model stages to their respective number of steps.
        """
        # Dictionary to store steps for each model stage
        steps_info = {}

        # Get all model stages
        model_stages = self.dataset.model_stages

        # Iterate through each model stage
        for stage in model_stages:
            try:
                # Try nodal results first
                nodal_results = self._get_node_results_names(stage)
                if nodal_results:
                    # Use the first nodal result to get the step count
                    results_name = nodal_results[0]
                    node_partition = self.dataset.nodes_info['dataframe']['file_id'][0]  # Get the partition
                    base_path = f"{stage}/RESULTS/ON_NODES/{results_name}/DATA"

                    with h5py.File(self.dataset.results_partitions[int(node_partition)], 'r') as results:
                        data_group = results.get(base_path)
                        if data_group is None:
                            raise ValueError(f"DATA group not found in path '{base_path}'.")
                        
                        # Number of datasets directly represents the number of steps
                        steps_info[stage] = len(data_group)
                    continue

                # If no nodal results, try element results
                element_results = self._get_elements_results_names(stage)
                
                if element_results:
                    results_name = element_results[0]
                    element_types = self._get_element_types(stage, results_name)[results_name]
                    if element_types:
                        element_partition = 0  # Adjust if partition handling is needed
                        base_path = f"{stage}/RESULTS/ON_ELEMENTS/{results_name}/{element_types[0]}/DATA"

                        with h5py.File(self.dataset.results_partitions[element_partition], 'r') as results:
                            data_group = results.get(base_path)
                            if data_group is None:
                                raise ValueError(f"DATA group not found in path '{base_path}'.")

                            # Number of datasets directly represents the number of steps
                            steps_info[stage] = len(data_group)

            except Exception as e:
                print(f"Error processing model stage '{stage}': {e}")
                steps_info[stage] = None

        return steps_info
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    