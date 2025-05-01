import h5py
import numpy as np
from collections import defaultdict
import os
import glob
import re
import pandas as pd


class GetModelInfo:
    """This is a mixin class to be used with the MCPO_VirtualDataset retreive model information fom the dataset."""
    
    def get_model_stages(self, verbose=False):
        """
        Retrieve model stages from all result partitions.

        Args:
            verbose (bool, optional): If True, prints the model stages.

        Returns:
            list: Sorted list of model stage names from all partitions.
        """
        model_stages = []

        # Use partition paths from the dictionary created by _get_results_partitions
        for _, partition_path in self.results_partitions.items():
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


    
    def get_elements_results_names(self, model_stage=None, verbose=False):
        """
        Retrieve the names of element results for a given model stage from all result partitions.
        
        Args:
            model_stage (str): Name of the model stage.
            verbose (bool, optional): If True, prints the element results names.
        
        Returns:
            list: List of element results names across all partitions.
        """
        if model_stage is None:
            model_stages=self.model_stages
        else:
            # Check for model stage errors
            self._model_stages_error(model_stage)
            model_stages=[model_stage]
        
        element_results_names = []
        for model_stage in model_stages:
            # Iterate over all result partitions
            for _, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    # Get the element results for the given model stage
                    ele_results = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage))
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
    
    def get_element_types(self, model_stage=None, results_name=None, verbose=False):
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
            model_stages = self.model_stages
        else:
            # Check for model stage errors
            self._model_stages_error(model_stage)
            model_stages = [model_stage]

        element_types_dict = {}
        for stage in model_stages:
            for partition, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    if results_name is None:
                        results_names = self.element_results_names
                    else:
                        self._element_results_name_error(results_name, stage)
                        results_names = [results_name]

                    for name in results_names:
                        ele_types = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=stage) + f"/{name}")
                        if ele_types is None:
                            raise ValueError(f"Element types group not found for {name} in partition {partition}")
                        if name not in element_types_dict:
                            element_types_dict[name] = []
                            element_types_dict[name].extend(list(ele_types.keys()))

        unique_element_types = set()
        
        # Remove duplicates in the lists of element types
        for name in element_types_dict:
            unique_element_types.update(element_types_dict[name])

        if verbose:
            print(f'The element types found are: {element_types_dict}')
            
        results = {'element_types_dict': element_types_dict, 'unique_element_types': unique_element_types}
        
        return results

        
    def _get_all_types(self, model_stage=None):
        
        if model_stage is None:
            model_stages = self.model_stages
        else:
            # Check for model stage errors
            self._model_stages_error(model_stage)
            model_stages = [model_stage]
        
        element_types = set()
        for model_stage in model_stages:
            for _, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    results_names = self.get_elements_results_names(model_stage)
                    for name in results_names:
                        ele_types = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{name}")
                        if ele_types is None:
                            raise ValueError(f"Element types group not found for {name}")
                        element_types.update([key.split('[')[0] for key in ele_types.keys()])
            
                
            return sorted(list(element_types))
        
    def get_node_results_names(self, model_stage=None, verbose=False):
        """
        Retrieve the names of node results names for a given model stage.

        Args:
            model_stage (str, optional): Name of the model stage. If None, retrieve results for all model stages.
            verbose (bool, optional): If True, prints the node results names.

        Returns:
            list: List of node results names.
        """
        if model_stage is None:
            model_stages = self.model_stages
        else:
            # Check for model stage errors
            self._model_stages_error(model_stage)
            model_stages = [model_stage]

        node_results_names = set()

        for stage in model_stages:
            for _, partition_path in self.results_partitions.items():
                with h5py.File(partition_path, 'r') as results:
                    nodes_groups = results.get(self.RESULTS_ON_NODES_PATH.format(model_stage=stage))
                    if nodes_groups is None:
                        continue  # Skip this partition if the nodes group is not found
                    node_results_names.update(nodes_groups.keys())

            if verbose:
                print(f"Node results names for model stage '{stage}': {list(node_results_names)}")

        if not node_results_names:
            raise ValueError(f"No node results found for model stage(s): {', '.join(model_stages)} in the result partitions.")

        return list(node_results_names)
    
    
    
    def get_nodes_by_z_coordinate_bak(self, model_stage, z_value, tolerance=1e-6):
        """
        Retrieve all nodes at a specific z-coordinate value within a given tolerance.
        
        Args:
            model_stage (str): The model stage to query.
            z_value (float): The target z-coordinate value.
            tolerance (float, optional): The tolerance for comparing z-coordinates. Defaults to 1e-6.
        
        Returns:
            list: A list of node IDs at the specified z-coordinate.
        """
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Access the nodes group
            nodes_group = h5file.get(self.MODEL_NODES_PATH.format(model_stage=model_stage))
            if nodes_group is None:
                raise ValueError(f"Nodes group not found for model stage '{model_stage}'.")

            node_ids = []
            for key in nodes_group.keys():
                if key.startswith("ID"):
                    ids = nodes_group[key][...]  # Node IDs
                    coord_key = key.replace("ID", "COORDINATES")
                    if coord_key in nodes_group:
                        coordinates = nodes_group[coord_key][...]  # Node coordinates
                        # Filter nodes by the z-coordinate
                        for i, coord in enumerate(coordinates):
                            if abs(coord[2] - z_value) <= tolerance:  # Compare z-coordinate
                                node_ids.append(int(ids[i]))
            return node_ids
        
    def _get_file_list(self, extension: str, verbose=False):
        """
        Retrieves and organizes files with the specified extension in the results directory.

        Args:
            extension (str): The file extension to search for (e.g., 'cdata', 'txt').
            verbose (bool, optional): If True, prints the found files. Default is False.

        Returns:
            defaultdict: A dictionary mapping base names to a dictionary with part numbers as keys 
                        and file paths as values. Example:
                        
                        {
                            "base_name_1": {
                                0: "/path/to/base_name_1.part-0.extension",
                                1: "/path/to/base_name_1.part-1.extension"
                            },
                            "base_name_2": {
                                0: "/path/to/base_name_2.part-0.extension"
                            },
                        }

        Raises:
            ValueError: If the extension is not provided or is empty.
            FileNotFoundError: If no files with the specified extension are found.
            Exception: For other unexpected errors.
        """
        # Validate inputs
        if not extension or not isinstance(extension, str):
            raise ValueError("Invalid file extension provided. Please provide a non-empty string.")
        
        results_directory = getattr(self, 'results_directory', None)
        if not results_directory or not os.path.isdir(results_directory):
            raise ValueError("The 'results_directory' attribute is not set or is not a valid directory.")
        
        try:
            # Use glob to find all files with the given extension
            files = glob.glob(os.path.join(results_directory, f"*.{extension}"))
            
            if not files:
                raise FileNotFoundError(
                    f"No files with the extension '.{extension}' were found in {results_directory}."
                )
            
            # Dictionary to store mapping: { base_name: { part_number: file_path, ... } }
            file_mapping = defaultdict(dict)
            
            for file in files:
                # Extract the filename without directory
                filename = os.path.basename(file)
                
                # Ensure the file has the expected part naming (e.g., ".part-XX")
                if ".part-" in filename:
                    try:
                        name, part_str = filename.split(".part-", 1)
                        part = int(part_str.split(".")[0])  # Extract part number
                        # Add to the mapping
                        file_mapping[name][part] = file
                    except (ValueError, IndexError):
                        print(f"Skipping file due to unexpected naming format: {file}")
            
            # Optionally print the mapping for debugging
            if verbose:
                print("\nFound files:")
                for name, parts in file_mapping.items():
                    print(f"\n{name}:")
                    for part, path in sorted(parts.items()):
                        print(f"  Part: {part}, File: {path}")
            
            return file_mapping

        except FileNotFoundError as fnf_error:
            print(f"Error: {fnf_error}")
            raise
        except ValueError as ve_error:
            print(f"Error: {ve_error}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise
        
    def find_elements_crossing_plane(self, plane_point, plane_normal):
        """
        Find all elements that cross a given plane.

        Args:
            plane_point (list or np.ndarray): A point [x, y, z] on the plane.
            plane_normal (list or np.ndarray): The normal vector [nx, ny, nz] of the plane.

        Returns:
            list: A list of element IDs that cross the plane.
        """
        # Validate and normalize inputs
        plane_point = np.array(plane_point, dtype=float)
        plane_normal = np.array(plane_normal, dtype=float)
        if plane_point.shape != (3,) or plane_normal.shape != (3,):
            raise ValueError("plane_point and plane_normal must be 3D vectors.")
        plane_normal /= np.linalg.norm(plane_normal)  # Normalize the plane normal

        crossing_elements = []

        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Load mappings
            nodes_mapping = h5file['/Mappings/Nodes'][:]
            element_mapping = h5file['/Mappings/Elements'][:]

            # Extract node coordinates for efficient access
            node_id_to_coords = {node['node_id']: (node['x'], node['y'], node['z']) for node in nodes_mapping}

            # Iterate over elements
            for element in element_mapping:
                element_id = element['element_id']
                node_list = element['node_list']
                # Filter out padding (-1)
                node_list = node_list[node_list != -1]

                # Get coordinates for nodes in the element
                try:
                    node_coords = np.array([node_id_to_coords[node_id] for node_id in node_list])
                except KeyError as e:
                    print(f"Warning: Node ID {e} not found in mappings. Skipping element {element_id}.")
                    continue

                # Compute signed distances to the plane
                signed_distances = np.dot(node_coords - plane_point, plane_normal)

                # Check if nodes exist on both sides of the plane
                if np.any(signed_distances < 0) and np.any(signed_distances > 0):
                    crossing_elements.append(element_id)

        return crossing_elements
    
    def get_number_of_steps(self):
        """
        Retrieves and stores the number of steps (datasets) for each model stage 
        based on available nodal or element results.

        Returns:
            dict: A dictionary mapping model stages to their respective number of steps.
        """
        # Dictionary to store steps for each model stage
        steps_info = {}

        # Get all model stages
        model_stages = self.get_model_stages()

        # Iterate through each model stage
        for stage in model_stages:
            try:
                # Try nodal results first
                nodal_results = self.get_node_results_names(stage)
                if nodal_results:
                    # Use the first nodal result to get the step count
                    results_name = nodal_results[0]
                    node_partition = self.nodes_info['dataframe']['file_id'][0]  # Get the partition
                    base_path = f"{stage}/RESULTS/ON_NODES/{results_name}/DATA"

                    with h5py.File(self.results_partitions[int(node_partition)], 'r') as results:
                        data_group = results.get(base_path)
                        if data_group is None:
                            raise ValueError(f"DATA group not found in path '{base_path}'.")
                        
                        # Number of datasets directly represents the number of steps
                        steps_info[stage] = len(data_group)
                    continue

                # If no nodal results, try element results
                element_results = self.get_elements_results_names(stage)
                if element_results:
                    results_name = element_results[0]
                    element_types = self.get_element_types(stage, results_name)[results_name]
                    if element_types:
                        element_partition = 0  # Adjust if partition handling is needed
                        base_path = f"{stage}/RESULTS/ON_ELEMENTS/{results_name}/{element_types[0]}/DATA"

                        with h5py.File(self.results_partitions[element_partition], 'r') as results:
                            data_group = results.get(base_path)
                            if data_group is None:
                                raise ValueError(f"DATA group not found in path '{base_path}'.")

                            # Number of datasets directly represents the number of steps
                            steps_info[stage] = len(data_group)

            except Exception as e:
                print(f"Error processing model stage '{stage}': {e}")
                steps_info[stage] = None

        return steps_info
    
    def _get_time_series_ON_NODES_for_stage(self, model_stage, results_name):
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

        for part_number, partition_path in self.results_partitions.items():
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

    def _get_time_series_ON_ELEMENTS_for_stage(self, model_stage, results_name, element_type):
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

        for part_number, partition_path in self.results_partitions.items():
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
    
    def get_time_series(self):
        """
        Retrieve and consolidate the unique time series data across all model stages,
        storing them in a MultiIndex Pandas DataFrame.

        Returns:
            pd.DataFrame: A MultiIndex DataFrame with index ['MODEL_STAGE', 'STEP'] 
                        and column ['TIME'], sorted by MODEL_STAGE and STEP.
        """

        model_stages = self.model_stages  # Get all model stages
        model_elements_types = self.element_types['element_types_dict']  # Result name -> Element types

        all_time_series = []  # List to store results for all model stages

        for model_stage in model_stages:
            time_series_df = None  # Initialize per-stage DataFrame

            # Check if nodal results exist
            if self.node_results_names is not None:
                # Get the time series from nodes
                time_series_df = self._get_time_series_ON_NODES_for_stage(model_stage, self.node_results_names[0])
            
            elif self.element_results_names is not None:
                # Find the first valid element result
                for result, element_list in model_elements_types.items():
                    if result is not None and element_list:
                        element_type = element_list[0]  # Use the first element type
                        time_series_df = self._get_time_series_ON_ELEMENTS_for_stage(model_stage, result, element_type)
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
            

                    

        

    
    
