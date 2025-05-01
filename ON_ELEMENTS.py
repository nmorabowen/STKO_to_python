import h5py
import numpy as np
import matplotlib.pyplot as plt
import re
import pandas as pd


class ON_ELEMENTS:
    """This is a mixin class to work with MPCO_VirtualDataset"""
    
    def _get_all_element_index(self, element_type=None, print_memory=False):
        """
        Fetch information for all elements of a given type in the partition files.
        If no element type is provided, fetch information for all element types.

        Args:
            element_type (str or None): The type of elements to fetch (e.g., 'ElasticBeam3d'). 
                                        If None, fetches all element types.
            print_memory (bool, optional): If True, prints memory usage for the structured array
                                        and DataFrame. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - 'array': Structured NumPy array with element data.
                - 'dataframe': Pandas DataFrame with element data.
        """
        model_stages = self.get_model_stages()

        # Get all element types if none specified
        if element_type is None:
            element_types = self._get_all_types(model_stage=model_stages[0])
        else:
            element_types = [element_type]

        # List to store all elements' information
        elements_info = []

        # Loop through partition files
        for part_number, partition_path in self.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                # Loop through each element type
                for etype in element_types:
                    element_group = partition.get(self.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=etype))

                    if element_group is None:
                        print(f"Warning: Element type '{etype}' not found in partition {partition_path}.")
                        continue

                    # Loop through each element in the group
                    for element in element_group.keys():
                        element_name = element.split('[')[0]
                        file_name = part_number

                        if element_name == etype:
                            dataset = element_group[element]
                            data = dataset[:]  # Load dataset

                            # Collect info for each element
                            for idx, element_data in enumerate(data):
                                elements_info.append({
                                    'element_id': element_data[0],
                                    'element_idx': idx,
                                    'node_list': element_data[1:].tolist(),  # Node connectivity data
                                    'file_name': file_name,
                                    'element_type': etype
                                })

        # Prepare structured array and DataFrame
        if elements_info:
            dtype = [
                ('element_id', int),
                ('element_idx', int),
                ('file_name', int),
                ('element_type', object),
                ('node_list', object)  # Include node_list as an object field
            ]
            structured_data = [
                (elem['element_id'], elem['element_idx'], elem['file_name'], elem['element_type'], elem['node_list'])
                for elem in elements_info
            ]
            results_array = np.array(structured_data, dtype=dtype)

            # Convert to a Pandas DataFrame
            df = pd.DataFrame.from_records(elements_info)

            # Optionally print memory usage
            if print_memory:
                array_memory = results_array.nbytes
                df_memory = df.memory_usage(deep=True).sum()
                print(f"Memory usage for structured array (ELEMENTS): {array_memory / 1024**2:.2f} MB")
                print(f"Memory usage for DataFrame (ELEMENTS): {df_memory / 1024**2:.2f} MB")

            return {
                'array': results_array,
                'dataframe': df
            }
        else:
            print("No elements found.")
            return {
                'array': np.array([], dtype=dtype),
                'dataframe': pd.DataFrame()
            }
        
    
    def get_element_results_for_type_and_id_bak(self, result_name, element_type, element_id, model_stage):
        
        # Check for errors in result name and element type, and model stage
        self._model_stages_error(model_stage)
        self._element_results_name_error(result_name, model_stage)
        self._element_type_name_error(element_type, model_stage)
        
        # Get the element index and info, and check if the element exists
        element_info = self._get_element_index(element_type, element_id, model_stage)
        if element_info['element_id'] is None:
            raise ValueError(f"Element ID {element_id} not found in the virtual dataset.")

        # element_info is a dictionary with the element_id, element_idx, node_coordinates_list, and file_name
        
        results_data=[]
        
        # Fetch the elements results for the element index
        with h5py.File(self.virtual_data_set, 'r') as results:
            
            element_results = results.get(self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage) + f"/{result_name}")
            
            if element_results is None:
                raise ValueError("Element results group not found in the virtual dataset.")
            
            # We have a list of element results, we need to get the results for the element type, for this we need to compare the propper file name without the info after '_'
            
            for ele_type in element_results.keys():
                
                ele_type_name=ele_type.split('[')[0]
                
                if ele_type_name == element_type:
                    
                    
                    # Navigate to the "DATA" folder
                    data_group_path = f"{self.RESULTS_ON_ELEMENTS_PATH.format(model_stage=model_stage)}/{result_name}/{ele_type}/DATA"
                    data_group = results.get(data_group_path)
                    
                    if data_group is None:
                        raise ValueError(f"The DATA group does not exist under the path'.")
                    
                    ####################################
                    # VER POR ACA
                    ####################################
                    
                    all_steps = list(data_group.keys())
                    # Filter steps ending with the specific file_id
                    relevant_steps = np.array([step for step in all_steps if step.endswith(element_info['file_name'])])
                    
                    if len(relevant_steps) == 0:
                        raise ValueError(f"No data found for Element ID {element_id} in file.")
                    
                    # Extract step numbers and sort using np.argsort
                    step_numbers = np.array([int(step.split("_")[1]) for step in relevant_steps])
                    sorted_indices = np.argsort(step_numbers)
                    sorted_steps = relevant_steps[sorted_indices]
                    num_steps = len(sorted_steps)
                    
                    # Determine the number of components from the first dataset
                    first_dataset = data_group[sorted_steps[0]]
                    num_components = first_dataset.shape[1] if len(first_dataset.shape) > 1 else 1
                    
                    # Preallocate NumPy array
                    results_data = np.zeros((num_steps, 1 + num_components))  # 1 column for step + components
                    
                    # Extract data for the node index
                    for i, step_name in enumerate(sorted_steps):
                        step_num = int(step_name.split("_")[1])
                        result_data = data_group[step_name][element_info['element_idx']]
                        results_data[i, 0] = step_num  # First column: step number
                        results_data[i, 1:] = result_data  # Remaining columns: result components
                        
                        
        return results_data
    
    def get_element_nodes(self, element_id):
        """
        Retrieve the node IDs for a specific element.

        Args:
            element_id (int): Element ID to query.

        Returns:
            list: A list of node IDs associated with the given element.
        """
        with h5py.File(self.virtual_data_set, 'r') as h5file:
            # Locate the element in the mapping
            element_mapping = h5file['/Mappings/Elements'][:]
            element_index = np.where(element_mapping['element_id'] == element_id)[0]
            if element_index.size == 0:
                raise ValueError(f"Element ID {element_id} not found.")
            
            element_file = element_mapping[element_index[0]]['file_name'].decode()

            # Fetch the connectivity information from the dataset
            element_group_path = f"/MODEL/ELEMENTS/{element_file}/CONNECTIVITY"
            connectivity_group = h5file.get(element_group_path)
            if connectivity_group is None:
                raise ValueError(f"Connectivity information not found for file {element_file}.")

            # Get the node connectivity for the specific element
            connectivity_data = connectivity_group[element_id]
            return connectivity_data.tolist()
        
    def get_element_results(self, model_stage=None, results_name=None, element_ids=None, selection_set_id=None):
        """
        Get element results optimized for numerical operations.
        Returns results as a structured NumPy array or DataFrame for efficient computation.

        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve.
            element_ids (int, list, or np.ndarray, optional): Single element ID, a list, or a NumPy array of element IDs.
            selection_set_id (int, optional): The ID of the selection set to retrieve elements from.

        Returns:
            pd.DataFrame: A DataFrame with MultiIndex (stage, element_id, step) if model_stage is None,
                        or Index (element_id, step) if model_stage is specified.
                        Columns represent the components of the results.
        """
        # Input validation
        element_ids = self._validate_and_prepare_element_inputs(
            model_stage, results_name, element_ids, selection_set_id, target='element'
        )

        # If no specific model stage is given, process all stages
        if model_stage is None:
            all_results = []
            for stage in self.model_stages:
                try:
                    stage_results = self._get_stage_element_results(
                        stage, results_name, element_ids
                    )
                    stage_results['stage'] = stage  # Add stage information
                    all_results.append(stage_results)
                except Exception as e:
                    print(f"Warning: Could not retrieve results for stage {stage}: {str(e)}")
                    continue

            if not all_results:
                raise ValueError("No results found for any stage")

            # Combine all stages into a single DataFrame
            return pd.concat(all_results, axis=0)

        # If specific model stage is given, process just that stage
        return self._get_stage_element_results(model_stage, results_name, element_ids)

    def _get_stage_element_results(self, element_type, model_stage, results_name, element_ids):
        """
        Helper function to get results for a specific model stage.
        """
        # Check if the element type is valid for the model stage
        self._element_type_name_error(element_type)
        
        # Get element files and indices information
        elements_info = self.get_element_files_and_indices(element_ids=element_ids)

        # Group elements by file_id for batch processing
        file_groups = elements_info.groupby('file_id')

        # Base path for results
        base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}/DATA"

        # List to store all results before combining
        all_results = []

        # Process each file only once, reading multiple elements
        for file_id, group in file_groups:
            with h5py.File(self.results_partitions[int(file_id)], 'r') as results:
                data_group = results.get(base_path)
                if data_group is None:
                    raise ValueError(f"DATA group not found in path '{base_path}'.")

                # Get all step names once
                step_names = list(data_group.keys())

                # Pre-fetch all element indices for this file
                element_indices = group['index'].values
                file_element_ids = group['element_id'].values

                # Process all steps
                for step_idx, step_name in enumerate(step_names):
                    dataset = data_group[step_name]
                    # Read all required indices at once
                    step_data = dataset[element_indices]

                    # Create DataFrame for this step
                    step_df = pd.DataFrame(
                        step_data,
                        index=file_element_ids,
                        columns=[f'component_{i}' for i in range(step_data.shape[1])]
                    )
                    step_df['step'] = step_idx
                    step_df['step_name'] = step_name
                    step_df['element_id'] = file_element_ids

                    all_results.append(step_df)

        if not all_results:
            raise ValueError(f"No results found for stage {model_stage}")

        # Combine all results into a single DataFrame
        combined_results = pd.concat(all_results, axis=0)

        # Set up MultiIndex
        combined_results.set_index(['element_id', 'step'], inplace=True)
        combined_results.sort_index(inplace=True)

        return combined_results

    

                        

                    
                    
                    


                
                    

                            


                    

    
    