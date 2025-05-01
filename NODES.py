import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



class NODES:
    """ 
    This is a mixin class to be used with the MCPO_VirtualDataset to handle the nodes of the dataset. 
    """
    
    def _get_all_nodes_ids(self, print_memory=False):
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

        for part_number, partition_path in self.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                nodes_group = partition.get(self.MODEL_NODES_PATH.format(model_stage=self.get_model_stages()[0]))
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

        if print_memory:
            array_memory = results_array.nbytes
            df_memory = df.memory_usage(deep=True).sum()
            print(f"Memory usage for structured array (NODES): {array_memory / 1024**2:.2f} MB")
            print(f"Memory usage for DataFrame (NODES): {df_memory / 1024**2:.2f} MB")

        return results_dict

    
    def get_nodal_results(self, model_stage=None, results_name=None, node_ids=None, selection_set_id=None):
        """
        Get nodal results optimized for numerical operations.
        Returns results as a structured NumPy array or DataFrame for efficient computation.
        
        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve.
            node_ids (int, list, or np.ndarray, optional): Single node ID, a list, or a NumPy array of node IDs.
            selection_set_id (int, optional): The ID of the selection set to retrieve nodes from.

        Returns:
            pd.DataFrame: A DataFrame with MultiIndex (stage, node_id, step) if model_stage is None,
                        or Index (node_id, step) if model_stage is specified.
                        Columns represent the components of the results.
        """
        # Input validation
        node_ids = self._validate_and_prepare_inputs(
            model_stage, results_name, node_ids, selection_set_id
        )

        # If no specific model stage is given, process all stages
        if model_stage is None:
            all_results = []
            for stage in self.model_stages:
                try:
                    stage_results = self._get_stage_results(
                        stage, results_name, node_ids
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
        return self._get_stage_results(model_stage, results_name, node_ids)

    def _get_stage_results(self, model_stage, results_name, node_ids):
        """Helper function to get results for a specific model stage."""
        # Get node files and indices information
        nodes_info = self.get_node_files_and_indices(node_ids=node_ids)
        
        # Group nodes by file_id for batch processing
        file_groups = nodes_info.groupby('file_id')
        
        # Base path for results
        base_path = f"{model_stage}/RESULTS/ON_NODES/{results_name}/DATA"
        
        # List to store all results before combining
        all_results = []
        
        # Process each file only once, reading multiple nodes
        for file_id, group in file_groups:
            with h5py.File(self.results_partitions[int(file_id)], 'r') as results:
                data_group = results.get(base_path)
                if data_group is None:
                    raise ValueError(f"DATA group not found in path '{base_path}'.")
                
                # Get all step names once
                step_names = list(data_group.keys())
                
                # Pre-fetch all node indices for this file
                node_indices = group['index'].values
                file_node_ids = group['node_id'].values
                
                # Process all steps
                for step_idx, step_name in enumerate(step_names):
                    dataset = data_group[step_name]
                    # Read all required indices at once
                    step_data = dataset[node_indices]
                    
                    # Create DataFrame for this step
                    step_df = pd.DataFrame(
                        step_data,
                        index=file_node_ids,
                        columns=[i + 1 for i in range(step_data.shape[1])]
                    )
                    step_df['step'] = step_idx
                    #step_df['step_name'] = step_name
                    step_df['node_id'] = file_node_ids
                    
                    all_results.append(step_df)
        
        if not all_results:
            raise ValueError(f"No results found for stage {model_stage}")
        
        # Combine all results into a single DataFrame
        combined_results = pd.concat(all_results, axis=0)
        
        # Set up MultiIndex
        combined_results.set_index(['node_id', 'step'], inplace=True)
        combined_results.sort_index(inplace=True)
        
        return combined_results

    def _validate_and_prepare_inputs(self, model_stage, results_name, node_ids, selection_set_id):
        """Helper function to validate inputs and prepare node_ids."""
        # Input validation
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Only one of 'node_ids' or 'selection_set_id' can be provided.")
        if node_ids is None and selection_set_id is None:
            raise ValueError("Either 'node_ids' or 'selection_set_id' must be provided.")
        
        # Results name validation
        if results_name not in self.node_results_names:
            raise ValueError(f"Results name '{results_name}' not found in the dataset. Available names: {self.node_results_names}")
        
        # Model stage validation (only if specified)
        if model_stage is not None and model_stage not in self.model_stages:
            raise ValueError(f"Model stage '{model_stage}' not found in the dataset. Available stages: {self.model_stages}")
        
        # Handle selection set
        if selection_set_id is not None:
            selection_set = self.selection_set[selection_set_id]
            if not selection_set or "NODES" not in selection_set:
                raise ValueError(f"Selection set ID '{selection_set_id}' does not contain nodes. Valid ids are: {list(self.selection_set.keys())}")
            return np.array(selection_set["NODES"])
        
        # Handle node_ids
        if isinstance(node_ids, int):
            return np.array([node_ids])
        elif isinstance(node_ids, list):
            return np.array(node_ids)
        elif isinstance(node_ids, np.ndarray) and node_ids.size > 0:
            return node_ids
        else:
            raise ValueError("node_ids must be a non-empty NumPy array, list, or a single integer.")
                
                



    


    