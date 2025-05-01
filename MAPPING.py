import h5py
import numpy as np
import pandas as pd

class MAPPING:
    def build_and_store_mappings(self):
        """
        Build and store node and element mappings for efficient lookups.
        Includes node lists for each element with -1 padding for consistency.
        """
        
        # Get the nodes array info
        nodes_array=self.nodes_info['array']
        # Fetch and map elements for this partition
        element_array = self.elements_info['dataframe']
        
        with h5py.File(self.virtual_data_set, 'r+') as h5file:

            # Save to HDF5 with compression and chunking
            mapping_group = h5file.require_group("/Mappings")

            # Save datasets
            mapping_group.create_dataset("Nodes", data=nodes_array)
            #mapping_group.create_dataset("Elements", data=element_array)
            node_lists = element_array.pop('node_list')  # remove node_list, but keep index alignment
            element_array.to_hdf(self.virtual_data_set, key="Elements", mode="a", format="table", data_columns=True)  

    def get_node_files_and_indices(self, node_ids):
        """
        Retrieve node file associations and indices for a list of node IDs.

        This method filters the `nodes_info` DataFrame to return only the rows
        corresponding to the specified node IDs. Each row includes details about
        the nodes, such as their indices and associated file names.

        Parameters
        ----------
        node_ids : list of int
            A list of node IDs for which the file and index information is requested.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing information about the specified nodes, including
            the `node_id`, associated file names, and indices. The structure depends
            on the columns available in `self.nodes_info['dataframe']`.

        Raises
        ------
        ValueError
            If `node_ids` is not a list or any of its elements are not integers.

        Examples
        --------
        Suppose `self.nodes_info['dataframe']` contains the following data:
        
        >>> nodes_info = pd.DataFrame({
        ...     'node_id': [1, 2, 3, 4],
        ...     'file': ['file1', 'file2', 'file1', 'file3'],
        ...     'index': [0, 1, 2, 3]
                'x': [0.0]
                'y': [0.0]
                'z': [0.0]
        ... })

        Calling the method with a list of node IDs:
        
        >>> get_node_files_and_indices([1, 3])
        Returns:
            node_id   file   index x y z
        0        1  file1       0
        1        3  file1       2

        Notes
        -----
        - This method relies on `self.nodes_info['dataframe']` being a valid pandas
        DataFrame with at least the column `node_id`.
        - Ensure `self.nodes_info` is correctly populated before calling this method.

        """
        
        if isinstance(node_ids, (int, float)):
            node_ids = [node_ids]
        
        if not isinstance(node_ids, (list, np.ndarray)):
            raise ValueError("node_ids should be a list of integers")

        nodes_info = self.nodes_info['dataframe']
        filter_nodes_info = nodes_info[nodes_info['node_id'].isin(node_ids)]

        return filter_nodes_info


    def get_element_files_and_indices(self, element_ids):
        """
        Retrieve element file associations and indices for a list of element IDs.

        This method filters the `elements_info` DataFrame to return only the rows
        corresponding to the specified element IDs. Each row includes details about
        the elements, such as their indices and associated file names.

        Parameters
        ----------
        element_ids : list of int
            A list of element IDs for which the file and index information is requested.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing information about the specified elements, including
            the `element_id`, associated file names, and indices. The structure depends
            on the columns available in `self.elements_info['dataframe']`.

        Raises
        ------
        ValueError
            If `element_ids` is not a list of integers.

        Examples
        --------
        Suppose `self.elements_info['dataframe']` contains the following data:
        
        >>> elements_info = pd.DataFrame({
        ...     'element_id': [101, 102, 103, 104],
        ...     'element_idx': ['file1', 'file2', 'file3', 'file4'],
        ...     'file_name': [0, 1, 2, 3]
                'element_type': ['beam', 'beam', 'beam', 'beam']
        ... })

        Calling the method with a list of element IDs:
        
        >>> get_element_files_and_indices([101, 103])
        Returns:
            element_id   element_idx   file_name element_type

        Notes
        -----
        - This method relies on `self.elements_info['dataframe']` being a valid pandas
        DataFrame with at least the column `element_id`.
        - Ensure `self.elements_info` is correctly populated before calling this method.

        """
        if element_ids is int:
            element_ids=[element_ids]
        
        if not isinstance(element_ids, (list, np.ndarray)):
            raise ValueError("element_ids should be a list or numpy array of integers")

        elements_info = self.elements_info['dataframe']
        filter_elements_info = elements_info[elements_info['element_id'].isin(element_ids)]

        return filter_elements_info
    
    def get_selection_set_nodes_elements(self, selection_set_ids):
        """
        Retrieve nodes and elements information for one or more selection sets.

        This method extracts the node and element IDs from the specified selection sets
        and retrieves their associated file and index information.

        Parameters
        ----------
        selection_set_ids : int or list of int
            A single selection set ID or a list of selection set IDs.

        Returns
        -------
        dict
            A dictionary containing:
                - 'NODES': A pandas DataFrame with node information for all selection sets.
                - 'ELEMENTS': A pandas DataFrame with element information for all selection sets.

        Raises
        ------
        ValueError
            If `selection_set_ids` is not an integer or a list of integers.

        Examples
        --------
        Suppose the selection set contains the following data:
        
        >>> self.extract_selection_set_ids(selection_set_ids=[1, 2])
        Returns:
            {
                1: {"NODES": [101, 102], "ELEMENTS": [201, 202]},
                2: {"NODES": [103, 104], "ELEMENTS": [203, 204]},
            }

        Calling the method with multiple selection sets:
        
        >>> get_selection_set_nodes_elements([1, 2])
        Returns:
            {
                'NODES': DataFrame with combined node information,
                'ELEMENTS': DataFrame with combined element information
            }
        """
        # Ensure selection_set_ids is a list
        if isinstance(selection_set_ids, int):
            selection_set_ids = [selection_set_ids]
        elif not isinstance(selection_set_ids, list) or not all(isinstance(id, int) for id in selection_set_ids):
            raise ValueError("selection_set_ids must be an integer or a list of integers.")

        # Initialize sets to collect unique nodes and elements
        all_nodes = set()
        all_elements = set()

        # Extract nodes and elements for each selection set
        for selection_set_id in selection_set_ids:
            selection_set_dict = self.extract_selection_set_ids(selection_set_ids=[selection_set_id])
            if selection_set_id not in selection_set_dict:
                raise ValueError(f"Selection set ID {selection_set_id} not found.")
            all_nodes.update(selection_set_dict[selection_set_id].get('NODES', []))
            all_elements.update(selection_set_dict[selection_set_id].get('ELEMENTS', []))

        # Retrieve node and element information
        nodes_info = self.get_node_files_and_indices(node_ids=list(all_nodes))
        elements_info = self.get_element_files_and_indices(element_ids=list(all_elements))

        # Combine results into a dictionary
        result_dict = {
            'NODES': nodes_info,
            'ELEMENTS': elements_info
        }

        return result_dict
