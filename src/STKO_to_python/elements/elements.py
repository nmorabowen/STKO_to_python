
from typing import TYPE_CHECKING
import h5py
import numpy as np
import pandas as pd 

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class Elements:
    
    def __init__(self, dataset:'MPCODataSet'):
        self.dataset = dataset
        
    def _get_all_element_index(self, element_type=None, verbose=False):
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
        model_stages = self.dataset.model_stages

        dtype = [
                ('element_id', 'i8'),
                ('file_id', 'i8'),
                ('index', 'i8'),
                ('type', 'U50')
            ]

        # Get all element types if none specified
        if element_type is None:
            element_types = self.dataset.element_types
        else:
            element_types = [element_type]

        # List to store all elements' information
        elements_info = []

        # Loop through partition files
        for part_number, partition_path in self.dataset.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                # Loop through each element type
                for etype in element_types:
                    element_group = partition.get(self.dataset.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=etype))

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
            if verbose:
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    