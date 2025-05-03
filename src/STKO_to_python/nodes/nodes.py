
from typing import TYPE_CHECKING
import h5py
import numpy as np
import pandas as pd


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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    