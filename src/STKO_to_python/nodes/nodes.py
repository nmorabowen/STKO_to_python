from typing import TYPE_CHECKING
import pandas as pd
import numpy as np
import h5py
from typing import Union, Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import time
import gc

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

# Set up logging instead of using print statements
logging.basicConfig(
    filename='log.log',  # <- this writes to a file
    filemode='w',               # 'w' to overwrite, 'a' to append
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Decorator for performance monitoring
def profile_execution(func):
    """Decorator to measure execution time and log it."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result
    return wrapper

class Nodes:
    # Maximum memory budget in MB (configurable)
    # This should be a fraction of your total available RAM to leave room for the operating system and other processes.
    MAX_MEMORY_BUDGET_MB = 2048
    # Chunk size for processing large node sets
    # Here's a guideline for setting the chunk size:
    # For simple results (e.g., displacements with 3 components): 10,000-25,000 nodes per chunk
    # For complex results (e.g., stress tensors with 6+ components): 5,000-10,000 nodes per chunk
    # For multi-step analyses: Divide the above numbers by the average number of steps
    DEFAULT_CHUNK_SIZE = 5000
    
    def __init__(self, dataset: 'MPCODataSet'):
        self.dataset = dataset
        # Cache for node information to avoid redundant lookups
        self._node_info_cache = {}
        
    def _estimate_node_count(self) -> int:
        """Estimate the total number of nodes without loading all data."""
        total_nodes = 0
        model_stage = self.dataset.model_stages[0]
        
        for part_number, partition_path in self.dataset.results_partitions.items():
            try:
                with h5py.File(partition_path, 'r') as partition:
                    nodes_group = partition.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                    if nodes_group is None:
                        continue
                    
                    for key in nodes_group.keys():
                        if key.startswith("ID"):
                            node_count = len(nodes_group[key])
                            total_nodes += node_count
                            # No need to load actual data, just get the count
                            break
            except Exception as e:
                logger.warning(f"Error estimating nodes in partition {part_number}: {str(e)}")
                
        return total_nodes

    @profile_execution
    def _get_all_nodes_ids(self, verbose=False, max_workers=4) -> Dict[str, Any]:
        """
        Retrieve all node IDs, file names, indices, and coordinates from the partition files.
        
        Optimized to use pre-allocation, vectorized operations, and parallel processing.

        Args:
            verbose (bool): If True, prints the memory usage of the structured array and DataFrame.
            max_workers (int): Maximum number of worker threads for parallel processing.

        Returns:
            dict: A dictionary containing:
                - 'array': A structured NumPy array with all node IDs, file names, indices, and coordinates.
                - 'dataframe': A pandas DataFrame with the same data.
        """
        # Estimate total node count first to pre-allocate arrays
        estimated_node_count = self._estimate_node_count()
        if estimated_node_count == 0:
            logger.warning("No nodes found in any partition")
            return {'array': np.array([], dtype=self._get_node_dtype()), 'dataframe': pd.DataFrame()}
        
        # Check if we need chunked processing based on memory budget
        estimated_memory_mb = estimated_node_count * 48 / 1024 / 1024  # Rough estimate: 6 fields * 8 bytes each
        chunked_processing = estimated_memory_mb > self.MAX_MEMORY_BUDGET_MB
        
        if verbose and chunked_processing:
            logger.info(f"Estimated memory usage ({estimated_memory_mb:.2f} MB) exceeds budget. Using chunked processing.")
        
        # Define dtype for structured array
        dtype = self._get_node_dtype()
        
        # Function to process a single partition in parallel
        def process_partition(partition_info):
            part_number, partition_path = partition_info
            model_stage = self.dataset.model_stages[0]
            partition_data = []
            
            try:
                with h5py.File(partition_path, 'r') as partition:
                    nodes_group = partition.get(self.dataset.MODEL_NODES_PATH.format(model_stage=model_stage))
                    if nodes_group is None:
                        return []
                    
                    for key in nodes_group.keys():
                        if key.startswith("ID"):
                            file_id = part_number
                            node_ids = nodes_group[key][...]  # Use [...] for immediate loading
                            coord_key = key.replace("ID", "COORDINATES")
                            
                            if coord_key in nodes_group:
                                coords = nodes_group[coord_key][...]
                                
                                # Vectorized operation to create structured data
                                indices = np.arange(len(node_ids))
                                file_ids = np.full_like(node_ids, file_id)
                                
                                # Create structured array directly
                                part_data = np.zeros(len(node_ids), dtype=dtype)
                                part_data['node_id'] = node_ids
                                part_data['file_id'] = file_ids
                                part_data['index'] = indices
                                part_data['x'] = coords[:, 0]
                                part_data['y'] = coords[:, 1]
                                part_data['z'] = coords[:, 2]
                                
                                return part_data
            except Exception as e:
                logger.warning(f"Error processing partition {part_number}: {str(e)}")
            
            return []
        
        # Process partitions in parallel
        all_data = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_partition, self.dataset.results_partitions.items()))
            all_data = [r for r in results if len(r) > 0]
        
        # Combine arrays
        if not all_data:
            return {'array': np.array([], dtype=dtype), 'dataframe': pd.DataFrame()}
        
        results_array = np.concatenate(all_data)
        
        # Convert to DataFrame efficiently using the structured array
        df = pd.DataFrame({
            'node_id': results_array['node_id'],
            'file_id': results_array['file_id'],
            'index': results_array['index'],
            'x': results_array['x'],
            'y': results_array['y'],
            'z': results_array['z']
        })
        
        results_dict = {
            'array': results_array,
            'dataframe': df
        }
        
        if verbose:
            array_memory = results_array.nbytes
            df_memory = df.memory_usage(deep=True).sum()
            logger.info(f"Memory usage for structured array (NODES): {array_memory / 1024**2:.2f} MB")
            logger.info(f"Memory usage for DataFrame (NODES): {df_memory / 1024**2:.2f} MB")
        
        # Store in cache for future use
        self._node_info_cache['all_nodes'] = results_dict
        
        return results_dict
    
    def _get_node_dtype(self):
        """Return the NumPy dtype for node data."""
        return [
            ('node_id', 'i8'),
            ('file_id', 'i8'),
            ('index', 'i8'),
            ('x', 'f8'),
            ('y', 'f8'),
            ('z', 'f8')
        ]
    
    @lru_cache(maxsize=32)
    def get_node_files_and_indices(self, node_ids=None):
        """
        Get file IDs and indices for specified node IDs.
        Cached for repeated calls with the same node IDs.
        
        Args:
            node_ids: Node IDs to find file and index information for
            
        Returns:
            pd.DataFrame: DataFrame with node_id, file_id, and index columns
        """
        # Convert node_ids to hashable tuple for caching
        if isinstance(node_ids, np.ndarray):
            node_ids = tuple(node_ids.tolist())
        elif isinstance(node_ids, list):
            node_ids = tuple(node_ids)
            
        # Get all nodes if not already in cache
        if 'all_nodes' not in self._node_info_cache:
            self._get_all_nodes_ids()
        
        all_nodes_df = self._node_info_cache['all_nodes']['dataframe']
        
        # Filter nodes efficiently
        if node_ids is None:
            return all_nodes_df[['node_id', 'file_id', 'index']]
            
        # Use numpy for faster filtering
        node_id_array = np.array(node_ids)
        mask = np.isin(all_nodes_df['node_id'].to_numpy(), node_id_array)
        filtered_df = all_nodes_df.loc[mask, ['node_id', 'file_id', 'index']]
        
        if len(filtered_df) == 0:
            raise ValueError(f"None of the provided node IDs were found in the dataset")
            
        if len(filtered_df) < len(node_id_array):
            missing_nodes = set(node_id_array) - set(filtered_df['node_id'])
            logger.warning(f"Some node IDs were not found: {missing_nodes}")
            
        return filtered_df
    
    def _validate_and_prepare_inputs(self, model_stage, results_name, node_ids, selection_set_id):
        """
        Validate inputs and return a NumPy array of node IDs.
        Optimized with improved validation flow.

        Raises:
            ValueError: On invalid combinations or missing/unknown inputs.
        Returns:
            np.ndarray: Array of node IDs.
        """
        # --- Check required parameters ---
        if results_name is None:
            raise ValueError("results_name is a required parameter")
        
        # --- Quick path for mutually exclusive inputs ---
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Provide only one of 'node_ids' or 'selection_set_id', not both.")
        
        if node_ids is None and selection_set_id is None:
            raise ValueError("You must specify either 'node_ids' or 'selection_set_id'.")
        
        # --- Check available results ---
        if not hasattr(self.dataset, 'node_results_names'):
            # Dynamically get available results if not already cached
            self._cache_available_results()
            
        if results_name not in self.dataset.node_results_names:
            raise ValueError(
                f"Result name '{results_name}' not found. Available options: {self.dataset.node_results_names}"
            )
        
        # --- Validate model_stage if provided ---
        if model_stage is not None and model_stage not in self.dataset.model_stages:
            raise ValueError(
                f"Model stage '{model_stage}' not found. Available stages: {self.dataset.model_stages}"
            )
        
        # --- Resolve selection_set efficiently ---
        if selection_set_id is not None:
            if selection_set_id not in self.dataset.selection_set:
                raise ValueError(f"Selection set ID '{selection_set_id}' not found.")
            selection = self.dataset.selection_set[selection_set_id]
            if "NODES" not in selection or not selection["NODES"]:
                raise ValueError(f"Selection set {selection_set_id} does not contain nodes.")
            return np.asarray(selection["NODES"], dtype=np.int64)
        
        # --- Resolve node_ids with efficient type checking ---
        if isinstance(node_ids, (int, np.integer)):
            return np.array([node_ids], dtype=np.int64)
        
        # Handle list and ndarray efficiently with asarray instead of multiple conversions
        try:
            result = np.asarray(node_ids, dtype=np.int64)
            if result.size == 0:
                raise ValueError("'node_ids' is empty.")
            return result
        except (TypeError, ValueError):
            raise ValueError("Invalid 'node_ids' format. Must be int, non-empty list, or NumPy array.")
    
    def _cache_available_results(self):
        """Dynamically discover available result types and cache them."""
        self.dataset.node_results_names = set()
        model_stage = self.dataset.model_stages[0]
        
        for partition_path in self.dataset.results_partitions.values():
            try:
                with h5py.File(partition_path, 'r') as h5file:
                    results_path = f"{model_stage}/RESULTS/ON_NODES"
                    if results_path in h5file:
                        self.dataset.node_results_names.update(h5file[results_path].keys())
            except Exception as e:
                logger.warning(f"Error caching results: {str(e)}")
    
    @profile_execution
    def _get_stage_results(
        self,
        model_stage: str,
        results_name: str,
        node_ids: Union[np.ndarray, list, int],
        chunk_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve nodal results for a given model stage and result type.
        Optimized with chunked processing and parallel file access.

        Args:
            model_stage (str): Name of the model stage.
            results_name (str): Type of result to retrieve (e.g., 'Displacement').
            node_ids (np.ndarray | list | int): Node IDs to retrieve.
            chunk_size (int, optional): Size of chunks for processing large node sets.

        Returns:
            pd.DataFrame: DataFrame indexed by (node_id, step) with result components as columns.
        """
        if chunk_size is None:
            chunk_size = self.DEFAULT_CHUNK_SIZE
            
        # Check if node set is large enough to warrant chunked processing
        node_ids_array = np.asarray(node_ids, dtype=np.int64)
        if len(node_ids_array) > chunk_size:
            return self._get_chunked_stage_results(model_stage, results_name, node_ids_array, chunk_size)
            
        # Resolve node indices and file mapping
        nodes_info = self.get_node_files_and_indices(node_ids=tuple(node_ids_array.tolist()))
        base_path = f"{model_stage}/RESULTS/ON_NODES/{results_name}/DATA"
        
        # Group all entries by file_id to minimize file access
        file_groups = {
            file_id: group for file_id, group in nodes_info.groupby('file_id')
        }
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=min(len(file_groups), 4)) as executor:
            future_to_file = {
                executor.submit(
                    self._process_file_results, 
                    file_id, 
                    group, 
                    base_path
                ): file_id for file_id, group in file_groups.items()
            }
            
            # Collect results as they complete
            all_results = []
            for future in future_to_file:
                try:
                    result = future.result()
                    if result is not None:
                        all_results.extend(result)
                except Exception as e:
                    logger.error(f"Error processing file: {str(e)}")
        
        if not all_results:
            raise ValueError(f"No results found for model stage '{model_stage}'.")
        
        # Create the final DataFrame efficiently
        combined_df = pd.concat(all_results, axis=0, copy=False)
        
        # Create the index directly, avoiding multiple operations
        combined_df = combined_df.set_index(['node_id', 'step'])
        
        # Sort the index in a single operation
        return combined_df.sort_index()
    
    def _process_file_results(self, file_id, group, base_path):
        """
        Process results for a single file and all its steps.
        Extracted to a separate method for parallel processing.
        
        Args:
            file_id: File ID to process
            group: DataFrame group containing node indices for this file
            base_path: HDF5 path to the result data
            
        Returns:
            list: List of DataFrames for each step
        """
        file_results = []
        file_path = self.dataset.results_partitions[int(file_id)]
        
        try:
            with h5py.File(file_path, 'r') as results_file:
                data_group = results_file.get(base_path)
                if data_group is None:
                    logger.warning(f"DATA group not found in path '{base_path}'.")
                    return None
                
                step_names = list(data_group.keys())
                
                # Convert to NumPy arrays for faster indexing
                node_indices = group['index'].to_numpy(dtype=np.int64)
                node_id_vals = group['node_id'].to_numpy(dtype=np.int64)
                
                # Check the first step to get component dimensions
                first_dataset = data_group[step_names[0]]
                sample_data = first_dataset[node_indices[0:1]]
                component_count = sample_data.shape[1]
                
                # Create column names once
                columns = [f"val_{i + 1}" for i in range(component_count)]
                
                # Process all steps for this file
                for step_idx, step_name in enumerate(step_names):
                    dataset = data_group[step_name]
                    
                    # Get all data in one operation
                    try:
                        step_data = dataset[node_indices]
                    except Exception as e:
                        logger.warning(f"Error reading step data for step {step_name}: {str(e)}")
                        continue
                    
                    # Verify data shape consistency
                    if step_data.shape[1] != component_count:
                        logger.warning(f"Step {step_name} has inconsistent component count. Expected {component_count}, got {step_data.shape[1]}")
                        continue
                    
                    # Create DataFrame with pre-defined columns
                    step_df = pd.DataFrame(
                        step_data,
                        columns=columns
                    )
                    
                    # Add index columns without modifying the index yet
                    step_df['step'] = step_idx
                    step_df['node_id'] = node_id_vals
                    file_results.append(step_df)
                    
                    # Explicitly delete step_data to manage memory
                    del step_data
                    
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return None
            
        return file_results
    
    def _get_chunked_stage_results(self, model_stage, results_name, node_ids, chunk_size):
        """
        Process large node sets in manageable chunks to control memory usage.
        
        Args:
            model_stage: Model stage to retrieve results for
            results_name: Type of result to retrieve
            node_ids: Array of node IDs
            chunk_size: Number of nodes to process in each chunk
            
        Returns:
            pd.DataFrame: Combined results from all chunks
        """
        node_chunks = [node_ids[i:i+chunk_size] for i in range(0, len(node_ids), chunk_size)]
        logger.info(f"Processing {len(node_ids)} nodes in {len(node_chunks)} chunks of size {chunk_size}")
        
        all_chunk_results = []
        
        for i, chunk in enumerate(node_chunks):
            logger.info(f"Processing chunk {i+1}/{len(node_chunks)} ({len(chunk)} nodes)")
            try:
                # Process each chunk individually
                chunk_df = self._get_stage_results(model_stage, results_name, chunk, None)
                all_chunk_results.append(chunk_df)
                
                # Force garbage collection after each chunk
                gc.collect()
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
        
        if not all_chunk_results:
            raise ValueError(f"No results found for any chunk in model stage '{model_stage}'")
        
        # Combine all chunks
        return pd.concat(all_chunk_results, axis=0)
    
    @profile_execution
    def get_nodal_results(
        self, 
        model_stage=None, 
        results_name=None, 
        node_ids=None, 
        selection_set_id=None,
        chunk_size=None,
        memory_limit_mb=None
    ):
        """
        Get nodal results optimized for numerical operations.
        Returns results as a structured DataFrame for efficient computation.

        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve (e.g., 'Displacement', 'Reaction').
            node_ids (int, list, or np.ndarray, optional): Specific node IDs to filter. Ignored if selection_set_id is used.
            selection_set_id (int, optional): The ID of the selection set to use for filtering node IDs.
            chunk_size (int, optional): Size of chunks for processing large node sets.
            memory_limit_mb (int, optional): Memory limit in MB to control chunking.

        Returns:
            pd.DataFrame: If model_stage is None, returns MultiIndex DataFrame (stage, node_id, step).
                        Otherwise, returns Index (node_id, step). Columns represent result components.
        """
        # Override default memory settings if provided
        if memory_limit_mb is not None:
            self.MAX_MEMORY_BUDGET_MB = memory_limit_mb
        
        if chunk_size is not None:
            self.DEFAULT_CHUNK_SIZE = chunk_size
        
        # --- Validate and determine node_ids ---
        node_ids = self._validate_and_prepare_inputs(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=node_ids,
            selection_set_id=selection_set_id
        )
        
        # Estimate the result size to determine chunking
        estimated_steps = 10  # Conservative estimate
        estimated_components = 6  # Common value for displacements, stresses, etc.
        estimated_memory_mb = (len(node_ids) * estimated_steps * estimated_components * 8) / (1024 * 1024)
        should_use_chunking = estimated_memory_mb > self.MAX_MEMORY_BUDGET_MB
        
        if should_use_chunking:
            logger.info(f"Estimated memory {estimated_memory_mb:.2f} MB exceeds budget. Using chunked processing.")
        
        # Process all stages or a specific stage
        if model_stage is None:
            return self._get_all_stages_results(results_name, node_ids)
        
        # If a specific model stage is requested, delegate to _get_stage_results
        df = self._get_stage_results(
            model_stage, 
            results_name, 
            node_ids,
            self.DEFAULT_CHUNK_SIZE if should_use_chunking else None
        )
        
        return df
    
    def _get_all_stages_results(self, results_name, node_ids):
        """
        Get results for all stages with improved memory management.
        
        Args:
            results_name: Type of result to retrieve
            node_ids: Array of node IDs
            
        Returns:
            pd.DataFrame: Combined results with stage, node_id, step index
        """
        all_results = []
        
        for stage in self.dataset.model_stages:
            try:
                logger.info(f"Processing stage '{stage}'")
                stage_df = self._get_stage_results(stage, results_name, node_ids)
                
                # Add stage column to the data (not the index yet)
                stage_df = stage_df.reset_index()
                stage_df['stage'] = stage
                all_results.append(stage_df)
                
                # Force garbage collection after each stage
                gc.collect()
            except Exception as e:
                logger.warning(f"Could not retrieve results for stage '{stage}': {str(e)}")
        
        if not all_results:
            raise ValueError("No results found for any model stage.")
        
        # Combine all stages and create the hierarchical index in one operation
        combined_df = pd.concat(all_results, axis=0, copy=False)
        combined_df.set_index(['stage', 'node_id', 'step'], inplace=True)
        
        return combined_df.sort_index()
        
    def iter_nodal_results(
        self, 
        model_stage=None, 
        results_name=None, 
        node_ids=None, 
        selection_set_id=None, 
        chunk_size=1000
    ):
        """
        Iterator version of get_nodal_results that yields results in chunks.
        Useful for processing very large datasets without loading everything into memory.
        
        Args:
            model_stage (str, optional): The model stage name. If None, gets results for all stages.
            results_name (str): The name of the result to retrieve.
            node_ids (int, list, or np.ndarray, optional): Specific node IDs to filter.
            selection_set_id (int, optional): The ID of the selection set to use for filtering.
            chunk_size (int): Number of nodes to process in each chunk.
            
        Yields:
            pd.DataFrame: Chunks of results with appropriate index structure.
        """
        # Validate inputs
        node_ids = self._validate_and_prepare_inputs(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=node_ids,
            selection_set_id=selection_set_id
        )
        
        # Create chunks of node IDs
        node_chunks = [node_ids[i:i+chunk_size] for i in range(0, len(node_ids), chunk_size)]
        
        # Process chunks for all stages or a specific stage
        if model_stage is None:
            for stage in self.dataset.model_stages:
                for chunk in node_chunks:
                    try:
                        # Get results for this chunk and stage
                        chunk_df = self._get_stage_results(stage, results_name, chunk)
                        
                        # Add stage column and set appropriate index
                        chunk_df = chunk_df.reset_index()
                        chunk_df['stage'] = stage
                        chunk_df.set_index(['stage', 'node_id', 'step'], inplace=True)
                        
                        yield chunk_df
                    except Exception as e:
                        logger.warning(f"Could not retrieve results for stage '{stage}' chunk: {str(e)}")
        else:
            # Process chunks for a specific stage
            for chunk in node_chunks:
                try:
                    chunk_df = self._get_stage_results(model_stage, results_name, chunk)
                    yield chunk_df
                except Exception as e:
                    logger.warning(f"Could not retrieve results for stage '{model_stage}' chunk: {str(e)}")
    
    def save_to_hdf5(self, output_file, model_stage=None, results_name=None, 
                    node_ids=None, selection_set_id=None, chunk_size=1000):
        """
        Save nodal results directly to an HDF5 file without keeping all data in memory.
        
        Args:
            output_file (str): Path to output HDF5 file
            model_stage (str, optional): The model stage name. If None, saves results for all stages.
            results_name (str): The name of the result to retrieve.
            node_ids (int, list, or np.ndarray, optional): Specific node IDs to filter.
            selection_set_id (int, optional): The ID of the selection set to use for filtering.
            chunk_size (int): Number of nodes to process in each chunk.
            
        Returns:
            str: Path to the output file
        """
        with pd.HDFStore(output_file, mode='w') as store:
            for chunk_df in self.iter_nodal_results(
                model_stage, results_name, node_ids, selection_set_id, chunk_size
            ):
                # For each chunk, append to the store
                store.append('nodal_results', chunk_df, format='table')
        
        logger.info(f"Results saved to {output_file}")
        return output_file
    
    