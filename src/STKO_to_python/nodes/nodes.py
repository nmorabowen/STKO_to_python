from __future__ import annotations  # lets you annotate TimeHistoryResults before it's defined
import warnings
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np
import h5py
from dataclasses import dataclass
from typing import Union, Dict, List, Tuple, Optional, Any, Sequence, Callable, Union
AggregateFn = Callable[[np.ndarray], float]

from concurrent.futures import ThreadPoolExecutor

from .nodal_results_dataclass import NodalResults

import logging
from functools import lru_cache
import time
import gc

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet
    from .nodes_results_dataclass import NodalResults

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
        
        if verbose:
            print(f"Estimated total nodes across all partitions: {estimated_node_count}")
        
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
                                if coords.shape[1] == 3:
                                    part_data['x'] = coords[:, 0]
                                    part_data['y'] = coords[:, 1]
                                    part_data['z'] = coords[:, 2]
                                elif coords.shape[1] == 2:
                                    part_data['x'] = coords[:, 0]
                                    part_data['y'] = coords[:, 1]
                                    part_data['z'] = 0.0  # Pad with zeros for 2D models
                                else:
                                    raise ValueError(f"Unexpected number of coordinate components: {coords.shape[1]}")
                                
                                return part_data
            except Exception as e:
                logger.warning(f"Node Error processing partition {part_number}: {str(e)}")
            
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
    def get_node_files_and_indices(self, node_ids=None) -> pd.DataFrame:
        """
        Return a canonical mapping (one row per node_id) to (file_id, index).

        Policy:
            - If a node appears in multiple partitions, choose the smallest file_id.
            - Results are sorted by node_id and include columns: ['node_id', 'file_id', 'index'].

        Notes:
            - This function is cached. Do NOT mutate the returned DataFrame in place.
            If you need to modify it, call `.copy()` first.
        """
        # Normalize node_ids to a hashable tuple for the cache key
        if isinstance(node_ids, np.ndarray):
            node_ids = tuple(node_ids.tolist())
        elif isinstance(node_ids, list):
            node_ids = tuple(node_ids)

        # Ensure node metadata is cached
        if "all_nodes" not in self._node_info_cache:
            self._get_all_nodes_ids()

        all_nodes_df: pd.DataFrame = self._node_info_cache["all_nodes"]["dataframe"]

        # Base frame (only the columns we need)
        base = all_nodes_df[["node_id", "file_id", "index"]]

        if node_ids is None:
            filtered = base
        else:
            node_id_array = np.asarray(node_ids, dtype=np.int64)
            filtered = base[base["node_id"].isin(node_id_array)]

            if filtered.empty:
                raise ValueError("None of the provided node IDs were found in the dataset")

            if filtered["node_id"].nunique() < len(node_id_array):
                missing = set(node_id_array) - set(filtered["node_id"].to_numpy())
                logger.warning(f"Some node IDs were not found: {sorted(list(missing))[:10]}"
                            f"{'…' if len(missing) > 10 else ''}")

        # Canonicalize: pick one file per node (smallest file_id), deterministic
        before = len(filtered)
        filtered = (
            filtered
            .sort_values(["node_id", "file_id", "index"], kind="mergesort")
            .drop_duplicates(subset="node_id", keep="first")
            .sort_values("node_id", kind="mergesort")
            .reset_index(drop=True)
        )
        dropped = before - len(filtered)
        if dropped > 0:
            logger.info(f"Dropped {dropped} duplicate (node_id, file_id) rows while canonicalizing nodes.")

        # Important because this function is cached: avoid returning a frame
        # that callers might mutate in-place.
        return filtered.copy(deep=False)
    
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
                columns = [i+1 for i in range(component_count)]
                
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
    
    @lru_cache(maxsize=128)
    def get_nodes_in_selection_set(self, selection_set_id: int) -> np.ndarray:
        """
        Return the node IDs belonging to a given STKO selection set.

        Parameters
        ----------
        selection_set_id
            Numeric ID of the selection set (as shown in STKO).

        Returns
        -------
        np.ndarray
            Sorted, **unique** node IDs (dtype=int64).

        Raises
        ------
        AttributeError
            If `self.dataset` has no `.selection_set` attribute.
        ValueError
            If the selection set is missing or does not contain nodes.
        """
        if not hasattr(self.dataset, "selection_set"):
            raise AttributeError(
                "Dataset object has no 'selection_set' attribute. "
                "Make sure you parse selection-set metadata when "
                "building the MPCODataSet."
            )

        sel_dict = self.dataset.selection_set

        if selection_set_id not in sel_dict:
            raise ValueError(f"Selection set ID '{selection_set_id}' not found.")

        sel_entry = sel_dict[selection_set_id]

        if "NODES" not in sel_entry or not sel_entry["NODES"]:
            raise ValueError(f"Selection set {selection_set_id} does not contain any nodes.")

        # Return as a clean, unique, sorted NumPy array
        return np.unique(np.asarray(sel_entry["NODES"], dtype=np.int64))

    @profile_execution
    def get_time_history(
        self,
        model_stage: str,
        results_name: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        scaling_factor: float = 1.0,
        sort_by: str = "z",
        reverse_sort: bool = False,
    ):
        """
        Fetch and prepare nodal time-history data (batched) for plotting/analysis.

        Notes
        -----
        - Returns raw values (optionally scaled), *not* reduced to a single component.
        - Preserves a MultiIndex by node_id so callers can xs(node_id, level=0).
        """

        # ---- resolve node_ids -------------------------------------------- #
        ds = getattr(self, "_dataset", None) or getattr(self, "dataset", None)
        if ds is None:
            raise RuntimeError("[nodes.get_time_history] Missing dataset handle on Nodes object.")

        if node_ids is None and selection_set_id is not None:
            node_ids = self.get_nodes_in_selection_set(selection_set_id)
        elif node_ids is None:
            base_df = ds.nodes_info["dataframe"] if isinstance(ds.nodes_info, dict) else ds.nodes_info
            node_ids = base_df["node_id"].to_numpy()

        node_ids = tuple(np.unique(node_ids))
        if not node_ids:
            raise RuntimeError("[nodes.get_time_history] No node IDs to fetch.")

        # ---- TIME --------------------------------------------------------- #
        time_df = ds.time.loc[model_stage]
        time_arr = (time_df["TIME"] if "TIME" in time_df else time_df.index).to_numpy()

        # ---- coordinates & sorting --------------------------------------- #
        coords_df = (
            ds.nodes_info["dataframe"]
            if isinstance(ds.nodes_info, dict) else ds.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        if sort_by not in {"x", "y", "z"}:
            warnings.warn(f"[nodes.get_time_history] sort_by '{sort_by}' not recognised. Using 'z'.",
                        RuntimeWarning)
            sort_by = "z"

        coords_subset = coords_df.loc[list(node_ids), ["x", "y", "z"]].sort_values(
            by=sort_by, ascending=not reverse_sort
        )
        node_ids_sorted = tuple(coords_subset.index)
        coords_map = coords_subset.to_dict("index")

        # ---- batched results fetch --------------------------------------- #
        df_all = self.get_nodal_results(
            model_stage=model_stage,
            results_name=results_name,
            node_ids=list(node_ids_sorted),
        )
        if df_all is None or df_all.empty:
            raise RuntimeError(f"[nodes.get_time_history] No data for result '{results_name}'.")

        # Ensure columns are a simple, nameable sequence
        if isinstance(df_all.columns, pd.MultiIndex):
            df_all.columns = ["|".join(map(str, c)) for c in df_all.columns.to_list()]

        comp_names = tuple(map(str, df_all.columns.to_list()))

        # ---- scaling (vectorized) ---------------------------------------- #
        if scaling_factor != 1.0:
            # scale a copy to avoid mutating any shared frame
            df_all = df_all * float(scaling_factor)


        return TimeHistoryResults(
            time=np.asarray(time_arr, dtype=float).reshape(-1),
            steps=self.dataset.number_of_steps,
            df=df_all,
            node_ids=node_ids_sorted,
            coords_map=coords_map,
            component_names=comp_names,
        )

    @profile_execution
    def get_roof_drift(
        self,
        model_stage: str,
        direction: str | int,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        normalize: bool = True,
        scaling_factor: float = 1.0,
        z_round: int = 3,
        top_z: float | None = None,
        bottom_z: float | None = None,
        aggregate: Union[str, AggregateFn] = "Mean",  # 'Mean'|'Median'|'Max'|'Min' or callable(np.ndarray)->float
    ) -> RoofDriftResults:
        """
        Compute roof drift time-history from nodal displacements.
        Parameters
        ----------
        model_stage
            Model stage name (as in STKO).
        direction
            Direction of displacement to consider: 'x', 'y', 'z' or 1, 2, 3.
        node_ids
            Specific node IDs to consider. Provide either this or `selection_set_id`.
        selection_set_id
            Selection set ID to consider. Provide either this or `node_ids`.
        normalize
            If True, divide drift by the height (top_z - bottom_z).
        scaling_factor
            Scaling factor to apply to displacements (e.g. 1000 for m->mm).
        z_round
            Number of decimals to round Z-coordinates to when grouping levels.
        top_z
            Z-coordinate of the top level. If None, uses the highest Z-level found.
        bottom_z
            Z-coordinate of the bottom level. If None, uses the lowest Z-level found.
        aggregate
            Aggregation method across nodes at each level. One of:
            - 'Mean' (default)
            - 'Median'
            - 'Max'
            - 'Min'
            - Callable function that takes a 1D np.ndarray and returns a float.
        Returns
        -------
        RoofDriftResults
            Named tuple with fields:
            - time: 1D np.ndarray of time values.
            - height: float, height between top and bottom levels.
            - drift: 1D np.ndarray of roof drift values (scaled, not normalized).
            - normalized_drift: 1D np.ndarray of normalized roof drift (if normalize=True).
            - component_name: str, name of the displacement component used.
            - top_node_ids: tuple of node IDs at the top level.
            - bottom_node_ids: tuple of node IDs at the bottom level.
            - all_node_ids: tuple of all involved node IDs (top + bottom).
        """
        # ---- direction mapping ---------------------------------------------- #
        if isinstance(direction, str):
            dmap = {"x": 0, "y": 1, "z": 2}
            dkey = direction.lower()
            if dkey not in dmap:
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = dmap[dkey]
            dir_lbl = dkey
        else:
            if direction not in (1, 2, 3):
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = direction - 1
            dir_lbl = ["x", "y", "z"][dir_idx]

        # ---- resolve node set & Z grouping ---------------------------------- #
        if node_ids is None and selection_set_id is None:
            raise ValueError("Provide either `node_ids` or `selection_set_id`.")
        if node_ids is not None and selection_set_id is not None:
            raise ValueError("Provide only one of `node_ids` or `selection_set_id`.")

        if node_ids is None:
            node_ids = self.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(np.asarray(node_ids, dtype=int))
        if node_ids.size < 2:
            raise ValueError("Need at least two nodes (top & bottom) to compute roof drift.")

        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict) else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        try:
            coords_sub = coords_df.loc[node_ids, ["x", "y", "z"]].copy()
        except KeyError as e:
            missing = set(node_ids) - set(coords_df.index)
            raise ValueError(f"Some nodes missing in nodes_info: {sorted(missing)[:10]} ...") from e

        coords_sub["_z_group"] = coords_sub["z"].round(z_round)
        level_groups = coords_sub.groupby("_z_group")
        if len(level_groups) < 2:
            raise ValueError("Need at least two Z-levels to compute roof drift.")

        z_levels = np.array(sorted(level_groups.groups.keys()), dtype=float)
        z_bot = float(bottom_z) if bottom_z is not None else float(z_levels.min())
        z_top = float(top_z)    if top_z    is not None else float(z_levels.max())

        if z_bot == z_top:
            raise ValueError("Top and bottom Z-levels coincide (height = 0).")
        if z_bot not in level_groups.groups or z_top not in level_groups.groups:
            raise ValueError(
                f"Requested z-levels not found. Available: {z_levels.tolist()}, "
                f"requested top={z_top}, bottom={z_bot} (rounded to {z_round})."
            )

        ids_bot = tuple(level_groups.get_group(z_bot).index.to_list())
        ids_top = tuple(level_groups.get_group(z_top).index.to_list())
        height = abs(z_top - z_bot)

        # ---- fetch displacements for all involved nodes --------------------- #
        all_ids = np.unique(np.concatenate([ids_bot, ids_top]))
        bundle = self.get_time_history(
            model_stage=model_stage,
            results_name="DISPLACEMENT",
            node_ids=all_ids.tolist(),
            scaling_factor=1.0,     # keep raw; scale at end
            sort_by="z",
            reverse_sort=False,
        )
        time_arr = bundle.time
        df_all   = bundle.df
        comp_names = bundle.component_names

        # choose component by position; fall back to a reasonable name
        comp_name = comp_names[dir_idx] if dir_idx < len(comp_names) else f"#{dir_idx+1}"
        series_all = df_all.iloc[:, dir_idx]  # Series with MultiIndex (node_id, step)

        # collapse any duplicate (node_id, step) rows (ghost/boundary nodes)
        series_all = series_all.groupby(level=["node_id", "step"]).mean()

        # Build wide tables: index=step, columns=node_id (top & bottom)
        def wide(ids: Sequence[int]) -> pd.DataFrame:
            if len(ids) == 0:
                return pd.DataFrame(index=pd.Index([], name="step"))
            sub = series_all.loc[list(ids)] if len(ids) > 1 else series_all.loc[ids[0]]
            if isinstance(sub, pd.Series) and getattr(sub.index, "nlevels", 1) == 2:
                w = sub.unstack(level=0)           # rows: step, cols: node_id
            elif isinstance(sub, pd.Series):        # single node: index=step
                w = sub.to_frame(name=ids[0])
            else:
                w = sub
            w.index.name = "step"
            return w

        top_wide = wide(ids_top)
        bot_wide = wide(ids_bot)

        # Align steps on the union, keep sorted
        steps_union = top_wide.index.union(bot_wide.index).sort_values()
        top_wide = top_wide.reindex(steps_union)
        bot_wide = bot_wide.reindex(steps_union)

        # ---- aggregator across nodes (per step) ----------------------------- #
        if isinstance(aggregate, str):
            k = aggregate.lower()
            if k == "mean":
                agg_fn = np.nanmean
            elif k == "median":
                agg_fn = np.nanmedian
            elif k == "max":
                agg_fn = np.nanmax
            elif k == "min":
                agg_fn = np.nanmin
            else:
                raise ValueError("aggregate must be 'Mean','Median','Max','Min' or a callable.")
            u_top = agg_fn(top_wide.to_numpy(dtype=float), axis=1)
            u_bot = agg_fn(bot_wide.to_numpy(dtype=float), axis=1)
        else:
            # custom callable: pass a 1D float ndarray per step (NaNs removed)
            u_top = top_wide.apply(lambda r: aggregate(r.dropna().to_numpy(dtype=float)), axis=1).to_numpy()
            u_bot = bot_wide.apply(lambda r: aggregate(r.dropna().to_numpy(dtype=float)), axis=1).to_numpy()

        # ---- build aligned time for these steps ----------------------------- #
        steps_idx = steps_union.to_numpy(dtype=int)
        valid = (steps_idx >= 0) & (steps_idx < len(time_arr))
        if not np.all(valid):
            warnings.warn(
                f"[get_roof_drift] {np.count_nonzero(~valid)} step(s) outside time range; trimming.",
                RuntimeWarning,
            )
            steps_idx = steps_idx[valid]
            u_top = u_top[valid]
            u_bot = u_bot[valid]

        x_time = time_arr[steps_idx]

        # ---- drift ----------------------------------------------------------- #
        drift = (u_top - u_bot)
        if normalize:
            if height == 0:
                raise ValueError("Zero height between top and bottom levels.")
            drift = drift / float(height)
        if scaling_factor != 1.0:
            drift = drift * float(scaling_factor)

        return RoofDriftResults(
            time=np.asarray(x_time, dtype=float).reshape(-1),
            steps=np.asarray(steps_idx, dtype=int).reshape(-1),
            drift=np.asarray(drift, dtype=float).reshape(-1),
            u_top=np.asarray(u_top, dtype=float).reshape(-1),
            u_bot=np.asarray(u_bot, dtype=float).reshape(-1),
            top_ids=tuple(ids_top),
            bottom_ids=tuple(ids_bot),
            top_z=float(z_top),
            bottom_z=float(z_bot),
            height=float(height),
            direction=dir_lbl,
            component_name=str(comp_name),
        )

    @profile_execution
    def get_story_drifts(
        self,
        model_stage: str,
        *,
        node_ids: Sequence[int] | None = None,
        selection_set_id: int | None = None,
        direction: str | int = "x",
        normalize: bool = True,
        scaling_factor: float = 1.0,
        sort_by: str = "z",
        reverse_sort: bool = False,
        z_round: int = 3,
        aggregate: str | Callable[[np.ndarray], float] = "Mean",  # Mean|Median|Max|Min or callable(1D)->float
    ) -> StoryDriftsResults:
        """
        Compute inter-storey drifts Δu (or Δu/h if normalize=True) over time.
        Also returns per-storey min/max drift envelopes for drift-profile plots.
        """
        import warnings

        # --- pick nodes ------------------------------------------------------- #
        if (node_ids is None) == (selection_set_id is None):
            raise ValueError("Specify either `node_ids` or `selection_set_id`, not both.")
        if node_ids is None:
            node_ids = self.get_nodes_in_selection_set(selection_set_id)
        node_ids = np.unique(np.asarray(node_ids, dtype=int))
        if node_ids.size < 2:
            raise ValueError("Need ≥2 nodes to compute storey drifts.")

        # --- coords & z grouping --------------------------------------------- #
        coords_df = (
            self.dataset.nodes_info["dataframe"]
            if isinstance(self.dataset.nodes_info, dict) else self.dataset.nodes_info
        ).drop_duplicates("node_id").set_index("node_id")

        if sort_by not in {"x", "y", "z"}:
            warnings.warn(f"sort_by '{sort_by}' invalid. Using 'z'.", RuntimeWarning)
            sort_by = "z"

        coords_sub = coords_df.loc[list(node_ids), ["x", "y", "z"]].copy()
        coords_sub["_z_group"] = coords_sub[sort_by].round(z_round)
        level_groups = coords_sub.groupby("_z_group")
        z_levels = sorted(level_groups.groups.keys(), reverse=reverse_sort)
        if len(z_levels) < 2:
            raise ValueError("Need ≥2 Z levels to compute storey drifts.")

        # --- direction -> index/name ----------------------------------------- #
        if isinstance(direction, str):
            dmap = {"x": 0, "y": 1, "z": 2}
            dkey = direction.lower()
            if dkey not in dmap:
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = dmap[dkey]
            dir_lbl = dkey
        else:
            if direction not in (1, 2, 3):
                raise ValueError("direction must be 'x','y','z' or 1,2,3")
            dir_idx = direction - 1
            dir_lbl = ["x", "y", "z"][dir_idx]

        # --- fetch all node displacements once ------------------------------- #
        bundle = self.get_time_history(
            model_stage=model_stage,
            results_name="DISPLACEMENT",
            node_ids=list(node_ids),
            scaling_factor=1.0,  # scale at end
            sort_by="z",
            reverse_sort=False,
        )
        time_arr = bundle.time
        df_all = bundle.df
        comp_names = bundle.component_names
        comp_name = comp_names[dir_idx] if dir_idx < len(comp_names) else f"#{dir_idx+1}"

        if df_all is None or df_all.empty:
            raise ValueError("No displacement data available for requested nodes.")

        # chosen component as Series with MultiIndex (node_id, step)
        series_all = df_all.iloc[:, dir_idx]
        # collapse duplicate (node_id, step) if any
        series_all = series_all.groupby(level=["node_id", "step"]).mean()

        # --- aggregator across nodes of a level ------------------------------- #
        if isinstance(aggregate, str):
            k = aggregate.lower()
            if   k == "mean":   agg_fn = np.nanmean
            elif k == "median": agg_fn = np.nanmedian
            elif k == "max":    agg_fn = np.nanmax
            elif k == "min":    agg_fn = np.nanmin
            else:
                raise ValueError("aggregate must be 'Mean','Median','Max','Min' or a callable.")
            def reduce_rows(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                u = agg_fn(wide.to_numpy(dtype=float), axis=1)
                idx = wide.index.to_numpy(dtype=int)
                return idx, u
        else:
            # custom callable: reduce each step's row (1D ndarray w/o NaNs)
            def reduce_rows(wide: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
                vals = wide.apply(lambda r: aggregate(r.dropna().to_numpy(dtype=float)), axis=1).to_numpy()
                idx = wide.index.to_numpy(dtype=int)
                return idx, vals

        # helper: level average series (steps_idx, u_level)
        def level_series(ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
            if len(ids) == 0:
                return np.array([], dtype=int), np.array([], dtype=float)
            sub = series_all.loc[list(ids)] if len(ids) > 1 else series_all.loc[ids[0]]
            if isinstance(sub, pd.Series) and getattr(sub.index, "nlevels", 1) == 2:
                wide = sub.unstack(level=0)          # rows: step, cols: node_id
            elif isinstance(sub, pd.Series):          # single node
                wide = sub.to_frame(name=ids[0])
            else:
                wide = sub
            wide.index.name = "step"
            return reduce_rows(wide)

        # --- compute per-story drift & collect global step union -------------- #
        pairs: list[tuple[float,float]] = []
        labels: list[str] = []
        heights: list[float] = []
        per_story_steps: list[np.ndarray] = []
        per_story_drift: list[np.ndarray] = []

        for i in range(len(z_levels) - 1):
            z1, z2 = float(z_levels[i]), float(z_levels[i+1])
            ids1 = level_groups.get_group(z1).index.tolist()
            ids2 = level_groups.get_group(z2).index.tolist()
            h = abs(z2 - z1)
            if h == 0 and normalize:
                warnings.warn(f"Zero height between levels {z1} and {z2}; skipping.", RuntimeWarning)
                continue

            s1_idx, u1 = level_series(ids1)
            s2_idx, u2 = level_series(ids2)

            # align by step union for THIS story
            story_union = np.unique(np.concatenate([s1_idx, s2_idx]))

            def align(idx: np.ndarray, vals: np.ndarray, union: np.ndarray) -> np.ndarray:
                out = np.full(union.shape[0], np.nan, dtype=float)
                if idx.size:
                    pos = {int(k): j for j, k in enumerate(union)}
                    for k, v in zip(idx, vals):
                        j = pos.get(int(k))
                        if j is not None:
                            out[j] = v
                return out

            u1a = align(s1_idx, u1, story_union)
            u2a = align(s2_idx, u2, story_union)

            drift = (u2a - u1a) * float(scaling_factor)
            if normalize and h != 0:
                drift = drift / float(h)

            # keep only steps inside time array
            valid = (story_union >= 0) & (story_union < len(time_arr))
            story_union = story_union[valid]
            drift = drift[valid]

            per_story_steps.append(story_union)
            per_story_drift.append(drift)
            pairs.append((z1, z2))
            labels.append(f"{z1:.2f}→{z2:.2f}")
            heights.append(h)

        if not per_story_steps:
            raise ValueError("No valid storey drift could be computed.")

        # --- build a GLOBAL step union and reindex each story to it ----------- #
        global_steps = np.unique(np.concatenate(per_story_steps))
        time_used = time_arr[global_steps]

        def reindex_to_global(story_steps: np.ndarray, story_vals: np.ndarray) -> np.ndarray:
            out = np.full(global_steps.shape[0], np.nan, dtype=float)
            pos = {int(k): j for j, k in enumerate(global_steps)}
            for k, v in zip(story_steps, story_vals):
                j = pos.get(int(k))
                if j is not None:
                    out[j] = v
            return out

        drift_matrix = np.vstack([
            reindex_to_global(s, v) for s, v in zip(per_story_steps, per_story_drift)
        ])  # shape: (n_stories, n_steps_global)

        # --- envelope/profile (min/max over time for each storey) ------------- #
        envelope_min = np.nanmin(drift_matrix, axis=1)
        envelope_max = np.nanmax(drift_matrix, axis=1)
        z_tops = np.array([p[1] for p in pairs], dtype=float)
        z_base = float(min(z_levels))

        return StoryDriftsResults(
            time=np.asarray(time_used, dtype=float),
            steps=np.asarray(global_steps, dtype=int),
            drift=drift_matrix,
            labels=tuple(labels),
            z_pairs=tuple(pairs),
            heights=np.asarray(heights, dtype=float),
            direction=dir_lbl,
            component_name=str(comp_name),
            z_base=z_base,
            z_tops=z_tops,
            envelope_min=envelope_min,
            envelope_max=envelope_max,
        )


# REVISAR -----------------------------------------------------
    def get_nodes_at_z_levels(self, list_z: list[float], tol: float = 1e-3) -> dict:
        """
        Devuelve los node_id presentes en cada altura especificada.

        Args:
            list_z (list): Lista de alturas Z en mm.
            tol (float): Tolerancia para comparación de altura.

        Returns:
            dict: Diccionario {z: [node_id1, node_id2, ...]} ordenado por Z.
        """
        import numpy as np

        # Asegurar que las coordenadas estén disponibles
        if 'all_nodes' not in self._node_info_cache:
            self._get_all_nodes_ids()

        node_df = self._node_info_cache['all_nodes']['dataframe']

        # Inicializar diccionario
        nodes_by_z = {}

        # Recorrer cada altura Z
        for z_val in sorted(list_z):
            mask = np.isclose(node_df['z'], z_val, atol=tol)
            ids = node_df.loc[mask, 'node_id'].sort_values().tolist()
            nodes_by_z[z_val] = ids

        return nodes_by_z

    def get_results_from_node_dict(
        self,
        model_stage: str,
        results_name: str,
        nodes_by_level: dict,
        reduction: str = "sum",
        direction: int = 1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calcula los resultados por nivel Z utilizando un diccionario de nodos prefiltrados por altura.

        Args:
            model_stage (str): Etapa del modelo (ej: 'MODEL_STAGE[3]').
            results_name (str): Nombre del resultado (ej: 'REACTION_FORCE').
            nodes_by_level (dict): Diccionario con alturas Z como clave y lista de node_ids como valor.
            reduction (str): Operación a aplicar por step: 'sum', 'mean', 'max', 'min'.
            direction (int): Componente (1, 2 o 3) a extraer del resultado.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - df_full: Resultados por nodo, step, x, y, z y valor.
                - df_summary: Una fila por z con min y max luego de aplicar la reducción por step.
        """
        import pandas as pd

        reduction = reduction.lower()
        if reduction not in ['sum', 'mean', 'max', 'min']:
            raise ValueError("Reducción no válida. Usa: 'sum', 'mean', 'max', 'min'.")

        # Cargar geometría de nodos si no está en caché
        if 'all_nodes' not in self._node_info_cache:
            self._get_all_nodes_ids()
        node_df = self._node_info_cache['all_nodes']['dataframe']

        full_rows = []
        summary_rows = []

        for z, node_ids in nodes_by_level.items():
            if not node_ids:
                continue

            df_res = self.get_nodal_results(
                model_stage=model_stage,
                results_name=results_name,
                node_ids=node_ids
            ).reset_index()  # columnas: ['node_id', 'step', 1, 2, 3]

            if direction not in df_res.columns:
                continue

            # Agregar coordenadas x, y, z
            df_coords = node_df[node_df['node_id'].isin(node_ids)][['node_id', 'x', 'y', 'z']]
            df_merged = pd.merge(df_res, df_coords, on='node_id', how='left')

            # Renombrar componente
            df_merged = df_merged[['step', 'node_id', 'x', 'y', 'z', direction]].rename(columns={direction: 'value'})
            full_rows.append(df_merged)

            # Reducción por step
            grouped = df_merged.groupby("step")['value']
            if reduction == "sum":
                reduced = grouped.sum()
            elif reduction == "mean":
                reduced = grouped.mean()
            elif reduction == "max":
                reduced = grouped.max()
            elif reduction == "min":
                reduced = grouped.min()
            
            summary_rows.append({
                "z": z,
                "min_comp": reduced.min(),
                "max_comp": reduced.max()
            })

        df_full = pd.concat(full_rows, ignore_index=True) if full_rows else pd.DataFrame()
        df_summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

        return df_full, df_summary

# ───────────────────────────────────────────────────────────────────── #
# Helper @dataclass to store info

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class TimeHistoryResults:
    """
    Container returned by get_time_history() with results and metadata.

    time : (n_steps,) float array
    steps : (n_steps,) int array
    df : DataFrame indexed by (node_id, step); columns are components ['# 1', '# 2', ...]
    """
    time: np.ndarray
    steps: np.ndarray
    df: pd.DataFrame
    node_ids: Tuple[int, ...]
    coords_map: Dict[int, Dict[str, float]]
    component_names: Tuple[str, ...]

@dataclass(frozen=True, slots=True)
class RoofDriftResults:
    """
    Container returned by compute_roof_drift() with results and metadata.
    Notes
    -----
    - time: (n_steps,) float array
    - steps: (n_steps,) int array
    - drift: (n_steps,) float array
    - u_top: (n_steps,) float array
    - u_bot: (n_steps,) float array
    - top_ids: Tuple of node IDs at the top level
    - bottom_ids: Tuple of node IDs at the bottom level
    - top_z: float, Z coordinate of the top level
    - bottom_z: float, Z coordinate of the bottom level
    - height: float, height between top and bottom levels
    - direction: str, 'x', 'y', or 'z' indicating the displacement direction
    - component_name: str, column label used (e.g. '# 1')
    """
    time: np.ndarray              # aligned to the union of steps used
    steps: np.ndarray             # step indices used for the arrays below
    drift: np.ndarray             # (u_top - u_bot), optionally normalized & scaled
    u_top: np.ndarray             # aggregated top displacement (chosen direction)
    u_bot: np.ndarray             # aggregated bottom displacement (chosen direction)
    top_ids: Tuple[int, ...]
    bottom_ids: Tuple[int, ...]
    top_z: float
    bottom_z: float
    height: float
    direction: str                # 'x' | 'y' | 'z'
    component_name: str           # column label used (e.g. '# 1')

@dataclass(frozen=True, slots=True)
class StoryDriftsResults:
    """
    Container returned by compute_story_drifts() with results and metadata.
    Notes
    -----
    - time: (n_steps_global,) float array
    - steps: (n_steps_global,) int array
    - drift: (n_stories, n_steps_global) float array
    - labels: Tuple of str labels for each storey (e.g. "0.00→3.00", "3.00→6.00", ...)
    - z_pairs: Tuple of (z1, z2) pairs for each storey
    - heights: (n_stories,) float array of storey heights
    - direction: str, 'x', 'y', or 'z' indicating the displacement
    - component_name: str, column label used (e.g. '# 1')
    - z_base: float, absolute base Z (min of all Z levels)
    - z_tops: (n_stories,) float array of top Z of each store
    - envelope_min: (n_stories,) float array of min drift over time per storey
    - envelope_max: (n_stories,) float array of max drift over time per store
    """
    time: np.ndarray              # (n_steps_global,)
    steps: np.ndarray             # (n_steps_global,)
    drift: np.ndarray             # (n_stories, n_steps_global) — Δu or Δu/h at each step
    labels: Tuple[str, ...]       # e.g. ("0.00→3.00", "3.00→6.00", ...)
    z_pairs: Tuple[Tuple[float, float], ...]  # ((z1,z2), ...) for each storey (bottom->top)
    heights: np.ndarray           # (n_stories,)
    direction: str                # 'x' | 'y' | 'z'
    component_name: str           # e.g. '# 1'

    # Envelope / profile info (computed from `drift`)
    z_base: float                 # absolute base Z (min of all Z levels)
    z_tops: np.ndarray            # (n_stories,) — top Z of each storey in `z_pairs`
    envelope_min: np.ndarray      # (n_stories,) — min drift over time per storey
    envelope_max: np.ndarray      # (n_stories,) — max drift over time per storey
