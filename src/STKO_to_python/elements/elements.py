from typing import TYPE_CHECKING, Any
import h5py
import numpy as np
import pandas as pd 
from concurrent.futures import ThreadPoolExecutor


if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class Elements:
    def __init__(self, dataset: 'MPCODataSet'):
        self.dataset = dataset
    
    def _get_all_element_index(self, element_type=None, verbose=False):
        """
        Fetch information for all elements of a given type in the partition files.
        If no element type is provided, fetch information for all element types.

        Args:
            element_type (str or None): The type of elements to fetch (e.g., 'ElasticBeam3d'). 
                                        If None, fetches all element types.
            verbose (bool, optional): If True, prints memory usage for the 
                                      structured array and DataFrame. Defaults to False.

        Returns:
            dict: A dictionary containing:
                - 'array': Structured NumPy array with element data.
                - 'dataframe': Pandas DataFrame with element data.
        """
        model_stages = self.dataset.model_stages

        # Prepare a dictionary of node coordinates for centroid calculation
        # (node_id -> (x, y, z))
        node_coord_map = {}
        if hasattr(self.dataset, 'nodes_info') and 'dataframe' in self.dataset.nodes_info:
            df_nodes = self.dataset.nodes_info['dataframe']
            # Build the dictionary for quick lookup of coordinates by node_id
            for node_id, x, y, z in zip(df_nodes['node_id'], df_nodes['x'], df_nodes['y'], df_nodes['z']):
                node_coord_map[int(node_id)] = (float(x), float(y), float(z))
        else:
            # If node information is not available, we cannot compute centroids
            # Proceed without centroid calculation
            node_coord_map = None

        # Determine which element types to fetch
        if element_type is None:
            element_types = self.dataset.element_types.get('unique_element_types', [])
            if isinstance(element_types, set):
                element_types = list(element_types)

            # 🔁 Extract base names like '203-ASDShellQ4'
            base_elements = sorted(set(e.split('[')[0] for e in element_types))
            element_types = base_elements

        else:
            element_types = [element_type.split('[')[0]]  # just to be safe

        if verbose:
            print(f"Fetching elements of types: {element_types}")
        
        elements_info = []  # List to store info for all elements

        # Loop through each partition file
        for part_number, partition_path in self.dataset.results_partitions.items():
            with h5py.File(partition_path, 'r') as partition:
                # Loop through each requested element type
                for etype in element_types:
                    # Construct the HDF5 path for this element type in the current partition
                    elem_path = self.dataset.MODEL_ELEMENTS_PATH.format(model_stage=model_stages[0], element_type=etype)
                    element_group = partition.get(elem_path)
                    if element_group is None:
                        # If this partition does not contain the element type, skip it
                        if verbose:
                            print(f"Warning: Element type '{etype}' not found in partition {part_number}.")
                        continue

                    # Loop through each dataset in the element group
                    for element_name in element_group.keys():
                        # The dataset name might be like "ASDQuad4[1]" etc. 
                        # We use the part before '[' to match the type name
                        base_name = element_name.split('[')[0]
                        if base_name != etype:
                            continue  # skip any entries not matching the element type (just in case)

                        dataset = element_group[element_name]
                        data = dataset[:]  # Load all element data (IDs and connectivity)
                        # Each entry in data: [element_id, node1, node2, ..., nodeN]

                        for idx, element_data in enumerate(data):
                            element_id = int(element_data[0])
                            node_ids = element_data[1:]  # array of node IDs (possibly numpy types)
                            # Convert node IDs to a regular Python list of ints
                            node_list = [int(nid) for nid in node_ids]

                            # Calculate centroid if node coordinates are available
                            if node_coord_map:
                                # Sum coordinates of all nodes in this element
                                sx = sy = sz = 0.0
                                for nid in node_list:
                                    # Look up each node's coordinates
                                    x, y, z = node_coord_map.get(nid, (0.0, 0.0, 0.0))
                                    sx += x; sy += y; sz += z
                                num_nodes = len(node_list)
                                centroid_x = sx / num_nodes
                                centroid_y = sy / num_nodes
                                centroid_z = sz / num_nodes
                            else:
                                # If node coordinates unavailable, set centroid as None or 0
                                num_nodes = len(node_list)
                                centroid_x = centroid_y = centroid_z = None

                            # Append element info dictionary
                            elements_info.append({
                                'element_id': element_id,
                                'element_idx': idx,
                                'file_name': part_number,
                                'element_type': etype,
                                'node_list': node_list,
                                'num_nodes': num_nodes,
                                'centroid_x': centroid_x,
                                'centroid_y': centroid_y,
                                'centroid_z': centroid_z
                            })

        # Convert the collected info to structured numpy array and pandas DataFrame
        if elements_info:
            # Define dtype for structured array, matching keys of the dict
            dtype = [
                ('element_id', 'i8'),
                ('element_idx', 'i8'),
                ('file_name', 'i8'),
                ('element_type', object),
                ('node_list', object),
                ('num_nodes', 'i8'),
                ('centroid_x', 'f8'),
                ('centroid_y', 'f8'),
                ('centroid_z', 'f8')
            ]
            # Create structured array data
            structured_data = [
                (
                    elem['element_id'],
                    elem['element_idx'],
                    elem['file_name'],
                    elem['element_type'],
                    elem['node_list'],
                    elem['num_nodes'],
                    elem['centroid_x'] if elem['centroid_x'] is not None else np.nan,
                    elem['centroid_y'] if elem['centroid_y'] is not None else np.nan,
                    elem['centroid_z'] if elem['centroid_z'] is not None else np.nan
                )
                for elem in elements_info
            ]
            results_array = np.array(structured_data, dtype=dtype)
            # Create DataFrame from the list of dicts
            df = pd.DataFrame(elements_info)

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
            if verbose:
                print("No elements found.")
            return {
                'array': np.array([], dtype=[
                    ('element_id', 'i8'),
                    ('file_id', 'i8'),
                    ('index', 'i8'),
                    ('type', 'U50'),
                    ('node_list', object),
                    ('num_nodes', 'i8'),
                    ('centroid_x', 'f8'),
                    ('centroid_y', 'f8'),
                    ('centroid_z', 'f8')
                ]),
                'dataframe': pd.DataFrame()
            }

    def get_elements_at_z_levels(
        self,
        list_z: list[float],
        element_type: str | None = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Return a DataFrame with elements that intersect horizontal planes at multiple Z-levels.

        Parameters
        ----------
        list_z : list of float
            Z-coordinates (in mm) where horizontal slicing planes are defined.
        element_type : str or None, default=None
            Specific element type to filter (e.g., '203-ASDShellQ4').
            If None, all element types are considered.
        verbose : bool, default=False
            If True, prints the number of elements found at each Z-level.

        Returns
        -------
        pd.DataFrame
            A DataFrame of intersecting elements including a 'z_level' column.
        """
        # Get elements of the specified type (or all types if None)
        result = self._get_all_element_index(element_type=element_type, verbose=False)
        df_elements = result['dataframe']

        # Validate node info availability
        if not hasattr(self.dataset, 'nodes_info') or 'dataframe' not in self.dataset.nodes_info:
            raise ValueError("Node information is not available in the dataset.")

        df_nodes = self.dataset.nodes_info['dataframe']
        node_z_map = dict(zip(df_nodes['node_id'], df_nodes['z']))

        all_filtered = []

        for z_level in list_z:
            filtered_elements = []

            for _, row in df_elements.iterrows():
                node_ids = row['node_list']
                z_coords = [node_z_map.get(nid, None) for nid in node_ids]
                z_coords = [z for z in z_coords if z is not None]

                if not z_coords:
                    continue  # Skip if no Z coordinates available

                min_z = min(z_coords)
                max_z = max(z_coords)

                # Check if the Z-plane intersects the element
                if min_z <= z_level <= max_z:
                    filtered_elements.append(row)

            df_filtered = pd.DataFrame(filtered_elements)
            df_filtered['z_level'] = z_level
            all_filtered.append(df_filtered)

            if verbose:
                print(f"[Z = {z_level}] Elements found: {len(df_filtered)}")

        if all_filtered:
            return pd.concat(all_filtered, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_available_element_results(self, element_type: str = None) -> dict[str, dict[str, list[str]]]:
        """
        List available element result types across partitions.

        Parameters
        ----------
        element_type : str, optional
            Base name (e.g., '203-ASDShellQ4') or full decorated name
            (e.g., '203-ASDShellQ4[201:0:0]'). If None, includes all.

        Returns
        -------
        dict
            {
                partition_id: {
                    result_name: [list of matching decorated types]
                }
            }
        """
        results_by_partition = {}

        for part_id, filepath in self.dataset.results_partitions.items():
            with h5py.File(filepath, "r") as f:
                try:
                    partition_results = {}

                    for stage in self.dataset.model_stages:
                        group_path = f"{stage}/RESULTS/ON_ELEMENTS"
                        if group_path not in f:
                            continue

                        on_elements = f[group_path]

                        for result_name in on_elements:
                            result_group = on_elements[result_name]
                            matched_element_types = []

                            for etype_name in result_group:
                                if element_type is None:
                                    matched_element_types.append(etype_name)
                                elif etype_name == element_type:
                                    matched_element_types.append(etype_name)
                                elif etype_name.startswith(element_type):  # allow base match
                                    matched_element_types.append(etype_name)

                            if matched_element_types:
                                partition_results[result_name] = matched_element_types

                    if partition_results:
                        results_by_partition[part_id] = partition_results

                except Exception as e:
                    print(f"[{filepath}] → Error reading results: {e}")

        return results_by_partition


    def get_elements_in_selection_at_z_levels(
        self,
        selection_set_id: int,
        list_z: list[float],
        element_type: str | None = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Return a DataFrame with elements from a selection set that intersect horizontal Z planes.

        Parameters
        ----------
        selection_set_id : int
            ID of the selection set (e.g., 17).
        list_z : list of float
            Z-levels in mm for intersection planes.
        element_type : str or None, default=None
            Filter by base element type (e.g., '203-ASDShellQ4'). If None, includes all types.
        verbose : bool, default=False
            If True, prints count of intersecting elements per level.

        Returns
        -------
        pd.DataFrame
            Elements intersecting each Z-level, with added 'z_level' column.
        """
        # Load all or filtered element metadata
        result = self._get_all_element_index(element_type=element_type, verbose=False)
        df_elements = result['dataframe']

        # Validate selection set exists
        try:
            element_ids = self.dataset.selection_set[selection_set_id]['ELEMENTS']
        except (AttributeError, KeyError):
            raise ValueError(f"Selection set {selection_set_id} not found or has no 'ELEMENTS' key.")

        # Filter element list
        df_elements = df_elements[df_elements['element_id'].isin(element_ids)]

        # Validate node data
        if not hasattr(self.dataset, 'nodes_info') or 'dataframe' not in self.dataset.nodes_info:
            raise ValueError("Node information is not available in the dataset.")

        df_nodes = self.dataset.nodes_info['dataframe']
        node_z_map = dict(zip(df_nodes['node_id'], df_nodes['z']))

        all_filtered = []

        for z_level in list_z:
            filtered_elements = []

            for _, row in df_elements.iterrows():
                node_ids = row['node_list']
                z_coords = [node_z_map.get(nid, None) for nid in node_ids]
                z_coords = [z for z in z_coords if z is not None]

                if not z_coords:
                    continue

                if min(z_coords) <= z_level <= max(z_coords):
                    filtered_elements.append(row)

            df_filtered = pd.DataFrame(filtered_elements)
            df_filtered['z_level'] = z_level
            all_filtered.append(df_filtered)

            if verbose:
                print(f"[Z = {z_level}] Elements in selection set: {len(df_filtered)}")

        if all_filtered:
            return pd.concat(all_filtered, ignore_index=True)
        else:
            return pd.DataFrame()

    def get_element_results(
        self,
        results_name: str,
        element_type: str,
        element_ids: list[int],
        model_stage: str = None,
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Retrieve element results using base element type for filtering and dynamic
        decorated-name matching from HDF5 structure.

        Parameters
        ----------
        results_name : str
            e.g. 'globalForces'
        element_type : str
            Base type (e.g. '203-ASDShellQ4') — not the decorated name
        element_ids : list[int]
            Element IDs to retrieve
        model_stage : str, optional
            Defaults to the first model stage
        verbose : bool
            Print debug info

        Returns
        -------
        pd.DataFrame
            Results indexed by (element_id, step)
        """
        if not element_ids:
            raise ValueError("No element IDs provided.")

        if model_stage is None:
            model_stage = self.dataset.model_stages[0]

        # Step 1: Filter from memory
        df_info = self.dataset.elements_info['dataframe']
        df_info = df_info[df_info['element_type'].str.startswith(element_type)]
        df_info = df_info[df_info['element_id'].isin(element_ids)]

        if df_info.empty:
            raise ValueError(f"No matching elements found for base type '{element_type}'.")

        if verbose:
            print(f"[INFO] {len(df_info)} matching elements found for '{element_type}'")
            print(df_info[['element_id', 'file_name', 'element_type']].to_string(index=False))

        collected = []

        # Step 2: group by partition only (not by element_type)
        for file_id, df_group in df_info.groupby('file_name'):
            file_path = self.dataset.results_partitions[file_id]
            idx_list = df_group['element_idx'].to_numpy(dtype=int)
            id_list = df_group['element_id'].to_numpy(dtype=int)

            with h5py.File(file_path, 'r') as f:
                base_results_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}"
                if base_results_path not in f:
                    if verbose:
                        print(f"[WARN] '{base_results_path}' not found in partition {file_id}")
                    continue

                candidates = list(f[base_results_path].keys())
                matching_names = [name for name in candidates if name.startswith(element_type)]

                if not matching_names:
                    if verbose:
                        print(f"[WARN] No match for base '{element_type}' under '{base_results_path}'")
                    continue

                for decorated_type in matching_names:
                    h5_path = f"{base_results_path}/{decorated_type}/DATA"
                    if h5_path not in f:
                        if verbose:
                            print(f"[WARN] Path not found: {h5_path}")
                        continue

                    for step_idx, step_name in enumerate(f[h5_path]):
                        dset = f[f"{h5_path}/{step_name}"]
                        values = dset[idx_list]

                        df = pd.DataFrame(
                            values,
                            columns=[f"val_{i+1}" for i in range(values.shape[1])]
                        )
                        df['step'] = step_idx
                        df['element_id'] = id_list
                        collected.append(df)

        if not collected:
            if verbose:
                print("[INFO] No result data collected.")
            return pd.DataFrame()

        out = pd.concat(collected, axis=0)
        out.set_index(['element_id', 'step'], inplace=True)
        return out.sort_index()

    def get_element_results_by_selection_and_z(
        self,
        results_name: str,
        selection_set_id: int,
        list_z: list[float],
        element_type: str | None = None,
        model_stage: str | None = None,
        verbose: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Filter elements by selection set + Z-levels, then delegate to get_element_results().
        Returns result DataFrames grouped by decorated element type.
        """
        # Step 1: Get all elements in selection set intersecting Z
        df_filtered = self.get_elements_in_selection_at_z_levels(
            selection_set_id=selection_set_id,
            list_z=list_z,
            element_type=element_type,
            verbose=verbose
        )

        if df_filtered.empty:
            if verbose:
                print("[INFO] No elements found at Z-levels in selection set.")
            return {}

        # Step 2: Get mapping of element_id → element_type (decorated) and file_name
        df_info = self.dataset.elements_info['dataframe'][['element_id', 'file_name', 'element_type']]
        df_merged = pd.merge(df_filtered, df_info, on=['element_id', 'file_name'], suffixes=('', '_decorated'))
        df_merged['element_type'] = df_merged['element_type_decorated']

        # Step 3: Group by real decorated type
        results_by_type: dict[str, pd.DataFrame] = {}

        for decorated_type, df_group in df_merged.groupby('element_type'):
            element_ids = df_group['element_id'].unique().tolist()

            if verbose:
                print(f"\n↳ {decorated_type}: {len(element_ids)} elements to fetch")

            df_result = self.get_element_results(
                results_name=results_name,
                element_type=decorated_type,  # <- full name for HDF5
                element_ids=element_ids,
                model_stage=model_stage,
                verbose=verbose
            )

            if df_result.empty:
                continue

            # Attach z_level info
            df_with_z = pd.merge(
                df_result.reset_index(),
                df_group[['element_id', 'z_level']].drop_duplicates(),
                on='element_id',
                how='left'
            ).set_index(['element_id', 'step'])

            results_by_type[decorated_type] = df_with_z.sort_index()

        return results_by_type





    
    
    
    
    
    
    
    
    
    
    
    