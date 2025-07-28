from typing import TYPE_CHECKING
import h5py
import numpy as np
import pandas as pd 

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
            # If element_types is a dict, use its keys; if list, use directly
            element_types = self.dataset.element_types
            if isinstance(element_types, dict):
                element_types = list(element_types.keys())
        else:
            element_types = [element_type]

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


    def get_elements_at_z_levels(self, list_z: list[float], element_type: str, verbose: bool = False) -> pd.DataFrame:
        """
        Devuelve un DataFrame con los elementos que intersectan planos horizontales en múltiples niveles Z.

        Parámetros:
        -----------
        list_z : list of float
            Lista de valores Z (mm) donde se definen los planos horizontales de corte.
        element_type : str
            Tipo de elemento (ej. '203-ASDShellQ4') a considerar.
        verbose : bool
            Si es True, imprime la cantidad de elementos encontrados por nivel Z.

        Retorna:
        --------
        pd.DataFrame:
            DataFrame con los elementos que intersectan cada plano, incluyendo columna 'z_level'.
        """
        # Obtener todos los elementos del tipo especificado
        result = self._get_all_element_index(element_type=element_type, verbose=False)
        df_elements = result['dataframe']

        # Obtener coordenadas de los nodos
        if not hasattr(self.dataset, 'nodes_info') or 'dataframe' not in self.dataset.nodes_info:
            raise ValueError("Información de nodos no disponible en el dataset.")

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
                    continue  # Si no hay coordenadas disponibles, se omite

                min_z = min(z_coords)
                max_z = max(z_coords)

                # Verifica si el plano Z intersecta el elemento
                if min_z <= z_level <= max_z:
                    filtered_elements.append(row)

            df_filtered = pd.DataFrame(filtered_elements)
            df_filtered['z_level'] = z_level
            all_filtered.append(df_filtered)

            if verbose:
                print(f"[Z = {z_level}] Elementos encontrados: {len(df_filtered)}")

        if all_filtered:
            return pd.concat(all_filtered, ignore_index=True)
        else:
            return pd.DataFrame()



    def get_available_element_results(self, element_type: str = None):
        """
        Explora los archivos de partición para listar los tipos de resultados disponibles por elemento.
        
        Args:
            element_type (str, optional): Tipo de elemento a consultar (por ejemplo, '203-ASDShellQ4').
                                        Si es None, muestra todos los tipos disponibles.
        
        Returns:
            dict: Diccionario {partition_id: [lista de resultados disponibles]}
        """
        model_stages = self.dataset.model_stages
        results_by_partition = {}

        for part_id, filepath in self.dataset.results_partitions.items():
            with h5py.File(filepath, 'r') as f:
                try:
                    stage = model_stages[0]
                    results_path = f"/MODEL/{stage}/ELEMENTS"
                    if results_path not in f:
                        continue

                    element_results = f[results_path]
                    results_for_type = {}

                    for etype_name in element_results:
                        # Si se solicita un tipo específico, saltar los otros
                        if element_type is not None and not etype_name.startswith(element_type):
                            continue

                        group = element_results[etype_name]
                        result_names = list(group.keys())
                        results_for_type[etype_name] = result_names

                    results_by_partition[part_id] = results_for_type
                except Exception as e:
                    print(f"Error leyendo {filepath}: {e}")

        return results_by_partition


    def get_element_results(self, results_name: str, element_type: str, element_ids: list[int] = None) -> pd.DataFrame:
        """
        Devuelve resultados de un tipo específico de elemento para todos los model_stages disponibles.

        Parámetros:
        -----------
        results_name : str
            Nombre exacto del resultado (ej. 'STRESS_TENSOR', 'STRAIN_TENSOR', etc.).
        element_type : str
            Tipo de elemento (ej. '203-ASDShellQ4').
        element_ids : list[int], opcional
            Lista de IDs de elementos a extraer (si se desea filtrar). Si None, se extraen todos.

        Retorna:
        --------
        DataFrame con columnas: ['model_stage', 'step', 'frame', 'element_id', 'data']
        """
        results = []

        for model_stage in self.dataset.model_stages:
            for part_number, path in self.dataset.results_partitions.items():
                with h5py.File(path, 'r') as f:
                    base_path = f'results/{model_stage}/ELEMENT/{element_type}/{results_name}'
                    if base_path not in f:
                        continue

                    group = f[base_path]
                    for step_key in group:
                        step_group = group[step_key]
                        for frame_key in step_group:
                            frame_data = step_group[frame_key]
                            element_ids_data = frame_data['element_id'][:]
                            result_data = frame_data['data'][:]

                            for eid, data in zip(element_ids_data, result_data):
                                if (element_ids is None) or (eid in element_ids):
                                    results.append({
                                        'model_stage': model_stage,
                                        'step': step_key,
                                        'frame': frame_key,
                                        'element_id': int(eid),
                                        'data': data
                                    })

        if not results:
            print(f"No se encontraron resultados para '{results_name}' en '{element_type}'")
            return pd.DataFrame()

        return pd.DataFrame(results)