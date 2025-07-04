from typing import TYPE_CHECKING
import h5py
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet

class Elements:
    def __init__(self, dataset: 'MPCODataSet'):
        self.dataset = dataset

    def _get_all_element_index(self, element_type=None, model_stage=None, verbose=False):
        """
        Extrae la conectividad y ubicación de elementos de un tipo específico o todos los tipos conocidos.

        Parámetros:
        -----------
        element_type : str, opcional
            Tipo de elemento a leer (nombre completo incluyendo corchetes).
        model_stage : str, opcional
            Etapa del modelo a usar (por defecto usa la primera disponible).
        verbose : bool
            Si True, imprime información de memoria.

        Retorna:
        --------
        dict: {'array': ndarray estructurado, 'dataframe': DataFrame}
        """
        if model_stage is None:
            model_stage = self.dataset.model_stages[0]

        if element_type is None:
            element_types = self.dataset.element_types['unique_element_types']
        else:
            element_types = [element_type]

        node_coord_map = {}
        if hasattr(self.dataset, 'nodes_info') and 'dataframe' in self.dataset.nodes_info:
            df_nodes = self.dataset.nodes_info['dataframe']
            for node_id, x, y, z in zip(df_nodes['node_id'], df_nodes['x'], df_nodes['y'], df_nodes['z']):
                node_coord_map[int(node_id)] = (x, y, z)

        elements_info = []

        for part_id, path in self.dataset.results_partitions.items():
            with h5py.File(path, 'r') as f:
                base_path = f"{model_stage}/MODEL/ELEMENTS"
                for etype in element_types:
                    dataset_path = f"{base_path}/{etype}"
                    if dataset_path not in f:
                        continue

                    data = f[dataset_path][:]
                    for idx, row in enumerate(data):
                        element_id = int(row[0])
                        node_list = row[1:].tolist()
                        centroid_x = centroid_y = centroid_z = np.nan

                        if node_coord_map:
                            coords = [node_coord_map.get(nid, (0, 0, 0)) for nid in node_list]
                            coords = np.array(coords)
                            centroid_x, centroid_y, centroid_z = coords.mean(axis=0)

                        elements_info.append({
                            'element_id': element_id,
                            'element_idx': idx,
                            'file_name': part_id,
                            'element_type': etype,
                            'node_list': node_list,
                            'num_nodes': len(node_list),
                            'centroid_x': centroid_x,
                            'centroid_y': centroid_y,
                            'centroid_z': centroid_z
                        })

        if not elements_info:
            if verbose:
                print("No elements found.")
            return {'array': np.array([]), 'dataframe': pd.DataFrame()}

        df = pd.DataFrame(elements_info)
        return {'array': df.to_records(index=False), 'dataframe': df}



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
    

    def get_shell_result_components(self, result_type: str, model_stage: str = 'MODEL_STAGE[3]', verbose: bool = True):
        """
        Retorna los componentes disponibles para un tipo de resultado aplicado a elementos tipo Shell.

        Parámetros:
        -----------
        result_type : str
            Tipo de resultado (ej. 'section.force', 'section.deformation').
        model_stage : str
            Etapa del modelo donde buscar (por defecto 'MODEL_STAGE[3]').
        verbose : bool
            Si True, imprime los resultados encontrados.

        Retorna:
        --------
        dict:
            Diccionario {element_type: [componentes]} para cada tipo de elemento tipo Shell encontrado.
        """
        shell_components = {}

        for part_id, path in self.dataset.results_partitions.items():
            with h5py.File(path, 'r') as f:
                base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{result_type}"
                if base_path not in f:
                    if verbose:
                        print(f"[{part_id}] Resultado '{result_type}' no encontrado.")
                    continue

                for etype in f[base_path]:
                    if 'Shell' not in etype:
                        continue

                    meta_path = f"{base_path}/{etype}/META/COMPONENTS"
                    if meta_path not in f:
                        if verbose:
                            print(f"[{part_id}] {etype} no tiene COMPONENTS definidos.")
                        continue

                    comps = [c.decode() for c in f[meta_path][()]]
                    shell_components[etype] = comps

                    if verbose:
                        print(f"🟩 Archivo: {part_id}")
                        print(f"   • Tipo elemento : {etype}")
                        print(f"   • Componentes   : {comps}")

        # === Agregado: imprimir resumen de resultados disponibles ===
        print("\n📋 Resultados disponibles en ModelAnalysis_01:")
        if hasattr(self.dataset, "element_results_names"):
            for rtype in sorted(self.dataset.element_results_names):
                print(f"   • {rtype}")
        else:
            print("   [X] No se encontró 'element_results_names' en el dataset.")

        return shell_components



    def get_element_results_by_ids(
        self,
        model_stage: str,
        results_name: str,
        element_ids: list[int],
        desired_components: list[str],
        verbose: bool = False
    ) -> pd.DataFrame:
        """
        Extrae resultados específicos por componente de elementos tipo Shell.

        Parámetros:
        -----------
        model_stage : str
            Etapa del modelo (ej. 'MODEL_STAGE[3]').
        results_name : str
            Nombre del resultado (ej. 'section.deformation').
        element_ids : list[int]
            Lista de IDs de elementos deseados.
        desired_components : list[str]
            Componentes deseadas (ej. ['epsXX', 'epsYY']).
        verbose : bool
            Si True, imprime información adicional.

        Retorna:
        --------
        pd.DataFrame con columnas ['element_id', 'step', 'point_id', comp_1, comp_2, ...]
        """
        import pandas as pd

        all_results = []

        for file_id, file_path in self.dataset.results_partitions.items():
            with h5py.File(file_path, 'r') as f:
                base_path = f"{model_stage}/RESULTS/ON_ELEMENTS/{results_name}"
                if base_path not in f:
                    continue

                for element_type in f[base_path]:
                    if 'Shell' not in element_type:
                        continue

                    group_path = f"{base_path}/{element_type}"
                    ids_path = f"{group_path}/ID"
                    meta_path = f"{group_path}/META/COMPONENTS"
                    data_path = f"{group_path}/DATA"

                    if ids_path not in f or meta_path not in f or data_path not in f:
                        continue

                    # Obtener IDs
                    element_ids_all = f[ids_path][()]
                    # Obtener componentes
                    raw_labels = f[meta_path][()][0].decode()
                    point_blocks = raw_labels.split(';')
                    per_point_labels = point_blocks[0].split(',')
                    per_point_labels = [l.strip().split('.')[-1] for l in per_point_labels]
                    n_points = len(point_blocks)
                    n_comp = len(per_point_labels)

                    # Verificar componentes
                    missing = [c for c in desired_components if c not in per_point_labels]
                    if missing:
                        if verbose:
                            print(f"[{file_id}] {element_type} → componentes no encontradas: {missing}")
                        continue

                    comp_indices = [per_point_labels.index(c) for c in desired_components]

                    # Revisar elementos encontrados
                    mask = np.isin(element_ids_all, element_ids)
                    if not np.any(mask):
                        continue

                    element_idx = np.where(mask)[0]
                    element_real_ids = element_ids_all[element_idx]

                    for step_key in f[data_path].keys():
                        step_data = f[f"{data_path}/{step_key}"][()]

                        for idx, eid in zip(element_idx, element_real_ids):
                            full_vector = step_data[idx]
                            for p in range(n_points):
                                row = {
                                    'element_id': int(eid),
                                    'step': int(step_key.replace("STEP_", "")),
                                    'point_id': p
                                }
                                for ci, cname in zip(comp_indices, desired_components):
                                    row[cname] = full_vector[p * n_comp + ci]
                                all_results.append(row)


        if not all_results:
            if verbose:
                print("No se encontraron resultados para los elementos y componentes indicados.")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        return df.sort_values(by=['element_id', 'step', 'point_id']).reset_index(drop=True)

    def plot_shell_elements(self, df_elements: pd.DataFrame, highlight_ids: list[int] = None):
        """
        Dibuja elementos tipo Shell y nodos, con opción de resaltar ciertos elementos.

        Parámetros:
        -----------
        df_elements : pd.DataFrame
            DataFrame con columnas ['element_id', 'node_list', 'centroid_x', 'centroid_y', 'centroid_z'].

        highlight_ids : list[int], opcional
            Lista de IDs de elementos a resaltar. Si es None, se dibujan todos igual.
        """
        if not hasattr(self.dataset, 'nodes_info') or 'dataframe' not in self.dataset.nodes_info:
            raise ValueError("Información de nodos no disponible en el dataset.")

        df_nodes = self.dataset.nodes_info['dataframe']
        coord_map = dict(zip(df_nodes['node_id'], zip(df_nodes['x'], df_nodes['y'], df_nodes['z'])))
        highlight_ids = set(highlight_ids) if highlight_ids else set()

        fig = plt.figure(figsize=(10, 8))
        ax = plt.subplot(111, projection='3d')

        for _, row in df_elements.iterrows():
            node_ids = row['node_list']
            coords = [coord_map[nid] for nid in node_ids]
            if coords[0] != coords[-1]:
                coords.append(coords[0])

            is_selected = row['element_id'] in highlight_ids
            color = 'lightcoral' if is_selected else 'skyblue'
            alpha = 0.4 if is_selected else 0.7

            poly = Poly3DCollection([coords], facecolor=color, edgecolor='k', alpha=alpha)
            ax.add_collection3d(poly)

            ax.text(row['centroid_x'], row['centroid_y'], row['centroid_z'],
                    str(row['element_id']), color='black', fontsize=8, ha='center')

        ax.scatter(df_nodes['x'], df_nodes['y'], df_nodes['z'], color='black', s=5)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Shell Elements')

        ax.view_init(elev=0, azim=-90)
        ax.set_proj_type('ortho')
        plt.tight_layout()
        plt.show()