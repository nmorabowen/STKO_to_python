


from ..nodes.nodes import Nodes
from ..elements.elements import Elements
from ..model.model_info import ModelInfo
from ..model.cdata import CData
from ..plotting.plot import Plot


class MPCODataSet:
    
    """
    A dataset management class for MPCO (Model, Partition, Component, Operation) simulation files.

    This class serves as the main access point to HDF5-based MPCO data files, providing
    functionality to load, organize, and access simulation data through a composition
    of specialized classes (Nodes, Elements, ModelInfo, CData).

    The class implements a "friend" pattern with its component classes, allowing it
    to access their protected methods which are indicated by a leading underscore.
    These protected methods are used internally within the dataset implementation
    but are not intended to be part of the public API.

    Upon initialization, this class automatically loads directory information,
    extracts partitions, model stages, results names, and other essential dataset
    attributes to provide convenient access to the simulation data.

    Attributes
    ----------
    nodes : Nodes
    Component handling node-related operations and data access.
    elements : Elements
    Component handling element-related operations and data access.
    model_info : ModelInfo
    Component handling model metadata and information.
    cdata : CData
    Component handling component data information.
    hd5f_directory : str
    Path to the directory containing HDF5 files.
    recorder_name : str
    Base name of the recorder files to load.
    file_extension : str
    File extension pattern for dataset files.
    verbose : bool
    Flag controlling verbose output.
    results_partitions : dict
    Mapping of partition indices to result file paths.
    cdata_partitions : dict
    Mapping of partition indices to cdata file paths.
    model_stages : list
    Available model stages in the dataset.
    node_results_names : list
    Names of nodal results available in the dataset.
    element_results_names : list
    Names of element results available in the dataset.
    element_types : dict
    Dictionary containing element type information.
    unique_element_types : list
    List of all unique element types in the dataset.
    time : pandas.DataFrame
    Time series data for the simulation.
    nodes_info : dict
    Node mapping information for efficient data access.
    elements_info : dict
    Element mapping information for efficient data access.
    number_of_steps : dict
    Number of simulation steps per model stage.
    selection_set : dict
    Selection set mappings from the dataset.

    Parameters
    ----------
    hdf5_directory : str
    Path to the directory containing HDF5 files.
    recorder_name : str
    Base name of the recorder files to load.
    file_extension : str, optional
    File extension pattern for dataset files, default is '*.mpco'.
    verbose : bool, optional
    Enable verbose output, default is False.

    Examples
    --------
    >>> dataset = MPCODataSet('/path/to/hdf5_files', 'simulation_results')
    >>> dataset.print_summary()
    >>> # Access model stages
    >>> print(dataset.model_stages)
    >>> # Get information about nodal results
    >>> dataset.print_nodal_results()

    Notes
    -----
    The class uses common path templates for accessing data within the HDF5 structure:
    - MODEL_NODES_PATH: "/{model_stage}/MODEL/NODES"
    - MODEL_ELEMENTS_PATH: "/{model_stage}/MODEL/ELEMENTS"
    - RESULTS_ON_ELEMENTS_PATH: "/{model_stage}/RESULTS/ON_ELEMENTS"
    - RESULTS_ON_NODES_PATH: "/{model_stage}/RESULTS/ON_NODES"
    """

    # Common path templates
    MODEL_NODES_PATH = "/{model_stage}/MODEL/NODES"
    MODEL_ELEMENTS_PATH = "/{model_stage}/MODEL/ELEMENTS"
    RESULTS_ON_ELEMENTS_PATH = "/{model_stage}/RESULTS/ON_ELEMENTS"
    RESULTS_ON_NODES_PATH = "/{model_stage}/RESULTS/ON_NODES"
    
    def __init__(
        self,
        hdf5_directory: str,
        recorder_name: str,
        file_extension='*.mpco',
        verbose=False,
    ):
        
        self.hd5f_directory = hdf5_directory
        self.recorder_name = recorder_name
        self.file_extension = file_extension
        self.verbose = verbose
        
        # Instanciate the composite classes
        self.nodes=Nodes(self)
        self.elements=Elements(self)
        self.model_info = ModelInfo(self)
        self.cdata=CData(self)
        self.plot=Plot(self)
        
        # Create the object attributes
        self._create_object_attributes()
        
    def _create_object_attributes(self):
        """
        Helper method to create object attributes.
        """
        # Extract the results partition for the given recorder name
        self.results_partitions=self.model_info._get_file_list_for_results_name(extension='mpco', verbose=False)
        self.cdata_partitions=self.model_info._get_file_list_for_results_name(extension='cdata', verbose=False)
        
        # Extract the model stages information
        self.model_stages= self.model_info._get_model_stages()
        
        # Get model results names for nodes
        self.node_results_names=self.model_info._get_node_results_names()
        
        # Get model results names, types, and unique names for the elements
        self.element_results_names= self.model_info._get_elements_results_names()
        self.element_types= self.model_info._get_element_types()
        self.unique_element_types=self.model_info._get_all_types()
        
        # Extract the model time information
        self.time=self.model_info._get_time_series()
        
        # Get node and element information
        # In order to query the data efficiently, we will store the mappings in a structured numpy array and df
        # Usually the size of this arrays is not too big, so we can store them in memory, the method contain a print_memory=True statement to check the size of the arrays
        self.nodes_info=self.nodes._get_all_nodes_ids(verbose=True)
        self.elements_info=self.elements._get_all_element_index(verbose=True)
        
        # Extract the model steps information
        self.number_of_steps=self.model_info._get_number_of_steps()
        
        # Get the selection set mapping (This is the only place cdata is used)
        self.selection_set=self.cdata._extract_selection_set_ids()
        
        if self.verbose:
            self.print_summary()
        
    def print_summary(self):
        """
        Print a summary of the virtual dataset.
        ---------------------------------------
        """
        print(f'File name: {self.recorder_name}')
        print(f'Number of partitions: {len(self.results_partitions)}')
        
        print('------------------------------------------------------')
        
        print(f"Number of model stages: {len(self.model_stages)}")
        print(f'Model stages: {self.model_stages}')
        for stage in self.model_stages:
            print(f"  - {stage}")
            
        print('------------------------------------------------------')
        print(f'Number of nodal results: {len(self.node_results_names)}')
        for name in self.node_results_names:
            print(f"  - {name}")
            
        print('------------------------------------------------------')
        print(f'Number of element results: {len(self.element_results_names)}')
        for name in self.element_results_names:
            print(f"  - {name}")
        print(f'Number of unique element types: {len(self.unique_element_types)}')
        for name in self.unique_element_types:
            print(f"  - {name}")
        
        print('------------------------------------------------------')
        print('General model information:')
        
        print(f"Number of nodes: {len(self.nodes_info)}")
        print(f"Number of element types: {len(self.unique_element_types)}")
        print(f"Number of elements: {len(self.elements_info)}")
        print(f"Number of steps: {self.number_of_steps}")
        print(f"Number of selection sets: {len(self.selection_set)}")

    def print_selection_set_info(self):
        """Method to print the selection set information.
        """
        
        for key in self.selection_set.keys():
            print(f"Selection set: {key}")
            print(f"Selection Set name: {self.selection_set[key]['SET_NAME']}")
            print('------------------------------------------------------')
    
    def print_model_stages(self):
        """Method to print the model stages information.
        """
        
        print(f"Number of model stages: {len(self.model_stages)}")
        for stage in self.model_stages:
            print(f"  - {stage}")
            
    def print_nodal_results(self):
        """Method to print the nodal results information.
        """
        
        print(f"Number of nodal results: {len(self.node_results_names)}")
        for name in self.node_results_names:
            print(f"  - {name}")
            
    def print_element_results(self):
        """Method to print the element results information.
        """
        
        print(f"Number of element results: {len(self.element_results_names)}")
        for name in self.element_results_names:
            print(f"  - {name}")
            
    def print_element_types(self):
        """Method to print the element types information.
        """
        
        print(f"Number of unique element types: {len(self.element_types['unique_element_types'])}")
        for result, types in self.element_types['element_types_dict'].items():
            print(f"  - {result}")
            for type in types:
                print(f"    - {type}")
                        
    def print_unique_element_types(self):
        """Method to print the unique element types information.
        """
        
        print(f"Number of unique element types: {len(self.unique_element_types)}")
        for name in self.unique_element_types:
            print(f"  - {name}")
        
        