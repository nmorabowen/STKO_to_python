from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from ..nodes.nodes import Nodes
from ..elements.elements import Elements
from ..model.model_info import ModelInfo
from ..model.cdata import CData
from ..plotting.plot import Plot
from ..io.info import Info
from ..io.partition_pool import Hdf5PartitionPool
from ..io.format_policy import MpcoFormatPolicy
from ..selection import SelectionSetResolver
from .dataclasses import MetaData
from ..plotting.plot_dataclasses import ModelPlotSettings

logger = logging.getLogger(__name__)


class MPCODataSet:
    
    """
    A dataset management class for MPCO (Model, Partition, Component, Operation) simulation files.

    This class serves as the main access point to HDF5-based MPCO data files, providing
    functionality to load, organize, and access simulation data through a composition
    of specialized classes (Nodes, Elements, ModelInfo, CData).

    The class implements a "friend" pattern with its component classes, allowing it
    to access their protected methods which are indicated by a leading underscore.

    Friend-method convention
    ------------------------
    Methods prefixed with a single underscore on the composite classes
    (``Nodes``, ``Elements``, ``ModelInfo``, ``CData``) are of two kinds:

    1. **Manager-facing "friend" methods.** These are called by
       ``MPCODataSet`` (and only by ``MPCODataSet``) to assemble dataset
       attributes during ``_create_object_attributes``. Examples:
       ``ModelInfo._get_file_list_for_results_name``,
       ``ModelInfo._get_model_stages``, ``Nodes._get_all_nodes_ids``,
       ``Elements._get_all_element_index``, ``CData._extract_selection_set_ids``.
       These are intentionally not public, but the dataset is allowed to
       reach in. Renaming them requires updating the dataset too.

    2. **Truly internal helpers.** Everything else starting with ``_`` is
       private to its owning class and must not be called from outside.

    The upcoming refactor (see ``docs/architecture-refactor-proposal.md``)
    replaces this convention with explicit, named collaborator objects
    passed through constructors. Until then, treat category (1) as a
    stable internal contract between ``MPCODataSet`` and its managers.

    Context-manager support
    -----------------------
    ``MPCODataSet`` supports the ``with`` statement. ``__enter__`` returns
    the dataset; ``__exit__`` is a no-op today but will close pooled HDF5
    handles once the partition pool lands in Phase 1 of the refactor.
    Existing code that does not use ``with`` keeps working unchanged.

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
    hdf5_directory : str
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
        name=None, # The model name, if None it will be extracted from the folder name
        file_extension='*.mpco',
        verbose=False,
        plot_settings: Optional[ModelPlotSettings] = None,
        pool_size: Optional[int] = None,
    ):

        self.hdf5_directory = hdf5_directory
        self.recorder_name = recorder_name
        self.name=name
        self.file_extension = file_extension
        self.verbose = verbose
        # pool_size=None → performance-first default of min(16, n_partitions),
        # applied in _create_object_attributes once the partition count is
        # known. Users who need the legacy open-per-call behavior (e.g.
        # another process is writing to the file and they want coherent
        # reads on every query) pass pool_size=0 explicitly.
        self._pool_size: Optional[int] = pool_size

        if verbose:
            logger.setLevel(logging.INFO)

        # Phase 1 Layer 1 collaborators. The pool is constructed with the
        # partition map after `_create_object_attributes` discovers the
        # files; the format policy is stateless and can be built up front.
        self._format_policy = MpcoFormatPolicy()
        self._pool: Optional[Hdf5PartitionPool] = None

        # Initialize the metadata
        self.metadata = MetaData()
        
        # Instanciate the composite classes
        self.nodes=Nodes(self)
        self.elements=Elements(self)
        self.model_info = ModelInfo(self)
        self.cdata=CData(self)
        self.plot=Plot(self)
        self.info=Info(self)
        self.plot_settings = plot_settings or ModelPlotSettings()
        
        # Create the object attributes
        self._create_object_attributes()
        
        # Print welcome message
        self._print_welcome_message()

        
    def _create_object_attributes(self):
        """
        Helper method to create object attributes.
        """
        # Extract the results partition for the given recorder name
        self.results_partitions=self.model_info._get_file_list_for_results_name(extension='mpco', verbose=False)
        self.cdata_partitions=self.model_info._get_file_list_for_results_name(extension='cdata', verbose=False)

        # Build the partition pool now that we know the partition map.
        # Default to min(16, n_partitions) for the performance-first
        # baseline (refactor proposal §6). Every consumer (model_info,
        # nodes, elements) now routes through the pool, so a pooled
        # default is safe; pool_size=0 remains available for callers
        # that want the legacy open-per-call semantics.
        if self._pool_size is None:
            effective_pool_size = min(16, len(self.results_partitions))
        else:
            effective_pool_size = int(self._pool_size)
        self._pool = Hdf5PartitionPool(
            self.results_partitions,
            pool_size=effective_pool_size,
        )
        
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

        # Phase 2: centralized resolver (side-by-side with legacy per-manager
        # helpers; consumers will switch over in a later phase).
        self._selection_resolver = SelectionSetResolver(self.selection_set)

        if self.verbose:
            self.print_summary()
        
    def print_summary(self):
        """
        Emit a summary of the virtual dataset at INFO level on the
        module logger. Enable ``verbose=True`` on construction or
        call ``logging.basicConfig(level=logging.INFO)`` to see output.
        """
        logger.info('File name: %s', self.recorder_name)
        logger.info('Number of partitions: %d', len(self.results_partitions))

        logger.info('------------------------------------------------------')

        logger.info("Number of model stages: %d", len(self.model_stages))
        logger.info('Model stages: %s', self.model_stages)
        for stage in self.model_stages:
            logger.info("  - %s", stage)

        logger.info('------------------------------------------------------')
        logger.info('Number of nodal results: %d', len(self.node_results_names))
        for name in self.node_results_names:
            logger.info("  - %s", name)

        logger.info('------------------------------------------------------')
        logger.info('Number of element results: %d', len(self.element_results_names))
        for name in self.element_results_names:
            logger.info("  - %s", name)
        logger.info('Number of unique element types: %d', len(self.unique_element_types))
        for name in self.unique_element_types:
            logger.info("  - %s", name)

        logger.info('------------------------------------------------------')
        logger.info('General model information:')

        logger.info("Number of nodes: %d", len(self.nodes_info))
        logger.info("Number of element types: %d", len(self.unique_element_types))
        logger.info("Number of elements: %d", len(self.elements_info))
        logger.info("Number of steps: %s", self.number_of_steps)
        logger.info("Number of selection sets: %d", len(self.selection_set))

    def print_selection_set_info(self):
        """Emit selection-set information at INFO level on the module logger."""

        for key in self.selection_set.keys():
            logger.info("Selection set: %s", key)
            logger.info("Selection Set name: %s", self.selection_set[key]['SET_NAME'])
            logger.info('------------------------------------------------------')

    def print_model_stages(self):
        """Emit model-stages information at INFO level on the module logger."""

        logger.info("Number of model stages: %d", len(self.model_stages))
        for stage in self.model_stages:
            logger.info("  - %s", stage)

    def print_nodal_results(self):
        """Emit nodal-results information at INFO level on the module logger."""

        logger.info("Number of nodal results: %d", len(self.node_results_names))
        for name in self.node_results_names:
            logger.info("  - %s", name)

    def print_element_results(self):
        """Emit element-results information at INFO level on the module logger."""

        logger.info("Number of element results: %d", len(self.element_results_names))
        for name in self.element_results_names:
            logger.info("  - %s", name)

    def print_element_types(self):
        """Emit element-types information at INFO level on the module logger."""

        logger.info(
            "Number of unique element types: %d",
            len(self.element_types['unique_element_types']),
        )
        for result, types in self.element_types['element_types_dict'].items():
            logger.info("  - %s", result)
            for type in types:
                logger.info("    - %s", type)

    def print_unique_element_types(self):
        """Emit unique element-types information at INFO level on the module logger."""

        logger.info("Number of unique element types: %d", len(self.unique_element_types))
        for name in self.unique_element_types:
            logger.info("  - %s", name)

    def _print_welcome_message(self):

        text=f"""
            ============================================================================
            Hola Ladruños!
            Working on file:{self.hdf5_directory}
            Model Name: {self.name if self.name else 'Not specified'}
            ============================================================================
            """

        logger.info(text)
    
    def __str__(self) -> str:
        return f"{self.name or Path(self.hdf5_directory).name} ({self.recorder_name})"

    def __repr__(self) -> str:
        return (f"MPCODataSet(hdf5_directory={str(self.hdf5_directory)!r}, "
                f"recorder_name={self.recorder_name!r}, "
                f"name={self.name!r}, "
                f"file_extension={self.file_extension!r}, "
                f"verbose={self.verbose})")

    def __enter__(self) -> "MPCODataSet":
        """Enter the runtime context; returns ``self``.

        Phase 0 stub: no resources are held yet. When the partition pool
        lands in Phase 1, ``__exit__`` will close pooled HDF5 handles.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the runtime context.

        Closes every handle held by the partition pool. Safe to call
        even when ``pool_size=0`` (the legacy default) — the pool holds
        no handles in that configuration, so ``close_all()`` is a no-op.
        Returning ``None`` (implicitly) means exceptions raised inside
        the ``with`` block propagate unchanged.
        """
        pool = getattr(self, "_pool", None)
        if pool is not None:
            pool.close_all()
        return None
        
        