from .core.dataset import MPCODataSet

from .io.hdf5_utils import HDF5Utils

from .nodes.node_manager import NodeManager
from .elements.element_manager import ElementManager
from .elements.element_results import ElementResults
from .model.model_info_reader import ModelInfoReader
from .model.cdata_reader import CDataReader

from .plotting.plot import Plot
from .plotting.plot_settings import PlotSettings

# Back-compat aliases preserved on the top-level package surface
# (quiet); each deprecated deep path emits a ``DeprecationWarning`` via
# PEP 562 ``__getattr__`` in its respective shim.
Nodes = NodeManager
Elements = ElementManager
ModelInfo = ModelInfoReader
CData = CDataReader
ModelPlotSettings = PlotSettings

from .dataprocess import Aggregator, StrOp

from.utilities import H5RepairTool
from .utilities.attribute_dictionary_class import AttrDict

from .results.nodal_results_dataclass import NodalResults
from .results.nodal_results_plotting import NodalResultsPlotter

from .MPCOList import MPCOResults
from .MPCOList import MPCO_df

__all__ = [
    "MPCODataSet",
    "HDF5Utils",
    "ModelInfo",
    "CData",
    "Nodes",
    "Elements",
    "ElementResults",
    "Plot",
    "Aggregator",
    "StrOp",
    "H5RepairTool",
    "AttrDict",
    "NodalResults",
    "NodalResultsPlotter",
    "MPCOResults",
    "MPCO_df"
]