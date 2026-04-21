"""Phase 3.1 contract: canonical manager names exist; legacy names are aliases.

The refactor proposal introduces ``NodeManager`` and ``ElementManager``
as the canonical Layer 3 class names. The legacy names ``Nodes`` and
``Elements`` must remain importable from their current paths as aliases
of the canonical classes (not separate subclasses) — pickle compat and
``isinstance`` checks depend on the alias being the same class object.
"""
from __future__ import annotations


def test_node_manager_is_importable_from_module():
    from STKO_to_python.nodes.nodes import NodeManager
    assert NodeManager.__name__ == "NodeManager"


def test_legacy_nodes_is_alias_of_node_manager():
    from STKO_to_python.nodes.nodes import NodeManager, Nodes
    assert Nodes is NodeManager


def test_node_manager_importable_from_package():
    from STKO_to_python.nodes import NodeManager, Nodes
    assert Nodes is NodeManager


def test_element_manager_is_importable_from_module():
    from STKO_to_python.elements.elements import ElementManager
    assert ElementManager.__name__ == "ElementManager"


def test_legacy_elements_is_alias_of_element_manager():
    from STKO_to_python.elements.elements import ElementManager, Elements
    assert Elements is ElementManager


def test_element_manager_importable_from_package():
    from STKO_to_python.elements import ElementManager, Elements
    assert Elements is ElementManager


def test_top_level_names_still_work():
    """Top-level re-exports under ``STKO_to_python`` must still expose
    the legacy names."""
    from STKO_to_python import Nodes, Elements
    from STKO_to_python.nodes.nodes import NodeManager
    from STKO_to_python.elements.elements import ElementManager
    assert Nodes is NodeManager
    assert Elements is ElementManager


# ---------------------------------------------------------------------- #
# Layer 3 readers (Phase 3.2)
# ---------------------------------------------------------------------- #
def test_model_info_reader_is_importable():
    from STKO_to_python.model.model_info import ModelInfoReader
    assert ModelInfoReader.__name__ == "ModelInfoReader"


def test_legacy_model_info_is_alias():
    from STKO_to_python.model.model_info import ModelInfo, ModelInfoReader
    assert ModelInfo is ModelInfoReader


def test_model_info_reader_package_import():
    from STKO_to_python.model import ModelInfo, ModelInfoReader
    assert ModelInfo is ModelInfoReader


def test_cdata_reader_is_importable():
    from STKO_to_python.model.cdata import CDataReader
    assert CDataReader.__name__ == "CDataReader"


def test_legacy_cdata_is_alias():
    from STKO_to_python.model.cdata import CData, CDataReader
    assert CData is CDataReader


def test_cdata_reader_package_import():
    from STKO_to_python.model import CData, CDataReader
    assert CData is CDataReader
