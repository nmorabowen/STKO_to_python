"""Phase 3.1 + Group-B contract: canonical manager names live at their
own modules; legacy names remain importable as aliases.

After the Group-B refactor:
- The canonical class lives at e.g. ``STKO_to_python.nodes.node_manager``.
- The package surface (``STKO_to_python.nodes``) exposes both the new
  and the legacy names quietly.
- The legacy *deep* path (``STKO_to_python.nodes.nodes``) keeps
  resolving the legacy name but emits a ``DeprecationWarning`` via
  PEP 562 ``__getattr__``.
- ``isinstance`` and pickle compatibility require the alias to be the
  same class object as the canonical class — verified here.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------- #
# NodeManager / Nodes
# ---------------------------------------------------------------------- #
def test_node_manager_importable_from_canonical_module():
    from STKO_to_python.nodes.node_manager import NodeManager
    assert NodeManager.__name__ == "NodeManager"


def test_node_manager_importable_from_legacy_module_quietly():
    """The shim re-exports ``NodeManager`` directly so the canonical
    name resolves without any warning even from the deprecated path."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.nodes.nodes import NodeManager
    assert NodeManager.__name__ == "NodeManager"


def test_legacy_nodes_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="Nodes.*deprecated"):
        from STKO_to_python.nodes.nodes import Nodes
    from STKO_to_python.nodes.node_manager import NodeManager
    assert Nodes is NodeManager


def test_node_manager_importable_from_package_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.nodes import NodeManager, Nodes
    assert Nodes is NodeManager


# ---------------------------------------------------------------------- #
# ElementManager / Elements
# ---------------------------------------------------------------------- #
def test_element_manager_importable_from_canonical_module():
    from STKO_to_python.elements.element_manager import ElementManager
    assert ElementManager.__name__ == "ElementManager"


def test_element_manager_importable_from_legacy_module_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.elements.elements import ElementManager
    assert ElementManager.__name__ == "ElementManager"


def test_legacy_elements_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="Elements.*deprecated"):
        from STKO_to_python.elements.elements import Elements
    from STKO_to_python.elements.element_manager import ElementManager
    assert Elements is ElementManager


def test_element_manager_importable_from_package_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.elements import ElementManager, Elements
    assert Elements is ElementManager


# ---------------------------------------------------------------------- #
# Top-level package
# ---------------------------------------------------------------------- #
def test_top_level_names_still_work_quietly():
    """Top-level re-exports under ``STKO_to_python`` expose the legacy
    names without any deprecation warning."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python import Nodes, Elements
        from STKO_to_python.nodes.node_manager import NodeManager
        from STKO_to_python.elements.element_manager import ElementManager
    assert Nodes is NodeManager
    assert Elements is ElementManager


# ---------------------------------------------------------------------- #
# Layer 3 readers (Phase 3.2)
# ---------------------------------------------------------------------- #
def test_model_info_reader_importable_from_canonical_module():
    from STKO_to_python.model.model_info_reader import ModelInfoReader
    assert ModelInfoReader.__name__ == "ModelInfoReader"


def test_model_info_reader_importable_from_legacy_module_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.model.model_info import ModelInfoReader
    assert ModelInfoReader.__name__ == "ModelInfoReader"


def test_legacy_model_info_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="ModelInfo.*deprecated"):
        from STKO_to_python.model.model_info import ModelInfo
    from STKO_to_python.model.model_info_reader import ModelInfoReader
    assert ModelInfo is ModelInfoReader


def test_model_info_reader_package_import_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.model import ModelInfo, ModelInfoReader
    assert ModelInfo is ModelInfoReader


def test_cdata_reader_importable_from_canonical_module():
    from STKO_to_python.model.cdata_reader import CDataReader
    assert CDataReader.__name__ == "CDataReader"


def test_cdata_reader_importable_from_legacy_module_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.model.cdata import CDataReader
    assert CDataReader.__name__ == "CDataReader"


def test_legacy_cdata_alias_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="CData.*deprecated"):
        from STKO_to_python.model.cdata import CData
    from STKO_to_python.model.cdata_reader import CDataReader
    assert CData is CDataReader


def test_cdata_reader_package_import_quietly():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        from STKO_to_python.model import CData, CDataReader
    assert CData is CDataReader
