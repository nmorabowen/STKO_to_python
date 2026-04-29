"""Hard backward-compat contract: every public class documented in the
refactor proposal (§ non-negotiable #4) must stay importable from its
original path.

If any of these imports fail, the refactor has broken the public API and
downstream notebooks/scripts will break. Do not "fix" these tests by
changing the import targets — fix the package so the imports work.
"""
from __future__ import annotations

import importlib

import pytest


# (import_path, attribute_name) tuples. The attribute must resolve to a
# class (or object) after importing the module.
PUBLIC_SYMBOLS = [
    # Top-level re-exports — see src/STKO_to_python/__init__.py
    ("STKO_to_python", "MPCODataSet"),
    ("STKO_to_python", "HDF5Utils"),
    ("STKO_to_python", "ModelInfo"),
    ("STKO_to_python", "CData"),
    ("STKO_to_python", "Nodes"),
    ("STKO_to_python", "Elements"),
    ("STKO_to_python", "ElementResults"),
    ("STKO_to_python", "Plot"),
    ("STKO_to_python", "ModelPlotSettings"),
    ("STKO_to_python", "Aggregator"),
    ("STKO_to_python", "StrOp"),
    ("STKO_to_python", "H5RepairTool"),
    ("STKO_to_python", "AttrDict"),
    ("STKO_to_python", "NodalResults"),
    ("STKO_to_python", "NodalResultsPlotter"),
    ("STKO_to_python", "MPCOResults"),
    ("STKO_to_python", "MPCO_df"),
    # Deep imports that existing notebooks use.
    ("STKO_to_python.core.dataset", "MPCODataSet"),
    ("STKO_to_python.io.hdf5_utils", "HDF5Utils"),
    ("STKO_to_python.nodes.nodes", "Nodes"),
    ("STKO_to_python.elements.elements", "Elements"),
    ("STKO_to_python.elements.element_results", "ElementResults"),
    ("STKO_to_python.model.model_info", "ModelInfo"),
    ("STKO_to_python.model.cdata", "CData"),
    ("STKO_to_python.plotting.plot", "Plot"),
    ("STKO_to_python.utilities.attribute_dictionary_class", "AttrDict"),
    ("STKO_to_python.results.nodal_results_dataclass", "NodalResults"),
    ("STKO_to_python.results.nodal_results_plotting", "NodalResultsPlotter"),
    # NOTE: deprecated deep paths (e.g. STKO_to_python.plotting.plot_dataclasses)
    # are exercised below with explicit DeprecationWarning expectations.
]


@pytest.mark.parametrize(("module_path", "attr"), PUBLIC_SYMBOLS)
def test_public_symbol_importable(module_path: str, attr: str) -> None:
    """Every documented public symbol resolves from its documented path."""
    module = importlib.import_module(module_path)
    assert hasattr(module, attr), (
        f"Public symbol {module_path}.{attr} is missing — this breaks "
        f"backward compatibility. See refactor proposal §non-negotiable #4."
    )


def test_package_all_contract() -> None:
    """``STKO_to_python.__all__`` lists every top-level re-export."""
    import STKO_to_python as pkg

    required = {
        "MPCODataSet", "HDF5Utils", "ModelInfo", "CData",
        "Nodes", "Elements", "ElementResults",
        "Plot", "Aggregator", "StrOp", "H5RepairTool", "AttrDict",
        "NodalResults", "NodalResultsPlotter", "MPCOResults", "MPCO_df",
    }
    declared = set(pkg.__all__)
    missing = required - declared
    assert not missing, f"Public __all__ is missing symbols: {missing}"


DEPRECATED_DEEP_IMPORTS = [
    # (module_path, attr, canonical_module, canonical_attr)
    (
        "STKO_to_python.core.dataclasses",
        "MetaData",
        "STKO_to_python.core.metadata",
        "ModelMetadata",
    ),
    (
        "STKO_to_python.plotting.plot_dataclasses",
        "ModelPlotSettings",
        "STKO_to_python.plotting.plot_settings",
        "PlotSettings",
    ),
]


@pytest.mark.parametrize(
    ("module_path", "attr", "canonical_module", "canonical_attr"),
    DEPRECATED_DEEP_IMPORTS,
)
def test_deprecated_deep_path_still_resolves(
    module_path: str,
    attr: str,
    canonical_module: str,
    canonical_attr: str,
) -> None:
    """Deprecated deep paths must keep resolving (hard-compat) but emit
    a ``DeprecationWarning`` and resolve to the same object as the
    canonical path."""
    module = importlib.import_module(module_path)
    with pytest.warns(DeprecationWarning, match=attr):
        legacy = getattr(module, attr)
    canonical = getattr(importlib.import_module(canonical_module), canonical_attr)
    assert legacy is canonical


def test_pickle_module_qualname_pins() -> None:
    """NodalResults' (module, qualname) must not drift — pickles depend on it.

    Refactor proposal §non-negotiable #6 (pickle stability): the pickle
    unpickler looks up ``<module>.<qualname>``. If either changes, every
    pickle ever produced by the library fails to load.
    """
    from STKO_to_python.results.nodal_results_dataclass import NodalResults

    assert NodalResults.__module__ == "STKO_to_python.results.nodal_results_dataclass"
    assert NodalResults.__qualname__ == "NodalResults"
