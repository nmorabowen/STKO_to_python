"""Back-compat shim — emits ``DeprecationWarning`` when ``Nodes`` is accessed.

The canonical class lives at :mod:`STKO_to_python.nodes.node_manager`
as ``NodeManager``. ``Nodes`` is preserved as a back-compat name on
this module and (without warning) on the top-level
``STKO_to_python.nodes`` package, so

    >>> from STKO_to_python import Nodes               # quiet
    >>> from STKO_to_python.nodes.nodes import Nodes   # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.nodes.node_manager import NodeManager
    >>> # or, equivalently:
    >>> from STKO_to_python.nodes import NodeManager

The lookup is implemented via the PEP 562 module ``__getattr__`` so
plain ``import STKO_to_python.nodes.nodes`` remains silent — only the
attribute access fires the warning.
"""
from __future__ import annotations

import warnings
from typing import Any

from .node_manager import NodeManager


def __getattr__(name: str) -> Any:
    if name == "Nodes":
        warnings.warn(
            "`STKO_to_python.nodes.nodes.Nodes` is deprecated; "
            "import `NodeManager` from "
            "`STKO_to_python.nodes.node_manager` instead (or use the "
            "top-level `STKO_to_python.Nodes`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return NodeManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Nodes"]
