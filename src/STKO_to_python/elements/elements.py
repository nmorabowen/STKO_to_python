"""Back-compat shim — emits ``DeprecationWarning`` when ``Elements`` is accessed.

The canonical class lives at :mod:`STKO_to_python.elements.element_manager`
as ``ElementManager``. ``Elements`` is preserved as a back-compat name on
this module and (without warning) on the top-level
``STKO_to_python.elements`` package, so

    >>> from STKO_to_python import Elements                     # quiet
    >>> from STKO_to_python.elements.elements import Elements   # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.elements.element_manager import ElementManager
    >>> # or, equivalently:
    >>> from STKO_to_python.elements import ElementManager

The lookup is implemented via the PEP 562 module ``__getattr__``.
"""
from __future__ import annotations

import warnings
from typing import Any

from .element_manager import ElementManager


def __getattr__(name: str) -> Any:
    if name == "Elements":
        warnings.warn(
            "`STKO_to_python.elements.elements.Elements` is deprecated; "
            "import `ElementManager` from "
            "`STKO_to_python.elements.element_manager` instead (or use "
            "the top-level `STKO_to_python.Elements`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return ElementManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Elements"]
