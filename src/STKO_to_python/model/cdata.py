"""Back-compat shim — emits ``DeprecationWarning`` when ``CData`` is accessed.

The canonical class lives at :mod:`STKO_to_python.model.cdata_reader`
as ``CDataReader``. ``CData`` is preserved as a back-compat name on
this module and (without warning) on the top-level
``STKO_to_python.model`` package, so

    >>> from STKO_to_python import CData               # quiet
    >>> from STKO_to_python.model.cdata import CData   # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.model.cdata_reader import CDataReader
    >>> # or, equivalently:
    >>> from STKO_to_python.model import CDataReader
"""
from __future__ import annotations

import warnings
from typing import Any

from .cdata_reader import CDataReader


def __getattr__(name: str) -> Any:
    if name == "CData":
        warnings.warn(
            "`STKO_to_python.model.cdata.CData` is deprecated; "
            "import `CDataReader` from "
            "`STKO_to_python.model.cdata_reader` instead (or use the "
            "top-level `STKO_to_python.CData`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return CDataReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CData"]
