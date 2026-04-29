"""Back-compat shim — emits ``DeprecationWarning`` when ``ModelInfo`` is accessed.

The canonical class lives at :mod:`STKO_to_python.model.model_info_reader`
as ``ModelInfoReader``. ``ModelInfo`` is preserved as a back-compat
name on this module and (without warning) on the top-level
``STKO_to_python.model`` package, so

    >>> from STKO_to_python import ModelInfo                  # quiet
    >>> from STKO_to_python.model.model_info import ModelInfo # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.model.model_info_reader import ModelInfoReader
    >>> # or, equivalently:
    >>> from STKO_to_python.model import ModelInfoReader
"""
from __future__ import annotations

import warnings
from typing import Any

from .model_info_reader import ModelInfoReader


def __getattr__(name: str) -> Any:
    if name == "ModelInfo":
        warnings.warn(
            "`STKO_to_python.model.model_info.ModelInfo` is deprecated; "
            "import `ModelInfoReader` from "
            "`STKO_to_python.model.model_info_reader` instead (or use "
            "the top-level `STKO_to_python.ModelInfo`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return ModelInfoReader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelInfo"]
