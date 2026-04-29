"""Back-compat shim — emits ``DeprecationWarning`` when ``MetaData`` is accessed.

The canonical class lives at :mod:`STKO_to_python.core.metadata` as
``ModelMetadata``. ``MetaData`` is preserved as a back-compat name on
this module and (without warning) on the top-level package, so

    >>> from STKO_to_python import MetaData            # quiet
    >>> from STKO_to_python.core.dataclasses import MetaData   # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.core.metadata import ModelMetadata
    >>> # or, equivalently, the top-level export:
    >>> from STKO_to_python import ModelMetadata

The lookup is implemented via the PEP 562 module ``__getattr__`` so the
warning only fires when ``MetaData`` is actually imported from this
specific deep path — plain ``import STKO_to_python.core.dataclasses``
remains silent.
"""
from __future__ import annotations

import warnings
from typing import Any

from .metadata import ModelMetadata


def __getattr__(name: str) -> Any:
    if name == "MetaData":
        warnings.warn(
            "`STKO_to_python.core.dataclasses.MetaData` is deprecated; "
            "import `ModelMetadata` from `STKO_to_python.core.metadata` "
            "instead (or use the top-level `STKO_to_python.ModelMetadata`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return ModelMetadata
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MetaData"]
