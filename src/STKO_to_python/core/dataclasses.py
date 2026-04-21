"""Back-compat shim — re-exports ``ModelMetadata`` as ``MetaData``.

The canonical class now lives in :mod:`STKO_to_python.core.metadata`.
``MetaData`` is an alias so existing callers continue to work
unchanged. Both names resolve to the *same* class object:

    >>> from STKO_to_python.core.dataclasses import MetaData
    >>> from STKO_to_python.core.metadata import ModelMetadata
    >>> MetaData is ModelMetadata
    True

A future phase may emit a ``DeprecationWarning`` on import; for now the
shim is silent so the strict-warning filter under
``pyproject.toml[tool.pytest]`` stays happy.
"""
from __future__ import annotations

from .metadata import ModelMetadata as MetaData

__all__ = ["MetaData"]
