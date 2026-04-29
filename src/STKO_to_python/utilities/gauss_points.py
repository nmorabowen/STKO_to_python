"""Back-compat shim — emits ``DeprecationWarning`` on attribute access.

The Gauss-point catalog and quadrature primitives moved to
:mod:`STKO_to_python.format.gauss_points` as part of the architecture
refactor. The legacy import path

    from STKO_to_python.utilities.gauss_points import get_ip_layout

continues to work but now emits a ``DeprecationWarning`` per access via
PEP 562 module-level ``__getattr__``. New code should prefer

    from STKO_to_python.format.gauss_points import get_ip_layout
    # or, equivalently:
    from STKO_to_python.format import get_ip_layout
"""
from __future__ import annotations

import warnings
from typing import Any

from ..format import gauss_points as _canonical


_REEXPORTED_NAMES = frozenset(_canonical.__all__)


def __getattr__(name: str) -> Any:
    if name in _REEXPORTED_NAMES:
        warnings.warn(
            f"`STKO_to_python.utilities.gauss_points.{name}` is deprecated; "
            f"import from `STKO_to_python.format.gauss_points` (or "
            f"`STKO_to_python.format`) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_canonical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_REEXPORTED_NAMES)
