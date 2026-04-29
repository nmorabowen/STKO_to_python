"""Back-compat shim — emits ``DeprecationWarning`` on attribute access.

Shape functions, Jacobian primitives, and the natural-to-physical
coordinate mapping moved to
:mod:`STKO_to_python.format.shape_functions` as part of the
architecture refactor. The legacy import path

    from STKO_to_python.utilities.shape_functions import compute_physical_coords

continues to work but now emits a ``DeprecationWarning`` per access via
PEP 562 module-level ``__getattr__``. New code should prefer

    from STKO_to_python.format.shape_functions import compute_physical_coords
    # or, equivalently:
    from STKO_to_python.format import compute_physical_coords
"""
from __future__ import annotations

import warnings
from typing import Any

from ..format import shape_functions as _canonical


_REEXPORTED_NAMES = frozenset(_canonical.__all__)


def __getattr__(name: str) -> Any:
    if name in _REEXPORTED_NAMES:
        warnings.warn(
            f"`STKO_to_python.utilities.shape_functions.{name}` is deprecated; "
            f"import from `STKO_to_python.format.shape_functions` (or "
            f"`STKO_to_python.format`) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return getattr(_canonical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_REEXPORTED_NAMES)
