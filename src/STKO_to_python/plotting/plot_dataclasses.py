"""Back-compat shim — emits ``DeprecationWarning`` when ``ModelPlotSettings``
is accessed.

The canonical class lives at :mod:`STKO_to_python.plotting.plot_settings`
as ``PlotSettings``. ``ModelPlotSettings`` is preserved as a back-compat
name on this module and (without warning) on the top-level
``STKO_to_python.plotting`` package, so

    >>> from STKO_to_python.plotting import ModelPlotSettings   # quiet
    >>> from STKO_to_python.plotting.plot_dataclasses import ModelPlotSettings
    ...                                                         # DeprecationWarning

both keep working. New code should prefer

    >>> from STKO_to_python.plotting.plot_settings import PlotSettings
    >>> # or, equivalently:
    >>> from STKO_to_python.plotting import PlotSettings

The lookup is implemented via the PEP 562 module ``__getattr__`` so the
warning only fires when ``ModelPlotSettings`` is actually imported from
this specific deep path — plain ``import
STKO_to_python.plotting.plot_dataclasses`` remains silent.
"""
from __future__ import annotations

import warnings
from typing import Any

from .plot_settings import PlotSettings


def __getattr__(name: str) -> Any:
    if name == "ModelPlotSettings":
        warnings.warn(
            "`STKO_to_python.plotting.plot_dataclasses.ModelPlotSettings` "
            "is deprecated; import `PlotSettings` from "
            "`STKO_to_python.plotting.plot_settings` instead (or use the "
            "top-level `STKO_to_python.ModelPlotSettings`).",
            DeprecationWarning,
            stacklevel=2,
        )
        return PlotSettings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelPlotSettings"]
