"""Back-compat shim — re-exports ``PlotSettings`` as ``ModelPlotSettings``.

The canonical class now lives in :mod:`STKO_to_python.plotting.plot_settings`.
``ModelPlotSettings`` is an alias; both names resolve to the **same
class object**:

    >>> from STKO_to_python.plotting.plot_dataclasses import ModelPlotSettings
    >>> from STKO_to_python.plotting.plot_settings import PlotSettings
    >>> ModelPlotSettings is PlotSettings
    True
"""
from __future__ import annotations

from .plot_settings import PlotSettings as ModelPlotSettings

__all__ = ["ModelPlotSettings"]
