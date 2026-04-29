from .plot import Plot
from .plot_settings import PlotSettings

# Back-compat alias preserved on the package surface (quiet); the deep
# path ``STKO_to_python.plotting.plot_dataclasses.ModelPlotSettings``
# emits a ``DeprecationWarning`` via PEP 562 ``__getattr__``.
ModelPlotSettings = PlotSettings

__all__=[
    'Plot',
    'ModelPlotSettings',
    'PlotSettings',
]