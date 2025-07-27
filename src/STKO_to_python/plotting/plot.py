# ── src/STKO_to_python/plot/__init__.py ─────────────────────────────────
from __future__ import annotations
from typing import TYPE_CHECKING

from .plot_nodes import PlotNodes
# from .plot_elements import PlotElements  # ← when ready

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


class Plot:
    """
    Composite façade that groups plotting helpers under one roof.

    Examples
    --------
    >>> model.plot.nodes.plot_nodal_results(...)
    >>> model.plot.plot_nodal_results(...)   # tunneled for back-compat
    """

    # ------------------------------------------------------------------ #
    def __init__(self, dataset: "MPCODataSet") -> None:
        self._dataset: MPCODataSet = dataset

        # private storage for helpers
        self._nodes = PlotNodes(dataset)
        # self._elems = PlotElements(dataset)

    # ------------------------------------------------------------------ #
    # public handles (discoverable via tab-completion) ------------------ #
    @property
    def nodes(self) -> PlotNodes:
        """Node-related plotting utilities."""
        return self._nodes

    # @property
    # def elements(self) -> PlotElements:
    #     return self._elems

    # ------------------------------------------------------------------ #
    # attribute tunnelling (legacy support) ----------------------------- #
    def __getattr__(self, name: str):
        """
        Delegate unknown attributes to sub-helpers so old code like
        `model.plot.plot_nodal_results()` keeps working.
        """
        for helper in (self._nodes,):          # add self._elems later
            if hasattr(helper, name):
                return getattr(helper, name)
        raise AttributeError(name)

    # nice console repr -------------------------------------------------- #
    def __repr__(self) -> str:                 # pragma: no cover
        subs = ["nodes"]                       # add "elements" when ready
        return f"<Plot helpers: {', '.join(subs)}>"

# convenience export
__all__ = ["Plot"]
