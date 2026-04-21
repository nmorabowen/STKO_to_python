"""Placeholder ``PlotNodes`` — Phase 4.4.2.

Every prior method on this class (``plot_nodal_results``,
``plot_time_history``, ``plot_roof_drift``, ``plot_story_drifts``,
``plot_drift_profile``, ``plot_orbit``) was non-functional against the
current data model — they either called manager methods that never
existed (``get_time_history``, ``get_roof_drift``, ``get_story_drifts``,
``get_nodes_in_selection_set``) or assumed ``NodeManager.get_nodal_results``
returned a raw DataFrame (it has returned a ``NodalResults`` view since
Phase 2).

The bodies are removed rather than migrated because there is nothing
working to preserve under the "hard back-compat" rule. The class itself
is retained as a stub so ``Plot.nodes`` remains importable; a full
removal lands in Phase 4.4.3 together with a dataset-level convenience
wrapper on the ``Plot`` facade (Phase 4.4.4).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


class PlotNodes:
    """Node-level plotting helpers (stub).

    Intentionally empty in the current release — all prior methods
    were dead code. Use ``NodalResultsPlotter`` (via
    ``NodalResults.plot``) for per-result plotting, and the dataset-level
    convenience on ``Plot`` (added in Phase 4.4.4) for one-shot fetch+plot.
    """

    def __init__(self, dataset: "MPCODataSet"):
        self.dataset = dataset
