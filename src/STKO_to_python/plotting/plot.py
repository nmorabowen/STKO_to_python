"""Dataset-level plotting facade.

Per-result plotting lives on ``NodalResults.plot`` (a
``NodalResultsPlotter``). ``Plot`` is a thin dataset-bound facade for
one-shot "fetch + plot" convenience. Phase 4.4.4 adds a working
``xy(...)`` method here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.dataset import MPCODataSet


class Plot:
    """Dataset-level plotting facade.

    Held on ``MPCODataSet.plot``. Kept minimal: per-result plotting is
    accessed through ``NodalResults.plot``; this class is reserved for
    convenience methods that fetch a result and delegate.
    """

    def __init__(self, dataset: "MPCODataSet") -> None:
        self._dataset = dataset

    def __repr__(self) -> str:
        return f"<Plot facade for {type(self._dataset).__name__}>"


__all__ = ["Plot"]
